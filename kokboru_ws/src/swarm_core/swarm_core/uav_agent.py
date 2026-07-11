#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import math
import time
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                          VehicleOdometry, VehicleCommand,
                          VehicleGlobalPosition)

EARTH_RADIUS = 6371000.0

def project_to_ned(lat, lon, lat0, lon0):
    """(lat, lon) derecesini, (lat0, lon0) referansına göre lokal NED (north, east) metreye çevirir."""
    lat_r,  lon_r  = math.radians(lat),  math.radians(lon)
    lat0_r, lon0_r = math.radians(lat0), math.radians(lon0)

    sin_lat,  cos_lat  = math.sin(lat_r),  math.cos(lat_r)
    sin_lat0, cos_lat0 = math.sin(lat0_r), math.cos(lat0_r)
    cos_dlon = math.cos(lon_r - lon0_r)

    arg = max(-1.0, min(1.0, sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon))
    c = math.acos(arg)
    k = 1.0 if abs(c) < 1e-9 else c / math.sin(c)

    north = k * (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon) * EARTH_RADIUS
    east  = k * cos_lat * math.sin(lon_r - lon0_r) * EARTH_RADIUS
    return north, east

class UAVAgent(Node):
    def __init__(self):
        super().__init__('uav_agent')
        
        self.declare_parameter('agent_id', 'uav_1')
        self.agent_id = self.get_parameter('agent_id').get_parameter_value().string_value
        self.get_logger().info(f'UAV Agent {self.agent_id} başlatıldı.')

        # --- ORTAK ÇERÇEVE REFERANSI ---
        # TÜM ajanlarda AYNI olmalı! Default = PX4 SITL'in varsayılan home'u
        # (Gazebo dünya origin'i tam bu noktaya denk gelir)
        self.declare_parameter('ref_lat', 47.397742)
        self.declare_parameter('ref_lon', 8.545594)
        self.declare_parameter('align_samples', 100)   # ~2 sn @ 50Hz

        self.ref_lat = self.get_parameter('ref_lat').get_parameter_value().double_value
        self.ref_lon = self.get_parameter('ref_lon').get_parameter_value().double_value
        self.align_samples = self.get_parameter('align_samples').get_parameter_value().integer_value

        self.latest_global = None     # (north_gps, east_gps)
        self.offset_n = 0.0
        self.offset_e = 0.0
        self._align_buf = []
        self.frame_aligned = False

        # --- SÜREKLİ GPS (CONTINUOUS RTK) ÇAPASI ---
        self.anchor_lat = 47.397742
        self.anchor_lon = 8.545594
        self.global_x = None
        self.global_y = None

        # ROS 2 Abonelikleri ve Yayıncıları
        self.gps_sub = self.create_subscription(
            VehicleGlobalPosition,
            f'/{self.agent_id}/fmu/out/vehicle_global_position',
            self.gps_callback,
            qos_profile_sensor_data
        )

        self.pose_pub = self.create_publisher(PoseStamped, f'/{self.agent_id}/ap/pose/filtered', 10)
        
        self.target_sub = self.create_subscription(
            TwistStamped, f'/{self.agent_id}/target_velocity', self.coordinator_velocity_callback, 10
        )

        self.odom_sub = self.create_subscription(
            VehicleOdometry, f'/{self.agent_id}/fmu/out/vehicle_odometry', self.odometry_callback, qos_profile_sensor_data
        )

        self.arm_sub = self.create_subscription(
            String, f'/{self.agent_id}/arm_cmd', self.arm_command_callback, 10
        )

        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, f'/{self.agent_id}/fmu/in/offboard_control_mode', 10
        )

        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, f'/{self.agent_id}/fmu/in/trajectory_setpoint', 10
        )

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, f'/{self.agent_id}/fmu/in/vehicle_command', 10
        )

        self.global_pos_sub = self.create_subscription(
            VehicleGlobalPosition,
            f'/{self.agent_id}/fmu/out/vehicle_global_position',
            self.global_position_callback,
            qos_profile_sensor_data
        )
        
        self.timer = self.create_timer(0.1, self.publish_offboard_heartbeat)

        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vz = 0.0
        self.target_yaw = 0.0
        

    def global_position_callback(self, msg: VehicleGlobalPosition):
        if not math.isfinite(msg.lat) or not math.isfinite(msg.lon):
            return
        self.latest_global = project_to_ned(msg.lat, msg.lon, self.ref_lat, self.ref_lon)

    def gps_callback(self, msg: VehicleGlobalPosition):
        if msg.lat == 0.0 or msg.lon == 0.0:
            return

        # SÜREKLİ GPS -> METRE DÖNÜŞÜMÜ (EKF2 Sıçramalarından Etkilenmez)
        delta_lat = msg.lat - self.anchor_lat
        delta_lon = msg.lon - self.anchor_lon

        self.global_x = delta_lat * 111320.0
        self.global_y = delta_lon * 111320.0 * math.cos(math.radians(self.anchor_lat))

    def odometry_callback(self, msg: VehicleOdometry):
        local_n = float(msg.position[0])
        local_e = float(msg.position[1])
        local_d = float(msg.position[2])

        # --- ÇERÇEVE HİZALAMA (yalnızca bir kez, yerdeyken) ---
        if not self.frame_aligned:
            if self.latest_global is None:
                return   # GPS fix yok, henüz pose basma
            gps_n, gps_e = self.latest_global
            self._align_buf.append((gps_n - local_n, gps_e - local_e))

            if len(self._align_buf) >= self.align_samples:
                self.offset_n = sum(o[0] for o in self._align_buf) / len(self._align_buf)
                self.offset_e = sum(o[1] for o in self._align_buf) / len(self._align_buf)
                self.frame_aligned = True
                self._align_buf.clear()
                self.get_logger().info(
                    f"✅ ÇERÇEVE HİZALANDI [{self.agent_id}] → "
                    f"Offset N:{self.offset_n:+.2f}m E:{self.offset_e:+.2f}m "
                    f"(bu değer Gazebo spawn pozisyonunla eşleşmeli!)")
            return   # hizalanana kadar pose YAYINLAMA

        # --- ORTAK ÇERÇEVEDE POSE ---
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = local_n + self.offset_n   # North (ortak)
        pose_msg.pose.position.y = local_e + self.offset_e   # East  (ortak)
        pose_msg.pose.position.z = -local_d                  # Up (kendi kalkış zemininden)

        pose_msg.pose.orientation.w = float(msg.q[0])
        pose_msg.pose.orientation.x = float(msg.q[1])
        pose_msg.pose.orientation.y = float(msg.q[2])
        pose_msg.pose.orientation.z = float(msg.q[3])

        self.pose_pub.publish(pose_msg)

    def coordinator_velocity_callback(self, msg: TwistStamped):
        self.target_vx = msg.twist.linear.x
        self.target_vy = msg.twist.linear.y
        self.target_vz = msg.twist.linear.z
        self.target_yaw = msg.twist.angular.z

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_mode_pub.publish(msg)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        setpoint.position = [float('nan'), float('nan'), float('nan')]
        setpoint.acceleration = [float('nan'), float('nan'), float('nan')]
        setpoint.jerk = [float('nan'), float('nan'), float('nan')]
        setpoint.yawspeed = float('nan')
        
        setpoint.velocity = [float(self.target_vx), float(self.target_vy), float(-self.target_vz)]
        setpoint.yaw = float(self.target_yaw)
        self.trajectory_pub.publish(setpoint)

    def arm_command_callback(self, msg: String):
        try:
            drone_sys_id = int(self.agent_id.split('_')[1])
        except IndexError:
            drone_sys_id = 1

        offboard_cmd = VehicleCommand()
        offboard_cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        offboard_cmd.param1 = 1.0 
        offboard_cmd.param2 = 6.0 
        offboard_cmd.target_system = drone_sys_id
        offboard_cmd.target_component = 1
        offboard_cmd.source_system = 255 
        offboard_cmd.source_component = 0
        offboard_cmd.from_external = True
        self.vehicle_command_pub.publish(offboard_cmd)

        time.sleep(0.1) 

        arm_cmd = VehicleCommand()
        arm_cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        arm_cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        arm_cmd.target_system = drone_sys_id
        arm_cmd.target_component = 1
        arm_cmd.source_system = 255 
        arm_cmd.source_component = 0
        arm_cmd.from_external = True
        
        if msg.data == "ARM":
            arm_cmd.param1 = 1.0 
            self.get_logger().info(f"SİSTEM ONAYLANDI: OFFBOARD MOD VE ARM AKTİF! (SysID: {drone_sys_id})")
        elif msg.data == "DISARM":
            arm_cmd.param1 = 0.0 
            self.get_logger().info(f"MOTORLAR ACİL KAPATILIYOR! (SysID: {drone_sys_id})")
        elif msg.data == "LAND":
            land_cmd = VehicleCommand()
            land_cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            land_cmd.command = VehicleCommand.VEHICLE_CMD_NAV_LAND 
            land_cmd.target_system = drone_sys_id
            land_cmd.target_component = 1
            land_cmd.source_system = 255 
            land_cmd.source_component = 0
            land_cmd.from_external = True
            
            self.vehicle_command_pub.publish(land_cmd)
            self.get_logger().info(f"🛬 PX4 OTONOM İNİŞ (LAND) MODUNA GEÇTİ! (SysID: {drone_sys_id})")
            return
            
        self.vehicle_command_pub.publish(arm_cmd)

def main(args=None):
    rclpy.init(args=args)
    agent = UAVAgent()
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()