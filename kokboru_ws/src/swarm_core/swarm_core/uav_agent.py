#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import math
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleOdometry, VehicleCommand
import time

class UAVAgent(Node):
    def __init__(self):
        super().__init__('uav_agent')
        
        # Nesne özellikleri
        self.declare_parameter('agent_id', 'uav_1')
        self.agent_id = self.get_parameter('agent_id').get_parameter_value().string_value

        self.get_logger().info(f'UAV Agent {self.agent_id} başlatıldı.')

        # Swarm Koordinatore konum verisini bas
        self.pose_pub = self.create_publisher(
            PoseStamped,
            f'/{self.agent_id}/ap/pose/filtered',
            10
        )

        # Swarm Coordinatordan hız emri al
        self.target_sub = self.create_subscription(
            TwistStamped,
            f'/{self.agent_id}/target_velocity',
            self.coordinator_velocity_callback,
            10
        )

        # PX4'ten gelen konumu dinle
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            f'/{self.agent_id}/fmu/out/vehicle_odometry',
            self.odometry_callback,
            qos_profile_sensor_data
        )

        # --- EKSİK PARÇA: Beyinden gelen ARM/DISARM emrini dinle ---
        self.arm_sub = self.create_subscription(
            String,
            f'/{self.agent_id}/arm_cmd',
            self.arm_command_callback,
            10
        )

        # PX4'e heartbeat bas 
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            f'/{self.agent_id}/fmu/in/offboard_control_mode',
            10
        )

        # PX4'e hız emri bas
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            f'/{self.agent_id}/fmu/in/trajectory_setpoint',
            10
        )

        # --- EKSİK PARÇA: PX4'e Otonom (Offboard) ve Motor (ARM) komutunu bas ---
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            f'/{self.agent_id}/fmu/in/vehicle_command',
            10
        )
        
        # Offboard kalma koşulu 10Hz heartbeat
        self.timer = self.create_timer(0.1, self.publish_offboard_heartbeat)

        # Son alınan hız verilerini saklamak için
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vz = 0.0
        self.target_yaw = 0.0

    # ---------------- CALLBACK FONKSİYONLARI ----------------

    def arm_command_callback(self, msg: String):
        """ Swarm Agent'tan gelen emri PX4'ün anlayacağı offboard ve arm komutlarına çevirir """
        
        try:
            drone_sys_id = int(self.agent_id.split('_')[1])
        except IndexError:
            drone_sys_id = 1

        # 1. ADIM: Otonom Moda (Offboard) Geçiş İsteği
        offboard_cmd = VehicleCommand()
        offboard_cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        offboard_cmd.param1 = 1.0 
        offboard_cmd.param2 = 6.0 
        offboard_cmd.target_system = drone_sys_id
        offboard_cmd.target_component = 1
        offboard_cmd.source_system = 255 # <--- KRİTİK: Ben dış dünyadan (Bilgisayar) komut veren bir beyinim!
        offboard_cmd.source_component = 0
        offboard_cmd.from_external = True
        self.vehicle_command_pub.publish(offboard_cmd)

        # 2. ADIM: PX4'ün modu sindirmesi için nefes alma süresi (Event loop'u tıkamayacak kadar kısa)
        time.sleep(0.1) 

        # 3. ADIM: Motorları ARM Et
        arm_cmd = VehicleCommand()
        arm_cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        arm_cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        arm_cmd.target_system = drone_sys_id
        arm_cmd.target_component = 1
        arm_cmd.source_system = 255 # <--- KRİTİK
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
            land_cmd.command = VehicleCommand.VEHICLE_CMD_NAV_LAND # PX4 Otonom İniş Komutu
            land_cmd.target_system = drone_sys_id
            land_cmd.target_component = 1
            land_cmd.source_system = 255 
            land_cmd.source_component = 0
            land_cmd.from_external = True
            
            self.vehicle_command_pub.publish(land_cmd)
            self.get_logger().info(f"🛬 PX4 OTONOM İNİŞ (LAND) MODUNA GEÇTİ! (SysID: {drone_sys_id})")
            return
            
        self.vehicle_command_pub.publish(arm_cmd)
            
    def publish_offboard_heartbeat(self):
        # PX4 Ayarları Sadece Hız Değişecek
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
        
        setpoint.velocity = [
            float(self.target_vx),
            float(self.target_vy),
            float(-self.target_vz) # PX4 NED Z Ekseni Ters
        ]
        setpoint.yaw = float(self.target_yaw)
        
        self.trajectory_pub.publish(setpoint)


    def coordinator_velocity_callback(self, msg: TwistStamped):
        self.target_vx = msg.twist.linear.x
        self.target_vy = msg.twist.linear.y
        self.target_vz = msg.twist.linear.z
        self.target_yaw = msg.twist.angular.z


    def odometry_callback(self, msg: VehicleOdometry):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = float(msg.position[0])
        pose_msg.pose.position.y = float(msg.position[1])
        pose_msg.pose.position.z = float(-msg.position[2])

        pose_msg.pose.orientation.w = float(msg.q[0])
        pose_msg.pose.orientation.x = float(msg.q[1])
        pose_msg.pose.orientation.y = float(msg.q[2])
        pose_msg.pose.orientation.z = float(msg.q[3])

        self.pose_pub.publish(pose_msg)

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