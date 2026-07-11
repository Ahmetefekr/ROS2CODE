#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
import math
import json
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

# ===========================================================================
# ŞARTNAME QR SÖZLÜĞÜ
# ===========================================================================
FRM_MAP = {          # QR kodu -> iç formasyon adı
    "ok": "okbasi",
    "v":  "v",
    "l":  "cizgi",
}

RENK_MAP = {         # QR kodu -> camera_node'un bastığı renk anahtarı
    "r": "KIRMIZI",
    "b": "MAVI",
}

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

# ===========================================================================
# TargetMemory — gorev1.py'den taşındı
# EMA (Üstel Hareketli Ortalama) + Sigma Filtresi ile güvenilir hedef hafızası
# ===========================================================================
class TargetMemory:
    """
    Her renk hedefi için NED koordinatlarını birden fazla ölçümden
    ağırlıklı ortalama (EMA) ile birleştirir.

    ema_alpha  : Yeni ölçümün ağırlığı (0–1). Küçük → geçmişe daha çok güven.
    sigma_gate : Bu kadar std sapma dışındaki ölçümler outlier olarak atılır.
    min_obs    : Hedef "güvenilir" sayılmadan önce gereken minimum gözlem sayısı.
    """

    def __init__(self, ema_alpha: float = 0.35,
                 sigma_gate: float = 2.5,
                 min_obs: int = 1):
        self.alpha      = ema_alpha
        self.sigma_gate = sigma_gate
        self.min_obs    = min_obs
        self._data: dict = {}

    def update(self, color: str, north: float, east: float):
        """Yeni GPS+FOV ölçümü ile hafızayı güncelle."""
        now = time.time()

        # 5+ gözlemli seed varsa kamera ölçümü kabul etme (SDF hassasiyeti > kamera)
        if color in self._data and self._data[color].get("count", 0) >= 5:
            self._data[color]["last_ts"] = now
            return

        if color not in self._data:
            self._data[color] = {
                "north": north, "east": east,
                "var_n": 25.0, "var_e": 25.0,   # başlangıç varyansı: 5m std
                "count": 1, "last_ts": now,
            }
            return

        d = self._data[color]
        std_n = math.sqrt(d["var_n"])
        std_e = math.sqrt(d["var_e"])
        diff_n = abs(north - d["north"])
        diff_e = abs(east  - d["east"])

        # Sigma gate: outlier rejection
        if d["count"] >= self.min_obs:
            if (diff_n > self.sigma_gate * std_n or
                    diff_e > self.sigma_gate * std_e):
                return  # Outlier — kabul etme

        # EMA güncelle
        a = self.alpha
        new_n = a * north + (1 - a) * d["north"]
        new_e = a * east  + (1 - a) * d["east"]
        new_var_n = (1 - a) * (d["var_n"] + a * (north - d["north"]) ** 2)
        new_var_e = (1 - a) * (d["var_e"] + a * (east  - d["east"])  ** 2)

        d.update({
            "north":   new_n,
            "east":    new_e,
            "var_n":   max(0.04, new_var_n),
            "var_e":   max(0.04, new_var_e),
            "count":   d["count"] + 1,
            "last_ts": now,
        })

    def get(self, color: str):
        """Güvenilir tahmin varsa (north, east) döndür, yoksa None."""
        if color not in self._data:
            return None
        d = self._data[color]
        if d["count"] < self.min_obs:
            return None
        return d["north"], d["east"]

    def get_raw(self, color: str):
        """min_obs kontrolü olmadan ham tahmini döndür (fallback)."""
        if color not in self._data:
            return None
        return self._data[color]["north"], self._data[color]["east"]

    def __contains__(self, color: str) -> bool:
        return color in self._data

    def std(self, color: str):
        """(std_n, std_e) döndür — log için."""
        if color not in self._data:
            return 9.9, 9.9
        d = self._data[color]
        return math.sqrt(d["var_n"]), math.sqrt(d["var_e"])


# ===========================================================================
# SwarmAgent
# ===========================================================================
class SwarmAgent(Node):

    # Kuantizasyon adımları — TÜM ajanlarda AYNI olmak zorunda
    Q_POS = 0.25                 # metre
    Q_YAW = math.radians(2.0)    # radyan

    @staticmethod
    def _q(deger: float, adim: float) -> float:
        """Deterministik yuvarlama (Python'un banker's rounding'inden kaçınır)."""
        return math.floor(deger / adim + 0.5) * adim

    def __init__(self):
        super().__init__('swarm_agent')

        self.declare_parameter('uav_count', 3)
        self.uav_count = self.get_parameter('uav_count').get_parameter_value().integer_value

        self.declare_parameter('team_id', 'team_1')
        _raw_team = self.get_parameter('team_id').get_parameter_value().string_value
        self.team_id = _raw_team
        _digits = ''.join(ch for ch in _raw_team if ch.isdigit())
        self.team_key = _digits if _digits else "1"

        self.declare_parameter('my_id', 1)
        self.my_id = self.get_parameter('my_id').get_parameter_value().integer_value
        self.my_uav_name = f'uav_{self.my_id}'

        # ---------- STATE MACHINE ----------
        self.uav_positions = {}
        self.uav_stamps    = {}
        self.uav_frames    = {}
        self.uav_states    = {i: "SURUDE" for i in range(1, self.uav_count + 1)}

        self.swarm_state       = "YERDE_BEKLIYOR"
        self.active_task_queue = []
        self.current_task      = None
        self.task_wait_start_time = 0.0

        self.declare_parameter('ref_lat', 47.397742)
        self.declare_parameter('ref_lon', 8.545594)
        self.ref_lat = self.get_parameter('ref_lat').get_parameter_value().double_value
        self.ref_lon = self.get_parameter('ref_lon').get_parameter_value().double_value

        # 1) ÖNCE ham GPS haritası
        self.qr_gps = {
            1: (41.000054, 28.800000),
            2: (41.000000, 28.800062),
            3: (40.999946, 28.800000),
            4: (41.000027, 28.800062),
            5: (40.999973, 28.799938),
            6: (41.000027, 28.799938)
        }

        self.qr_map = {
            qr_id: project_to_ned(lat, lon, self.ref_lat, self.ref_lon)
            for qr_id, (lat, lon) in self.qr_gps.items()
        }
        self.get_logger().info(f"QR haritası (lokal NED): { {k: (round(v[0],1), round(v[1],1)) for k,v in self.qr_map.items()} }")

        # -----------------------------------------------------------------------

        self.qr_varis_zamani = 0.0
        self.qr_genel_bekleme = 0.0

        # ---------------------------------------------------------------
        # TargetMemory — EMA + sigma filtresi (gorev1.py'den)
        # ---------------------------------------------------------------
        self.target_memory = TargetMemory(ema_alpha=0.85, sigma_gate=2.5, min_obs=1)
        self.current_qr_id = 1

        self.global_target_n = 0.0
        self.global_target_e = 0.0
        self.next_target_n   = 0.0
        self.next_target_e   = 0.0
        self.yon_donus_baslangic = 0.0
        self.last_yaw_rad    = 0.0

        self.base_altitude   = 5.0
        self.target_altitude = self.base_altitude

        # ----- FORMASYON -----
        self.current_formation  = "okbasi"
        self.formation_spacing  = 3

        self.pitch_deg = 0.0
        self.roll_deg  = 0.0

        self.formations = ["okbasi", "v", "cizgi"]
        self.uav_to_slot = {i: i for i in range(1, self.uav_count + 1)}

        # ---------------------------------------------------------------
        # Kamera kalibrasyon parametreleri (gorev1.py'den)
        # SDF dosyanızdaki horizontal_fov değeriyle eşleşmeli!
        # ---------------------------------------------------------------
        self.camera_hfov_rad    = 1.74   # Gazebo SDF: <horizontal_fov>1.74</horizontal_fov>
        self.camera_img_width   = 1280    # Kamera çözünürlüğü
        self.camera_img_height  = 960

        # Şartnameye ayrılma bekleme süresi
        self.uav_timers            = {}

        # Home tutma
        self.home_n = 0.0
        self.home_e = 0.0

        # ----- ROS2 ABONELİKLER -----
        self.telemetry_subs = []

        self.get_logger().info(
            f"Ajan Başlatıldı. Ben: {self.my_uav_name}, Takım: {self.team_id}")

        for i in range(1, self.uav_count + 1):
            uav_id = f'uav_{i}'
            sub = self.create_subscription(
                PoseStamped,
                f'/{uav_id}/ap/pose/filtered',
                lambda msg, drone_id=i: self.telemetry_callback(msg, drone_id),
                qos_profile_sensor_data
            )
            self.telemetry_subs.append(sub)

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            f'/{self.my_uav_name}/target_velocity',
            10
        )

        self.arm_pub = self.create_publisher(
            String,
            f'/{self.my_uav_name}/arm_cmd',
            10
        )

        self.vision_sub = self.create_subscription(
            String,
            '/camera/swarm_commands',
            self.vision_command_callback,
            10
        )

        self.circle_sub = self.create_subscription(
            String,
            '/uav_1/camera/circles',
            self.circle_callback,
            10
        )

        self.manual_form_sub = self.create_subscription(
            String,
            '/swarm/override_formation',
            self.override_formation_callback,
            10
        )

        self.land_sub = self.create_subscription(
            String, '/swarm/land', self.land_callback, 10)

        # YER KONTROL QR İÇERİĞİ BASMAK İÇİN

        self.gcs_qr_pub = self.create_publisher(String, '/swarm/gcs/qr_content', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.takeoff_sub = self.create_subscription(
            String, '/swarm/takeoff', self.takeoff_callback, 10)
        
        self.TELEMETRI_TIMEOUT = 1.0     # saniye

    # -----------------------------------------------------------------------
    # TELEMETRY
    # -----------------------------------------------------------------------
    def telemetry_callback(self, msg: PoseStamped, uav_id: int):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        # Yaw'ı quaternion'dan hesapla ve pozisyona ekle (4. eleman)
        qw = msg.pose.orientation.w
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(min(sinp, 1.0), -1.0))
        
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        self.uav_positions[uav_id] = (x, y, z, yaw, pitch, roll)
        self.uav_stamps[uav_id]    = time.time()
        self.uav_frames[uav_id]    = msg.header.frame_id

        if not hasattr(self, 'first_telemetry'):
            self.get_logger().info(
                f"DDS TELEMETRİ ALINDI! X:{x:.2f}, Y:{y:.2f}, Yaw:{math.degrees(yaw):.1f}°")
            self.first_telemetry = True

    # -----------------------------------------------------------------------
    # TAKEOFF
    # -----------------------------------------------------------------------
    def takeoff_callback(self, msg: String):
        if self.swarm_state == "YERDE_BEKLIYOR":
            
            # 🛠️ YENİ GÜVENLİK KALKANI: Tüm sürünün GPS'i hazır mı?
            if len(self.uav_positions) < self.uav_count:
                eksikler = set(range(1, self.uav_count + 1)) - set(self.uav_positions.keys())
                self.get_logger().error(f"🚨 KALKIŞ REDDEDİLDİ: Sürü eksik! GPS kilidi almayan İHA'lar: {eksikler}. Lütfen bekleyin.")
                return

            uav1_pos = self.uav_positions.get(1)
            if uav1_pos:
                self.home_n = uav1_pos[0]
                self.home_e = uav1_pos[1]
            else:
                surudeki = [pos for idd, pos in self.uav_positions.items() if self.uav_states.get(idd) == "SURUDE"]
                if surudeki:
                    self.home_n = sum(p[0] for p in surudeki) / len(surudeki)
                    self.home_e = sum(p[1] for p in surudeki) / len(surudeki)

            self.global_target_n = self.home_n
            self.global_target_e = self.home_e

            self.get_logger().info(f"🚀 Kalkış Emri Alındı! Tüm sürü ARM ediliyor. Home N:{self.home_n:.2f}, E:{self.home_e:.2f}")
            self.swarm_state = "KALKIS"
            self.arm_pub.publish(String(data="ARM"))

    # -----------------------------------------------------------------------
    # MANUEL İNİŞ (YKİ)
    # -----------------------------------------------------------------------
    def land_callback(self, msg: String):
        if self.swarm_state in ("YERDE_BEKLIYOR", "EVE_DONUS", "TOPLU_INIS"):
            return

        self.active_task_queue.clear()
        self.current_task = None
        self.pitch_deg = 0.0
        self.roll_deg  = 0.0

        if msg.data.strip().upper() == "NOW":
            # ACİL: olduğun yerde in, eve dönme
            surudeki = [p for idd, p in self.uav_positions.items()
                        if self.uav_states.get(idd) == "SURUDE"]
            if surudeki:
                self.global_target_n = sum(p[0] for p in surudeki) / len(surudeki)
                self.global_target_e = sum(p[1] for p in surudeki) / len(surudeki)
            self.current_qr_id = 0
            self.swarm_state = "TOPLU_INIS"
            self.get_logger().warn("🛬 ACİL İNİŞ! Mevcut konumda alçalınıyor.")
            return

        # NORMAL: eve dön, sonra in
        self.current_qr_id = 0
        self.next_target_n = self.home_n
        self.next_target_e = self.home_e
        self.swarm_state = "YONE_DONUS"
        self.yon_donus_baslangic = time.time()
        self.assign_slots_hungarian()
        self.get_logger().warn(f"🛬 MANUEL İNİŞ! Home'a dönülüyor.")

    # -----------------------------------------------------------------------
    # QR CALLBACK — ŞARTNAME FORMATI
    # {"qr":1,"w":4,"mis":[paket0,paket1,paket2],"team":{"1":[paket_no, sonraki_qr]}}
    # -----------------------------------------------------------------------
    def vision_command_callback(self, msg: String):
        if self.swarm_state != "QR_BEKLIYOR":
            return
        if time.time() - self.qr_varis_zamani < 3.0:
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"QR JSON parse edilemedi: {e}")
            return

        try:
            okunan_qr_id = int(payload.get("qr", -1))

            # Yanlış QR'a bakıyoruz (rota dışı bir QR görüldü) -> yoksay
            if okunan_qr_id != self.current_qr_id:
                self.get_logger().warn(
                    f"QR-{okunan_qr_id} okundu ama biz QR-{self.current_qr_id} bekliyoruz. Yoksayıldı.")
                return

            wait_s = float(payload.get("w", 0))
            mis_listesi = payload.get("mis", [])
            team_tablosu = payload.get("team", {})

            # --- TAKIM SATIRI: "1": [paket_no, sonraki_qr] ---
            if self.team_key not in team_tablosu:
                self.get_logger().error(
                    f"QR-{okunan_qr_id} içinde takım '{self.team_key}' bulunamadı! "
                    f"Mevcut anahtarlar: {list(team_tablosu.keys())}")
                return

            satir = team_tablosu[self.team_key]
            paket_no    = int(satir[0])   # 1-tabanlı
            sonraki_qr  = int(satir[1])

            if not (1 <= paket_no <= len(mis_listesi)):
                self.get_logger().error(
                    f"Paket no {paket_no} geçersiz (mis uzunluğu {len(mis_listesi)}).")
                return

            paket = mis_listesi[paket_no - 1]

            self.get_logger().info(
                f"QR-{okunan_qr_id} ÇÖZÜLDÜ | Takım {self.team_key} → "
                f"paket #{paket_no}, w={wait_s}s, sonraki QR={sonraki_qr}")
            self.get_logger().info(f"   Görev paketi: {paket}")

            # --- YKİ'ye çözülen içeriği bas (şartname: en az 1 kez gösterilmeli) ---
            self.gcs_qr_pub.publish(String(data=json.dumps({
                "qr": okunan_qr_id,
                "team": self.team_key,
                "paket_no": paket_no,
                "gorevler": paket,
                "sonraki_qr": sonraki_qr,
                "raw": payload,
            }, ensure_ascii=False)))

            # --- KUYRUĞU KUR ---
            self.swarm_state = "QR_ISLENIYOR"
            self.active_task_queue.clear()
            self.qr_genel_bekleme = wait_s

            for komut in paket:
                if not komut:
                    continue
                cmd = str(komut[0]).strip().lower()

                # ["frm", "ok", 6]
                if cmd == "frm":
                    frm_kod = str(komut[1]).strip().lower()
                    if frm_kod not in FRM_MAP:
                        self.get_logger().error(f"Bilinmeyen formasyon kodu: {frm_kod}")
                        continue
                    self.active_task_queue.append({
                        "tip":    "FORMASYON",
                        "deger":  FRM_MAP[frm_kod],
                        "mesafe": float(komut[2]),
                    })

                # ["mnv", pitch, roll]
                elif cmd == "mnv":
                    self.active_task_queue.append({
                        "tip":   "MANEVRA",
                        "pitch": float(komut[1]),
                        "roll":  float(komut[2]),
                    })

                # ["alt", 20]
                elif cmd == "alt":
                    self.active_task_queue.append({
                        "tip":   "IRTIFA",
                        "deger": float(komut[1]),
                    })

                # ["leav", 3, "r"]
                elif cmd == "leav":
                    renk_kod = str(komut[2]).strip().lower()
                    if renk_kod not in RENK_MAP:
                        self.get_logger().error(f"Bilinmeyen renk kodu: {renk_kod}")
                        continue
                    self.active_task_queue.append({
                        "tip":        "AYRILMA",
                        "drone_id":   int(komut[1]),
                        "hedef_renk": RENK_MAP[renk_kod],   # KIRMIZI / MAVI
                        "bekleme":    wait_s,
                    })

                else:
                    self.get_logger().warn(f"Tanınmayan QR komutu: {cmd}")
                    continue

                # ŞARTNAME: her görevin ardından "w" kadar bekle
                if wait_s > 0:
                    self.active_task_queue.append({"tip": "BEKLEME", "sure": wait_s})

            # Son olarak sonraki hedef
            self.active_task_queue.append({
                "tip":   "SONRAKI_HEDEF",
                "qr_id": sonraki_qr,
            })

            self.siradaki_gorevi_baslat()

        except (KeyError, IndexError, TypeError, ValueError) as e:
            self.get_logger().error(f"QR görev kurgulama hatası: {e}")


    # -----------------------------------------------------------------------
    # GÖREV KUYRUĞU YÜRÜTÜCÜSÜ (FSM TASK EXECUTOR)
    # -----------------------------------------------------------------------
    def siradaki_gorevi_baslat(self):
        if len(self.active_task_queue) == 0:
            return

        self.current_task = self.active_task_queue.pop(0)
        tip = self.current_task["tip"]
        self.get_logger().info(f"Görev icra ediliyor: {tip}")

        if tip == "FORMASYON":
            yeni_tip = self.current_task["deger"]
            yeni_mesafe = self.current_task.get("mesafe")
            if yeni_tip in self.formations:
                self.current_formation = yeni_tip
            if yeni_mesafe and yeni_mesafe > 0.5:
                self.formation_spacing = float(yeni_mesafe)
            self.get_logger().info(
                f"FORMASYON → {self.current_formation.upper()} / aralık {self.formation_spacing} m")
            self.assign_slots_hungarian()
            self.siradaki_gorevi_baslat()

        elif tip == "MANEVRA":
            MAX_ACI = 30.0
            self.pitch_deg = max(-MAX_ACI, min(float(self.current_task["pitch"]), MAX_ACI))
            self.roll_deg  = max(-MAX_ACI, min(float(self.current_task["roll"]),  MAX_ACI))
            self.get_logger().info(
                f"MANEVRA → pitch {self.pitch_deg}° / roll {self.roll_deg}° (eğim korunacak)")
            self.siradaki_gorevi_baslat()

        elif tip == "IRTIFA":
            istenen = float(self.current_task["deger"])
            self.target_altitude = max(10.0, min(istenen, 30.0))   # şartname alt sınır 10 m
            if abs(istenen - self.target_altitude) > 0.01:
                self.get_logger().warn(f"İrtifa {istenen}m → sınırlara clamp edildi: {self.target_altitude}m")
            self.siradaki_gorevi_baslat()

        elif tip == "AYRILMA":
            hedef_id   = self.current_task["drone_id"]
            hedef_renk = self.current_task["hedef_renk"]      # KIRMIZI / MAVI

            if self.uav_states.get(hedef_id) == "SURUDE":
                pos = (self.target_memory.get(hedef_renk) or
                       self.target_memory.get_raw(hedef_renk))
                if pos:
                    self.uav_states[hedef_id] = f"INIS_{hedef_renk}"
                    self.get_logger().warn(
                        f"UAV-{hedef_id} sürüden ayrılıyor → {hedef_renk} "
                        f"({pos[0]:.1f}, {pos[1]:.1f}). Yuvası REZERVE.")
                    self.assign_slots_hungarian()
                else:
                    self.get_logger().error(
                        f"{hedef_renk} alan hafızada YOK — ayrılma görevi icra edilemiyor!")
            else:
                self.get_logger().warn(
                    f"UAV-{hedef_id} zaten sürüde değil ({self.uav_states.get(hedef_id)}), ayrılma atlandı.")
            self.siradaki_gorevi_baslat()

        elif tip == "BEKLEME":
            self.task_wait_start_time = time.time()
            self.swarm_state = "BEKLEMEDE"
            self.get_logger().info(f"Sürü {self.current_task['sure']} saniye boyunca mevcut görevde bekliyor (w süresi).")

        elif tip == "SONRAKI_HEDEF":
            yeni_qr = self.current_task["qr_id"]
            self.current_qr_id = yeni_qr

            if yeni_qr == 0:
                self.pitch_deg = 0.0      # eve dönüşte düz uç (güvenli iniş)
                self.roll_deg  = 0.0
                self.next_target_n = self.home_n
                self.next_target_e = self.home_e
                self.get_logger().info("Görevler bitti. Eve (0,0) dönülüyor.")
            else:
                hedef_koord = self.qr_map.get(yeni_qr, (0.0, 0.0))
                self.next_target_n = hedef_koord[0]
                self.next_target_e = hedef_koord[1]
                self.get_logger().info(
                    f"Hedef QR-{yeni_qr}. Eğim korunuyor (p={self.pitch_deg}° r={self.roll_deg}°).")

            self.swarm_state = "EKSİK_AJAN_BEKLENIYOR"
            self.yon_donus_baslangic = time.time()
            self.assign_slots_hungarian()

    # -----------------------------------------------------------------------
    # CIRCLE CALLBACK — KUSURSUZ 3D PROJEKSİYON (PITCH/ROLL + OFSET DÜZELTMELİ)
    # -----------------------------------------------------------------------
    def circle_callback(self, msg: String):
        try:
            data    = json.loads(msg.data)
            circles = data.get("circles", {})
            if not circles:
                return

            # YENİ: Veriyi HANGİ DRONE gönderdiyse hesaplamayı onun konumuna göre yap!
            goren_ajan = data.get("agent_id", "uav_1")
            goren_id = int(goren_ajan.split('_')[1])

            goren_pos = self.uav_positions.get(goren_id)
            if not goren_pos:
                return

            north_d1 = goren_pos[0]
            east_d1  = goren_pos[1]
            alt      = goren_pos[2]
            yaw_rad  = goren_pos[3] if len(goren_pos) > 3 else self.last_yaw_rad
            
            # YENİ: Eğim verilerini çek (Pitch ve Roll)
            pitch_rad = goren_pos[4] if len(goren_pos) > 4 else 0.0
            roll_rad  = goren_pos[5] if len(goren_pos) > 5 else 0.0

            # ── KATMAN 1: State filtresi ──────────────────────────────────
            if self.swarm_state not in ("QR_BEKLIYOR", "BEKLEMEDE", "QR_ISLENIYOR"):
                inis_aktif = any(
                    str(durum).startswith("INIS_")
                    for durum in self.uav_states.values()
                )
                if inis_aktif:
                    # 🛠️ FİZİKSEL HİLE: Biri iniyorsa kamerayı KAPAT! 
                    # İnen drone D1'in görüşünü kapatıp dairenin merkezini kaydırır.
                    # Bu yüzden hafızadaki o son "kusursuz" hedefe kilitli kalıyoruz.
                    return

            # ── KATMAN 2: İrtifa stabil mi? ───────────────────────────────
            if abs(alt - self.target_altitude) > 0.5:
                # İniş modunda bu kontrolü atla
                inis_aktif = any(
                    str(d).startswith("INIS_")
                    for d in self.uav_states.values()
                )
                if not inis_aktif:
                    return

            # ── KATMAN 3: Alt sınır ───────────────────────────────────────
            if alt < 1.5:
                return

            # FOV tabanlı ölçek
            m_per_px_x = (alt * 2.0 * math.tan(self.camera_hfov_rad / 2.0)) / self.camera_img_width
            hfov_v     = 2.0 * math.atan(math.tan(self.camera_hfov_rad / 2.0) * self.camera_img_height / self.camera_img_width)
            m_per_px_y = (alt * 2.0 * math.tan(hfov_v / 2.0)) / self.camera_img_height

            cos_y = math.cos(yaw_rad)
            sin_y = math.sin(yaw_rad)

            for renk, info in circles.items():
                offset_x = info.get("offset_x", 0.0)
                offset_y = info.get("offset_y", 0.0)
                conf     = info.get("confidence", 0.0)

                if conf < 0.75:
                    continue

                # --- 🛠️ KUSURSUZ PROJEKSİYON MATEMATİĞİ ---
                
                # 1. Pikselleri Metreye Çevir
                # X pikseli (sağ-sol) -> Dronun Doğu (East) eksenidir
                # Y pikseli (üst-alt) -> Dronun Güney (-North) eksenidir
                body_e_pixel =  offset_x * m_per_px_x
                body_n_pixel = -offset_y * m_per_px_y
                
                # 3. Dronun eğikliğinden (Pitch/Roll) kaynaklanan kayma (Gimbal-less kompansasyonu)
                # Pitch > 0 (Burun yukarı) -> Kamera geriye (Güney) bakar -> Hedef eksi North'tadır
                # Roll > 0 (Sağ kanat aşağı) -> Kamera sağa (Doğu) bakar -> Hedef artı East'tedir
                tilt_offset_n = -alt * math.tan(pitch_rad)
                tilt_offset_e =  alt * math.tan(roll_rad)
                
                # 4. Toplam Gerçek Body Ofseti
                body_n = body_n_pixel + tilt_offset_n
                body_e = body_e_pixel + tilt_offset_e

                world_n = body_n * cos_y - body_e * sin_y
                world_e = body_n * sin_y + body_e * cos_y

                hedef_n = north_d1 + world_n
                hedef_e = east_d1  + world_e

                self.target_memory.update(renk, hedef_n, hedef_e)

                std_n, std_e = self.target_memory.std(renk)
                self.get_logger().info(
                    f"[TgtMem] {renk} → N:{hedef_n:.2f} E:{hedef_e:.2f} "
                    f"(Eğim Düzeltmesi: N: {tilt_offset_n:.2f}m, E: {tilt_offset_e:.2f}m)")

        except Exception as e:
            self.get_logger().error(f"Circle callback hatası: {e}")

    # -----------------------------------------------------------------------
    # OVERRIDE FORMATION
    # -----------------------------------------------------------------------
    def override_formation_callback(self, msg: String):
        """Format: 'okbasi' veya 'okbasi:8'  (tip:aralık_metre)"""
        parca = msg.data.strip().lower().split(':')
        yeni_tip = parca[0]

        if yeni_tip not in self.formations:
            self.get_logger().error(
                f"Bilinmeyen formasyon: {yeni_tip}. Geçerli: {self.formations}")
            return

        self.current_formation = yeni_tip
        if len(parca) > 1:
            try:
                yeni_mesafe = float(parca[1])
                if yeni_mesafe > 0.5:
                    self.formation_spacing = yeni_mesafe
            except ValueError:
                pass

        self.assign_slots_hungarian()      # ← KRİTİK
        self.get_logger().warn(
            f"YKİ MÜDAHALESİ: {self.current_formation.upper()} / "
            f"aralık {self.formation_spacing}m")

    # -----------------------------------------------------------------------
    # SAF GEOMETRİK SLOT (YUVA) HESAPLAYICI (KUSURSUZ FİZİK)
    # -----------------------------------------------------------------------
    def get_slot_offset(self, slot_id: int, n: int):
        s = self.formation_spacing
        
        if self.current_formation == "v":
            # --- KRİTİK YAMA 2: NORMAL V FORMASYONU ---
            # Kanatlar liderin gerisinde (-k * s) kalacak şekilde düzeltildi
            if n % 2 != 0:
                if slot_id == 1: return 0.0, 0.0
                k = slot_id // 2
                yon = 1 if slot_id % 2 == 0 else -1
                return float(-k * s), float(yon * k * s)
            else:
                k = (slot_id + 1) // 2
                yon = 1 if slot_id % 2 != 0 else -1
                return float(-k * s), float(yon * k * s)
            
        elif self.current_formation == "cizgi":
            y_offset = float((slot_id - (n + 1) / 2.0) * s)
            return 0.0, y_offset
            
        else: # okbasi
            if slot_id == 1: return 0.0, 0.0
            r = 1
            kapasite = 1
            while kapasite < slot_id:
                r += 1
                kapasite += r
            ilk_eleman = kapasite - r + 1
            j = slot_id - ilk_eleman + 1 
            return float(-(r - 1) * s), float((j - (r + 1) / 2.0) * s)

    # -----------------------------------------------------------------------
    # İHA'NIN ATANDIĞI YUVAYI DÖNDÜREN SİSTEM
    # -----------------------------------------------------------------------
    def get_formation_offset(self, uav_id: int):
        # 🛠️ DÜZELTME: Havadaki aktif sayıyı (N) değil, TOPLAM sayıyı baz alıyoruz.
        # Böylece formasyon asla daralmaz, ayrılanın yuvası fiziksel olarak boş kalır.
        n = self.uav_count 
        slot_id = self.uav_to_slot.get(uav_id, uav_id)
        return self.get_slot_offset(slot_id, n)
    
    def calculateAPFOffset(self, uav_id: int, d_des: float = None):
        # Güvenlik yarıçapı formasyon aralığıyla ölçeklenir (QR'dan 5-9m geliyor)
        if d_des is None:
            d_des = max(2.0, min(3.5, 0.5 * self.formation_spacing))

        k_taban = 2.0     # sabit itme kazancı
        lam     = 3.5     # yaklaşma hızına fren çarpanı
        F_MAX   = 3.0     # m/s — tek komşudan gelebilecek maks itme (patlama koruması)
        F_MIN   = 0.35    # m  — 1/f² için mesafe tabanı

        my_pos = self.uav_positions.get(uav_id)
        if not my_pos:
            return 0.0, 0.0

        if not hasattr(self, '_apf_prev'):
            self._apf_prev = {}          # {pair_key: (mesafe, zaman)}

        now = time.time()
        rep_n = rep_e = 0.0

        for other_id, other_pos in self.uav_positions.items():
            if other_id == uav_id:
                continue

            # Dikey ayrım varsa (manevra eğimi vs.) yatay itmeye gerek yok
            if abs(my_pos[2] - other_pos[2]) > 2.0:
                continue

            dn = my_pos[0] - other_pos[0]
            de = my_pos[1] - other_pos[1]
            raw = math.hypot(dn, de)
            if raw < 1e-6:
                continue                  # tam üst üste — yön tanımsız

            f_ij = max(raw, F_MIN)        # sadece BÜYÜKLÜK için taban
            ux, uy = dn / raw, de / raw   # yön ham mesafeden

            # --- Göreli hız (türev) — her çift için KENDİ zaman damgasıyla ---
            pair_key = tuple(sorted((uav_id, other_id)))
            prev = self._apf_prev.get(pair_key)
            if prev is None:
                f_dot = 0.0
            else:
                prev_f, prev_t = prev
                dt = now - prev_t
                f_dot = (f_ij - prev_f) / dt if dt > 1e-3 else 0.0
            self._apf_prev[pair_key] = (f_ij, now)

            if f_ij >= d_des:
                continue                  # güvenli mesafedeyiz

            # Yaklaşıyorsa (f_dot < 0) kazanç artar
            k_a = k_taban + lam * max(0.0, -f_dot)
            force_mag = -k_a * (1.0 - (d_des ** 2) / (f_ij ** 2))
            force_mag = min(force_mag, F_MAX)     # doyum

            rep_n += ux * force_mag
            rep_e += uy * force_mag

        return rep_n, rep_e
    
    # -----------------------------------------------------------------------
    # SÜRÜ OPTİMİZASYONU (MACAR ALGORİTMASI) — DETERMİNİSTİK
    # Tüm ajanlar aynı girdiden aynı atamayı üretmeli, yoksa iki drone
    # aynı slotu hedefler. Kuantizasyon telemetri jitter'ını öldürür.
    # -----------------------------------------------------------------------
    
    def assign_slots_hungarian(self):
        n = self.uav_count
        slotlar = list(range(1, n + 1))
        cost_matrix = np.zeros((n, n), dtype=np.int64)   # INT: float belirsizliği yok

        # Yaw ve sürü merkezini kuantize et — rotasyon matrisi her ajanda birebir aynı olsun
        yaw = self._q(getattr(self, 'last_yaw_rad', 0.0), self.Q_YAW)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        gt_n = self._q(self.global_target_n, self.Q_POS)
        gt_e = self._q(self.global_target_e, self.Q_POS)

        for i in range(n):
            uav_id = i + 1
            pos = self.uav_positions.get(uav_id)

            if not pos:
                cost_matrix[i, :] = 9_999_900
                continue

            # Pozisyonu kuantize et — ajanlar arası telemetri farkı burada ölür
            uav_n = self._q(pos[0], self.Q_POS)
            uav_e = self._q(pos[1], self.Q_POS)

            for j in range(n):
                slot_id = slotlar[j]

                # REZERVE YUVA: sürüde olmayan drone SADECE kendi yuvasında kalabilir.
                # Böylece ayrılanın yeri fiziksel olarak boş kalır, formasyon daralmaz.
                if self.uav_states.get(uav_id) != "SURUDE":
                    if self.uav_to_slot.get(uav_id) == slot_id:
                        cost_matrix[i, j] = 0            # kendi yuvası bedava
                    else:
                        cost_matrix[i, j] = 99_999_900   # başka yuva yasak
                    continue

                off_n, off_e = self.get_slot_offset(slot_id, n)
                rot_n = off_n * cos_y - off_e * sin_y
                rot_e = off_n * sin_y + off_e * cos_y

                hedef_n = gt_n + rot_n
                hedef_e = gt_e + rot_e

                mesafe = math.hypot(hedef_n - uav_n, hedef_e - uav_e)

                # cm'e yuvarla (×100), son 2 hane deterministik tie-breaker
                cost_matrix[i, j] = int(round(mesafe * 100.0)) * 100 + (i * n + j)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        yeni_atama = {}
        for idx in range(n):
            yeni_atama[int(row_ind[idx]) + 1] = slotlar[int(col_ind[idx])]

        # Güvenlik: iki drone aynı slota düşmüş mü?
        if len(set(yeni_atama.values())) != n:
            self.get_logger().error(f"🛑 SLOT ÇAKIŞMASI! Atama: {yeni_atama}")
            return

        if yeni_atama != self.uav_to_slot:
            self.uav_to_slot = yeni_atama
            self.get_logger().warn(
                f"SLOT ATAMASI [{self.my_uav_name}] "
                f"form={self.current_formation} yaw={math.degrees(yaw):.0f}° → {self.uav_to_slot}")
    
    # -----------------------------------------------------------------------
    # SWARM VELOCİTY 
    # -----------------------------------------------------------------------
    def calculateSwarmVelocity(self, i, pos_i, hedef_n, hedef_e,
                                yaw_rad, msg, apf_n, apf_e):
        
        off_n, off_e = self.get_formation_offset(i)
        
        # 🚨 KUSURSUZ 3D RIGID BODY PROJEKSİYONU 🚨
        pitch_rad = math.radians(self.pitch_deg)
        roll_rad  = math.radians(self.roll_deg)
        
        # Drone Pitch veya Roll yaptıkça yataydaki izdüşümü (kapladığı alan) DARALMALIDIR (Cos)!
        off_n_3d = off_n * math.cos(pitch_rad)
        off_e_3d = off_e * math.cos(roll_rad)

        # Yeni daraltılmış ofsetleri YAW'a göre çevir
        rot_n = off_n_3d * math.cos(yaw_rad) - off_e_3d * math.sin(yaw_rad)
        rot_e = off_n_3d * math.sin(yaw_rad) + off_e_3d * math.cos(yaw_rad)

        hedef_i_n = hedef_n + rot_n
        hedef_i_e = hedef_e + rot_e

        Kp  = 0.6
        v_n = (hedef_i_n - pos_i[0]) * Kp
        v_e = (hedef_i_e - pos_i[1]) * Kp

        cek_hiz = math.hypot(v_n, v_e)
        if cek_hiz > 2.5:
            v_n = (v_n / cek_hiz) * 2.5
            v_e = (v_e / cek_hiz) * 2.5

        toplam_n = v_n + apf_n
        toplam_e = v_e + apf_e

        final_hiz = math.hypot(toplam_n, toplam_e)
        if final_hiz > 2.5:
            toplam_n = (toplam_n / final_hiz) * 2.5
            toplam_e = (toplam_e / final_hiz) * 2.5

        # -----------------------------------------------------------------------
        # PITCH/ROLL (İrtifa Ayarı)
        # -----------------------------------------------------------------------
        # Tanjant (uzatan) SİLİNDİ, Sinüs (sabit hipotenüs) eklendi!
        delta_z_pitch = off_n * math.sin(pitch_rad)
        delta_z_roll  = -off_e * math.sin(roll_rad)

        bireysel_hedef_irtifa = self.target_altitude + delta_z_pitch + delta_z_roll

        Kp_z = 0.5
        v_z  = (bireysel_hedef_irtifa - pos_i[2]) * Kp_z
        msg.twist.linear.z = float(max(min(v_z, 2.0), -2.0))

        if pos_i[2] < 1.5:
            # Yere yakın: formasyon çekimi kapalı, ama ÇARPIŞMA ÖNLEME açık
            msg.twist.linear.x = float(apf_n)
            msg.twist.linear.y = float(apf_e)
        else:
            msg.twist.linear.x = float(toplam_n)
            msg.twist.linear.y = float(toplam_e)

    # -----------------------------------------------------------------------
    # CONTROL LOOP
    # -----------------------------------------------------------------------
    def control_loop(self):

        now = time.time()
        bayat = [i for i in range(1, self.uav_count + 1)
                 if now - self.uav_stamps.get(i, 0.0) > self.TELEMETRI_TIMEOUT]

        if bayat:
            self.get_logger().error(f"🛑 TELEMETRİ KAYBI: UAV {bayat} — ACİL DURUM")
            # Kendi dronunu güvenli hale getir: yatayda dur, irtifayı koru
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = 0.0
            msg.twist.linear.y = 0.0
            msg.twist.linear.z = 0.0
            self.cmd_pub.publish(msg)
            return

        if len(self.uav_positions) == 0:
            return 

        # --- GLOBAL DURUM GEÇİŞLERİ ---

        if self.swarm_state == "KALKIS":
            surudeki = [pos for idd, pos in self.uav_positions.items()
                        if self.uav_states.get(idd) == "SURUDE"]
            hedef_irtifaya = sum(1 for pos in surudeki if pos[2] > self.target_altitude - 0.5)
            
            if hedef_irtifaya == len(surudeki) and len(surudeki) > 0:
                self.get_logger().info("İrtifaya Ulaşıldı! Rota QR-1'e çevrilmeden önce sürü havada toplanıyor.")
                self.current_qr_id = 1
                self.swarm_state   = "FORMASYON_TOPLANMASI"
                self.assign_slots_hungarian() 

        if self.swarm_state == "BEKLEMEDE":
            if self.current_task and \
               time.time() - self.task_wait_start_time >= self.current_task["sure"]:
                self.get_logger().info("Bekleme süresi doldu, sıradaki göreve geçiliyor.")
                self.swarm_state = "QR_ISLENIYOR"
                self.siradaki_gorevi_baslat()

        if self.swarm_state == "FORMASYON_TOPLANMASI":
            herkes_hazir = True
            for i in range(1, self.uav_count + 1):
                if self.uav_states.get(i) != "SURUDE":
                    herkes_hazir = False
                    break
                pos_i = self.uav_positions.get(i)
                if pos_i:
                    if pos_i[2] < self.target_altitude - 1.5:
                        herkes_hazir = False
                        break
                    off_n, off_e = self.get_formation_offset(i)
                    rot_n = off_n * math.cos(self.last_yaw_rad) - off_e * math.sin(self.last_yaw_rad)
                    rot_e = off_n * math.sin(self.last_yaw_rad) + off_e * math.cos(self.last_yaw_rad)
                    hedef_i_n = self.global_target_n + rot_n
                    hedef_i_e = self.global_target_e + rot_e
                    hata = math.hypot(hedef_i_n - pos_i[0], hedef_i_e - pos_i[1])
                    if hata > 1.0:
                        herkes_hazir = False
                        break

            if herkes_hazir:
                if self.current_qr_id == 0:
                    self.get_logger().info("Sürü eksiksiz toplandı! Eve dönüş rotası için burunlar hizalanıyor.")
                    self.next_target_n   = self.home_n
                    self.next_target_e   = self.home_e
                    self.swarm_state     = "YONE_DONUS"
                    self.yon_donus_baslangic = time.time()
                else:
                    hedef_koord = self.qr_map.get(self.current_qr_id, (0.0, 0.0))
                    self.next_target_n   = hedef_koord[0]
                    self.next_target_e   = hedef_koord[1]
                    self.swarm_state     = "YONE_DONUS"
                    self.yon_donus_baslangic = time.time()
                    self.get_logger().info(f"SÜRÜ TAMAMLANDI! YENİ ROTA (QR-{self.current_qr_id}): Önce burunlar hizalanıyor.")

        # --- EKSİK AJAN BEKLENİYOR (ASENKRON KAVUŞMA) ---
        if self.swarm_state == "EKSİK_AJAN_BEKLENIYOR":
            eksik_var_mi = False
            
            for i in range(1, self.uav_count + 1):
                durum = self.uav_states.get(i, "SURUDE")
                
                if durum != "SURUDE" and durum != "MOTORLAR_KAPALI": 
                    eksik_var_mi = True
                    break
                    
                if durum == "SURUDE":
                    pos_i = self.uav_positions.get(i)
                    if pos_i:
                        off_n, off_e = self.get_formation_offset(i)
                        rot_n = off_n * math.cos(self.last_yaw_rad) - off_e * math.sin(self.last_yaw_rad)
                        rot_e = off_n * math.sin(self.last_yaw_rad) + off_e * math.cos(self.last_yaw_rad)
                        
                        hedef_i_n = self.global_target_n + rot_n
                        hedef_i_e = self.global_target_e + rot_e
                        
                        hata = math.hypot(hedef_i_n - pos_i[0], hedef_i_e - pos_i[1])
                        # Ajan sürüye katılmış ama hala yuvasına (1.5m) tırmanıyorsa bekle!
                        if hata > 1.5: 
                            eksik_var_mi = True
                            break

            # SÜRÜ TAM KADROYSA ROTASYON (YONE_DONUS) AŞAMASINA GEÇ
            if not eksik_var_mi:
                self.swarm_state = "YONE_DONUS"
                self.yon_donus_baslangic = time.time()
                self.get_logger().info("Sürü tam kadro! Yeni hedefe yönelim (Yaw) hizalaması başlatılıyor.")

        # --- YAW HESAPLAMASI VE YÖNELİM ---
        hedef_n = self.global_target_n
        hedef_e = self.global_target_e

        surudeki = [pos for idd, pos in self.uav_positions.items()
                    if self.uav_states.get(idd) == "SURUDE"]
        if surudeki:
            yaw_rad = self.last_yaw_rad

            # --- KRİTİK YAMA 3: TİTREŞİMSİZ (SABİT) YAW HESABI ---
            if self.swarm_state in ["YONE_DONUS", "EKSİK_AJAN_BEKLENIYOR"]:
                fark_e = self.next_target_e - self.global_target_e
                fark_n = self.next_target_n - self.global_target_n
                hedef_yaw = self.last_yaw_rad
                
                # Hedef değiştiyse kesin matematiğe göre yönel, aynı yerdeyse dönme
                if abs(fark_e) > 0.1 or abs(fark_n) > 0.1:
                    hedef_yaw = math.atan2(fark_e, fark_n)
                    # Kısa yoldan, sınırlı hızla dön (max ~30°/s @ 10Hz)
                    delta = math.atan2(math.sin(hedef_yaw - self.last_yaw_rad),
                                       math.cos(hedef_yaw - self.last_yaw_rad))
                    MAX_YAW_ADIM = math.radians(3.0)   # 10Hz × 3° = 30°/s
                    delta = max(-MAX_YAW_ADIM, min(delta, MAX_YAW_ADIM))
                    yaw_rad = self.last_yaw_rad + delta
                    
                # Ancak fırlatma kararı (Navigasyon/Eve Dönüş) sadece YONE_DONUS durumundayken 5sn dolunca verilir
                # Fırlatma kararı: yaw OTURDU MU? (sabit 5sn yerine yakınsama kontrolü)
                if self.swarm_state == "YONE_DONUS":
                    yaw_hatasi = abs(math.atan2(
                        math.sin(hedef_yaw - yaw_rad),
                        math.cos(hedef_yaw - yaw_rad)))
                    gecen = time.time() - self.yon_donus_baslangic

                    # Yaw hedefe oturdu (< 5°) VEYA güvenlik timeout'u (12s)
                    if yaw_hatasi < math.radians(5.0) or gecen > 12.0:
                        if gecen > 12.0:
                            self.get_logger().warn(
                                f"Yaw hizalama TIMEOUT! Kalan hata: {math.degrees(yaw_hatasi):.0f}°")

                        self.global_target_n = self.next_target_n
                        self.global_target_e = self.next_target_e

                        if self.current_qr_id == 0:
                            self.swarm_state = "EVE_DONUS"
                            self.eve_donus_baslangic = time.time()
                            self.get_logger().info("Hizalama tamamlandı! Sürü eve dönüyor.")
                        else:
                            self.swarm_state = "NAVIGASYON"
                            self.get_logger().info(
                                f"Hizalama tamamlandı ({math.degrees(yaw_hatasi):.1f}° hata)! Hedefe fırlatılıyor.")

            # QR_BEKLIYOR ve EVE_VARIS kontrolü (Toplu Hata Hesabı)
            if self.swarm_state in ["NAVIGASYON", "EVE_DONUS"]:
                toplam_hata   = 0.0
                aktif_iha_say = 0
                for i in range(1, self.uav_count + 1):
                    if self.uav_states.get(i) == "SURUDE":
                        pos_i = self.uav_positions.get(i)
                        if pos_i:
                            off_n, off_e = self.get_formation_offset(i)
                            rot_n = off_n * math.cos(yaw_rad) - off_e * math.sin(yaw_rad)
                            rot_e = off_n * math.sin(yaw_rad) + off_e * math.cos(yaw_rad)
                            hata  = math.hypot(
                                hedef_n + rot_n - pos_i[0],
                                hedef_e + rot_e - pos_i[1])
                            toplam_hata   += hata
                            aktif_iha_say += 1

                if aktif_iha_say > 0:
                    ort_hata = toplam_hata / aktif_iha_say
                    
                    if self.swarm_state == "NAVIGASYON" and ort_hata < 0.8:
                        self.swarm_state     = "QR_BEKLIYOR"
                        self.qr_varis_zamani = time.time()
                        self.get_logger().info("Sürü hedefe oturdu! Stabilizasyon için 3 saniye bekleniyor.")
                        
                    elif self.swarm_state == "EVE_DONUS":
                        # KRİTİK YAMA 2: Toleransı artırdık ve 15 saniye Timeout ekledik
                        gecen_sure = time.time() - getattr(self, 'eve_donus_baslangic', time.time())
                        if ort_hata < 2.0 or gecen_sure > 15.0:
                            self.swarm_state = "TOPLU_INIS"
                            self.get_logger().info(f"Sürü eve ulaştı (Ort Hata: {ort_hata:.2f}m, Süre: {gecen_sure:.1f}s)! Toplu iniş başlatılıyor.")

            self.last_yaw_rad = yaw_rad
        else:
            yaw_rad = 0.0

        # --- HER DRONE İÇİN KONTROL ---
        for i in range(1, self.uav_count + 1):
            pos_i = self.uav_positions.get(i)
            if not pos_i:
                continue

            msg   = TwistStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            durum = self.uav_states.get(i, "SURUDE")

            apf_n, apf_e = 0.0, 0.0
            if i == self.my_id and not durum.startswith(
                    ("LAND_BEKLEMESI", "YERDE_BEKLIYOR", "MOTORLAR_KAPALI")):
                apf_n, apf_e = self.calculateAPFOffset(i)

            # ── SURUDE (Ana Sürü İşçileri) ────────────────────────────
            if durum == "SURUDE":
                if self.swarm_state == "KALKIS":
                    msg.twist.linear.x = float(apf_n)
                    msg.twist.linear.y = float(apf_e)
                    Kp_z = 0.5
                    v_z  = (self.target_altitude - pos_i[2]) * Kp_z
                    msg.twist.linear.z  = float(max(min(v_z, 2.0), -2.0))
                    msg.twist.angular.z = 0.0
                    
                elif self.swarm_state == "EVE_DONUS":
                    self.calculateSwarmVelocity(
                        i, pos_i, hedef_n, hedef_e, yaw_rad, msg, apf_n, apf_e)
                    msg.twist.angular.z = float(yaw_rad)
                    
                elif self.swarm_state == "TOPLU_INIS":
                    self.calculateSwarmVelocity(i, pos_i, hedef_n, hedef_e, yaw_rad, msg, apf_n, apf_e)
                    msg.twist.angular.z = float(yaw_rad)
                    
                    if pos_i[2] > 1.2:
                        msg.twist.linear.z = -0.5  # Senkronize alçalış
                    else:
                        if self.uav_states[i] != "LAND_BEKLEMESI_EVE":
                            self.uav_states[i] = "LAND_BEKLEMESI_EVE"
                            if i == self.my_id:
                                self.arm_pub.publish(String(data="LAND"))
                                self.get_logger().info(f"UAV-{i} güvenli mesafeye (1.2m) indi. LAND tetiklendi, temas bekleniyor...")
                                
                else: # Normal Navigasyon, Yön Dönüşü ve Bekleme durumları
                    self.calculateSwarmVelocity(
                        i, pos_i, hedef_n, hedef_e, yaw_rad, msg, apf_n, apf_e)
                    msg.twist.angular.z = float(yaw_rad)

            # ── INIS_ (Sürüden Tırmanarak Ayrılma ve Hedefe İniş) ──
            elif durum.startswith("INIS_"):
                hedef_renk = durum.split("INIS_")[1]
                hedef_pos  = (self.target_memory.get(hedef_renk) or
                              self.target_memory.get_raw(hedef_renk))

                if not hedef_pos:
                    msg.twist.linear.x = float(apf_n)
                    msg.twist.linear.y = float(apf_e)
                    msg.twist.linear.z = 0.0
                else:
                    target_n, target_e = hedef_pos
                    fark_n  = target_n - pos_i[0]
                    fark_e  = target_e - pos_i[1]
                    mesafe  = math.hypot(fark_n, fark_e)

                    guvenli_ayrilma_irtifasi = self.target_altitude + 2.0 

                    if pos_i[2] < (guvenli_ayrilma_irtifasi - 0.2) and mesafe > 0.5:
                        msg.twist.linear.x = float(apf_n)   # ← tırmanırken de kaçın
                        msg.twist.linear.y = float(apf_e)
                        msg.twist.linear.z = 1.5
                    else:
                        Kp_inis = 1.0 if mesafe > 1.0 else 0.8
                        v_n = fark_n * Kp_inis + apf_n
                        v_e = fark_e * Kp_inis + apf_e
                        
                        hiz = math.hypot(v_n, v_e)
                        max_hiz = 2.5 if mesafe > 1.0 else 1.0
                        if hiz > max_hiz:
                            v_n = (v_n / hiz) * max_hiz
                            v_e = (v_e / hiz) * max_hiz

                        alcisa_gecti = (pos_i[2] < guvenli_ayrilma_irtifasi - 0.5)

                        if mesafe > 0.25 and not alcisa_gecti:
                            Kp_z = 0.8
                            v_z  = (guvenli_ayrilma_irtifasi - pos_i[2]) * Kp_z
                            msg.twist.linear.z = float(max(min(v_z, 1.5), -1.5))
                            
                            msg.twist.linear.x = float(v_n)
                            msg.twist.linear.y = float(v_e)
                        else:
                            if pos_i[2] > 1.2:
                                msg.twist.linear.z = -0.6  
                                msg.twist.linear.x = float(v_n)  
                                msg.twist.linear.y = float(v_e)
                            else:
                                msg.twist.linear.x = 0.0
                                msg.twist.linear.y = 0.0
                                msg.twist.linear.z = 0.0
                                self.uav_states[i] = f"LAND_BEKLEMESI_{hedef_renk}"
                                if i == self.my_id:
                                    self.arm_pub.publish(String(data="LAND"))
                                    self.get_logger().info(f"{hedef_renk} rotasında 1.2m'ye inildi. Otonom LAND başlatıldı...")

            # ── LAND_BEKLEMESI_ (Otonom İnişin Bitmesini Bekle) ────────────────
            elif durum.startswith("LAND_BEKLEMESI_"):
                msg.twist.linear.x = 0.0
                msg.twist.linear.y = 0.0
                msg.twist.linear.z = 0.0
                
                hedef_str = durum.split("LAND_BEKLEMESI_")[1] 
                
                if pos_i[2] < 0.25:
                    if hedef_str == "EVE":
                        self.uav_states[i] = "MOTORLAR_KAPALI"
                        if i == self.my_id:
                            self.get_logger().info(f"UAV-{i} EVE yere temas etti. PX4'ün otonom DISARM atması bekleniyor...")
                    else:
                        self.uav_states[i] = f"YERDE_BEKLIYOR_{hedef_str}"
                        self.uav_timers[i] = time.time() # Yerde bekleme kronometresini ŞİMDİ başlat
                        if i == self.my_id:
                            self.get_logger().info(f"KUSURSUZ İNİŞ! {hedef_str} hedefine temas edildi. Kronometre başladı, otonom DISARM bekleniyor...")

            # ── MOTORLAR_KAPALI (Güvenli Ölü Bekleme) ────────────────────
            elif durum == "MOTORLAR_KAPALI":
                msg.twist.linear.x = 0.0
                msg.twist.linear.y = 0.0
                msg.twist.linear.z = 0.0

            # ── YERDE_BEKLIYOR_ ───────────────────────────────────────
            elif durum.startswith("YERDE_BEKLIYOR_"):
                msg.twist.linear.x = 0.0
                msg.twist.linear.y = 0.0
                msg.twist.linear.z = 0.0
                
                gecen = time.time() - self.uav_timers.get(i, 0.0)
                
                # Yeni Şartname Uyarınca: QR'dan gelen 'w' (self.qr_genel_bekleme) süresi doldu mu?
                if gecen >= self.qr_genel_bekleme:
                    self.uav_states[i] = "SURUDE"
                    self.assign_slots_hungarian() # Macar algoritmasını onar
                    if i == self.my_id:
                        self.arm_pub.publish(String(data="ARM"))
                        self.get_logger().info(f"w SÜRESİ ({self.qr_genel_bekleme}s) DOLDU! ARM EDİLDİM VE SÜRÜYE TIRMANDIM!")

            if i == self.my_id:
                if not durum.startswith(("LAND_BEKLEMESI", "YERDE_BEKLIYOR", "MOTORLAR_KAPALI")):
                    self.cmd_pub.publish(msg)


# ===========================================================================
def main(args=None):
    rclpy.init(args=args)
    coordinator = SwarmAgent()
    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()