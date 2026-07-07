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
    def __init__(self):
        super().__init__('swarm_agent')

        self.declare_parameter('uav_count', 3)
        self.uav_count = self.get_parameter('uav_count').get_parameter_value().integer_value

        self.declare_parameter('team_id', 'team_1')
        self.team_id = self.get_parameter('team_id').get_parameter_value().string_value

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

        # QR Koordinat Haritası
        self.qr_map = {
            1: (6.0,  0.0),
            4: (3.0,  5.2),
            2: (-3.0, 5.2),
            3: (-6.0, 0.0),
            5: (-3.0, -5.2),
            6: (3.0,  -5.2),
        }

        self.qr_varis_zamani = 0.0

        # ---------------------------------------------------------------
        # TargetMemory — EMA + sigma filtresi (gorev1.py'den)
        # ---------------------------------------------------------------
        self.target_memory = TargetMemory(ema_alpha=0.85, sigma_gate=2.5, min_obs=1)
        # Eski target_errors artık gerekmiyor — TargetMemory içinde var
        # (geriye dönük uyumluluk için boş bırakıyoruz)
        self.target_errors = {}

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
        self.formation_spacing  = 2

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
        self.ayrilma_bekleme_suresi = 0.0

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

        self.timer = self.create_timer(0.1, self.control_loop)

        self.takeoff_sub = self.create_subscription(
            String, '/swarm/takeoff', self.takeoff_callback, 10)

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

        self.uav_positions[uav_id] = (x, y, z, yaw)
        self.uav_stamps[uav_id]    = msg.header.stamp
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

            self.get_logger().info(f"Kalkış Emri Alındı! Kaydedilen Home Üssü (UAV-1): N:{self.home_n:.2f}, E:{self.home_e:.2f}")
            self.swarm_state = "KALKIS"
            self.arm_pub.publish(String(data="ARM"))

    # -----------------------------------------------------------------------
    # QR CALLBACK
    # -----------------------------------------------------------------------
    def vision_command_callback(self, msg: String):
        if self.swarm_state != "QR_BEKLIYOR":
            return
        if time.time() - self.qr_varis_zamani < 3.0:
            return

        try:
            payload      = json.loads(msg.data)
            okunan_qr_id = payload.get("qr_id")

            if okunan_qr_id != self.current_qr_id:
                return

            self.get_logger().info(
                f"HEDEF QR-{self.current_qr_id} okundu. Görevler yerine getiriliyor.")

            self.swarm_state = "QR_ISLENIYOR"
            self.active_task_queue.clear()

            gorev = payload.get("gorev", {})

            # Formasyon
            formasyon = gorev.get("formasyon", {})
            if formasyon.get("aktif"):
                self.active_task_queue.append({
                    "tip":   "FORMASYON",
                    "deger": formasyon.get("tip").strip().lower()
                })

            # Pitch/Roll Manevrası
            manevra = gorev.get("manevra_pitch_roll", {})
            if manevra.get("aktif"):
                self.active_task_queue.append({
                    "tip":   "MANEVRA",
                    "pitch": float(manevra.get("pitch_deg", 0)),
                    "roll":  float(manevra.get("roll_deg",  0))
                })

            # İrtifa
            irtifa = gorev.get("irtifa_degisim", {})
            if irtifa.get("aktif"):
                self.active_task_queue.append({
                    "tip":   "IRTIFA",
                    "deger": float(irtifa.get("deger", self.base_altitude))
                })

            # Sürüden Ayrılma
            ayrilma = gorev.get("suruden_ayrilma", {})
            if ayrilma.get("aktif"):
                self.active_task_queue.append({
                    "tip":        "AYRILMA",
                    "drone_id":   ayrilma.get("ayrilacak_drone_id"),
                    "hedef_renk": ayrilma.get("hedef_renk").upper(),
                    "bekleme":    ayrilma.get("bekleme_suresi_s", 0)
                })

            # Bekleme
            bekleme_s = gorev.get("bekleme_suresi_s", 0)
            if bekleme_s > 0:
                self.active_task_queue.append({
                    "tip":  "BEKLEME",
                    "sure": bekleme_s
                })

            # Sonraki QR
            sonraki_qr_dict = payload.get("sonraki_qr", {})
            sonraki_qr_id   = sonraki_qr_dict.get(self.team_id, 0)
            self.active_task_queue.append({
                "tip":   "SONRAKI_HEDEF",
                "qr_id": sonraki_qr_id
            })

            self.siradaki_gorevi_baslat()

        except Exception as e:
            self.get_logger().error(f"QR json ayrıştırma hatası: {e}")

    # -----------------------------------------------------------------------
    # GÖREV KUYRUĞU
    # -----------------------------------------------------------------------
    def siradaki_gorevi_baslat(self):
        if len(self.active_task_queue) == 0:
            return

        self.current_task = self.active_task_queue.pop(0)
        tip = self.current_task["tip"]
        self.get_logger().info(f"Görev yerine getiriliyor: {tip}")

        if tip == "FORMASYON":
            yeni_tip = self.current_task["deger"]
            if yeni_tip in self.formations:
                self.current_formation = yeni_tip
                self.assign_slots_hungarian() 
            self.siradaki_gorevi_baslat()

        elif tip == "MANEVRA":
            self.pitch_deg = self.current_task["pitch"]
            self.roll_deg  = self.current_task["roll"]
            self.siradaki_gorevi_baslat()

        elif tip == "IRTIFA":
            self.target_altitude = self.current_task["deger"]
            self.siradaki_gorevi_baslat()

        elif tip == "AYRILMA":
            hedef_id   = self.current_task["drone_id"]
            hedef_renk = self.current_task["hedef_renk"]
            self.ayrilma_bekleme_suresi = self.current_task["bekleme"]

            if hedef_id in self.uav_states and self.uav_states[hedef_id] == "SURUDE":
                pos = self.target_memory.get(hedef_renk) or \
                      self.target_memory.get_raw(hedef_renk)
                if pos:
                    self.uav_states[hedef_id] = f"INIS_{hedef_renk}"
                    self.get_logger().warn(f"Drone {hedef_id} ayrılıyor. Yuvası BOŞ bırakılacak!")
                    self.assign_slots_hungarian() # Macar çalışacak ama ayrılanın yerini boş bırakacak
                else:
                    self.get_logger().error(f"Haritada {hedef_renk} alan bulunamadı, dron ayrılamıyor.")
            self.siradaki_gorevi_baslat()

        elif tip == "BEKLEME":
            self.task_wait_start_time = time.time()
            self.swarm_state = "BEKLEMEDE"
            self.get_logger().info(
                f"Sürü {self.current_task['sure']} saniye boyunca pozisyonunu koruyor.")

        elif tip == "SONRAKI_HEDEF":
            yeni_qr = self.current_task["qr_id"]
            if yeni_qr == 0:
                self.get_logger().info("Tüm görevler bitti. Eve dönüş için formasyon hizalanması bekleniyor.")
                self.current_qr_id = 0
                self.swarm_state   = "FORMASYON_TOPLANMASI"
                self.assign_slots_hungarian()
            else:
                self.current_qr_id = yeni_qr
                self.swarm_state   = "FORMASYON_TOPLANMASI"
                self.get_logger().info(f"Sıradaki hedefe (QR-{yeni_qr}) fırlamadan önce sürünün hizalanması bekleniyor.")
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
        yeni_tip = msg.data.strip().lower()
        if yeni_tip in self.formations:
            self.current_formation = yeni_tip
            self.get_logger().info(
                f"YKİ MÜDAHALESİ: Sürü anında '{yeni_tip.upper()}' formasyonuna geçiyor!")
        else:
            self.get_logger().error(
                f"Bilinmeyen formasyon emri: {yeni_tip}. "
                f"Geçerli olanlar: okbasi, v, cizgi")

    # -----------------------------------------------------------------------
    # SAF GEOMETRİK SLOT (YUVA) HESAPLAYICI (KUSURSUZ FİZİK)
    # -----------------------------------------------------------------------
    def get_slot_offset(self, slot_id: int, n: int):
        s = self.formation_spacing
        
        if self.current_formation == "v":
            # --- 🛠️ DÜZELTME 1: TAKTİKSEL V (Sivri Uç Geride) ---
            # PDF'teki '-k * s' kısımları '+k * s' yapıldı. Kanatlar artık ileri atılacak.
            if n % 2 != 0:
                if slot_id == 1: return 0.0, 0.0
                k = slot_id // 2
                yon = 1 if slot_id % 2 == 0 else -1
                return float(k * s), float(yon * k * s)
            else:
                k = (slot_id + 1) // 2
                yon = 1 if slot_id % 2 != 0 else -1
                return float(k * s), float(yon * k * s)
            
        elif self.current_formation == "cizgi":
            # --- 🛠️ DÜZELTME 2: DİKİNE (ARKA ARKAYA) ÇİZGİ ---
            # Y ekseni sıfırlandı. Formül X eksenine uygulandı.
            # slot_id=1 en önde, slot_id=N en arkada olacak şekilde merkeze hizalanır.
            x_offset = float(((n + 1) / 2.0 - slot_id) * s)
            return x_offset, 0.0
            
        else: # okbasi
            # Dinamik üçgensel sayı dizilimi (Değişmedi, kusursuz çalışıyor)
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
    
    # -----------------------------------------------------------------------
    # SÜRÜ OPTİMİZASYONU (MACAR ALGORİTMASI)
    # -----------------------------------------------------------------------
    def assign_slots_hungarian(self):
        # 🛠️ DÜZELTME: Tüm dronlar optimizasyon matrisine girer
        n = self.uav_count
        slotlar = list(range(1, n + 1))
        cost_matrix = np.zeros((n, n))
        
        for i in range(n):
            uav_id = i + 1
            pos = self.uav_positions.get(uav_id)
            if not pos:
                cost_matrix[i, :] = 9999.0
                continue
                
            uav_n, uav_e = pos[0], pos[1]
            
            for j in range(n):
                slot_id = slotlar[j]
                
                # 🛠️ KORUMA KALKANI: Eğer drone sürüden ayrılmışsa (SURUDE değilse), 
                # sadece ve sadece kendi yuvasına eşleşebilir. Diğer yuvalar ona yasaklanır!
                if self.uav_states.get(uav_id) != "SURUDE":
                    if self.uav_to_slot.get(uav_id) == slot_id:
                        cost_matrix[i, j] = 0.0      # Kendi yuvasında kalması bedava
                    else:
                        cost_matrix[i, j] = 99999.0  # Başka yuvaya geçmesi, başkasının oraya girmesi yasak!
                    continue

                # Sürüde olanlar için normal maliyet hesabı
                off_n, off_e = self.get_slot_offset(slot_id, n)
                yaw = getattr(self, 'last_yaw_rad', 0.0)
                rot_n = off_n * math.cos(yaw) - off_e * math.sin(yaw)
                rot_e = off_n * math.sin(yaw) + off_e * math.cos(yaw)
                
                hedef_n = self.global_target_n + rot_n
                hedef_e = self.global_target_e + rot_e
                
                cost_matrix[i, j] = math.hypot(hedef_n - uav_n, hedef_e - uav_e)
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for idx in range(n):
            atanan_uav = row_ind[idx] + 1
            atanan_slot = slotlar[col_ind[idx]]
            self.uav_to_slot[atanan_uav] = atanan_slot
            
        self.get_logger().warn(f"SÜRÜ OPTİMİZE EDİLDİ (Rezerve Yuva Sistemi)! Eşleşmeler: {self.uav_to_slot}")

    # -----------------------------------------------------------------------
    # IAPF (DİNAMİK SÖNÜMLEMELİ - GÖRELİ HIZ TABANLI)
    # -----------------------------------------------------------------------
    def calculateAPFOffset(self, uav_id: int, d_des: float = 1.6):
        # 🛠️ GÖRSELDEKİ DİNAMİK K PARAMETRELERİ
        k_taban = 2.0  # Formasyon stabilken gereksiz titremeyi önleyen düşük taban
        lam = 3.5      # Lambda: Yaklaşma hızına verilecek agresif tepki (fren) çarpanı
        rep_n = 0.0
        rep_e = 0.0

        my_pos = self.uav_positions.get(uav_id)
        if not my_pos:
            return 0.0, 0.0

        # Hafıza oluştur (Türev/Hız hesabı için önceki mesafeler)
        if not hasattr(self, 'prev_distances'):
            self.prev_distances = {}
            self.last_apf_time = time.time()
            
        now = time.time()
        dt = now - self.last_apf_time
        if dt <= 0.01: dt = 0.1 # Sıfıra bölme koruması

        for other_id, other_pos in self.uav_positions.items():
            if other_id == uav_id:
                continue
            
            # İrtifa (Z) filtresi
            if abs(my_pos[2] - other_pos[2]) > 2.0:
                continue

            # Anlık mesafe
            f_ij = math.hypot(my_pos[0] - other_pos[0], my_pos[1] - other_pos[1])
            
            # 🛠️ GÖRELİ HIZ HESABI (TÜREV)
            pair_key = tuple(sorted((uav_id, other_id)))
            prev_f_ij = self.prev_distances.get(pair_key, f_ij)
            
            # Mesafenin değişim hızı (f_dot). Negatifse birbirlerine yaklaşıyorlar demektir.
            f_dot = (f_ij - prev_f_ij) / dt 
            self.prev_distances[pair_key] = f_ij
            
            # Güvenlik çemberi ihlali
            if 0.01 < f_ij < d_des:
                # 🛠️ GÖRSELDEKİ DİNAMİK FORMÜLÜN UYGULANMASI
                # Sadece tehlikeli şekilde yaklaşıyorlarsa (f_dot < 0) k_a artar!
                k_a_dinamik = k_taban + lam * max(0.0, -f_dot)
                
                # Altuncu & Canım Denklem (9) Gradyanı
                force_mag = -k_a_dinamik * (1.0 - (d_des**2 / f_ij**2))
                
                dx = (my_pos[0] - other_pos[0]) / f_ij
                dy = (my_pos[1] - other_pos[1]) / f_ij
                
                rep_n += dx * force_mag
                rep_e += dy * force_mag

        self.last_apf_time = now
        return rep_n, rep_e
    
    # -----------------------------------------------------------------------
    # SWARM VELOCİTY 
    # -----------------------------------------------------------------------
    def calculateSwarmVelocity(self, i, pos_i, hedef_n, hedef_e,
                                yaw_rad, msg, apf_n, apf_e):
        # 1. Yatay (X, Y) Formasyon Ofsetleri ve Hedef Çekimi
        off_n, off_e = self.get_formation_offset(i)
        rot_n = off_n * math.cos(yaw_rad) - off_e * math.sin(yaw_rad)
        rot_e = off_n * math.sin(yaw_rad) + off_e * math.cos(yaw_rad)

        hedef_i_n = hedef_n + rot_n
        hedef_i_e = hedef_e + rot_e

        # P-Kontrolcü ile Hedef Vektörü
        Kp  = 1.0
        v_n = (hedef_i_n - pos_i[0]) * Kp
        v_e = (hedef_i_e - pos_i[1]) * Kp

        cek_hiz = math.hypot(v_n, v_e)
        
        # Ön Fren: Hedef çekim hızını standart seyir hızına (2.5) limitle
        if cek_hiz > 2.5:
            v_n = (v_n / cek_hiz) * 2.5
            v_e = (v_e / cek_hiz) * 2.5

        # 2. Makale Denklem 11: Toplam Kontrol Girdisi (Hedef Çekimi + IAPF İtmesi)
        toplam_n = v_n + apf_n
        toplam_e = v_e + apf_e

        final_hiz = math.hypot(toplam_n, toplam_e)
        
        # 3. Makale Denklem 13: Velocity Normalization (Mutlak Hız Sınırı)
        # APF ve hedef çekimi birleştiğinde bile hız 2.5 m/s'yi asla aşamaz
        if final_hiz > 2.5:
            toplam_n = (toplam_n / final_hiz) * 2.5
            toplam_e = (toplam_e / final_hiz) * 2.5

        # -----------------------------------------------------------------------
        # PITCH/ROLL
        # -----------------------------------------------------------------------
        pitch_rad = math.radians(self.pitch_deg)
        roll_rad  = math.radians(self.roll_deg)

        # Şartname Doğrulaması: 
        # Pitch: Sürünün önündeki İHA'lar (off_n > 0) eksi açıyla aşağı, arkadakiler yukarı.
        # Roll : Sürünün sağındaki İHA'lar (off_e > 0) eksi açıyla aşağı, soldakiler yukarı.
        delta_z_pitch = off_n * math.tan(pitch_rad)
        delta_z_roll  = -off_e * math.tan(roll_rad)

        # Her dronun formasyondaki (X,Y) konumuna göre özgün 3D hedef irtifası
        # Sürü merkezi (self.target_altitude) matematiksel olarak SABİT kalır!
        bireysel_hedef_irtifa = self.target_altitude + delta_z_pitch + delta_z_roll

        # Z Ekseni P-Kontrolcüsü (İrtifaya yönelim)
        Kp_z = 0.5
        v_z  = (bireysel_hedef_irtifa - pos_i[2]) * Kp_z
        msg.twist.linear.z = float(max(min(v_z, 2.0), -2.0)) # Yükselme/Alçalma hız limiti

        # 5. Güvenlik ve Hız Komutlarının Atanması
        if pos_i[2] < 1.5:
            # Yere çok yakınken (kalkış/iniş manevraları) yatayda kaymayı engelle
            msg.twist.linear.x = 0.0
            msg.twist.linear.y = 0.0
        else:
            # Uçuş irtifasında hesaplanan nihai vektörleri bas
            msg.twist.linear.x = float(toplam_n)
            msg.twist.linear.y = float(toplam_e)

    # -----------------------------------------------------------------------
    # CONTROL LOOP
    # -----------------------------------------------------------------------
    def control_loop(self):
        if len(self.uav_positions) < self.uav_count:
            return

        # --- GLOBAL DURUM GEÇİŞLERİ ---

        # KALKIŞ → FORMASYON_TOPLANMASI
        if self.swarm_state == "KALKIS":
            surudeki = [pos for idd, pos in self.uav_positions.items()
                        if self.uav_states.get(idd) == "SURUDE"]
            hedef_irtifaya = sum(1 for pos in surudeki if pos[2] > self.target_altitude - 0.5)
            
            if hedef_irtifaya == len(surudeki) and len(surudeki) > 0:
                self.get_logger().info("İrtifaya Ulaşıldı! Rota QR-1'e çevrilmeden önce sürü havada toplanıyor.")
                self.current_qr_id = 1
                self.swarm_state   = "FORMASYON_TOPLANMASI"
                self.assign_slots_hungarian() 

        # BEKLEME süresi kontrolü
        if self.swarm_state == "BEKLEMEDE":
            if self.current_task and \
               time.time() - self.task_wait_start_time >= self.current_task["sure"]:
                self.get_logger().info("Bekleme süresi doldu, sıradaki göreve geçiliyor.")
                self.swarm_state = "QR_ISLENIYOR"
                self.siradaki_gorevi_baslat()

        # FORMASYON_TOPLANMASI — herkes slotuna oturdu mu?
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
                    self.pitch_deg       = 0.0
                    self.roll_deg        = 0.0
                    self.swarm_state     = "YONE_DONUS"
                    self.yon_donus_baslangic = time.time()
                else:
                    hedef_koord = self.qr_map.get(self.current_qr_id, (0.0, 0.0))
                    self.next_target_n   = hedef_koord[0]
                    self.next_target_e   = hedef_koord[1]
                    self.pitch_deg       = 0.0
                    self.roll_deg        = 0.0
                    self.swarm_state     = "YONE_DONUS"
                    self.yon_donus_baslangic = time.time()
                    self.get_logger().info(
                        f"SÜRÜ TAMAMLANDI! YENİ ROTA (QR-{self.current_qr_id}): "
                        f"Önce burunlar hizalanıyor.")

        # --- YAW ve HEDEF ---
        hedef_n = self.global_target_n
        hedef_e = self.global_target_e

        surudeki = [pos for idd, pos in self.uav_positions.items()
                    if self.uav_states.get(idd) == "SURUDE"]
        if surudeki:
            avg_n = sum(p[0] for p in surudeki) / len(surudeki)
            avg_e = sum(p[1] for p in surudeki) / len(surudeki)

            yaw_rad = self.last_yaw_rad

            if self.swarm_state == "YONE_DONUS":
                yaw_rad = math.atan2(   
                    self.next_target_e - avg_e,
                    self.next_target_n - avg_n)
                if time.time() - self.yon_donus_baslangic > 5.0:
                    self.global_target_n = self.next_target_n
                    self.global_target_e = self.next_target_e
                    
                    # 🛠️ YENİ: Bekleme süresi bitti. Hedef 0 ise eve, değilse QR'a!
                    if self.current_qr_id == 0:
                        self.swarm_state = "EVE_DONUS"
                        self.get_logger().info("Hizalama tamamlandı! Sürü eve (0,0) dönüyor.")
                    else:
                        self.swarm_state = "NAVIGASYON"
                        self.get_logger().info("Hizalama tamamlandı! Tüm sürü tek vücut olarak hedefe fırlatılıyor.")

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
                        
                    elif self.swarm_state == "EVE_DONUS" and ort_hata < 1.0:
                        self.swarm_state = "TOPLU_INIS"
                        self.get_logger().info("Sürü eve ulaştı ve formasyon tam olarak oturdu! Toplu iniş başlatılıyor.")

            self.last_yaw_rad = yaw_rad

            # QR_BEKLIYOR kontrolü
            if self.swarm_state == "NAVIGASYON":
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
                    if ort_hata < 0.8 and self.swarm_state != "QR_BEKLIYOR":
                        self.swarm_state     = "QR_BEKLIYOR"
                        self.qr_varis_zamani = time.time()
                        self.get_logger().info(
                            "Sürü hedefe oturdu! Stabilizasyon için 3 saniye bekleniyor.")
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
            if pos_i[2] > 1.5:
                apf_n, apf_e = self.calculateAPFOffset(i, 1.6)

            # ── SURUDE (Ana Sürü İşçileri) ────────────────────────────
            if durum == "SURUDE":
                if self.swarm_state == "KALKIS":
                    msg.twist.linear.x = float(apf_n)
                    msg.twist.linear.y = float(apf_e)
                    Kp_z = 0.5
                    v_z  = (self.target_altitude - pos_i[2]) * Kp_z
                    msg.twist.linear.z  = float(max(min(v_z, 2.0), -2.0))
                    msg.twist.angular.z = 0.0
                    
                # 🛠️ DÜZELTME 2: Eve dönüş ve Toplu iniş sadece "SURUDE" olanları etkiler!
                elif self.swarm_state == "EVE_DONUS":
                    # Eve giderken sadece yatayda uçar, ASLA ALÇALMAZ. Formasyonu korur.
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
                                
                else: # Normal Navigasyon ve Bekleme durumları
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

                    # TAVAN İRTİFASI (Sürünün 2 metre üstü)
                    guvenli_ayrilma_irtifasi = self.target_altitude + 2.0 

                    # 1. AŞAMA: POP-UP (Dikine Tırmanış)
                    if pos_i[2] < (guvenli_ayrilma_irtifasi - 0.2) and mesafe > 0.5:
                        msg.twist.linear.x = 0.0
                        msg.twist.linear.y = 0.0
                        msg.twist.linear.z = 1.5  
                    else:
                        # 2. AŞAMA: Çatı Katından Yatay Uçuş ve Merkeze Kilitlenerek İniş
                        Kp_inis = 1.0 if mesafe > 1.0 else 0.8
                        v_n = fark_n * Kp_inis + apf_n
                        v_e = fark_e * Kp_inis + apf_e
                        
                        hiz = math.hypot(v_n, v_e)
                        max_hiz = 2.5 if mesafe > 1.0 else 1.0
                        if hiz > max_hiz:
                            v_n = (v_n / hiz) * max_hiz
                            v_e = (v_e / hiz) * max_hiz

                        # 🛠️ ÇÖZÜM: Yoyo (Sürekli Tırmanma) Etkisini Engelleme!
                        # Drone çatı irtifasından 0.5m aşağı düştüyse artık inişe komit olmuştur.
                        alcisa_gecti = (pos_i[2] < guvenli_ayrilma_irtifasi - 0.5)

                        if mesafe > 0.25 and not alcisa_gecti:
                            # Henüz merkeze gelmedik, yatayda uçarken çatı irtifasını koru
                            Kp_z = 0.8
                            v_z  = (guvenli_ayrilma_irtifasi - pos_i[2]) * Kp_z
                            msg.twist.linear.z = float(max(min(v_z, 1.5), -1.5))
                            
                            msg.twist.linear.x = float(v_n)
                            msg.twist.linear.y = float(v_e)
                        else:
                            # 3. AŞAMA: Merkezdeyiz VEYA zaten alçalışa geçtik!
                            if pos_i[2] > 1.2:
                                msg.twist.linear.z = -0.6  # İnmeye devam et
                                msg.twist.linear.x = float(v_n)  # 🛠️ İnerken sağa sola kayarsa merkeze çekmeye devam et!
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
                # Drona karışmıyoruz, PX4'ün otonom LAND algoritması kendisi süzülerek iniyor.
                msg.twist.linear.x = 0.0
                msg.twist.linear.y = 0.0
                msg.twist.linear.z = 0.0
                
                hedef_str = durum.split("LAND_BEKLEMESI_")[1] 
                
                # Yere değdiği an (0.25m) HİÇBİR KOMUT GÖNDERMEDEN sadece kronometreyi başlatıyoruz.
                # PX4 kendi 2 saniyelik güvenlik süresini bekleyip motorları DOĞAL yolla kapatacak.
                if pos_i[2] < 0.25:
                    if hedef_str == "EVE":
                        self.uav_states[i] = "MOTORLAR_KAPALI"
                        if i == self.my_id:
                            self.get_logger().info(f"UAV-{i} EVE yere temas etti. PX4'ün otonom DISARM atması bekleniyor...")
                    else:
                        self.uav_states[i] = f"YERDE_BEKLIYOR_{hedef_str}"
                        self.uav_timers[i] = time.time() # Şartnamedeki yerde bekleme kronometresini ŞİMDİ başlat
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
                if gecen >= self.ayrilma_bekleme_suresi:
                    self.uav_states[i] = "SURUDE"
                    self.assign_slots_hungarian()
                    if i == self.my_id:
                        # Bekleme bittiğinde ARM gönderiyoruz, uav_agent.py zaten bunu Offboard moda geçirip ARM edecektir.
                        self.arm_pub.publish(String(data="ARM"))
                        self.get_logger().info(
                            "SÜREM DOLDU, ARM EDİLDİM VE SÜRÜYE KATILIYORUM!")

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