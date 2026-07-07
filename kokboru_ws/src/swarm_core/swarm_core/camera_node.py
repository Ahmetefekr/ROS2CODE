#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import json
import math
import time

from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage
from pyzbar.pyzbar import decode as pyzbar_decode

# ---------------------------------------------------------------------------
# Renk Profilleri (Kökbörü Takımı Algoritması)
# Satürasyon eşiği düşürüldü (60->40): açık havada soluk renkler kaçmasın
# Hue aralıkları genişletildi: kırmızı 165->160, mavi 95-130->90-135
# ---------------------------------------------------------------------------
COLOR_PROFILES = {
    "KIRMIZI": {
        "ranges": [
            (np.array([0,   40,  40],  np.uint8), np.array([10,  255, 255], np.uint8)),
            (np.array([160, 40,  40],  np.uint8), np.array([180, 255, 255], np.uint8)),
        ],
        "bgr": (0, 0, 220),
        "display": "KIRMIZI DAIRE",
    },
    "MAVI": {
        "ranges": [
            (np.array([90, 40, 40],  np.uint8), np.array([135, 255, 255], np.uint8)),
        ],
        "bgr": (220, 60, 0),
        "display": "MAVI DAIRE",
    },
}

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

IMAGE_TOPIC = '/world/default/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image'


class CameraVisionNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('agent_id', 'uav_1')
        self.agent_id = self.get_parameter('agent_id').get_parameter_value().string_value

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Swarm agent'taki projeksiyon hesabıyla eşleşmeli
        self.img_width  = 1280
        self.img_height = 960

        self.display_width = 1280
        self.display_height = 960

        # QR spam koruması için durum değişkenleri
        self._last_qr_data = ""
        self._last_qr_time = 0.0

        # Pencereyi aç
        self.window_name = f"Kamera - {self.agent_id}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)

        # Bekleme ekranı
        self.latest_image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        cv2.putText(self.latest_image, "Gazebo Verisi Bekleniyor...", (50, self.display_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # ROS Publisher'lar
        self.qr_pub = self.create_publisher(String, '/camera/swarm_commands', 10)
        self.circles_pub = self.create_publisher(String, f'/{self.agent_id}/camera/circles', 10)

        # Gazebo Transport Node — direkt bağlantı, bridge yok
        self.gz_node = GzNode()
        result = self.gz_node.subscribe(GzImage, IMAGE_TOPIC, self._gz_image_callback)

        if result:
            self.get_logger().info(f"✅ Gazebo topic'e direkt bağlandı: {IMAGE_TOPIC}")
        else:
            self.get_logger().error(f"❌ Gazebo topic subscribe başarısız: {IMAGE_TOPIC}")

        # 20Hz ekran yenileme
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.last_frame_time = 0.0

        self.get_logger().info(f"✅ CameraVisionNode başlatıldı: {self.agent_id}")

    def _gz_image_callback(self, gz_img):
        """Gazebo'dan gelen protobuf Image mesajını OpenCV frame'e çevir."""
        try:
            now = time.time()

            if now - self.last_frame_time < 0.2:  # Saniyede sadece 5 kare (5 FPS) işle
                return

            self.last_frame_time = now
            width  = gz_img.width
            height = gz_img.height
            raw    = np.frombuffer(gz_img.data, dtype=np.uint8)

            # Gazebo RGB_INT8 formatı → BGR
            if len(raw) == width * height * 3:
                frame = raw.reshape((height, width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif len(raw) == width * height * 4:
                frame = raw.reshape((height, width, 4))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                self.get_logger().warn(f"Beklenmeyen piksel boyutu: {len(raw)} (w={width}, h={height})")
                return

            processed = self.process_cv(frame)

            if processed.shape[1] != self.display_width or processed.shape[0] != self.display_height:
                self.latest_image = cv2.resize(processed, (self.display_width, self.display_height))
            else:
                self.latest_image = processed

        except Exception as e:
            self.get_logger().error(f"Görüntü işleme hatası: {e}")

    def timer_callback(self):
        cv2.imshow(self.window_name, self.latest_image)
        cv2.waitKey(1)

    def _detect_color_circles(self, frame: np.ndarray, hsv: np.ndarray) -> dict:
        cx_img, cy_img = frame.shape[1] / 2.0, frame.shape[0] / 2.0
        min_area, min_circ, min_pts = 300, 0.50, 5
        detected = {}

        for color_name, profile in COLOR_PROFILES.items():
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in profile["ranges"]:
                combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv, lo, hi))

            mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  MORPH_KERNEL)
            mask = cv2.morphologyEx(mask,          cv2.MORPH_CLOSE, MORPH_KERNEL)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best = None
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area or len(c) < min_pts:
                    continue

                peri = cv2.arcLength(c, True)
                if peri < 1.0:
                    continue

                circ = 4.0 * math.pi * area / (peri * peri)
                if circ < min_circ:
                    continue

                (x, y), r = cv2.minEnclosingCircle(c)
                if r < 5:
                    continue

                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = min(bw, bh) / max(bw, bh, 1)
                if aspect < 0.45:
                    continue

                try:
                    hull = cv2.convexHull(c.astype(np.float32))
                    if hull is None or len(hull) < 3:
                        continue
                    hull_area = cv2.contourArea(hull)
                    if hull_area < 1.0:
                        continue

                    solidity = area / hull_area
                    if solidity < 0.65:
                        continue

                    conf = circ * solidity
                    if best is None or conf > best["confidence"]:
                        best = {"x": x, "y": y, "r": r, "confidence": conf,
                                "ox": x - cx_img, "oy": y - cy_img}
                except Exception:
                    continue

            if best:
                bgr = profile["bgr"]
                cv2.circle(frame, (int(best["x"]), int(best["y"])), int(best["r"]), bgr, 3)
                cv2.putText(frame, f"{color_name} ({best['confidence']:.2f})",
                            (int(best["x"] - 40), int(best["y"] - best["r"] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
                detected[color_name] = {
                    "offset_x": best["ox"],
                    "offset_y": best["oy"],
                    "confidence": best["confidence"]
                }
        return detected

    def process_cv(self, frame: np.ndarray) -> np.ndarray:
        # --- 1. QR İŞLEME (PyZbar 3 Kademeli Tarama Stratejisi) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Strateji 1: Saf Gri Tonlama (PyZbar'ın en sevdiği format)
        qr_results = pyzbar_decode(gray)

        # Strateji 2: Bulamazsa, gölgeleri yok etmek için Binary Threshold uygula
        if not qr_results:
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            qr_results = pyzbar_decode(thresh)

        # Strateji 3: 7 metreden pikseller çok küçük geliyorsa, görüntüyü 2X BÜYÜT (Asıl Hile Bu!)
        scale_factor = 1
        if not qr_results:
            gray_large = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            qr_results = pyzbar_decode(gray_large)
            scale_factor = 2  # Çerçeve çizerken koordinatları geri küçültmek için

        # Sonuçları İşle
        for qr in qr_results:
            qr_data = qr.data.decode('utf-8')

            # QR'ın etrafına yeşil çerçeve çiz
            points = qr.polygon
            if len(points) == 4:
                # Eğer 3. strateji çalıştıysa (görüntü büyütüldüyse) koordinatları aslına döndür
                pts = np.array([(p.x // scale_factor, p.y // scale_factor) for p in points], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                top_left = (points[0].x // scale_factor, points[0].y // scale_factor)
            else:
                top_left = (50, 100)

            # Aynı QR'ı 3 saniyede bir gönder, spam yapma
            now = time.time()
            if qr_data != self._last_qr_data or (now - self._last_qr_time) > 3.0:
                self._last_qr_data = qr_data
                self._last_qr_time = now
                # Sürü beynine (swarm_agent) görevleri bas!
                self.qr_pub.publish(String(data=qr_data))

            # Kamera ekranına yazdır
            cv2.putText(frame, f"QR: {qr_data[:20]}", (max(10, top_left[0]), max(30, top_left[1] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- 2. DAİRE TESPİTİ ---
        # CLAHE ile parlaklık normalize et: gölge ve güneş geçişlerine karşı sağlamlık
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b_ch))
        frame_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame_eq, (5, 5), 0), cv2.COLOR_BGR2HSV)
        circles = self._detect_color_circles(frame_eq, hsv)
        self.circles_pub.publish(String(data=json.dumps({
            "circles": circles,
            "timestamp": time.time()
        })))

        # --- 3. OVERLAY (Gözlem Ekranı) ---
        cv2.putText(frame, f"Goz: {self.agent_id} | {time.strftime('%H:%M:%S')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Ekranın tam ortasına hedef imleci (Crosshair) ekle
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

        return frame

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()