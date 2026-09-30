"""
video_receiver.py
-----------------
Modul Praktikum: Multi-Drone Video Networking + AI Dashboard
Dashboard GCS (Ground Control Station) dengan Tactical HUD.
"""

import argparse
import select
import socket
import struct
import time
import datetime
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="GCS Tactical Dual-Video Receiver (UDP)")
    parser.add_argument("--port1", type=int, default=5001, help="Port UDP untuk Drone 1")
    parser.add_argument("--port2", type=int, default=5002, help="Port UDP untuk Drone 2")
    parser.add_argument("--timeout", type=float, default=2.0, help="Batas waktu deteksi OFFLINE")
    parser.add_argument("--yolo", action="store_true", help="Aktifkan deteksi YOLOv8")
    parser.add_argument("--model", default="yolov8n.pt", help="Path model YOLO")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold YOLO")
    return parser.parse_args()

def draw_tactical_hud(frame, drone_id, port, fps, status="ONLINE"):
    """Menggambar antarmuka industrial (HUD) di atas frame video."""
    h, w = frame.shape[:2]
    
    # 1. Alpha Blended Overlays (Transparansi) untuk Header & Footer
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (15, 15, 15), -1)      # Top bar
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 15), -1)  # Bottom bar
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 2. Tactical Grid Lines (Garis Bantuan Tata Ruang)
    grid_color = (0, 50, 0)
    for i in range(1, 4):
        cv2.line(frame, (w // 4 * i, 45), (w // 4 * i, h - 35), grid_color, 1)
        cv2.line(frame, (0, h // 4 * i), (w, h // 4 * i), grid_color, 1)

    # 3. Target Crosshair (Tengah)
    cx, cy = w // 2, h // 2
    cross_color = (0, 255, 0)
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), cross_color, 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), cross_color, 1)
    cv2.circle(frame, (cx, cy), 10, cross_color, 1)

    # 4. Telemetry Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    if status == "ONLINE":
        status_color = (0, 255, 0)
        # Efek kedip lambat untuk indikator REC/LIVE
        if int(time.time()) % 2 == 0:
            cv2.circle(frame, (w - 25, 22), 6, (0, 0, 255), -1) 
    else:
        status_color = (0, 0, 255)
        # Layar merah transparan jika offline
        offline_overlay = frame.copy()
        cv2.rectangle(offline_overlay, (0, 45), (w, h - 35), (0, 0, 50), -1)
        cv2.addWeighted(offline_overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "NO SIGNAL", (cx - 70, cy), font, 1, (0, 0, 255), 2)

    # Header Text
    cv2.putText(frame, f"UAV-{drone_id} | UDP:{port}", (15, 28), font, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"SYS: {status}", (200, 28), font, 0.6, status_color, 2)
    cv2.putText(frame, "LIVE", (w - 65, 28), font, 0.6, (255, 255, 255), 1)

    # Footer Text
    cv2.putText(frame, f"ALT: N/A | GPS: N/A", (15, h - 12), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {fps}", (w // 2 - 25, h - 12), font, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, timestamp, (w - 220, h - 12), font, 0.5, (255, 255, 255), 1)

    # Bingkai Sisi
    cv2.rectangle(frame, (0, 0), (w, h), (100, 100, 100), 2)

    return frame

def draw_industrial_yolo_box(frame, x1, y1, x2, y2, label, conf):
    """Menggambar kotak YOLO bergaya sci-fi/industrial corner-brackets."""
    color = (0, 215, 255)  # Amber/Cyan khas sistem penargetan
    thick = 2
    L = 20 # Panjang siku (corner length)

    # Sudut Kiri Atas
    cv2.line(frame, (x1, y1), (x1 + L, y1), color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + L), color, thick)
    # Sudut Kanan Atas
    cv2.line(frame, (x2, y1), (x2 - L, y1), color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + L), color, thick)
    # Sudut Kiri Bawah
    cv2.line(frame, (x1, y2), (x1 + L, y2), color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - L), color, thick)
    # Sudut Kanan Bawah
    cv2.line(frame, (x2, y2), (x2 - L, y2), color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - L), color, thick)

    # Titik pusat deteksi
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 2, color, -1)

    # Label dengan latar belakang semi-transparan
    text = f"TGT: {label.upper()} [{conf:.2f}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
    
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4), font, 0.4, color, 1)

def main():
    args = parse_args()
    PORT_1, PORT_2 = args.port1, args.port2
    TIMEOUT_OFFLINE = args.timeout
    USE_YOLO, MODEL_PATH, CONF_THRESH = args.yolo, args.model, args.conf

    print("=" * 60)
    print("  GCS TACTICAL DASHBOARD - INITIALIZING")
    print(f"  Port UAV-1 : {PORT_1} | Port UAV-2 : {PORT_2}")
    print(f"  AI Module  : {'ENGAGED (' + MODEL_PATH + ')' if USE_YOLO else 'STANDBY'}")
    print("=" * 60)

    yolo_model = None
    if USE_YOLO:
        from ultralytics import YOLO
        yolo_model = YOLO(MODEL_PATH)

    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock1.bind(("0.0.0.0", PORT_1))
    sock2.bind(("0.0.0.0", PORT_2))
    sock1.setblocking(False)
    sock2.setblocking(False)

    frame1, frame2 = None, None
    last1, last2 = 0, 0
    fps1, fps2, fps_count1, fps_count2 = 0, 0, 0, 0
    fps_timer1, fps_timer2 = time.time(), time.time()

    def process_yolo(frame):
        if yolo_model is None: return frame
        results = yolo_model(frame, conf=CONF_THRESH, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_name = yolo_model.names[int(box.cls[0])] if hasattr(yolo_model, "names") else "OBJ"
            draw_industrial_yolo_box(frame, x1, y1, x2, y2, cls_name, conf)
        return frame

    try:
        while True:
            readable, _, _ = select.select([sock1, sock2], [], [], 0.05)
            for sock in readable:
                try:
                    data, _ = sock.recvfrom(65535)
                except OSError: continue

                if len(data) < 16: continue
                jpeg_data = data[16:]
                array = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                
                if frame is None: continue

                if sock == sock1:
                    frame1, last1, fps_count1 = frame, time.time(), fps_count1 + 1
                    if time.time() - fps_timer1 >= 1.0:
                        fps1, fps_count1, fps_timer1 = fps_count1, 0, time.time()
                elif sock == sock2:
                    frame2, last2, fps_count2 = frame, time.time(), fps_count2 + 1
                    if time.time() - fps_timer2 >= 1.0:
                        fps2, fps_count2, fps_timer2 = fps_count2, 0, time.time()

            now = time.time()

            # Proses UAV 1
            if frame1 is not None and (now - last1 < TIMEOUT_OFFLINE):
                display1 = process_yolo(frame1.copy()) if yolo_model else frame1.copy()
                display1 = draw_tactical_hud(display1, "01", PORT_1, fps1, "ONLINE")
            else:
                display1 = np.zeros((480, 640, 3), dtype=np.uint8)
                display1 = draw_tactical_hud(display1, "01", PORT_1, 0, "OFFLINE")

            # Proses UAV 2
            if frame2 is not None and (now - last2 < TIMEOUT_OFFLINE):
                display2 = process_yolo(frame2.copy()) if yolo_model else frame2.copy()
                display2 = draw_tactical_hud(display2, "02", PORT_2, fps2, "ONLINE")
            else:
                display2 = np.zeros((480, 640, 3), dtype=np.uint8)
                display2 = draw_tactical_hud(display2, "02", PORT_2, 0, "OFFLINE")

            # H-Stack untuk UI
            display = np.hstack((display1, display2))
            
            # Garis pembatas tengah vertikal
            h, w = display.shape[:2]
            cv2.line(display, (w//2, 0), (w//2, h), (100, 100, 100), 2)

            cv2.imshow("GCS TACTICAL DASHBOARD", display)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    except KeyboardInterrupt:
        print("\n[INFO] GCS System Shutdown.")
    finally:
        sock1.close(); sock2.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
