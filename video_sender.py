"""
video_sender.py
---------------
Modul Praktikum: Multi-Drone Video Networking + AI Detection (YOLO)
Dijalankan pada Laptop Companion-01 (Drone 1) dan Laptop Companion-02 (Drone 2).
"""

import argparse
import struct
import time
import cv2
import socket

def parse_args():
    parser = argparse.ArgumentParser(description="Companion Computer Video Sender (UDP) with Edge AI")
    parser.add_argument("--ip", default="10.42.0.1", help="IP Address Ground Control Station (GCS)")
    parser.add_argument("--port", type=int, default=5001, help="Port UDP tujuan")
    parser.add_argument("--width", type=int, default=640, help="Lebar resolusi frame (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Tinggi resolusi frame (default: 480)")
    parser.add_argument("--fps", type=int, default=15, help="Frame Rate (default: 15)")
    parser.add_argument("--quality", type=int, default=60, help="Kualitas kompresi JPEG 1-100 (default: 60)")
    parser.add_argument("--camera", type=int, default=0, help="Index kamera webcam (default: 0)")
    parser.add_argument("--yolo", action="store_true", help="Aktifkan Edge AI YOLOv8 pada drone")
    parser.add_argument("--model", default="yolov8n.pt", help="Path model YOLO (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold YOLO (default: 0.5)")
    return parser.parse_args()

def draw_industrial_yolo_box(frame, x1, y1, x2, y2, label, conf):
    """Menggambar kotak YOLO bergaya tactical/industrial di sisi Edge/Drone."""
    color = (0, 215, 255)  # Amber/Cyan
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

    # Titik pusat deteksi target
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(frame, (cx - 5, cy), (cx + 5, cy), color, 1)
    cv2.line(frame, (cx, cy - 5), (cx, cy + 5), color, 1)

    # Label data
    text = f"TGT: {label.upper()} [{conf:.2f}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
    
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4), font, 0.4, color, 1)

def main():
    args = parse_args()

    GCS_IP, GCS_PORT = args.ip, args.port
    WIDTH, HEIGHT, FPS = args.width, args.height, args.fps
    JPEG_QUALITY, CAMERA_IDX = args.quality, args.camera
    USE_YOLO, MODEL_PATH, CONF_THRESH = args.yolo, args.model, args.conf

    print("=" * 60)
    print("  EDGE COMPUTE - UAV VIDEO TRANSMITTER")
    print(f"  Target GCS   : {GCS_IP}:{GCS_PORT}")
    print(f"  Resolution   : {WIDTH}x{HEIGHT} @ {FPS} FPS")
    print(f"  Edge AI      : {'ACTIVE (' + MODEL_PATH + ')' if USE_YOLO else 'STANDBY'}")
    print("=" * 60)

    yolo_model = None
    if USE_YOLO:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(MODEL_PATH)
            print("[INFO] Edge AI Engine Initialized.")
        except Exception as e:
            print(f"[ERROR] Edge AI Failure: {e}")
            return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    camera = cv2.VideoCapture(CAMERA_IDX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    if not camera.isOpened():
        print(f"[ERROR] Camera Index {CAMERA_IDX} Offline.")
        return

    frame_id = 0
    frame_interval = 1.0 / FPS
    print(f"[INFO] Uplink established to {GCS_IP}:{GCS_PORT}... Ctrl+C to abort.")

    try:
        while True:
            start_time = time.time()
            ret, frame = camera.read()
            
            if not ret:
                print("[WARNING] Frame capture dropped.")
                break

            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # Render Edge AI (dibakar langsung ke frame sebelum dikirim)
            if yolo_model is not None:
                results = yolo_model(frame, conf=CONF_THRESH, verbose=False)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_name = yolo_model.names[int(box.cls[0])] if hasattr(yolo_model, "names") else "OBJ"
                    draw_industrial_yolo_box(frame, x1, y1, x2, y2, cls_name, conf)

            # Kompresi dan Enkode
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not success:
                continue

            data = encoded.tobytes()
            timestamp = time.time()
            header = struct.pack("!IdI", frame_id, timestamp, len(data))
            packet = header + data

            try:
                sock.sendto(packet, (GCS_IP, GCS_PORT))
            except OSError as e:
                print(f"[ERROR] Transmission Failure: {e}")
                break

            frame_id += 1

            # Limitasi FPS Sinkron
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Transmission aborted by operator.")
    finally:
        camera.release()
        sock.close()
        print("[INFO] Hardware resources released.")

if __name__ == "__main__":
    main()
