<div align="center">

# SENTINEL UAV GCS
### Tactical Multi-UAV Video Ingestion & Edge AI Ground Control Station

```
──────────────────────────────────────────────────────────────────────────────────────────
  MISSION PROFILE : SEARCH & RESCUE (SAR)  •  EDGE COMPUTE : YOLOV8  •  PROTOCOL : UDP
──────────────────────────────────────────────────────────────────────────────────────────
```

An industrial-grade, low-latency tactical monitoring platform engineered for coordinated 
multi-drone aerial surveillance, edge-computed object localization, and centralized HUD telemetry.

</div>

---

## System Topology

Sentinel UAV GCS ingests synchronized, low-latency UDP video streams from distributed UAV companion computers. Target detection inference is offloaded directly to the edge nodes, leaving the Ground Station dedicated to multi-channel telemetry rendering and operator observation.

```
       +-----------------------+              +-----------------------+
       |   UAV-01 (Drone 1)    |              |   UAV-02 (Drone 2)    |
       |  Companion Computer   |              |  Companion Computer   |
       |  Edge YOLOv8 Engine   |              |  Edge YOLOv8 Engine   |
       |  IP: 10.42.0.10:5001  |              |  IP: 10.42.0.20:5002  |
       +-----------+-----------+              +-----------+-----------+
                   \                                      /
                    \          UDP Packet Stream         /
                     \       (Header + JPEG Frame)      /
                      \                                /
                       +------------------------------+
                       |    Ground Control Station    |
                       |    Host Gateway: 10.42.0.1   |
                       |    Hotspot: DRONE-KPD        |
                       |    Dual Tactical HUD View    |
                       +------------------------------+
```

---

## Core Engineering Features

> **Edge Inference Offloading**  
> Object detection runs directly onboard each companion computer. Bounding brackets and classification labels are rendered onto the frame buffer prior to JPEG serialization, ensuring zero compute bottleneck on the GCS host.

> **Non-Blocking Dual Ingestion**  
> Employs `select.select` socket multiplexing across independent UDP ports (`5001` and `5002`), eliminating thread locking and packet drop under variable transmission conditions.

> **Industrial Heads-Up Display (HUD)**  
> Real-time overlay featuring crosshair reticles, grid reference axes, live FPS calculation, active heartbeat indicators (`ONLINE` / `OFFLINE`), and dynamic timestamp telemetry.

> **Air-Gapped Field Deployment**  
> Built for closed tactical wireless networks using local static DHCP lease mapping (`dnsmasq`) over dedicated WLAN interfaces.

---

## Network Configuration Matrix

| Station / Node | Operational Role | Network Interface | IP Allocation | Transmission Port |
| :--- | :--- | :--- | :--- | :--- |
| **GCS Master** | Command Gateway / Receiver | `wlp1s0` (`DRONE-KPD`) | `10.42.0.1` (Static) | Inbound `5001`, `5002` |
| **UAV-01** | Edge Transmitter 1 | Wi-Fi Client | `10.42.0.10` (DHCP Host) | Outbound `5001` |
| **UAV-02** | Edge Transmitter 2 | Wi-Fi Client | `10.42.0.20` (DHCP Host) | Outbound `5002` |

---

## Environment & Prerequisites

### System Requirements
* Linux / macOS / Windows with Python 3.10+
* Camera source (integrated webcam, USB UVC camera, or RTSP capture card)

### Setup Procedure
```bash
git clone git@github.com:rhankbrguw/sentinel-uav-gcs.git
cd sentinel-uav-gcs

# Initialize virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core runtime dependencies
pip install -r requirements.txt
```

---

## Model Weights Acquisition

Pretrained weights must be placed in the project root directory prior to mission execution:

| Checkpoint File | Training Domain | Target Use Case | Reference Source |
| :--- | :--- | :--- | :--- |
| `yolov8n.pt` | MS-COCO (80 Classes) | Eye-level / Indoor / Ground targets | [Ultralytics Official](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) |
| `model-udara.pt` | VisDrone (10 Classes) | Aerial nadir / High-altitude drone view | [HuggingFace Checkpoint](https://huggingface.co/mshamrai/yolov8n-visdrone/resolve/main/best.pt) |

```bash
# Automated retrieval
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
curl -L -o model-udara.pt https://huggingface.co/mshamrai/yolov8n-visdrone/resolve/main/best.pt
```

---

## Mission Execution Protocol

### Step 1: Launch Central Ground Station (GCS)
Start the dual-channel tactical receiver on the GCS laptop:
```bash
python video_receiver.py --port1 5001 --port2 5002
```

### Step 2: Deploy Edge Transmitter Nodes

**Node UAV-01 (Drone 1):**
```bash
python video_sender.py --ip 10.42.0.1 --port 5001 --yolo --model yolov8n.pt
```

**Node UAV-02 (Drone 2):**
```bash
python video_sender.py --ip 10.42.0.1 --port 5002 --yolo --model yolov8n.pt
```

> **Note:** Swap `--model yolov8n.pt` with `--model model-udara.pt` for aerial operations.

### Local Hardware Validation (Dry-Run Mode)
Verify video capture and local inference pipeline without transmitting over network:
```bash
python video_sender.py --yolo --model yolov8n.pt
```

---

## Runtime Flag Reference

| Flag | Argument Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `--ip` | string | `10.42.0.1` | Destination GCS host IP address |
| `--port` | integer | `5001` | Destination UDP listening port |
| `--width` | integer | `640` | Captured frame width in pixels |
| `--height` | integer | `480` | Captured frame height in pixels |
| `--fps` | integer | `15` | Framerate limit for network bandwidth control |
| `--quality` | integer | `60` | JPEG compression ratio (1-100) |
| `--yolo` | boolean flag | `False` | Enables onboard Edge AI target detection |
| `--model` | string | `yolov8n.pt` | Path to target YOLO checkpoint file |
| `--conf` | float | `0.5` | Detection confidence threshold |

---

## License
MIT License. Developed for Tactical Multi-UAV Video Surveillance and Search-and-Rescue missions.
