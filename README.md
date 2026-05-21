# Smart Attendance System

Real-time face recognition attendance pipeline using an RTSP IP camera, ArcFace, face anti-spoofing, and an ESP32 OLED display.

---

## Repository Structure

```
├── app.py               # Main attendance system
├── camera.py            # Live feed viewer (testing only)
├── FAS/                 # Face Anti-Spoofing model code
├── p1_resnet.pth        # anti-spoofing trained model
├── board.c              # ESP32 script
└── testing/             # Evaluation scripts
```
---

## Requirements

### Python

Tested on Python 3.9+.

```bash
pip install opencv-python
pip install insightface
pip install onnxruntime          # CPU-only
pip install torch torchvision    # CPU build is fine
pip install albumentations
pip install Pillow
pip install numpy
pip install onvif-zeep           # only needed for camera.py
pip install urllib3
```

Or install all at once:

```bash
pip install opencv-python insightface onnxruntime torch torchvision albumentations Pillow numpy onvif-zeep urllib3
```

> **ONNX thread patch** — `app.py` patches `onnxruntime` at the top to use 1 thread. This is intentional for single-core servers and must remain as the first lines of the file before any other imports.

### InsightFace model

On first run, InsightFace will automatically download the `buffalo_l` model pack (~300 MB) to `~/.insightface/models/`. Ensure internet access on first launch.

---

## Configuration

Open `app.py` and update the constants at the top:

```python
# ESP32 Display
ESP32_IP   = "10.x.x.x"        # ← see ESP32 setup section below
ESP32_PORT = 4210

# Thresholds
RECOGNITION_THRESHOLD = 0.65   # cosine similarity cutoff
SPOOF_THRESHOLD       = 1.0    # set <1.0 to enable anti-spoofing rejection
```
---

## ESP32 Setup

1. Power on the ESP32 — it will connect to `IITD_WIFI` using the WPA2 Enterprise credentials hardcoded in the sketch (change them if needed).
2. **On boot, the OLED display shows the assigned IP address for ~10 seconds.**
3. Note that IP and set it as `ESP32_IP` in `app.py`.

> The ESP32 gets its IP via DHCP, so the IP may change after a power cycle. If the display stops updating, repeat step 1-3.

---

## Running

### 1. Register a person

```bash
python app.py
# Enter mode: register
# Enter Name: Alice
# Enter Entry No: 2022CS11001
```

Stand in front of the camera. The system waits 3 seconds then captures the best face over a 4-second window. A cropped face photo is saved to `registered_faces/`.
### 2. View a registered photo

```bash
python app.py
# Enter mode: show
# Enter Entry No: 2022CS11001
```

Prints the file path.

### 3. Run attendance

```bash
python app.py
# Enter mode: run
```

The system will:
- Detect faces from the live RTSP feed
- Run anti-spoofing check
- Match against the database
- Log to `attendance_log.txt`
- Send result to the ESP32 display

Press `Ctrl+C` to stop.

### 4. View live camera feed (testing)

```bash
python camera.py
```

Press `P` to save a screenshot, `Q` to quit.

---

## Attendance Log

Matches are appended to `attendance_log.txt`:

```
2025-05-15 09:31:04 - 2022CS11001 - Alice
2025-05-15 09:35:22 - 2022CS11002 - Bob
```

A 30-second cooldown per person prevents duplicate entries.

---
