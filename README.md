# Smart Attendance System

Real-time face recognition attendance web application and pipeline using an RTSP IP camera, InsightFace (ArcFace), PostgreSQL (`pgvector`), Face Anti-Spoofing (FAS), and an optional ESP32 OLED display.

---

## Repository Structure

```
├── app.py               # Main Flask Web Application & Dashboard routes
├── utils/
│   ├── face_recog.py    # AttendanceSystem controller (InsightFace, Postgres DB, UDP display)
│   └── frame.py         # Thread-safe RTSP Frame Fetcher
├── database/
│   └── schema.py        # Database schema initialization & connection pool (PostgreSQL + pgvector)
├── template/
│   └── app_template.py  # HTML5/CSS3 Web Dashboard UI Template
├── FAS/                 # Face Anti-Spoofing model code
├── board.c              # ESP32 sketch code
└── testing/             # Evaluation & benchmarking scripts
```

---

## Requirements

### Python & Dependencies

Tested on Python 3.9+.

```bash
pip install -r requirements.txt
```

Or install key dependencies manually:

```bash
pip install flask psycopg2-binary opencv-python insightface onnxruntime torch torchvision albumentations Pillow numpy python-dotenv
```

### Docker

Run the app and database together with Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

The dashboard will be available at `http://localhost:5000`.

The container setup uses a `pgvector` PostgreSQL image, persists app runtime data in Docker volumes, and caches the InsightFace model directory so the face model is not downloaded on every start.

> **ONNX thread patch** — `app.py` patches `onnxruntime` at the top to configure thread counts. This ensures optimal execution for single-core / low-resource servers.

### Database Setup

The project uses PostgreSQL with the `pgvector` extension enabled for embedding search:
- Tables managed: `students`, `student_faces`, and `attendance_records`.

### InsightFace Model

On first run, InsightFace will automatically download the `buffalo_l` model pack (~300 MB) to `~/.insightface/models/`. Ensure internet access on first launch.

---

## Configuration

Environment variables can be configured via a `.env` file in the root directory:

```env
DATABASE_URL=postgresql://attendance:attendance@db:5432/attendance
ESP32_IP=10.194.17.254
ESP32_PORT=4210
CAM_IP=10.208.22.128
CAM_USER=admin
CAM_PASSWORD=your_password
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=0
REGISTERED_FACES_DIR=/data/registered_faces
ATTENDANCE_LOG_FILE=/data/attendance_log.txt
```

---

## Running the Web Dashboard

Start the web application server:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` (or `http://<server-ip>:5000`).

---

## Features & Usage

### 1. Continuous Recognition & Attendance Confirmation
- Face recognition runs continuously in the background using the live camera RTSP stream.
- Recognized users trigger attendance logging into PostgreSQL and `attendance_log.txt`.
- Live attendance confirmations appear on the Web Dashboard and trigger UDP display alerts on the connected ESP32 OLED module.

### 2. User Registration
- Enter a **Name** and **Kerberos ID** in the **Register User** section of the web dashboard.
- Captures face embeddings and stores cropped photos directly in PostgreSQL database (`student_faces`) and local cache directory (`registered_faces/`).
- Supports up to 3 face sample captures per student with two-step verification.

### 3. Show Registered Face
- Enter a **Kerberos ID** under **Show Registered Face** to search and preview the primary stored cropped face image.

### 4. Delete Registered Images
- Enter a **Kerberos ID** in the **Delete Registered Images** card section to purge all associated face samples, embeddings, and cached image files from the database and local storage.

---

## ESP32 Setup

1. Power on the ESP32 module — it connects to Wi-Fi using the credentials in `board.c`.
2. On boot, the OLED display shows its assigned IP address.
3. Update `ESP32_IP` in `.env` or `app.py` if the IP changes.

---

## Attendance Log

Attendance logs are stored in PostgreSQL (`attendance_records` table) and appended to `attendance_log.txt`:

```
2026-07-27 15:30:12 - 2022CS11001 - Alice - 0.8912 - 0.45s
```

