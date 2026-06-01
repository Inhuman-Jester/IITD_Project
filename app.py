import onnxruntime as ort
hpc_session_options = ort.SessionOptions()

NUM_THREADS = 1 # set to number of CPU threads 
hpc_session_options.intra_op_num_threads = NUM_THREADS
hpc_session_options.inter_op_num_threads = NUM_THREADS

orig_init = ort.InferenceSession.__init__

def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    orig_init(self, path_or_bytes, hpc_session_options, providers, provider_options, **kwargs)

ort.InferenceSession.__init__ = patched_init

import time
import numpy as np
import pickle
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2
import sys
import socket
import torch
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from insightface.app import FaceAnalysis
import threading
from flask import Flask, flash, redirect, render_template_string, request, send_from_directory, url_for, Response
import logging
from template.app_template import APP_TEMPLATE
from database.schema import init_db
from utils.face_recog import AttendanceSystem
from utils.frame import FrameFetcher
logging.getLogger("werkzeug").setLevel(logging.ERROR)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FAS_DIR = os.path.join(CURRENT_DIR, 'FAS')
FACES_DIR = "registered_faces"

if FAS_DIR not in sys.path:
    sys.path.insert(0, FAS_DIR)

from FAS.nets.utils import get_model

# --- Configuration ---
CAM_IP = "10.208.22.128"
USER = "admin"
PASSWORD = "SOumil%40%40btp1"
RTSP_URL = f"rtsp://{USER}:{PASSWORD}@{CAM_IP}:554/video/live?channel=1&subtype=0"
ESP32_IP = "10.194.17.254"
ESP32_PORT = 4210
RECOGNITION_THRESHOLD = 0.65
SPOOF_THRESHOLD = 1.0
DB_PATH = "attendance_db.pkl"
LOG_FILE = "attendance_log.txt"
# Registration tuning
# Countdown before first capture (seconds)
REGISTRATION_COUNTDOWN = 1
# Seconds to sample for each capture window
REG_CAPTURE_WINDOW = 2.0
# Require a second verification capture during registration
REG_VERIFICATION_REQUIRED = True
# Optional separate threshold for registration verification; if None, use RECOGNITION_THRESHOLD
REG_VERIFICATION_THRESHOLD = None


# class FrameFetcher:
#     """
#     Background thread that opens a fresh RTSP connection per frame,
#     always keeping the single latest frame available for the inference thread.
#     The inference thread never waits on network I/O — it just reads whatever
#     the fetcher last stored.
#     """
#     def __init__(self, rtsp_url: str, retry_delay: float = 2.0):
#         self._url = rtsp_url
#         self._retry_delay = retry_delay

#         self._frame = None         
#         self._lock = threading.Lock()   
#         self._running = True
#         self._frame_ready = threading.Event()

#         self._thread = threading.Thread(target=self._fetch_loop, daemon=True, name="FrameFetcher")
#         self._thread.start()

#     def _fetch_loop(self):
#         cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         while self._running:
#             ret, frame = cap.read()

#             if ret and frame is not None:
#                 with self._lock:
#                     self._frame = frame
#                 self._frame_ready.set()

#     def get_frame(self, timeout: float = 10.0):
#         """
#         Block until a frame is available (only on cold start), then return
#         a copy of the latest frame. Returns (True, frame) or (False, None).
#         """
#         if not self._frame_ready.wait(timeout=timeout):
#             return False, None
#         with self._lock:
#             return True, self._frame.copy()

#     def stop(self):
#         self._running = False
#         self._thread.join(timeout=3)


class FASPreprocessor:
    """Replicates the exact preprocessing pipeline for a single live camera frame."""
    def __init__(self, input_size=224, test_crop=False):
        self.input_size = input_size
        self.test_crop = test_crop

        if self.test_crop:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(self.input_size, self.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __call__(self, face_crop_bgr):
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=face_rgb)
        tensor = augmented["image"]
        return tensor.unsqueeze(0)


class AntiSpoofPredictor:
    """Handles Face Anti-Spoofing Inference"""
    def __init__(self, model_path, arch='resnet50', num_classes=2, device_id=0):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print(f"Loading Anti-Spoofing model ({arch}) on {self.device}...")

        self.model = get_model(arch, num_classes)
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = FASPreprocessor(input_size=224, test_crop=False)

    @torch.no_grad()
    def predict(self, frame, bbox, threshold=0.5, class_index=1):
        h_frame, w_frame = frame.shape[:2]
        xmin = max(0, int(bbox[0]))
        ymin = max(0, int(bbox[1]))
        xmax = min(w_frame, int(bbox[2]))
        ymax = min(h_frame, int(bbox[3]))

        face_crop = frame[ymin:ymax, xmin:xmax]
        if face_crop.size == 0:
            return False

        input_tensor = self.preprocessor(face_crop).to(self.device)
        outputs = self.model(input_tensor)[1]
        scores = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        spoof_score = scores[class_index]
        print(f"Anti-Spoofing Score (spoof): {spoof_score:.4f}")
        return bool(spoof_score > threshold)


# class ESPDisplay:
#     """Handles UDP Network Communication to the OLED Display"""
#     def __init__(self, ip, port=4210):
#         self.ip = ip
#         self.port = port
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#     def display_attendance(self, status, name=""):
#         message = f"{status}:{name}"
#         self.sock.sendto(bytes(message, "utf-8"), (self.ip, self.port))
#         print(f"UDP Sent to ESP32: {message}")


# class Database:
#     """Handles Saving and Loading Embeddings"""
#     def __init__(self, db_path):
#         self.db_path = db_path
#         self.known_faces = self.load_db()

#     def load_db(self):
#         known_faces = {}
#         if os.path.exists(self.db_path):
#             with open(self.db_path, "rb") as f:
#                 while True:
#                     try:
#                         data = pickle.load(f)
#                         if isinstance(data, dict):
#                             known_faces.update(data)
#                     except EOFError:
#                         break
#                     except Exception as e:
#                         print(f"Error loading DB: {e}")
#                         break
#         return known_faces

#     def save_db(self):
#         tmp_path = self.db_path + ".tmp"
#         with open(tmp_path, "wb") as f:
#             pickle.dump(self.known_faces, f)
#         os.replace(tmp_path, self.db_path)

#     def append_or_update(self, entry_no, user_data):
#         self.known_faces[entry_no] = user_data
#         self.save_db()

#     def exists(self, entry_no):
#         return entry_no in self.known_faces

#     def get_all(self):
#         return self.known_faces


# class AttendanceSystem:
#     """Main Controller: Bridges Camera, AI Models, DB, and UI"""
#     def __init__(self):
#         self.db = Database(db_path=DB_PATH)
#         self.ui = ESPDisplay(ip=ESP32_IP, port=ESP32_PORT)
#         self.log_file = LOG_FILE
#         self._face_lock = threading.Lock()
#         self._recognition_lock = threading.Lock()
#         self._recognition_thread = None
#         self._recognition_stop_event = threading.Event()
#         self._recognition_status = "stopped"
#         self._last_message = ""
#         self._last_attendance_message = ""
#         self._last_attendance_event_id = 0

#         print("Initializing ArcFace...")
#         self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
#         self.app.prepare(ctx_id=0, det_size=(640, 640))

#         # persistent preview fetcher for dashboard camera stream
#         try:
#             self.preview_fetcher = FrameFetcher(RTSP_URL)
#         except Exception:
#             self.preview_fetcher = None

#         self.anti_spoof = None  # temporarily disable anti-spoof

#     @property
#     def recognition_running(self):
#         return self._recognition_thread is not None and self._recognition_thread.is_alive()

#     def get_last_message(self):
#         return self._last_message

#     def _set_message(self, message):
#         self._last_message = message
#         print(message)

#     def _capture_best_face(self, fetcher, window_seconds=None):
#         start_time = time.time()
#         best_face = None
#         best_frame = None
#         highest_det_score = 0.0

#         if window_seconds is None:
#             window_seconds = REG_CAPTURE_WINDOW

#         while time.time() - start_time < window_seconds:
#             ret, frame = fetcher.get_frame()
#             if not ret:
#                 continue
#             with self._face_lock:
#                 faces = self.app.get(frame)
#             if faces:
#                 for face in faces:
#                     if face.det_score > highest_det_score:
#                         highest_det_score = face.det_score
#                         best_face = face
#                         best_frame = frame

#         return best_face, best_frame, highest_det_score

#     def register_user(self, name, entry_no, overwrite=True):
#         start_time = time.time()
#         if self.db.exists(entry_no) and not overwrite:
#             self._set_message(f"{entry_no} already exists. Update cancelled.")
#             return False

#         if self.db.exists(entry_no) and overwrite:
#             self._set_message(f"{entry_no} already exists. Overwriting with a fresh capture.")

#         self._set_message(f"Starting registration for {name} ({entry_no}) in {REGISTRATION_COUNTDOWN} seconds. Look at the camera!")
#         time.sleep(REGISTRATION_COUNTDOWN)

#         # Dedicated fetcher just for registration window
#         fetcher = FrameFetcher(RTSP_URL)
#         first_face, first_frame, first_score = self._capture_best_face(fetcher, window_seconds=REG_CAPTURE_WINDOW)

#         if not first_face or first_score <= RECOGNITION_THRESHOLD:
#             fetcher.stop()
#             self._set_message("Failed to capture a clear face. Please try again.")
#             return False

#         # If verification is disabled, accept the first capture immediately
#         if not REG_VERIFICATION_REQUIRED:
#             second_face = first_face
#             second_frame = first_frame
#             second_score = first_score
#             verification_sim = 1.0
#         else:
#             self._set_message("First capture saved. Hold still for a second verification capture.")
#             second_face, second_frame, second_score = self._capture_best_face(fetcher, window_seconds=REG_CAPTURE_WINDOW)
#             fetcher.stop()

#             if not second_face or second_score <= RECOGNITION_THRESHOLD:
#                 self._set_message("Failed to capture a clear verification face. Please try again.")
#                 return False

#             verification_sim = self.cosine_similarity(first_face.normed_embedding, second_face.normed_embedding)
#             self._set_message(f"Verification similarity: {verification_sim:.4f}")

#             threshold = REG_VERIFICATION_THRESHOLD if REG_VERIFICATION_THRESHOLD is not None else RECOGNITION_THRESHOLD
#             if verification_sim < threshold:
#                 self._set_message("Second verification did not match the first capture. Registration cancelled.")
#                 return False

#         os.makedirs(FACES_DIR, exist_ok=True)
#         x1, y1, x2, y2 = [int(v) for v in second_face.bbox]
#         h, w = second_frame.shape[:2]
#         face_crop = second_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
#         face_path = os.path.join(FACES_DIR, f"{entry_no}.jpg")
#         cv2.imwrite(face_path, face_crop)
#         self._set_message(f"Face photo saved: {face_path}")

#         self.db.append_or_update(entry_no, {"name": name, "embedding": first_face.normed_embedding})
#         time_taken = time.time() - start_time
#         self._set_message(f"Successfully registered {name} ({entry_no}). Time taken : {time_taken}")
#         return True

#     def show_user(self, entry_no):
#         if not self.db.exists(entry_no):
#             self._set_message(f"Entry {entry_no} not found in database.")
#             return None
#         face_path = os.path.join(FACES_DIR, f"{entry_no}.jpg")
#         if not os.path.exists(face_path):
#             self._set_message(f"No saved photo found for {entry_no}.")
#             return None
#         name = self.db.get_all()[entry_no]["name"]
#         self._set_message(f"Registered photo for {entry_no} - {name} ")
#         return face_path

#     def cosine_similarity(self, emb1, emb2):
#         return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

#     def mark_attendance(self, entry_no, name, similarity, time_taken):
#         timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#         log_entry = f"{timestamp} - {entry_no} - {name} - {similarity:.4f} - {time_taken:.2f}s\n"
#         with open(self.log_file, 'a') as f:
#             f.write(log_entry)
#         self._last_attendance_event_id += 1
#         self._last_attendance_message = f"Attendance marked for {entry_no} - {name} at {timestamp}"
#         self._set_message(self._last_attendance_message)
#         self.ui.display_attendance("SUCCESS", name)

#     def _recognition_loop(self):
#         recently_marked = {}
#         cooldown_period = 30

#         fetcher = FrameFetcher(RTSP_URL)
#         self._recognition_status = "starting"
#         self._set_message("Starting recognition loop.")
#         self._set_message("[Main] Waiting for first frame...")
        
#         ok, _ = fetcher.get_frame(timeout=15.0)
#         if not ok:
#             self._set_message("[Main] Timed out waiting for camera. Check RTSP URL.")
#             fetcher.stop()
#             self._recognition_status = "stopped"
#             return
#         self._set_message("[Main] Camera ready. Starting inference loop.")
#         self._recognition_status = "running"

#         try:
#             while not self._recognition_stop_event.is_set():
#                 start_time = time.time()
#                 ret, frame = fetcher.get_frame(timeout=1.0)
#                 if not ret:
#                     if self._recognition_stop_event.is_set():
#                         break
#                     self._set_message("[Main] No frame available yet...")
#                     time.sleep(0.1)
#                     continue

#                 with self._face_lock:
#                     faces = self.app.get(frame)
#                 if not faces:
#                     continue

#                 for face in faces:
#                     # Anti-spoofing disabled for testing (no model required)
#                     # Previously we checked: self.anti_spoof.predict(...)
#                     # if spoof detected we would ignore the face. That logic
#                     # is intentionally disabled now so recognition proceeds.

#                     if self.anti_spoof is not None and self.anti_spoof.predict(frame, face.bbox, threshold=SPOOF_THRESHOLD):
#                         self._set_message("Spoof detected! Ignoring.")
#                         self.ui.display_attendance("SPOOF", "ALERT")
#                         continue

#                     best_match_entry = None
#                     highest_sim = 0.0
#                     for entry_no, data in self.db.get_all().items():
#                         sim = self.cosine_similarity(face.normed_embedding, data["embedding"])
#                         if sim > highest_sim:
#                             highest_sim = sim
#                             best_match_entry = entry_no

#                     self._set_message(f"Best match: {best_match_entry} with similarity {highest_sim:.4f}")

#                     if highest_sim > RECOGNITION_THRESHOLD and best_match_entry:
#                         current_time = time.time()
#                         name = self.db.get_all()[best_match_entry]["name"]
#                         if best_match_entry not in recently_marked or \
#                            (current_time - recently_marked[best_match_entry]) > cooldown_period:
#                             end_time = time.time()
#                             self.mark_attendance(best_match_entry, name, highest_sim, end_time - start_time)
#                             recently_marked[best_match_entry] = current_time

#         except KeyboardInterrupt:
#             self._set_message("Shutting down gracefully...")
#         finally:
#             fetcher.stop()
#             self._recognition_status = "stopped"

#     def start_recognition(self):
#         with self._recognition_lock:
#             if self.recognition_running:
#                 return False, "Recognition is already running."

#             self._recognition_stop_event.clear()
#             self._recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True, name="RecognitionLoop")
#             self._recognition_thread.start()
#             return True, "Recognition started."

#     def stop_recognition(self):
#         self._recognition_stop_event.set()
#         thread = self._recognition_thread
#         if thread and thread.is_alive():
#             thread.join(timeout=5)
#         self._recognition_status = "stopped"
#         return True, "Recognition stopped."

#     def get_log_lines(self, limit=12):
#         if not os.path.exists(self.log_file):
#             return []

#         with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
#             lines = [line.strip() for line in f if line.strip()]
#         return lines[-limit:]


def create_flask_app(system: AttendanceSystem):
    flask_app = Flask(__name__)
    flask_app.secret_key = os.environ.get("FLASK_SECRET_KEY", "smart-attendance-dashboard")

    def _ensure_recognition_running():
        # Keep recognition running continuously; restart if it ever stops.
        if not system.recognition_running:
            system.start_recognition()

    _ensure_recognition_running()

    @flask_app.before_request
    def _maintain_recognition_worker():
        _ensure_recognition_running()

    @flask_app.route("/", methods=["GET"])
    def index():
        selected_entry_no = request.args.get("entry_no", "").strip()
        name_arg = request.args.get("name", "").strip()
        selected_name = None
        face_exists = False
        all_users = system.get_all()
        if selected_entry_no and system.exists(selected_entry_no):
            selected_name = all_users[selected_entry_no]["name"]
            face_exists = os.path.exists(os.path.join(FACES_DIR, f"{selected_entry_no}.jpg"))
        else:
            selected_name = name_arg

        # Suggest an entry_no: the count of known faces + 1
        suggested_entry_no = str(len(all_users) + 1)

        return render_template_string(
            APP_TEMPLATE,
            recognition_running=system.recognition_running,
            registered_count=len(all_users),
            log_count=len(system.get_log_lines(limit=1000)),
            registered_faces=sorted(all_users.items()),
            log_lines=system.get_log_lines(),
            selected_entry_no=selected_entry_no,
            selected_name=selected_name,
            face_exists=face_exists,
            suggested_entry_no=suggested_entry_no,
        )

    @flask_app.route("/register", methods=["POST"])
    def register_user_route():
        name = request.form.get("name", "").strip()
        entry_no = request.form.get("entry_no", "").strip()
        overwrite = request.form.get("overwrite") == "1"

        if not name or not entry_no:
            flash("Name and Entry No are required.")
            return redirect(url_for("index", entry_no=entry_no, name=name))

        success = system.register_user(name=name, entry_no=entry_no, overwrite=overwrite)
        flash(system.get_last_message() if system.get_last_message() else ("Registration complete." if success else "Registration failed."))
        return redirect(url_for("index", entry_no=entry_no, name=name))

    @flask_app.route("/show", methods=["GET"])
    def show_user_route():
        entry_no = request.args.get("entry_no", "").strip()
        if not entry_no:
            flash("Enter an Entry No to show a saved photo.")
            return redirect(url_for("index"))

        system.show_user(entry_no)
        flash(system.get_last_message() if system.get_last_message() else "Lookup complete.")
        return redirect(url_for("index", entry_no=entry_no))

    @flask_app.route("/faces/<entry_no>", methods=["GET"])
    def face_image(entry_no):
        if not system.exists(entry_no):
            flash(f"Entry {entry_no} not found in database.")
            return redirect(url_for("index"))

        response = send_from_directory(FACES_DIR, f"{entry_no}.jpg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @flask_app.route("/status", methods=["GET"])
    def status():
        return {
            "recognition_running": system.recognition_running,
            "recognition_status": system._recognition_status,
            "registered_users": len(system.get_all()),
            "last_attendance_event_id": system._last_attendance_event_id,
            "last_attendance_message": system._last_attendance_message,
            "recent_log_lines": system.get_log_lines(),
        }

    @flask_app.route('/camera_feed')
    def camera_feed():
        if not getattr(system, 'preview_fetcher', None):
            return "Camera not available", 503

        def gen():
            while True:
                ok, frame = system.preview_fetcher.get_frame(timeout=2.0)
                if not ok or frame is None:
                    if system._recognition_stop_event.is_set():
                        break
                    continue
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                chunk = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')
                time.sleep(0.03)

        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @flask_app.route('/camera_snapshot')
    def camera_snapshot():
        if not getattr(system, 'preview_fetcher', None):
            return "Camera not available", 503

        ok, frame = system.preview_fetcher.get_frame(timeout=2.0)
        if not ok or frame is None:
            return "Frame not available", 503

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return "Failed to encode frame", 500

        response = Response(jpeg.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    return flask_app


if __name__ == "__main__":
    init_db()
    system = AttendanceSystem()
    system.start_recognition()
    app = create_flask_app(system)
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=True, threaded=True)