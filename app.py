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
ESP32_IP = "10.194.31.166"
ESP32_PORT = 4210
RECOGNITION_THRESHOLD = 0.65
SPOOF_THRESHOLD = 1.0
DB_PATH = "attendance_db.pkl"
LOG_FILE = "attendance_log.txt"


class FrameFetcher:
    """
    Background thread that opens a fresh RTSP connection per frame,
    always keeping the single latest frame available for the inference thread.
    The inference thread never waits on network I/O — it just reads whatever
    the fetcher last stored.
    """
    def __init__(self, rtsp_url: str, retry_delay: float = 2.0):
        self._url = rtsp_url
        self._retry_delay = retry_delay

        self._frame = None         
        self._lock = threading.Lock()   
        self._running = True
        self._frame_ready = threading.Event()

        self._thread = threading.Thread(target=self._fetch_loop, daemon=True, name="FrameFetcher")
        self._thread.start()

    def _fetch_loop(self):
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self._running:
            ret, frame = cap.read()

            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                self._frame_ready.set()

    def get_frame(self, timeout: float = 10.0):
        """
        Block until a frame is available (only on cold start), then return
        a copy of the latest frame. Returns (True, frame) or (False, None).
        """
        if not self._frame_ready.wait(timeout=timeout):
            return False, None
        with self._lock:
            return True, self._frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=3)


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


class ESPDisplay:
    """Handles UDP Network Communication to the OLED Display"""
    def __init__(self, ip, port=4210):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def display_attendance(self, status, name=""):
        message = f"{status}:{name}"
        self.sock.sendto(bytes(message, "utf-8"), (self.ip, self.port))
        print(f"UDP Sent to ESP32: {message}")


class Database:
    """Handles Saving and Loading Embeddings"""
    def __init__(self, db_path):
        self.db_path = db_path
        self.known_faces = self.load_db()

    def load_db(self):
        known_faces = {}
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                while True:
                    try:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            known_faces.update(data)
                    except EOFError:
                        break
                    except Exception as e:
                        print(f"Error loading DB: {e}")
                        break
        return known_faces

    def save_db(self):
        tmp_path = self.db_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.known_faces, f)
        os.replace(tmp_path, self.db_path)

    def append_or_update(self, entry_no, user_data):
        self.known_faces[entry_no] = user_data
        self.save_db()

    def exists(self, entry_no):
        return entry_no in self.known_faces

    def get_all(self):
        return self.known_faces


class AttendanceSystem:
    """Main Controller: Bridges Camera, AI Models, DB, and UI"""
    def __init__(self):
        self.db = Database(db_path=DB_PATH)
        self.ui = ESPDisplay(ip=ESP32_IP, port=ESP32_PORT)
        self.log_file = LOG_FILE

        print("Initializing ArcFace...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.anti_spoof = AntiSpoofPredictor(
            model_path='p1_resnet50.pth',
            arch='resnet50',
            num_classes=2,
            device_id=0
        )

    def register_user(self, name, entry_no):
        if self.db.exists(entry_no):
            ans = input(f"{entry_no} already exists. Capture again and overwrite? (y/n): ").strip().lower()
            if ans != "y":
                print("Update cancelled.")
                return

        print(f"Starting registration for {name} ({entry_no}) in 3 seconds. Look at the camera!")
        time.sleep(3)

        # Dedicated fetcher just for registration window
        fetcher = FrameFetcher(RTSP_URL)
        start_time = time.time()
        best_face = None
        best_frame = None
        highest_det_score = 0.0

        while time.time() - start_time < 4.0:
            ret, frame = fetcher.get_frame()
            if not ret:
                continue
            faces = self.app.get(frame)
            if faces:
                for face in faces:
                    if face.det_score > highest_det_score:
                        highest_det_score = face.det_score
                        best_face = face
                        best_frame = frame

        fetcher.stop()

        if not best_face or highest_det_score <= RECOGNITION_THRESHOLD:
            print("Failed to capture a clear face. Please try again.")
            return

        os.makedirs(FACES_DIR, exist_ok=True)
        x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
        h, w = best_frame.shape[:2]
        face_crop = best_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        face_path = os.path.join(FACES_DIR, f"{entry_no}.jpg")
        cv2.imwrite(face_path, face_crop)
        print(f"Face photo saved: {face_path}")

        self.db.append_or_update(entry_no, {"name": name, "embedding": best_face.normed_embedding})
        print(f"Successfully registered {name} ({entry_no}).")

    def show_user(self, entry_no):
        if not self.db.exists(entry_no):
            print(f"Entry {entry_no} not found in database.")
            return
        face_path = os.path.join(FACES_DIR, f"{entry_no}.jpg")
        if not os.path.exists(face_path):
            print(f"No saved photo found for {entry_no}.")
            return
        name = self.db.get_all()[entry_no]["name"]
        print(f"Registered photo for {name} ({entry_no}): {os.path.abspath(face_path)}")

    def cosine_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def mark_attendance(self, entry_no, name):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {entry_no} - {name}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        print(f"Attendance marked for {entry_no} - {name} at {timestamp}")
        self.ui.display_attendance("SUCCESS", name)

    def run_recognition_loop(self):
        print("Starting recognition loop. Press Ctrl+C to stop.")
        recently_marked = {}
        cooldown_period = 30

        fetcher = FrameFetcher(RTSP_URL)
        print("[Main] Waiting for first frame...")
        
        ok, _ = fetcher.get_frame(timeout=15.0)
        if not ok:
            print("[Main] Timed out waiting for camera. Check RTSP URL.")
            fetcher.stop()
            return
        print("[Main] Camera ready. Starting inference loop.")

        try:
            while True:
                ret, frame = fetcher.get_frame()
                if not ret:
                    print("[Main] No frame available yet...")
                    time.sleep(0.1)
                    continue

                faces = self.app.get(frame)
                if not faces:
                    continue

                for face in faces:
                    if self.anti_spoof.predict(frame, face.bbox, threshold=SPOOF_THRESHOLD):
                        print("Spoof detected! Ignoring.")
                        self.ui.display_attendance("SPOOF", "ALERT")
                        continue

                    best_match_entry = None
                    highest_sim = 0.0
                    for entry_no, data in self.db.get_all().items():
                        sim = self.cosine_similarity(face.normed_embedding, data["embedding"])
                        if sim > highest_sim:
                            highest_sim = sim
                            best_match_entry = entry_no

                    print(f"Best match: {best_match_entry} with similarity {highest_sim:.4f}")

                    if highest_sim > RECOGNITION_THRESHOLD and best_match_entry:
                        current_time = time.time()
                        name = self.db.get_all()[best_match_entry]["name"]
                        if best_match_entry not in recently_marked or \
                           (current_time - recently_marked[best_match_entry]) > cooldown_period:
                            self.mark_attendance(best_match_entry, name)
                            recently_marked[best_match_entry] = current_time

        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        finally:
            fetcher.stop()


if __name__ == "__main__":
    system = AttendanceSystem()

    while True:
        mode = input("\nEnter mode (register/run/show): ").strip().lower()
        if mode == "register":
            name = input("Enter Name: ")
            entry_no = input("Enter Entry No: ")
            system.register_user(name, entry_no)
        elif mode == "run":
            system.run_recognition_loop()
            break
        elif mode == "show":
            entry_no = input("Enter Entry No: ")
            system.show_user(entry_no)
        else:
            print("Invalid input.")