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
from flask import Flask, flash, redirect, render_template_string, request, send_from_directory, url_for, Response, jsonify, session
import logging
from template.app_template import APP_TEMPLATE
from template.auth_templates import LOGIN_TEMPLATE, STUDENT_TEMPLATE
from database.schema import init_db
from utils.face_recog import AttendanceSystem
from utils.frame import FrameFetcher
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("werkzeug").setLevel(logging.ERROR)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FAS_DIR = os.path.join(CURRENT_DIR, 'FAS')
FACES_DIR = os.environ.get("REGISTERED_FACES_DIR", "registered_faces")

# User Credentials Configuration
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
STUDENT_USERNAME = os.environ.get("STUDENT_USERNAME", "student")
STUDENT_PASSWORD = os.environ.get("STUDENT_PASSWORD", "student123")

if FAS_DIR not in sys.path:
    sys.path.insert(0, FAS_DIR)

# from FAS.nets.utils import get_model

# --- Configuration ---
ESP32_IP = os.environ.get("ESP32_IP", "10.194.17.254")
ESP32_PORT = int(os.environ.get("ESP32_PORT", "4210"))
RECOGNITION_THRESHOLD = 0.65
SPOOF_THRESHOLD = 1.0
LOG_FILE = os.environ.get("ATTENDANCE_LOG_FILE", "attendance_log.txt")
# Registration tuning
# Countdown before first capture (seconds)
REGISTRATION_COUNTDOWN = 1
# Seconds to sample for each capture window
REG_CAPTURE_WINDOW = 2.0
# Require a second verification capture during registration
REG_VERIFICATION_REQUIRED = True
# Optional separate threshold for registration verification; if None, use RECOGNITION_THRESHOLD
REG_VERIFICATION_THRESHOLD = None


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

        # self.model = get_model(arch, num_classes)
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



def create_flask_app(system: AttendanceSystem):
    flask_app = Flask(__name__)
    flask_app.secret_key = os.environ.get("FLASK_SECRET_KEY", "smart-attendance-dashboard-secret-key")

    def _ensure_recognition_running():
        # Keep recognition running continuously; restart if it ever stops.
        if not system.recognition_running:
            system.start_recognition()

    _ensure_recognition_running()

    @flask_app.before_request
    def _maintain_recognition_worker():
        _ensure_recognition_running()
        # Protect routes with login requirement
        open_endpoints = ['login', 'logout', 'static']
        if request.endpoint and request.endpoint not in open_endpoints:
            if not session.get('logged_in'):
                return redirect(url_for('login'))
            # Restrict admin endpoints from student access
            if session.get('role') == 'student' and request.endpoint != 'student_home':
                return redirect(url_for('student_home'))

    @flask_app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()

            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                session['logged_in'] = True
                session['username'] = username
                session['role'] = 'admin'
                return redirect(url_for('index'))
            elif username == STUDENT_USERNAME and password == STUDENT_PASSWORD:
                session['logged_in'] = True
                session['username'] = username
                session['role'] = 'student'
                return redirect(url_for('student_home'))
            else:
                flash("Invalid username or password.")
                return render_template_string(LOGIN_TEMPLATE)

        if session.get('logged_in'):
            if session.get('role') == 'admin':
                return redirect(url_for('index'))
            return redirect(url_for('student_home'))

        return render_template_string(LOGIN_TEMPLATE)

    @flask_app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.")
        return redirect(url_for('login'))

    @flask_app.route("/student-home", methods=["GET"])
    def student_home():
        return render_template_string(STUDENT_TEMPLATE)

    @flask_app.route("/", methods=["GET"])
    def index():
        selected_entry_no = request.args.get("entry_no", "").strip()
        name_arg = request.args.get("name", "").strip()
        selected_name = None
        face_exists = False
        all_users = system.get_all()
        if selected_entry_no and system.exists(selected_entry_no):
            selected_name = all_users[selected_entry_no]["name"]
            photo_record = system.get_registered_photo(selected_entry_no)
            face_exists = (
                system.get_registered_photo(selected_entry_no) is not None
            )
        else:
            selected_name = name_arg

        # Suggest an entry_no: the count of known faces + 1
        suggested_entry_no = str(len(all_users) + 1)

        return render_template_string(
            APP_TEMPLATE,
            recognition_running=system.recognition_running,
            registered_count=len(all_users),
            selected_entry_no=selected_entry_no,
            selected_name=selected_name,
            face_exists=face_exists,
            suggested_entry_no=suggested_entry_no,
        )

    @flask_app.route("/register", methods=["POST"])
    def register_user_route():
        entry_no = request.form.get("entry_no", "").strip()
        name = request.form.get("name", "").strip()
        overwrite = request.form.get("overwrite") == "1"

        if not entry_no:
            flash("Kerberos ID is required.")
            return redirect(url_for("index", entry_no=entry_no))

        success = system.register_user(kerberos_id=entry_no, name=name, overwrite=overwrite)
        flash(system.get_last_message() if system.get_last_message() else ("Registration complete." if success else "Registration failed."))
        return redirect(url_for("index", entry_no=entry_no))

    @flask_app.route("/show", methods=["GET"])
    def show_user_route():
        entry_no = request.args.get("entry_no", "").strip()
        if not entry_no:
            flash("Enter a Kerberos ID to show a saved photo.")
            return redirect(url_for("index"))

        system.show_user(entry_no)
        flash(system.get_last_message() if system.get_last_message() else "Lookup complete.")
        return redirect(url_for("index", entry_no=entry_no))

    @flask_app.route("/delete-images", methods=["POST"])
    def delete_images_route():
        entry_no = request.form.get("entry_no", "").strip()
        if not entry_no:
            flash("Kerberos ID is required to delete images.")
            return redirect(url_for("index"))

        success, count = system.delete_user_images(entry_no)
        flash(system.get_last_message() if system.get_last_message() else f"Deleted {count} image(s) for Kerberos ID {entry_no}.")
        return redirect(url_for("index", entry_no=entry_no))

    @flask_app.route("/faces/<entry_no>", methods=["GET"])
    def face_image(entry_no):
        if not system.exists(entry_no):
            flash(f"Entry {entry_no} not found in database.")
            return redirect(url_for("index"))

        photo_record = system.get_registered_photo(entry_no)
        if photo_record:
            _, face_image, _ = photo_record
            response = Response(face_image, mimetype="image/jpeg")
        else:
            face_path = os.path.join(FACES_DIR, f"{entry_no}.jpg")
            if not os.path.exists(face_path):
                flash(f"No saved photo found for {entry_no}.")
                return redirect(url_for("index", entry_no=entry_no))

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

    @flask_app.route("/student/<kerberos_id>")
    def get_student_route(kerberos_id):
        student = system.get_student(kerberos_id)

        if student is None:
            return jsonify({
                "found": False
            })

        return jsonify({
            "found": True,
            "name": student["name"]
        })

    @flask_app.route("/attendance/<kerberos_id>", methods=["GET"])
    def get_attendance_route(kerberos_id):
        if not system.exists(kerberos_id):
            return jsonify({
                "found": False,
                "message": f"Student with Kerberos ID {kerberos_id} not found."
            }), 404

        records = system.get_attendance_records(kerberos_id)
        return jsonify({
            "found": True,
            "kerberos_id": kerberos_id,
            "count": len(records),
            "records": records
        })

    @flask_app.route("/confirm-attendance", methods=["POST"])
    def confirm_attendance():
        data = request.get_json() or {}
        event_id = data.get("event_id")
        confirmed = data.get("confirmed", False)

        if not event_id:
            return jsonify({"status": "error", "message": "Missing event ID"}), 400

        # Pop the data straight out of the system object memory space
        pending_data = system.pending_confirmation.pop(event_id, None)
        
        if not pending_data:
            return jsonify({"status": "error", "message": "Event expired or already processed"}), 404

        if confirmed:
            # Spawn the database worker using the method in your system class
            threading.Thread(
                target=system._insert_samples_to_db, 
                args=(pending_data['kerberos_id'], pending_data['samples']),
                daemon=True
            ).start()
            
            return jsonify({"status": "success", "message": "Attendance saved."})
        else:
            return jsonify({"status": "ignored", "message": "Attendance discarded."})
        
    return flask_app
    
if __name__ == "__main__":
    init_db()
    system = AttendanceSystem()
    system.start_recognition()
    app = create_flask_app(system)
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)