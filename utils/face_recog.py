import os
import time
import socket
import threading
from database.schema import pool
import numpy as np
from insightface.app import FaceAnalysis
from utils.frame import FrameFetcher


ESP32_IP = os.getenv("ESP32_IP")
ESP32_PORT = 4210
LOG_FILE = "attendance_log.txt"
REG_CAPTURE_WINDOW = 2
RTSP_URL = os.getenv("RTSP_URL")
RECOGNITION_THRESHOLD = 0.35


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


class AttendanceSystem:
    """Main Controller: Bridges Camera, AI Models, DB, and UI"""
    def __init__(self):
        self.db = pool
        self.ui = ESPDisplay(ip=ESP32_IP, port=ESP32_PORT)
        self.log_file = LOG_FILE
        self._face_lock = threading.Lock()
        self._recognition_lock = threading.Lock()
        self._recognition_thread = None
        self._recognition_stop_event = threading.Event()
        self._recognition_status = "stopped"
        self._last_message = ""
        self._last_attendance_message = ""
        self._last_attendance_event_id = 0

        print("Initializing ArcFace...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # persistent preview fetcher for dashboard camera stream
        try:
            self.preview_fetcher = FrameFetcher(RTSP_URL)
        except Exception:
            self.preview_fetcher = None

        self.anti_spoof = None  # temporarily disable anti-spoof

    @property
    def recognition_running(self):
        return self._recognition_thread is not None and self._recognition_thread.is_alive()

    def get_last_message(self):
        return self._last_message

    def _set_message(self, message):
        self._last_message = message
        print(message)

    def _capture_best_face(self, fetcher, window_seconds=None):
        start_time = time.time()
        best_face = None
        best_frame = None
        highest_det_score = 0.0

        if window_seconds is None:
            window_seconds = REG_CAPTURE_WINDOW

        while time.time() - start_time < window_seconds:
            ret, frame = fetcher.get_frame()
            if not ret:
                continue
            with self._face_lock:
                faces = self.app.get(frame)
            if faces:
                for face in faces:
                    if face.det_score > highest_det_score:
                        highest_det_score = face.det_score
                        best_face = face
                        best_frame = frame

        return best_face, best_frame, highest_det_score

    def mark_attendance(self, entry_no, name, similarity, time_taken):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {entry_no} - {name} - {similarity:.4f} - {time_taken:.2f}s\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        self._last_attendance_event_id += 1
        self._last_attendance_message = f"Attendance marked for {entry_no} - {name} at {timestamp}"
        self._set_message(self._last_attendance_message)
        self.ui.display_attendance("SUCCESS", name)
        print(self._last_attendance_message)

    def _recognition_loop(self):
        print("Starting recognition loop...")
        recently_marked = {}
        cooldown_period = 30

        fetcher = FrameFetcher(RTSP_URL)
        self._recognition_status = "starting"
        self._set_message("Starting recognition loop.")
        self._set_message("[Main] Waiting for first frame...")
        
        ok, _ = fetcher.get_frame(timeout=15.0)
        if not ok:
            self._set_message("[Main] Timed out waiting for camera. Check RTSP URL.")
            fetcher.stop()
            self._recognition_status = "stopped"
            return
        self._set_message("[Main] Camera ready. Starting inference loop.")
        self._recognition_status = "running"

        try:
            while not self._recognition_stop_event.is_set():
                start_time = time.time()
                ret, frame = fetcher.get_frame(timeout=1.0)
                if not ret:
                    if self._recognition_stop_event.is_set():
                        break
                    self._set_message("[Main] No frame available yet...")
                    time.sleep(0.1)
                    continue

                with self._face_lock:
                    faces = self.app.get(frame)
                if not faces:
                    continue

                for face in faces:
                    # Anti-spoofing disabled for testing (no model required)
                    # Previously we checked: self.anti_spoof.predict(...)
                    # if spoof detected we would ignore the face. That logic
                    # is intentionally disabled now so recognition proceeds.

                    # if self.anti_spoof is not None and self.anti_spoof.predict(frame, face.bbox, threshold=SPOOF_THRESHOLD):
                    #     self._set_message("Spoof detected! Ignoring.")
                    #     self.ui.display_attendance("SPOOF", "ALERT")
                    #     continue
                    embedding = face.normed_embedding.tolist()

                    best_match_entry = None
                    highest_sim = 0.0

                    result = None
                    with self.db.connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                SELECT s.student_name, s.roll_number, MIN(f.embedding <=> %s) AS min_distance
                                FROM student_faces f
                                JOIN students s ON f.student_id = s.id
                                GROUP BY s.id, s.student_name, s.roll_number
                                HAVING MIN(f.embedding <=> %s) <= %s
                                ORDER BY min_distance ASC
                                LIMIT 1;
                            """, (embedding, embedding, RECOGNITION_THRESHOLD))
                            
                            result = cur.fetchone()
                            
                    if result:
                        student_name, roll_number, min_distance = result
                        best_match_entry = roll_number
                    
                    self._set_message(f"Best match: {best_match_entry} with similarity {highest_sim:.4f}")

                    current_time = time.time()
                    if best_match_entry not in recently_marked or \
                        (current_time - recently_marked[best_match_entry]) > cooldown_period:
                        end_time = time.time()
                        self.mark_attendance(best_match_entry, student_name, highest_sim, end_time - start_time)
                        recently_marked[best_match_entry] = current_time

        except KeyboardInterrupt:
            self._set_message("Shutting down gracefully...")
        finally:
            fetcher.stop()
            self._recognition_status = "stopped"

    def start_recognition(self):
        with self._recognition_lock:
            if self.recognition_running:
                return False, "Recognition is already running."

            self._recognition_stop_event.clear()
            self._recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True, name="RecognitionLoop")
            self._recognition_thread.start()
            return True, "Recognition started."

    def stop_recognition(self):
        self._recognition_stop_event.set()
        thread = self._recognition_thread
        if thread and thread.is_alive():
            thread.join(timeout=5)
        self._recognition_status = "stopped"
        return True, "Recognition stopped."

    def get_log_lines(self, limit=12):
        if not os.path.exists(self.log_file):
            return []

        with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines[-limit:]
