import os
import time
import socket
import threading
import cv2
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
FACES_DIR = "registered_faces"


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

    def exists(self, entry_no):
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM students WHERE kerberos_id = %s LIMIT 1;",
                    (entry_no,),
                )
                return cur.fetchone() is not None

    def get_all(self):
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT kerberos_id, student_name FROM students ORDER BY kerberos_id ASC;"
                )
                rows = cur.fetchall()
        return {entry_no: {"name": student_name} for entry_no, student_name in rows}

    def register_user(self, name, entry_no, overwrite=True):
        entry_no = (entry_no or "").strip()
        name = (name or "").strip()
        if not entry_no or not name:
            self._set_message("Name and Entry No are required.")
            return False

        exists_already = self.exists(entry_no)
        if exists_already and not overwrite:
            self._set_message(f"{entry_no} already exists. Update cancelled.")
            return False

        fetcher = FrameFetcher(RTSP_URL)
        try:
            face, frame, det_score = self._capture_best_face(fetcher, window_seconds=REG_CAPTURE_WINDOW)
        finally:
            fetcher.stop()

        if not face or frame is None or det_score <= RECOGNITION_THRESHOLD:
            self._set_message("Failed to capture a clear face. Please try again.")
            return False

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = frame.shape[:2]
        face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if face_crop.size == 0:
            self._set_message("Captured face was invalid. Please try again.")
            return False

        os.makedirs(FACES_DIR, exist_ok=True)
        face_path = os.path.join(FACES_DIR, f"{entry_no}.jpg")
        cv2.imwrite(face_path, face_crop)

        embedding = face.normed_embedding.tolist()
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                if exists_already:
                    cur.execute(
                        """
                        UPDATE students
                        SET student_name = %s
                        WHERE kerberos_id = %s
                        RETURNING id;
                        """,
                        (name, entry_no),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO students (student_name, Kerberos_ID)
                        VALUES (%s, %s)
                        RETURNING id;
                        """,
                        (name, entry_no),
                    )

                student_id = cur.fetchone()[0]
                cur.execute("DELETE FROM student_faces WHERE student_id = %s;", (student_id,))
                cur.execute(
                    """
                    INSERT INTO student_faces (student_id, embedding)
                    VALUES (%s, %s::vector);
                    """,
                    (student_id, vector_literal),
                )
            conn.commit()

        self._set_message(f"Successfully registered {name} ({entry_no}).")
        return True

    def show_user(self, entry_no):
        entry_no = (entry_no or "").strip()
        if not self.exists(entry_no):
            self._set_message(f"Entry {entry_no} not found in database.")
            return None

        face_path = os.path.join(FACES_DIR, f"{entry_no}.jpg")
        if not os.path.exists(face_path):
            self._set_message(f"No saved photo found for {entry_no}.")
            return None

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT student_name FROM students WHERE Kerberos_ID = %s LIMIT 1;",
                    (entry_no,),
                )
                row = cur.fetchone()

        student_name = row[0] if row else "Unknown"
        self._set_message(f"Registered photo for {entry_no} - {student_name}")
        return face_path

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
                                SELECT s.student_name, s.Kerberos_ID , MIN(f.embedding <=> %s) AS min_distance
                                FROM student_faces f
                                JOIN students s ON f.student_id = s.id
                                GROUP BY s.id, s.student_name, s.kerberos_id
                                HAVING MIN(f.embedding <=> %s) <= %s
                                ORDER BY min_distance ASC
                                LIMIT 1;
                            """, (embedding, embedding, RECOGNITION_THRESHOLD))
                            
                            result = cur.fetchone()
                            
                    if result:
                        student_name, roll_number, min_distance = result
                        best_match_entry = roll_number
                        highest_sim = max(0.0, 1.0 - float(min_distance))
                    
                    self._set_message(f"Best match: {best_match_entry} with similarity {highest_sim:.4f}")

                    if not best_match_entry:
                        continue

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
