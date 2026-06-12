import os
import time
import socket
import threading
import cv2
from database.schema import pool
import numpy as np
from insightface.app import FaceAnalysis
from utils.frame import FrameFetcher
from dotenv import load_dotenv
from urllib.parse import quote, unquote

load_dotenv()

print("ESP_IP =", repr(os.getenv("ESP32_IP")))

ESP32_IP = os.getenv("ESP32_IP")
ESP32_PORT = 4210
LOG_FILE = "attendance_log.txt"
REG_CAPTURE_WINDOW = 2
CAM_IP = os.getenv("CAM_IP", "10.208.22.128")
CAM_USER = os.getenv("CAM_USER", "admin")
CAM_PASSWORD = os.getenv("CAM_PASSWORD", "SOumil@@btp1")
RTSP_URL = (
    f"rtsp://{quote(unquote(CAM_USER), safe='')}:{quote(unquote(CAM_PASSWORD), safe='')}"
    f"@{CAM_IP}:554/video/live?channel=1&subtype=0"
)
RECOGNITION_THRESHOLD = 0.65
REG_VERIFICATION_THRESHOLD = 0.65
FACES_DIR = "registered_faces"
CAPTURE_WINDOW_BINS = 10


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
    
    def registering_user(self, name, kerberos_id, message=None):
        if message is None:
            message = f"REGISTERING:{kerberos_id}:{name}"
        self.sock.sendto(bytes(message, "utf-8"), (self.ip, self.port))
        print(f"UDP Sent to ESP32: {message}")


class AttendanceSystem:
    """Main Controller: Bridges Camera, AI Models, DB, and UI"""
    def __init__(self):
        self.db = pool
        self.ui = ESPDisplay(ip=ESP32_IP, port=ESP32_PORT)
        print(f"LCD IP : {ESP32_IP}")
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

    def _capture_best_face(self, fetcher, window_seconds=None, bin_number = None):
        start_time = time.time()
        best_face = None
        best_frame = None
        highest_det_score = 0.0

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
    
    def cosine_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def exists(self, kerberos_id):
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM students WHERE kerberos_id = %s LIMIT 1;",
                    (kerberos_id,),
                )
                return cur.fetchone() is not None

    def get_all(self):
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT kerberos_id, student_name FROM students ORDER BY kerberos_id ASC;"
                )
                rows = cur.fetchall()
        return {kerberos_id: {"name": student_name} for kerberos_id, student_name in rows}

    def get_registered_photo(self, kerberos_id = None):
        if not kerberos_id:
            return None

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.student_name, f.face_image, f.sample_number
                    FROM students s
                    JOIN student_faces f ON f.student_id = s.id
                    WHERE s.kerberos_id = %s AND f.face_image IS NOT NULL
                    ORDER BY f.sample_number DESC, f.id DESC
                    LIMIT 1;
                    """,
                    (kerberos_id,),
                )
                row = cur.fetchone()

        if not row:
            return None

        student_name, face_image, sample_number = row
        return student_name, bytes(face_image), sample_number
    
    def get_student(self, kerberos_id):
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT kerberos_id, student_name
                    FROM students
                    WHERE kerberos_id = %s
                    """,
                    (kerberos_id,)
                )

                row = cur.fetchone()

                if not row:
                    return None

                return {
                    "kerberos_id": row[0],
                    "name": row[1]
                }

    def register_user(self, name, kerberos_id, overwrite=True):
        kerberos_id = (kerberos_id or "").strip()
        name = (name or "").strip()
        if not kerberos_id or not name:
            self._set_message("Name and Kerberos ID are required.")
            return False

        exists_already = self.exists(kerberos_id)
        if exists_already and not overwrite:
            self._set_message(f"{kerberos_id} already exists. Update cancelled.")
            return False

        # Commented two step capture
        # fetcher = FrameFetcher(RTSP_URL)
        # first_face, first_frame, first_score = self._capture_best_face(fetcher, window_seconds=REG_CAPTURE_WINDOW)

        # if not first_face or first_score <= RECOGNITION_THRESHOLD:
        #     fetcher.stop()
        #     self._set_message("Failed to capture a clear face. Please try again.")
        #     return False
        
        # self._set_message("First capture saved. Hold still for a second verification capture.")
        # second_face, second_frame, second_score = self._capture_best_face(fetcher, window_seconds=REG_CAPTURE_WINDOW)
        # fetcher.stop()

        # if not second_face or second_score <= RECOGNITION_THRESHOLD:
        #     self._set_message("Failed to capture a clear verification face. Please try again.")
        #     return False

        # verification_sim = self.cosine_similarity(first_face.normed_embedding, second_face.normed_embedding)
        # self._set_message(f"Verification similarity: {verification_sim:.4f}")

        # threshold = REG_VERIFICATION_THRESHOLD if REG_VERIFICATION_THRESHOLD is not None else RECOGNITION_THRESHOLD
        # if verification_sim < threshold:
        #     self._set_message("Second verification did not match the first capture. Registration cancelled.")
        #     return False
        sample = 1
        while sample <= 10:  # allow up to 10 captures within the window to find a good face
            fetcher = FrameFetcher(RTSP_URL)
            face, frame, score = self._capture_best_face(fetcher, window_seconds=REG_CAPTURE_WINDOW, bin_number=CAPTURE_WINDOW_BINS)
            fetcher.stop()

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            h, w = frame.shape[:2]
            face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if face_crop.size == 0:
                self._set_message("Captured face was invalid. Please try again.")
                return False

            success, encoded_image = cv2.imencode(".jpg", face_crop)
            if not success:
                self._set_message("Failed to encode captured face. Please try again.")
                return False

            embedding = face.normed_embedding.tolist()
            vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
            face_image = encoded_image.tobytes()

            if sample <= 3:
                check = self.register_main_db(kerberos_id, exists_already, name, face_image, vector_literal)
                if(not check):
                    return False
            else:
                self.register_to_test_db(kerberos_id, exists_already, name, face_image, vector_literal)

            self.ui.registering_user(name, kerberos_id, message=f"Sample:{sample}")
            
            print(f"Captured sample {sample} for {kerberos_id} with detection score {score:.4f}")
            sample+=1
                
        
        return True
    
    def register_main_db(self, kerberos_id, exists_already, name, face_image, vector_literal):
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                if exists_already:
                    cur.execute(
                        "SELECT id FROM students WHERE kerberos_id = %s FOR UPDATE;",
                        (kerberos_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        self._set_message(f"Kerberos ID {kerberos_id} not found in database.")
                        return False

                    student_id = row[0]
                    # Count existing face samples while holding the student row lock
                    cur.execute(
                        "SELECT COUNT(*) FROM student_faces WHERE student_id = %s;",
                        (student_id,),
                    )
                    face_count = cur.fetchone()[0]
                    if face_count >= 3:
                        self._set_message(f"{kerberos_id} already has 3 face samples. Add another face is blocked.")
                        return False

                    cur.execute(
                        """
                        UPDATE students
                        SET student_name = %s
                        WHERE id = %s;
                        """,
                        (name, student_id),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO students (student_name, kerberos_id)
                        VALUES (%s, %s)
                        ON CONFLICT (kerberos_id)
                        DO UPDATE SET student_name = EXCLUDED.student_name
                        RETURNING id;
                        """,
                        (name, kerberos_id),
                    )

                    student_id = cur.fetchone()[0]

                    cur.execute(
                        "SELECT COUNT(*) FROM student_faces WHERE student_id = %s;",
                        (student_id,),
                    )
                    face_count = cur.fetchone()[0]

                sample_number = face_count + 1
                try:
                    cur.execute(
                        """
                        INSERT INTO student_faces (
                            student_id,
                            sample_number,
                            face_image,
                            embedding,
                            captured_at
                        )
                        VALUES (%s, %s, %s, %s::vector, NOW());
                        """,
                        (student_id, sample_number, face_image, vector_literal),
                    )
                except Exception as e:
                    # Handle unique constraint or other DB errors gracefully
                    conn.rollback()
                    self._set_message(f"Failed to store face sample: {e}")
                    return False
            conn.commit()

        self._set_message(f"Successfully registered {name} ({kerberos_id}) with face sample {sample_number}/3.")
        return True

    def register_to_test_db(self, kerberos_id, exists_already, name, face_image, vector_literal):
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO test_students (student_id, face_image, embedding)
                    VALUES (%s, %s, %s::vector);
                    """,
                    (kerberos_id, face_image, vector_literal),
                )
            conn.commit()
        self._set_message(f"Successfully registered {name} ({kerberos_id}) to test DB.")
        return True

    def show_user(self, kerberos_id):
        kerberos_id = (kerberos_id or "").strip()
        if not self.exists(kerberos_id):
            self._set_message(f"Kerberos ID {kerberos_id} not found in database.")
            return None

        photo_record = self.get_registered_photo(kerberos_id)
        if photo_record:
            student_name, _, sample_number = photo_record
            self._set_message(f"Registered photo for {kerberos_id} - {student_name} (sample {sample_number}/3)")
            return photo_record

        face_path = os.path.join(FACES_DIR, f"{kerberos_id}.jpg")
        if os.path.exists(face_path):
            self._set_message(f"Registered photo for {kerberos_id} is still available locally.")
            return face_path

        self._set_message(f"No saved photo found for {kerberos_id}.")
        return None

    def mark_attendance(self, kerberos_id, name, similarity, time_taken):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {kerberos_id} - {name} - {similarity:.4f} - {time_taken:.2f}s\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO attendance_records (
                        student_id,
                        attendance_date,
                        attendance_time,
                        similarity,
                        time_taken
                    )
                    SELECT id, CURRENT_DATE, NOW(), %s, %s
                    FROM students
                    WHERE kerberos_id = %s
                    LIMIT 1
                    ON CONFLICT (student_id, attendance_date) DO NOTHING;
                    """,
                    (similarity, time_taken, kerberos_id),
                )
                if cur.rowcount > 0:
                    conn.commit()

        self._last_attendance_event_id += 1
        self._last_attendance_message = f"Attendance marked for {kerberos_id} - {name} at {timestamp}"
        self._set_message(self._last_attendance_message)
        self.ui.display_attendance("MARKED", name)
        print(self._last_attendance_message)

    def recognition_loop_test(self):
        print("Starting recognition loop test...")
        fetcher = FrameFetcher(RTSP_URL)
        self._recognition_status = "starting"
        self._set_message("Starting recognition loop test.")
        self._set_message("[Test] Waiting for first frame...")
        
        ok, _ = fetcher.get_frame(timeout=15.0)
        if not ok:
            self._set_message("[Main] Timed out waiting for camera. Check RTSP URL.")
            fetcher.stop()
            self._recognition_status = "stopped"
            return
        self._set_message("[Main] Camera ready. Starting inference loop.")
        self._recognition_status = "running"

        recently_marked = {}
        cooldown_period = 30
        collection_cache = {}   
        cache_lock = threading.Lock()

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
                    embedding = face.normed_embedding.tolist()
                    embedding_vector = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"

                    best_match_entry = None
                    highest_sim = 0.0

                    result = None
                    with self.db.connection() as conn:
                        with conn.cursor() as cur:
                                cur.execute("""
                                    SELECT s.student_name, s.kerberos_id, MIN(f.embedding <=> %s::vector) AS min_distance
                                    FROM student_faces f
                                    JOIN students s ON f.student_id = s.id
                                    GROUP BY s.id, s.student_name, s.kerberos_id
                                    HAVING MIN(f.embedding <=> %s::vector) <= %s
                                    ORDER BY min_distance ASC
                                    LIMIT 1;
                                """, (embedding_vector, embedding_vector, 1-RECOGNITION_THRESHOLD))

                                result = cur.fetchone()
                            
                    if result:
                        student_name, kerberos_id, min_distance = result
                        best_match_entry = kerberos_id
                        highest_sim = max(0.0, 1.0 - float(min_distance))

                    if not best_match_entry:
                        continue
                    
                    with cache_lock:
                        current_time = time.time()
                        if best_match_entry not in collection_cache:
                            collection_cache[best_match_entry] = {
                                'samples' : [],
                                'last_updated': 0.0
                            }

                        student_data = collection_cache[best_match_entry]
                        
                        if current_time - student_data['last_updated'] > 1:

                            new_sample = {
                                'embedding': embedding_vector,
                                'detection_score': face.det_score,
                                'name': student_name
                            }
                            student_data['samples'].append(new_sample)
                            student_data['last_updated'] = current_time

                            if(len(student_data['samples']) == 3):
                                data = collection_cache.pop(best_match_entry)

                                threading.Thread(
                                    target=self._insert_samples_to_db, 
                                    args=(best_match_entry, data['samples']),
                                    daemon=True
                                ).start()
        
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
        
    def _insert_samples_to_db(self, kerberos_id, samples):
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(MAX(data_point_number), 0) + 1 
                    FROM test_students 
                    WHERE student_id = %s;
                """,
                (kerberos_id,)
                )
                result = cur.fetchone()
                data_point_number = result[0] if result else 1

                for sample_number in range(len(samples)):
                    sample = samples[sample_number]
                    try:
                        cur.execute(
                            """
                            INSERT INTO test_students (student_id, embedding, det_score, sample_number, data_point_number)
                            VALUES (%s, %s::vector, %s, %s, %s);
                            """,
                            (kerberos_id, sample['embedding'], sample['detection_score'], sample_number, data_point_number),
                        )
                    except Exception as e:
                        print(f"Failed to insert sample for {kerberos_id}: {e}")
            conn.commit()
        print(f"Inserted {len(samples)} samples for {kerberos_id} into test DB.")

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

                    # Convert embedding to a pgvector literal so psycopg doesn't send it
                    # as a SQL array (double precision[]). The query casts this
                    # literal to type `vector` so the <=> operator is available.
                    embedding_vector = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"

                    best_match_entry = None
                    highest_sim = 0.0

                    result = None
                    with self.db.connection() as conn:
                        with conn.cursor() as cur:
                                cur.execute("""
                                    SELECT s.student_name, s.kerberos_id, MIN(f.embedding <=> %s::vector) AS min_distance
                                    FROM student_faces f
                                    JOIN students s ON f.student_id = s.id
                                    GROUP BY s.id, s.student_name, s.kerberos_id
                                    HAVING MIN(f.embedding <=> %s::vector) <= %s
                                    ORDER BY min_distance ASC
                                    LIMIT 1;
                                """, (embedding_vector, embedding_vector, 1-RECOGNITION_THRESHOLD))

                                result = cur.fetchone()
                            
                    if result:
                        student_name, kerberos_id, min_distance = result
                        best_match_entry = kerberos_id
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
            self._recognition_thread = threading.Thread(target=self.recognition_loop_test, daemon=True, name="RecognitionLoop")
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
