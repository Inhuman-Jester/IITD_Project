import os
import time
import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/dbname")


class LazyConnectionPool:
    def __init__(self, conninfo, min_size=1, max_size=10):
        self.conninfo = conninfo
        self.min_size = min_size
        self.max_size = max_size
        self._pool = None

    @property
    def closed(self):
        return self._pool is None or self._pool.closed

    def _wait_for_database(self, timeout_seconds=60, retry_delay=2):
        deadline = time.monotonic() + timeout_seconds
        last_error = None

        while time.monotonic() < deadline:
            try:
                with psycopg.connect(self.conninfo, connect_timeout=5) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                        cur.fetchone()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(retry_delay)

        raise last_error if last_error is not None else RuntimeError("Database could not be reached")

    def _ensure_pool(self):
        if self._pool is None:
            self._wait_for_database()
            self._pool = ConnectionPool(conninfo=self.conninfo, min_size=self.min_size, max_size=self.max_size)
        return self._pool

    def open(self):
        return self._ensure_pool().open()

    def connection(self):
        return self._ensure_pool().connection()

    def close(self):
        if self._pool is not None and not self._pool.closed:
            self._pool.close()


pool = LazyConnectionPool(DB_URL, min_size=1, max_size=10)

def init_db():
    pool.open()

    schema_sql = """
    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        student_name VARCHAR(100) NOT NULL,
        kerberos_id VARCHAR(20) UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS student_faces (
        id SERIAL PRIMARY KEY,
        kerberos_id VARCHAR(20) NOT NULL REFERENCES students(kerberos_id) ON DELETE CASCADE,
        sample_number SMALLINT NOT NULL,
        face_image BYTEA,
        embedding VECTOR(512) NOT NULL,
        captured_at TIMESTAMP NOT NULL DEFAULT NOW(),

        CONSTRAINT student_faces_sample_number_range
            CHECK (sample_number BETWEEN 1 AND 3),

        CONSTRAINT unique_student_face_sample
            UNIQUE (kerberos_id, sample_number)
    );

    CREATE TABLE IF NOT EXISTS attendance_records (
        id SERIAL PRIMARY KEY,
        kerberos_id VARCHAR(20) NOT NULL REFERENCES students(kerberos_id) ON DELETE CASCADE,
        attendance_date DATE NOT NULL DEFAULT CURRENT_DATE,
        attendance_time TIMESTAMP NOT NULL DEFAULT NOW(),
        similarity REAL NOT NULL,
        time_taken REAL NOT NULL,

        CONSTRAINT unique_student_attendance_per_day
            UNIQUE (kerberos_id, attendance_date)
    );

    CREATE TABLE IF NOT EXISTS test_students (
        id SERIAL PRIMARY KEY,
        kerberos_id VARCHAR(20) NOT NULL REFERENCES students(kerberos_id) ON DELETE CASCADE,
        embedding VECTOR(512) NOT NULL,
        det_score REAL NOT NULL,
        sample_number SMALLINT NOT NULL,
        data_point_number SMALLINT NOT NULL,

        CONSTRAINT unique_test_student_sample
            UNIQUE (kerberos_id, sample_number, data_point_number)

    );

    CREATE INDEX IF NOT EXISTS student_faces_hnsw_idx
    ON student_faces USING hnsw (embedding vector_cosine_ops);

    CREATE INDEX IF NOT EXISTS student_faces_kerberos_id_idx
    ON student_faces (kerberos_id);  
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

        with conn.cursor() as cur:
            cur.execute(schema_sql)
            conn.commit()
    print("Database schema successfully verified and initialized.")


def close_db():
    if not pool.closed:
        pool.close()
    print("Database connection pool closed.")