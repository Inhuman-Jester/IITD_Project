import os
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/dbname")

pool = ConnectionPool(conninfo=DB_URL, min_size=1, max_size=10)

def init_db():
    pool.open()

    schema_sql = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        student_name VARCHAR(100) NOT NULL,
        kerberos_id VARCHAR(20) UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS student_faces (
        id SERIAL PRIMARY KEY,
        student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
        sample_number SMALLINT NOT NULL,
        face_image BYTEA,
        embedding VECTOR(512) NOT NULL,
        captured_at TIMESTAMP NOT NULL DEFAULT NOW(),

        CONSTRAINT student_faces_sample_number_range
            CHECK (sample_number BETWEEN 1 AND 3),

        CONSTRAINT unique_student_face_sample
            UNIQUE (student_id, sample_number)
    );

    CREATE TABLE IF NOT EXISTS attendance_records (
        id SERIAL PRIMARY KEY,
        student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
        attendance_date DATE NOT NULL DEFAULT CURRENT_DATE,
        attendance_time TIMESTAMP NOT NULL DEFAULT NOW(),
        similarity REAL NOT NULL,
        time_taken REAL NOT NULL,

        CONSTRAINT unique_student_attendance_per_day
            UNIQUE (student_id, attendance_date)
    );

    CREATE TABLE IF NOT EXISTS test_students (
        id SERIAL PRIMARY KEY,
        student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
        face_image BYTEA,
        embedding VECTOR(512) NOT NULL
    );

    CREATE TABLE IF NOT EXISTS test_students_attendance (
        id SERIAL PRIMARY KEY,  
        student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
        face_image BYTEA,
        embedding VECTOR(512) NOT NULL
    );

    CREATE INDEX IF NOT EXISTS student_faces_hnsw_idx
    ON student_faces USING hnsw (embedding vector_cosine_ops);

    CREATE INDEX IF NOT EXISTS student_faces_student_id_idx
    ON student_faces (student_id);  
    """

    with pool.connection() as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(schema_sql)
            conn.commit()
    print("Database schema successfully verified and initialized.")


def close_db():
    pool.close()
    print("Database connection pool closed.")