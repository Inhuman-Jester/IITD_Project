import os
import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@host:port/dbname")
print(f"Using database URL: {DB_URL}")

pool = ConnectionPool(conninfo=DB_URL, min_size=1, max_size=10)

def init_db():
    pool.open()

    schema_sql = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        student_name VARCHAR(100) NOT NULL,
        entry_number VARCHAR(20) UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS student_faces (
        id SERIAL PRIMARY KEY,
        student_id INTEGER REFERENCES students(id) ON DELETE CASCADE,
        embedding VECTOR(512) NOT NULL
    );

    CREATE INDEX IF NOT EXISTS student_faces_hnsw_idx 
    ON student_faces USING hnsw (embedding vector_cosine_ops);
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