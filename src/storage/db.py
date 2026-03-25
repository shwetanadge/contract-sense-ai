import sqlite3
from datetime import datetime

DB_PATH = "compliance_rag.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            chunk_id INTEGER,
            text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create query_log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            faithfulness_score REAL,
            latency_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized!")

def insert_chunks(chunks):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO chunks (source, chunk_id, text)
            VALUES (?, ?, ?)
        """, (
            chunk['metadata']['source'],
            chunk['metadata']['chunk_id'],
            chunk['text']
        ))
    
    conn.commit()
    conn.close()
    print(f"Inserted {len(chunks)} chunks into SQLite")

def log_query(question, answer, latency_seconds, faithfulness_score=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO query_log (question, answer, faithfulness_score, latency_seconds)
        VALUES (?, ?, ?, ?)
    """, (question, answer, faithfulness_score, latency_seconds))
    
    conn.commit()
    conn.close()

def get_query_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_log")
    rows = cursor.fetchall()
    conn.close()
    return rows