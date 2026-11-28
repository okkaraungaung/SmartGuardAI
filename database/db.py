import json
import sqlite3
from datetime import datetime


def get_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn


def init_db(db_path, default_settings=None):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    default_settings = {
        "loitering_threshold": "15.0",
        "detection_confidence": "0.45",
        "enable_unknown": "1",
        "enable_loitering": "1",
        "enable_package": "1",
        "enable_vehicle": "0",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "roi_enabled": "1",
        "roi_polygon": "[[0.20, 0.35], [0.80, 0.35], [0.92, 0.98], [0.08, 0.98]]"
    }

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_image TEXT NOT NULL,
            face_embedding TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            duration REAL,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'completed'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            tracking_id INTEGER,
            object_type TEXT,
            person_name TEXT,
            confidence REAL,
            timestamp REAL,
            image_path TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            snapshot_path TEXT,
            video_timestamp REAL,
            event_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            telegram_sent INTEGER DEFAULT 0,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    if default_settings:
        for k, v in default_settings.items():
            cursor.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (k, str(v))
            )

    conn.commit()
    conn.close()


# --- Settings Helpers ---
def get_settings(db_path):
    conn = get_connection(db_path)
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def update_settings(db_path, settings_dict):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    for k, v in settings_dict.items():
        cursor.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (k, str(v))
        )
    conn.commit()
    conn.close()


# --- Users Helpers ---
def add_user(db_path, name, face_image, embedding_list):
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO users (name, face_image, face_embedding, created_at) VALUES (?, ?, ?, ?)",
        (name, face_image, json.dumps(embedding_list), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_all_users(db_path):
    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "face_image": r["face_image"],
            "embedding": json.loads(r["face_embedding"]),
            "created_at": r["created_at"]
        }
        for r in rows
    ]


def delete_user(db_path, user_id):
    conn = get_connection(db_path)
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


# --- Videos & Detections Helpers ---
def insert_video(db_path, filename, file_path, duration):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO videos (filename, file_path, duration, uploaded_at, status) VALUES (?, ?, ?, ?, ?)",
        (filename, file_path, duration, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "completed")
    )
    video_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return video_id


def insert_detections_bulk(db_path, video_id, records):
    if not records:
        return
    conn = get_connection(db_path)
    conn.executemany(
        """
        INSERT INTO detections (video_id, tracking_id, object_type, person_name, confidence, timestamp, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                video_id,
                r.get("tracking_id"),
                r.get("label"),
                r.get("person_name"),
                r.get("confidence"),
                r.get("timestamp"),
                r.get("image_path", "")
            )
            for r in records
        ]
    )
    conn.commit()
    conn.close()


# --- Security Events Helpers ---
def insert_security_event(db_path, video_id, event_type, severity, description, snapshot_path, video_timestamp, telegram_sent):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO security_events (video_id, event_type, severity, description, snapshot_path, video_timestamp, event_time, telegram_sent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            event_type,
            severity,
            description,
            snapshot_path,
            video_timestamp,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            1 if telegram_sent else 0
        )
    )
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_security_events(db_path, event_type=None, date_query=None):
    conn = get_connection(db_path)
    query = """
        SELECT e.*, v.filename AS video_filename
        FROM security_events e
        LEFT JOIN videos v ON e.video_id = v.id
        WHERE 1=1
    """
    params = []
    if event_type and event_type != "ALL":
        query += " AND e.event_type = ?"
        params.append(event_type)
    if date_query:
        query += " AND DATE(e.event_time) = ?"
        params.append(date_query)

    query += " ORDER BY e.id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_dashboard_stats(db_path):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM security_events WHERE event_type = 'Unknown Loitering'")
    loitering_alerts = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM security_events WHERE event_type = 'Package Detected'")
    package_alerts = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM security_events WHERE telegram_sent = 1")
    telegram_sent_count = cur.fetchone()[0]

    cur.execute("""
        SELECT id, event_type, severity, description, snapshot_path, video_timestamp, telegram_sent
        FROM security_events
        ORDER BY id DESC
        LIMIT 5
    """)
    recent_events = [dict(row) for row in cur.fetchall()]

    conn.close()
    return {
        "total_users": total_users,
        "loitering_alerts": loitering_alerts,
        "package_alerts": package_alerts,
        "telegram_sent_count": telegram_sent_count,
        "recent_events": recent_events
    }