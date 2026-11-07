import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha1 TEXT,
    taken_at TEXT,
    kind TEXT NOT NULL DEFAULT 'image',
    width INTEGER,
    height INTEGER,
    thumb TEXT,
    indexed_at TEXT NOT NULL,
    error TEXT,
    place TEXT,
    camera TEXT,
    dhash TEXT,
    duration REAL,
    favorite INTEGER NOT NULL DEFAULT 0,
    root TEXT
);
CREATE INDEX IF NOT EXISTS idx_media_sha1 ON media(sha1);
CREATE INDEX IF NOT EXISTS idx_media_taken ON media(taken_at);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    centroid BLOB NOT NULL,
    n INTEGER NOT NULL DEFAULT 0,
    auto_name INTEGER NOT NULL DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    bbox TEXT,
    det_score REAL,
    embedding BLOB NOT NULL,
    cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    age REAL,
    age_band TEXT,
    gender TEXT,
    thumb TEXT,
    frame_time REAL,
    age_conf REAL
);
CREATE INDEX IF NOT EXISTS idx_faces_media ON faces(media_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS idx_faces_band ON faces(age_band);

CREATE TABLE IF NOT EXISTS saved_searches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    query TEXT NOT NULL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

MIGRATIONS = {
    "media": {
        "place": "TEXT",
        "camera": "TEXT",
        "dhash": "TEXT",
        "duration": "REAL",
        "favorite": "INTEGER NOT NULL DEFAULT 0",
        "root": "TEXT",
    },
    "faces": {
        "frame_time": "REAL",
        "age_conf": "REAL",
    },
}


def connect():
    config.ensure_dirs()
    conn = sqlite3.connect(str(config.DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def db():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def migrate(conn):
    for table, columns in MIGRATIONS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_dhash ON media(dhash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_fav ON media(favorite)")


def init_db(conn=None):
    own = conn is None
    if own:
        conn = connect()
    try:
        conn.executescript(SCHEMA)
        migrate(conn)
        conn.commit()
    finally:
        if own:
            conn.close()


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn, key, value):
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def threshold(conn):
    raw = get_meta(conn, "cluster_threshold")
    return float(raw) if raw is not None else config.CLUSTER_THRESHOLD


def roots(conn):
    import json

    raw = get_meta(conn, "roots")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(p) for p in data if p]
        except ValueError:
            pass
    single = get_meta(conn, "last_scan_root")
    return [single] if single else []


def set_roots(conn, paths):
    import json

    cleaned = []
    for p in paths:
        p = str(p)
        if p and p not in cleaned:
            cleaned.append(p)
    set_meta(conn, "roots", json.dumps(cleaned))
    if cleaned:
        set_meta(conn, "last_scan_root", ", ".join(cleaned))
