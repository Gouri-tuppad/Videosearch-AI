"""Add new columns and tables for the pipeline refactor.

Run once before starting the server after updating to the new pipeline.
Safe to run multiple times — each ALTER / CREATE uses IF NOT EXISTS.

Usage:
    python -m app.migrate_db
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("app.db")


def migrate() -> None:
    if not DB_PATH.exists():
        print(f"Database {DB_PATH} not found — it will be created on first server start.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # ── Add missing columns to 'videos' ─────────────────────────────
    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(videos)").fetchall()}

    if "file_hash" not in existing_cols:
        cur.execute("ALTER TABLE videos ADD COLUMN file_hash TEXT")
        print("Added column: videos.file_hash")

    if "status" not in existing_cols:
        cur.execute("ALTER TABLE videos ADD COLUMN status TEXT NOT NULL DEFAULT 'uploading'")
        print("Added column: videos.status")

    if "progress" not in existing_cols:
        cur.execute("ALTER TABLE videos ADD COLUMN progress REAL NOT NULL DEFAULT 0.0")
        print("Added column: videos.progress")

    # ── Create 'events' table ───────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id  INTEGER NOT NULL REFERENCES videos(id),
            label     TEXT    NOT NULL,
            start     REAL    NOT NULL,
            end       REAL    NOT NULL,
            clip_path TEXT,
            max_conf  REAL    NOT NULL DEFAULT 0.0
        )
    """)
    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS ix_events_video_id ON events(video_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_events_label    ON events(label)")
    print("Ensured table: events")

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()

