#!/usr/bin/env python
from pathlib import Path
import shutil

# Clear old processing artifacts for random3.mp4
video_name = "random3.mp4"

paths_to_clear = [
    Path(f"faiss_indexes/{video_name}.index"),
    Path(f"faiss_indexes/{video_name}_metadata.json"),
    Path(f"embeddings/{video_name}"),
    Path(f"frames/{video_name}"),
    Path(f"processing/status.json"),  # Clear status to reset
]

for path in paths_to_clear:
    if path.is_dir():
        print(f"Removing directory: {path}")
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        print(f"Removing file: {path}")
        path.unlink(missing_ok=True)

print(f"\nCleared all artifacts for {video_name}")
print("Pipeline will now reprocess the video on next upload/process request")

# Also delete from database
from app.db.database import SessionLocal
from app.db.crud import get_video_by_filename, delete_video

session = SessionLocal()
video = get_video_by_filename(session, video_name)
if video:
    print(f"Deleting video record from database: ID={video.id}")
    delete_video(session, video.id)
else:
    print(f"Video record not found in database")
session.close()

print("Done! Ready for reprocessing.")
