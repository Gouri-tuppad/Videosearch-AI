#!/usr/bin/env python
from app.db.database import SessionLocal
from app.db.crud import get_video_by_filename, list_frames_for_video, list_detections_for_frame

session = SessionLocal()
video = get_video_by_filename(session, "random3.mp4")
if video:
    print(f"Video: {video.filename}")
    frames = list_frames_for_video(session, video.id)
    all_labels = set()
    for frame in frames:
        dets = list_detections_for_frame(session, frame.id)
        for det in dets:
            all_labels.add(det.object_name.lower())
    
    print(f"All detected labels: {sorted(all_labels)}")
else:
    print("Video not found in database")

session.close()
