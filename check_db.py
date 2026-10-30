#!/usr/bin/env python
from app.db.database import SessionLocal
from app.db.crud import get_video_by_filename, list_frames_for_video, list_detections_for_frame

session = SessionLocal()
video = get_video_by_filename(session, "random3.mp4")
if video:
    print(f"Video found: {video.filename}, ID: {video.id}")
    frames = list_frames_for_video(session, video.id)
    print(f"Frames count: {len(frames)}")
    if frames:
        for frame in frames[:3]:
            dets = list_detections_for_frame(session, frame.id)
            print(f"  Frame {frame.frame_name}: {len(dets)} detections")
            for det in dets[:3]:
                print(f"    - {det.object_name}: {det.confidence}")
    else:
        print("No frames in database!")
else:
    print("Video not found in database")

session.close()
