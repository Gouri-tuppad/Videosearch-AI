#!/usr/bin/env python
import sys
sys.path.insert(0, '/d:/PROJECTS/videoprompt2')

from pathlib import Path
from app.core.detector import detector
from PIL import Image

# Get a few frames from different parts of the video
frame_dir = Path('embeddings/random3.mp4')  # This is where embeddings metadata is
metadata_file = frame_dir / 'metadata.json'

import json
with open(metadata_file) as f:
    metadata = json.load(f)

# Extract frames to check
from app.frame_extractor import extract_frames
frames_output = Path('frames/check_video')
frames_output.mkdir(parents=True, exist_ok=True)

try:
    frames = extract_frames('videos/random3.mp4', frames_output)
    print(f"Extracted {len(frames)} frames")
    
    # Check frames at different timestamps
    for idx in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]:
        frame_info = frames[idx]
        frame_path = frames_output / frame_info['frame_filename']
        
        if frame_path.exists():
            print(f"\nFrame {idx} (timestamp {frame_info['timestamp']}s):")
            detections = detector.predict(frame_path)
            for det in detections:
                print(f"  - {det['object_name']}: {det['confidence']:.2f}")
            if not detections:
                print("  No detections")
        else:
            print(f"Frame {idx} not found: {frame_path}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
