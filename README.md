# AI Natural Language Video Search System

The **AI Natural Language Video Search System** is an intelligent end-to-end video processing and retrieval platform that empowers users to search through long-form video content using everyday natural language queries. By marrying state-of-the-art computer vision (YOLOv8) with multimodal semantic understanding (OpenAI CLIP) and high-speed vector indexing (FAISS), the system automatically samples frames, detects visual events, builds vector embeddings, pre-cuts relevant video segments, and retrieves precise timestamped clips in response to open-ended text searches.

![Demo](docs/demo.png)

---

## Features

- **Video Ingestion & Deduplication**: Upload video files (`.mp4`, `.avi`, `.mov`, etc.) via a responsive web dashboard or REST API. Deduplication based on SHA-256 file hashing avoids redundant reprocessing.
- **Automated Video Event Pipeline**: Background worker performs single-pass frame extraction (1 frame/sec plus scene change detection), running YOLOv8 object detection ($\ge 0.5$ confidence) and CLIP frame embedding simultaneously.
- **Smart Temporal Event Tracking & Pre-Cut Clips**: Automatically tracks detected objects over time (opens event after $\ge 2$ consecutive detections, accommodates 2-second gaps, closes when absent $> 2$ seconds or length exceeds 30 seconds). Cuts padded video clips ($1.5$s padding) with `ffmpeg` during indexing and stores structured events in SQLite.
- **Hierarchical Search Engine**:
  1. **Events-First with Synonyms**: Matches queries (and expanded synonyms like *"puppy"* $\rightarrow$ *"dog"*, *"automobile"* $\rightarrow$ *"car"*) against detected events in SQLite and returns pre-cut clips instantly.
  2. **CLIP + FAISS Semantic Fallback**: Fallback semantic search for open-ended queries (prompted as *"a photo of {query}"*) with score thresholding ($\ge 0.25$) and distribution margin checks to filter out false positives.
- **Graceful "Not Found" Experience**: When neither stage yields high-confidence results, the system rejects false positives and provides interactive suggestion chips based on objects actually present in the video.
- **Modern Interactive Dashboard**: React + TypeScript frontend featuring drag-and-drop uploads, real-time pipeline progress meters, confidence gauges, video previewers, and clip download capabilities.

---

## Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+, asynchronous REST API)
- **Object Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n.pt`)
- **Semantic Text/Image Embeddings**: [OpenAI CLIP](https://github.com/openai/CLIP) via HuggingFace Transformers (`clip-vit-base-patch32`)
- **Vector Search Engine**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`)
- **Video & Frame Processing**: [OpenCV](https://opencv.org/) (`opencv-python`) & [FFmpeg](https://ffmpeg.org/)
- **Database & ORM**: SQLite with [SQLAlchemy](https://www.sqlalchemy.org/)
- **Frontend UI**: React, TypeScript, Tailwind CSS, Vite, Framer Motion, Lucide / React Icons

---

## Project Structure

```
videoprompt2/
├── app/
│   ├── main.py                     # FastAPI application entrypoint & static mounts
│   ├── pipeline.py                 # Background pipeline: sampling, YOLOv8, CLIP, clipping
│   ├── search_service.py           # Events-first & CLIP semantic fallback search logic
│   ├── clip_generator.py           # FFmpeg video clip slicing utilities
│   ├── faiss_index.py              # FAISS vector index creation and persistence
│   ├── frame_extractor.py          # Frame extraction and scene change detection
│   ├── migrate_db.py               # Database migration & schema initialization
│   ├── api/
│   │   └── routes/
│   │       ├── videos.py           # Video upload and status endpoints
│   │       ├── search.py           # Search endpoint (POST /search, GET /search/{video_id})
│   │       ├── clips.py            # Clip retrieval endpoint (GET /clips/{video_id})
│   │       ├── detections.py       # Object detection endpoints
│   │       ├── health.py           # Health check and model readiness
│   │       └── video_info.py       # Video metadata endpoints
│   ├── core/
│   │   ├── config.py               # App configuration & environment settings
│   │   ├── detector.py             # YOLOv8 singleton detector with PyTorch 2.6 safe globals
│   │   ├── clip_loader.py          # Cached CLIP model & processor inference loader
│   │   └── logging.py              # Centralized logging configuration
│   ├── db/
│   │   ├── database.py             # SQLAlchemy engine & session maker
│   │   ├── session.py              # Database session dependency
│   │   ├── models.py               # Video and Event ORM models
│   │   ├── detection_models.py     # Frame and ObjectDetection ORM models
│   │   └── crud.py                 # Database query helpers & suggestion extraction
│   └── utils/
│       └── video_metadata.py       # Video duration, resolution, and FPS inspection
├── frontend/                       # React + TypeScript web application
│   ├── src/
│   │   ├── components/             # ResultCard, SearchBar, UploadDropzone, etc.
│   │   ├── api.ts                  # Axios API client and TypeScript models
│   │   ├── App.tsx                 # Main dashboard view
│   │   └── index.css               # Styling and glassmorphism themes
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   ├── check_models.py             # Script to verify YOLOv8 and CLIP model inference
│   ├── inspect_embeddings.py       # Utility to inspect extracted frame vectors
│   └── inspect_search.py           # Utility to run standalone query inspections
├── tests/                          # Pytest and unittest regression test suite
│   ├── test_search_behavior.py     # Event matches, synonyms, and not_found tests
│   ├── test_new_pipeline.py        # Event tracking, upload dedup, and pipeline tests
│   ├── test_search_endpoint.py     # Search API response unit tests
│   ├── test_upload.py              # File upload validation tests
│   └── ...
├── docs/
│   └── demo.png                    # Dashboard demo screenshot (placeholder)
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git ignore rules for media, weights, and caches
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## Setup & Installation

### Prerequisites

- **Python**: Version 3.10 or 3.11 recommended.
- **Node.js**: Version 18+ (for building/running the React frontend).
- **FFmpeg**: Must be installed and accessible in your system `PATH`.
  - Windows: Install via `winget install Gyan.FFmpeg` or `choco install ffmpeg`.
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/videoprompt2.git
cd videoprompt2
```

### 2. Set Up Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# (Or on Linux/macOS: source .venv/bin/activate)
```

### 3. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```powershell
cp .env.example .env
```

Adjust any folder paths or thresholds in `.env` if desired (defaults work out of the box).

### 5. Verify Model Loading (Optional but Recommended)

Run the verification script to verify that both YOLOv8 and CLIP load and execute dummy inferences without errors:

```powershell
python scripts/check_models.py
```

> **Note on Model Weights**: Both `yolov8n.pt` and OpenAI CLIP weights (`openai/clip-vit-base-patch32`) will download automatically on first run and cache locally.

### 6. Start the Backend API

```powershell
uvicorn app.main:app --reload
```

The FastAPI backend will start at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 7. Run the Frontend (Optional)

In a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend dashboard will be available at `http://localhost:5173`.

---

## Example Searches

- **Direct Event Search**: Query `"dog"` retrieves pre-cut clips of detected dogs with high confidence.
- **Synonym Expansion**: Query `"puppy"` automatically maps to the `"dog"` event clips; `"automobile"` maps to `"car"`; `"someone"` maps to `"person"`.
- **Descriptive Fallback Queries**: Query `"a person running across the road"` or `"red car on the highway"` uses CLIP embeddings to identify the most visually similar video intervals.
- **Nonsense Query Rejection**: Query `"unicorn"` or irrelevant keywords return a clean `not_found` response with suggested topics detected in the video:
  ```json
  {
    "status": "not_found",
    "query": "unicorn",
    "message": "Not found in this video. Try another search.",
    "suggestions": ["person", "car", "dog"]
  }
  ```

---

## Running Tests

Run the test suite with `pytest`:

```powershell
pytest -v tests
```

---

## Limitations & Privacy

- **Inference Hardware**: Processing speed depends on GPU availability. On CPU-only machines, frame extraction interval is set to $0.5$s–$1.0$s to maintain practical turnaround times.
- **YOLOv8 Vocabulary**: Automatic event tracking uses the COCO dataset's 80 object categories. Natural language queries outside this vocabulary automatically route to the CLIP multimodal semantic fallback.
- **Privacy Note**: All video processing, frame extraction, vector indexing, and database operations happen **entirely locally on your machine**. No video frames or media content are transmitted to third-party cloud APIs.
