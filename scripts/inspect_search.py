import sys
from pathlib import Path

# Ensure project root is on sys.path when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search_service import encode_text_query, search_video, search_video_safe
from app.faiss_index import search_index
from app.core.config import settings

VIDEO = "love.mp4"
QUERIES = [("person", True), ("elephant", False)]  # (query, expected_present)

for q, expected in QUERIES:
    print("\n--- TEST QUERY:\n", q)
    emb = encode_text_query(q)
    print("Query embedding length:", emb.shape)
    # raw FAISS search
    raw = search_index(VIDEO, emb, top_k=5)
    scores = [r['score'] for r in raw]
    print("Raw FAISS scores:", scores)
    print("Settings threshold:", settings.similarity_threshold)
    # run service-level search
    filtered = search_video(VIDEO, q, top_k=5, similarity_threshold=float(settings.similarity_threshold))
    print("Filtered results count:", len(filtered))
    if filtered:
        for r in filtered:
            print(r)
    else:
        print("No filtered matches")

    safe = search_video_safe(VIDEO, q, top_k=5, similarity_threshold=float(settings.similarity_threshold))
    print("Safe response:", safe)
