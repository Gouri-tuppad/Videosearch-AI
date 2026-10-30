import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from app.core.config import settings
from app.core.clip_loader import load_clip_model

VIDEO = 'love.mp4'
emb_path = Path(settings.embeddings_folder) / VIDEO / 'frame_embeddings.npy'
print('Embeddings file:', emb_path)
arr = np.load(emb_path)
print('shape', arr.shape, 'dtype', arr.dtype)

# compute norms
norms = np.linalg.norm(arr, axis=1)
print('norms sample (first 10):', norms[:10])
print('min norm', norms.min(), 'max norm', norms.max(), 'mean norm', norms.mean())

# load CLIP model and print name
model, processor = load_clip_model()
model_name = getattr(getattr(model, 'config', None), 'name_or_path', None)
print('CLIP model name:', model_name)

# also get text embedding for a sample query and its norm
import numpy as np
from app.search_service import encode_text_query
q_emb = encode_text_query('person')
print('text emb shape', q_emb.shape, 'norm', np.linalg.norm(q_emb))

# print first frame vector and dot with query
first = arr[0]
print('first vector dot query (cosine since normalized):', float(np.dot(first, q_emb)))
