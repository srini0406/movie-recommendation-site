"""
Regenerate similarity_matrix.npz from saved model artifacts.
Run during build if the matrix file is missing (too large for git).
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import save_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = Path(os.environ.get('MODEL_DIR', 'models'))
MATRIX_PATH = MODEL_DIR / 'similarity_matrix.npz'

if MATRIX_PATH.exists():
    print(f"similarity_matrix.npz already exists at {MATRIX_PATH}, skipping regeneration.")
    sys.exit(0)

print("Regenerating similarity matrix...")

# Load artifacts
print("  Loading metadata...")
df = pd.read_parquet(MODEL_DIR / 'movie_metadata.parquet')

print("  Loading TF-IDF vectorizer...")
with open(MODEL_DIR / 'tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

print("  Loading SVD model...")
with open(MODEL_DIR / 'svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)


def build_text(row):
    genres = ' '.join(list(row['genres'])) if hasattr(row['genres'], '__iter__') and not isinstance(row['genres'], str) else ''
    overview = str(row['overview']) if pd.notna(row['overview']) else ''
    company = str(row['primary_company']) if pd.notna(row['primary_company']) else ''
    return f"{genres} {overview} {company}"


print("  Building feature text...")
texts = df.apply(build_text, axis=1).tolist()

print("  Transforming with TF-IDF...")
tfidf_matrix = tfidf.transform(texts)

print("  Applying SVD dimensionality reduction...")
try:
    reduced = svd.transform(tfidf_matrix)
    print(f"  Reduced shape: {reduced.shape}")
except Exception as e:
    print(f"  SVD failed ({e}), using TF-IDF matrix directly...")
    reduced = tfidf_matrix.toarray()

print("  Computing cosine similarity (this may take a few minutes)...")
# Process in chunks to avoid OOM on free tier
n = len(reduced)
chunk_size = 500
sim_matrix = np.zeros((n, n), dtype=np.float32)

for i in range(0, n, chunk_size):
    end = min(i + chunk_size, n)
    chunk_sim = cosine_similarity(reduced[i:end], reduced).astype(np.float32)
    sim_matrix[i:end] = chunk_sim
    if i % 2000 == 0:
        print(f"    Progress: {i}/{n}")

print("  Saving similarity matrix...")
save_npz(str(MATRIX_PATH), csr_matrix(sim_matrix))
print(f"Done. Saved to {MATRIX_PATH}  shape={sim_matrix.shape}")
