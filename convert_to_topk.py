"""
convert_to_topk.py
------------------
Converts a full NxN cosine similarity matrix (similarity_matrix.npz) into a
compact top-K format that only stores the 20 most similar movies per movie.

Output: top_k_similarities.npz
  - indices : shape (N, 20) — movie indices, most similar first
  - scores  : shape (N, 20) — corresponding similarity scores

Memory-efficient: processes one row at a time, never duplicates the full matrix.
"""

import os
import sys
import numpy as np
from scipy.sparse import load_npz, issparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")
INPUT_FILE  = MODELS_DIR / "similarity_matrix.npz"
OUTPUT_FILE = MODELS_DIR / "top_k_similarities.npz"
K = 20


# ---------------------------------------------------------------------------
# 1. Load the matrix — auto-detect key
# ---------------------------------------------------------------------------

def load_similarity_matrix(path: Path):
    """
    Load the similarity matrix from an .npz file.

    Handles both:
    - scipy sparse matrix (keys: indices, indptr, data, format, shape)
    - dense numpy array  (key: arr_0 or similar)

    Returns a scipy sparse matrix (CSR) or dense ndarray.
    We keep it sparse here and slice rows lazily to avoid loading 3+ GB into RAM.
    """
    print(f"Loading {path}  ({path.stat().st_size / 1e9:.2f} GB on disk)...")

    # Peek at the keys to decide format
    probe = np.load(path, allow_pickle=False)
    keys = list(probe.files)
    probe.close()
    print(f"  Keys found in archive: {keys}")

    is_scipy_sparse = {'indices', 'indptr', 'data', 'format', 'shape'}.issubset(set(keys))

    if is_scipy_sparse:
        print("  Detected scipy sparse matrix — loading with load_npz")
        matrix = load_npz(path)
        # Convert to CSR for efficient row slicing
        matrix = matrix.tocsr()
        print(f"  Sparse matrix shape: {matrix.shape}  nnz: {matrix.nnz:,}")
    else:
        key = "arr_0" if "arr_0" in keys else keys[0]
        print(f"  Detected dense array — using key '{key}'")
        matrix = np.load(path, allow_pickle=False)[key]
        print(f"  Dense matrix shape: {matrix.shape}  dtype: {matrix.dtype}")

    return matrix


# ---------------------------------------------------------------------------
# 2. Build top-K arrays
# ---------------------------------------------------------------------------

def build_topk(matrix, k: int = K):
    """
    For each row, find the indices and scores of the top-k most similar movies,
    excluding the movie itself (self-similarity on the diagonal).

    Handles both scipy sparse CSR matrices and dense ndarrays.
    Processes one row at a time — no extra full-matrix copies.
    """
    N = matrix.shape[0]
    out_indices = np.empty((N, k), dtype=np.int32)
    out_scores  = np.empty((N, k), dtype=np.float32)

    sparse = issparse(matrix)
    print(f"\nBuilding top-{k} neighbors for {N:,} movies  (sparse={sparse})...")
    report_every = max(1, N // 20)

    for i in range(N):
        if sparse:
            # .toarray() on a single row is cheap — shape (1, N) -> (N,)
            row = matrix.getrow(i).toarray().ravel().astype(np.float32)
        else:
            row = matrix[i].astype(np.float32)

        row[i] = -np.inf   # exclude self

        # argpartition: O(N), only fully sorts the k winners
        top_k_unsorted = np.argpartition(row, -k)[-k:]
        sorted_order   = np.argsort(row[top_k_unsorted])[::-1]
        top_k_idx      = top_k_unsorted[sorted_order]

        out_indices[i] = top_k_idx
        out_scores[i]  = row[top_k_idx]

        if (i + 1) % report_every == 0 or i == N - 1:
            pct = (i + 1) / N * 100
            print(f"  {i+1:>6,} / {N:,}  ({pct:.0f}%)")

    return out_indices, out_scores


# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------

def save_topk(indices: np.ndarray, scores: np.ndarray, path: Path):
    print(f"\nSaving to {path} ...")
    np.savez_compressed(path, indices=indices, scores=scores)
    print(f"  Saved.  ({path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 4. Sanity check
# ---------------------------------------------------------------------------

def sanity_check(matrix, indices: np.ndarray, scores: np.ndarray, k: int = 5):
    N = matrix.shape[0]
    rng = np.random.default_rng(seed=42)
    sample_idx = int(rng.integers(0, N))

    print(f"\n{'='*60}")
    print(f"Sanity check — movie index {sample_idx}")
    print(f"{'='*60}")

    print(f"\n  Top-{k} from top_k_similarities.npz:")
    for rank in range(k):
        idx   = indices[sample_idx, rank]
        score = scores[sample_idx, rank]
        print(f"    rank {rank+1}: movie_idx={idx:<6}  score={score:.4f}")

    print(f"\n  Top-{k} from original similarity_matrix (ground truth):")
    if issparse(matrix):
        row = matrix.getrow(sample_idx).toarray().ravel().astype(np.float32)
    else:
        row = matrix[sample_idx].copy().astype(np.float32)
    row[sample_idx] = -np.inf
    top_orig = np.argsort(row)[::-1][:k]
    for rank, idx in enumerate(top_orig):
        score = row[idx]
        print(f"    rank {rank+1}: movie_idx={idx:<6}  score={score:.4f}")

    match = np.all(indices[sample_idx, :k] == top_orig)
    print(f"\n  Match: {'✓ YES — top-{} identical'.format(k) if match else '✗ MISMATCH — check K value'}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run this script from the project root.")
        sys.exit(1)

    input_size_mb = INPUT_FILE.stat().st_size / 1e6
    print(f"Input file size:  {input_size_mb:.1f} MB")

    # Load
    matrix = load_similarity_matrix(INPUT_FILE)

    # Build
    indices, scores = build_topk(matrix, k=K)

    # Free the (potentially large) matrix before saving
    del matrix

    # Save
    save_topk(indices, scores, OUTPUT_FILE)

    output_size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"\nSize comparison:")
    print(f"  Before: {input_size_mb:>10.1f} MB  ({INPUT_FILE.name})")
    print(f"  After:  {output_size_mb:>10.1f} MB  ({OUTPUT_FILE.name})")
    print(f"  Reduction: {100 * (1 - output_size_mb / input_size_mb):.1f}%")

    # Reload for sanity check (lightweight — only top-K)
    data = np.load(OUTPUT_FILE)
    topk_indices = data['indices']
    topk_scores  = data['scores']

    # Reload original for ground-truth comparison (just one row needed)
    matrix_check = load_similarity_matrix(INPUT_FILE)
    sanity_check(matrix_check, topk_indices, topk_scores, k=5)


if __name__ == "__main__":
    main()
