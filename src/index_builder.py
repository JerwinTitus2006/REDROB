"""
index_builder.py — Build FAISS index for ANN candidate retrieval.

This module is provided for the full pipeline architecture but is optional
when using the deterministic scoring approach (which scores all candidates
directly). For 100k candidates, brute-force scoring is fast enough (<10s).

For larger datasets (1M+), enable FAISS ANN retrieval to shortlist candidates
before scoring.
"""

import json
import os
import numpy as np


def build_faiss_index(embeddings_path: str, output_dir: str):
    """
    Build a FAISS IVF index from pre-computed embeddings.
    
    Args:
        embeddings_path: path to embeddings.npy (N x D)
        output_dir: directory to write faiss_index.bin and id_map.json
    """
    try:
        import faiss
    except ImportError:
        print("[index_builder] faiss-cpu not installed, skipping index build")
        return

    embeddings = np.load(embeddings_path).astype("float32")
    n, d = embeddings.shape
    print(f"[index_builder] Building FAISS index: {n} vectors, {d} dimensions")

    # Use IVF for large datasets, flat for small
    if n > 10000:
        nlist = min(int(np.sqrt(n)), 256)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(nlist // 4, 32)
    else:
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"[index_builder] Index saved to {index_path}")


def search_index(index_path: str, query_embedding: np.ndarray, top_k: int = 500):
    """Search the FAISS index for top-K nearest candidates."""
    try:
        import faiss
    except ImportError:
        raise RuntimeError("faiss-cpu required for index search")

    index = faiss.read_index(index_path)
    query = query_embedding.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query)
    distances, indices = index.search(query, top_k)
    return distances[0], indices[0]
