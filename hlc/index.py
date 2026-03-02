"""Sparse Activation: FAISS-based ANN index for finding relevant columns."""
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class SparseActivation:
    """
    The brain's sparse activation implemented as approximate nearest neighbor search.

    Only columns whose signatures are similar to the input get activated.
    Everything else stays silent. The cost doesn't scale linearly with
    column count — a model with a million columns and a billion columns
    activate roughly the same number for any given input.

    Uses FAISS for GPU-accelerated similarity search when available,
    falls back to a numpy-based implementation otherwise.
    """

    def __init__(self, config):
        self.config = config
        self.dim = config.signature_dim
        self._ids: List[str] = []               # column_id at each index position
        self._id_to_idx: Dict[str, int] = {}    # column_id -> index position
        self._vectors: List[np.ndarray] = []     # signature vectors
        self._faiss_index = None
        self._use_faiss = False

        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.dim)  # inner product = cosine on normalized vecs
            self._use_faiss = True
        except ImportError:
            pass  # fallback to numpy

        # Try to load existing index
        self._load()

    def add_column(self, column_id: str, signature: np.ndarray,
                   metadata: Optional[dict] = None):
        """Add a column's signature to the index."""
        sig = self._normalize(signature.astype(np.float32))

        if column_id in self._id_to_idx:
            # Update existing
            idx = self._id_to_idx[column_id]
            self._vectors[idx] = sig
            self._rebuild_faiss()
        else:
            # Add new
            idx = len(self._ids)
            self._ids.append(column_id)
            self._id_to_idx[column_id] = idx
            self._vectors.append(sig)
            if self._use_faiss:
                self._faiss_index.add(sig.reshape(1, -1))

    def remove_column(self, column_id: str):
        """Remove a column from the index."""
        if column_id not in self._id_to_idx:
            return
        idx = self._id_to_idx.pop(column_id)
        self._ids.pop(idx)
        self._vectors.pop(idx)
        # Rebuild id-to-idx mapping
        self._id_to_idx = {cid: i for i, cid in enumerate(self._ids)}
        self._rebuild_faiss()

    def query(self, input_vector: np.ndarray,
              top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Find the top_k most similar columns to the input.

        Returns list of (column_id, cosine_similarity) pairs,
        only including columns above the similarity threshold.
        Sorted by similarity descending.
        """
        if not self._ids:
            return []

        k = min(top_k or self.config.top_k_columns, len(self._ids))
        query_vec = self._normalize(input_vector.astype(np.float32))

        if self._use_faiss and self._faiss_index.ntotal > 0:
            scores, indices = self._faiss_index.search(query_vec.reshape(1, -1), k)
            pairs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._ids):
                    continue
                if score >= self.config.similarity_threshold:
                    pairs.append((self._ids[idx], float(score)))
        else:
            # Numpy fallback
            matrix = np.stack(self._vectors)  # (N, dim)
            scores = matrix @ query_vec       # (N,) cosine similarities
            top_indices = np.argsort(scores)[::-1][:k]
            pairs = []
            for idx in top_indices:
                if scores[idx] >= self.config.similarity_threshold:
                    pairs.append((self._ids[idx], float(scores[idx])))

        return pairs

    def count(self) -> int:
        return len(self._ids)

    def save(self):
        """Persist index to disk."""
        self.config.faiss_dir.mkdir(parents=True, exist_ok=True)
        if self._ids:
            np.save(
                self.config.faiss_dir / "vectors.npy",
                np.stack(self._vectors),
            )
            with open(self.config.faiss_dir / "ids.txt", "w") as f:
                f.write("\n".join(self._ids))

    def _load(self):
        """Load index from disk if it exists."""
        vec_path = self.config.faiss_dir / "vectors.npy"
        ids_path = self.config.faiss_dir / "ids.txt"
        if vec_path.exists() and ids_path.exists():
            vectors = np.load(vec_path)
            with open(ids_path) as f:
                ids = [line.strip() for line in f if line.strip()]
            for cid, vec in zip(ids, vectors):
                self.add_column(cid, vec)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """L2 normalize so inner product = cosine similarity."""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    def _rebuild_faiss(self):
        """Rebuild the FAISS index from scratch."""
        if not self._use_faiss:
            return
        import faiss
        self._faiss_index = faiss.IndexFlatIP(self.dim)
        if self._vectors:
            matrix = np.stack(self._vectors).astype(np.float32)
            self._faiss_index.add(matrix)
