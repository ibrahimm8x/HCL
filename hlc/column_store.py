"""ColumnStore: manages the lifecycle of all memory columns."""
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple

from hlc.config import Config
from hlc.column import MemoryColumn
from hlc.index import SparseActivation
from hlc.hebbian import HebbianGraph


class ColumnStore:
    """
    The infinite library. Manages all columns, their index, and their links.

    Columns are loaded on-demand from disk (lazy loading) and cached
    in memory for fast repeated access.
    """

    def __init__(self, config: Config, index: SparseActivation,
                 graph: HebbianGraph):
        self.config = config
        self.index = index
        self.graph = graph
        self.columns: Dict[str, MemoryColumn] = {}  # loaded column cache

    def create_column(
        self,
        pattern: torch.Tensor,
        source_text: str,
        column_type: str = "knowledge",
        signature: Optional[np.ndarray] = None,
    ) -> MemoryColumn:
        """
        Birth a new column.

        Creates an attractor network, trains it on the pattern,
        adds it to the index, and persists it to disk.
        """
        col = MemoryColumn(
            self.config,
            column_type=column_type,
            source_text=source_text,
        )

        # Train the attractor to store this pattern
        loss = col.train_on_pattern(pattern)
        col.metadata.confidence = max(0.0, 1.0 - loss * 10)  # scale loss to confidence

        # Set signature and add to index
        if signature is not None:
            col.signature = torch.from_numpy(signature)
            self.index.add_column(
                col.id,
                signature,
                {"type": column_type, "text": source_text[:200]},
            )

        # Cache and persist
        self.columns[col.id] = col
        col.save(self.config.columns_dir / f"{col.id}.pt")

        return col

    def get_column(self, column_id: str) -> Optional[MemoryColumn]:
        """Load a column (from cache or disk)."""
        if column_id in self.columns:
            return self.columns[column_id]

        path = self.config.columns_dir / f"{column_id}.pt"
        if path.exists():
            col = MemoryColumn.load(path, self.config)
            self.columns[column_id] = col
            return col

        return None

    def find_relevant(self, input_vector: np.ndarray) -> List[Tuple[str, float]]:
        """
        Sparse activation: find columns relevant to input.
        Returns (column_id, similarity) pairs above threshold.
        """
        return self.index.query(input_vector)

    def activate_column(
        self,
        column_id: str,
        input_pattern: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Activate a column and return its converged pattern."""
        col = self.get_column(column_id)
        if col is None:
            return None
        return col.activate(input_pattern)

    def column_count(self) -> int:
        """Total columns in the index."""
        return self.index.count()
