"""Hebbian Links: weighted connections between columns that fire together."""
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from pathlib import Path

from hlc.config import Config


class HebbianGraph:
    """
    "Neurons that fire together wire together."

    Manages weighted edges between column IDs. When two columns
    co-activate (appear in working memory together), their link
    strengthens. Links enable spreading activation: activating
    one column sends energy to linked columns, making them
    easier to trigger.
    """

    def __init__(self, config: Config):
        self.config = config
        # {column_id: {neighbor_id: weight}}
        self.links: Dict[str, Dict[str, float]] = defaultdict(dict)

    def strengthen(self, col_a: str, col_b: str):
        """Strengthen the bidirectional link between two co-activated columns."""
        if col_a == col_b:
            return
        for src, tgt in [(col_a, col_b), (col_b, col_a)]:
            current = self.links[src].get(tgt, 0.0)
            new_weight = min(
                current + self.config.hebbian_learning_rate,
                self.config.hebbian_max_weight,
            )
            self.links[src][tgt] = new_weight

    def get_neighbors(self, column_id: str,
                      min_weight: float = 0.0) -> List[Tuple[str, float]]:
        """Get linked columns sorted by weight (strongest first)."""
        neighbors = [
            (nid, w) for nid, w in self.links.get(column_id, {}).items()
            if w > min_weight
        ]
        return sorted(neighbors, key=lambda x: x[1], reverse=True)

    def get_spreading_activation(
        self,
        active_ids: Set[str],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Spreading activation: given currently active columns,
        find their strongest linked neighbors (not already active).

        Activation energy from multiple active columns accumulates
        toward shared neighbors — columns linked to many active
        columns get the highest score.
        """
        candidate_scores: Dict[str, float] = defaultdict(float)
        for col_id in active_ids:
            for neighbor_id, weight in self.links.get(col_id, {}).items():
                if neighbor_id not in active_ids:
                    candidate_scores[neighbor_id] += weight

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_weight(self, col_a: str, col_b: str) -> float:
        """Get the link weight between two columns (0.0 if no link)."""
        return self.links.get(col_a, {}).get(col_b, 0.0)

    def total_links(self) -> int:
        """Total number of directed links."""
        return sum(len(v) for v in self.links.values())

    def save(self, path: Path):
        """Persist graph to JSON."""
        with open(path, "w") as f:
            json.dump(dict(self.links), f)

    def load(self, path: Path):
        """Load graph from JSON."""
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.links = defaultdict(dict, {k: dict(v) for k, v in data.items()})
