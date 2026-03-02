"""Working Memory: the spotlight — which columns are currently active."""
import torch
from typing import Dict, Optional, List

from hlc.config import Config


class WorkingMemory:
    """
    Limited to ~5-9 active column patterns at once.
    The small desk in an infinite library.

    Working memory is NOT a software buffer. Conceptually it's
    which columns are currently firing. In practice, it's the
    active tensors loaded in GPU/RAM participating in the routing loop.

    The capacity constraint is a feature — it forces focus and
    prevents information overload, just like the brain.
    """

    def __init__(self, config: Config):
        self.config = config
        self.capacity = config.working_memory_capacity
        # Currently active: {column_id: activation_pattern (tensor)}
        self.active: Dict[str, torch.Tensor] = {}
        # Priority scores for eviction
        self.priorities: Dict[str, float] = {}

    def load_column(self, column_id: str, pattern: torch.Tensor,
                    priority: float = 1.0):
        """Load a column's activation into working memory."""
        if column_id in self.active:
            # Already active — just update priority
            self.priorities[column_id] = max(self.priorities[column_id], priority)
            return
        if len(self.active) >= self.capacity:
            self._evict_lowest()
        self.active[column_id] = pattern
        self.priorities[column_id] = priority

    def _evict_lowest(self):
        """Remove the lowest-priority column from working memory."""
        if not self.priorities:
            return
        lowest_id = min(self.priorities, key=self.priorities.get)
        del self.active[lowest_id]
        del self.priorities[lowest_id]

    def get_active_ids(self) -> List[str]:
        """Get IDs of all active columns."""
        return list(self.active.keys())

    def get_active_patterns(self) -> Dict[str, torch.Tensor]:
        """Get all active column patterns."""
        return dict(self.active)

    def get_combined_state(self) -> Optional[torch.Tensor]:
        """
        Combine all active patterns into a single state vector.
        Priority-weighted average — stronger columns contribute more.
        """
        if not self.active:
            return None

        patterns = []
        weights = []
        for cid, pattern in self.active.items():
            patterns.append(pattern)
            weights.append(self.priorities.get(cid, 1.0))

        stacked = torch.stack(patterns)  # (N, dim)
        w = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
        w = w / w.sum()
        combined = (stacked * w.unsqueeze(1)).sum(dim=0)  # (dim,)
        return combined

    def clear(self):
        """Clear all active columns."""
        self.active.clear()
        self.priorities.clear()

    def is_full(self) -> bool:
        return len(self.active) >= self.capacity

    def size(self) -> int:
        return len(self.active)
