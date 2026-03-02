"""Memory Column: a small attractor network that stores knowledge as a stable activation pattern."""
import torch
import torch.nn as nn
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from hlc.config import Config


@dataclass
class ColumnMetadata:
    column_id: str
    column_type: str          # "knowledge", "strategy", "cluster"
    created_at: float         # timestamp
    access_count: int = 0
    confidence: float = 0.5
    decay_level: float = 0.0
    tags: List[str] = field(default_factory=list)
    source_text: str = ""     # original text that created this column


class AttractorNetwork(nn.Module):
    """
    A gated attractor network that stores one pattern as a stable fixed point.

    The core mechanism: a learned gate determines how much the input is
    "attracted" toward the stored pattern. When iterated:
    - Input near stored pattern → gate opens → output pulled toward stored → converges
    - Input far from stored → gate stays closed → output stays near input → no attraction

    This naturally creates a bounded attractor basin. The gate network
    learns the basin boundary through training with positive (near-pattern)
    and negative (random) examples.

    Energy landscape analogy:
    - The stored pattern sits in a valley
    - The gate's sigmoid output defines the valley's shape
    - Nearby inputs roll in (high gate), distant inputs stay put (low gate)
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        # The stored pattern (set during training)
        self.pattern = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        # Gate network: takes input, outputs attraction strength (0-1)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single attraction step.
        Output = gate * stored_pattern + (1 - gate) * input
        """
        g = self.gate(x)  # (batch, 1)
        pattern = self.pattern.unsqueeze(0).expand_as(x)
        return g * pattern + (1.0 - g) * x

    def attract(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Run attractor dynamics: repeatedly apply gated attraction.

        Each step pulls the input closer to the stored pattern
        (if the gate opens) or leaves it in place (if gate stays shut).
        Iteration amplifies the effect — near inputs converge fully.
        """
        state = x
        for _ in range(num_steps):
            new_state = self.forward(state)
            delta = (new_state - state).norm().item()
            state = new_state
            if delta < 1e-4:
                break
        return state

    def get_gate_value(self, x: torch.Tensor) -> float:
        """Get the gate's attraction strength for an input (0-1)."""
        with torch.no_grad():
            g = self.gate(x)
        return g.mean().item()


class MemoryColumn:
    """
    A single memory column: attractor network + metadata + signature.

    This is the fundamental unit of knowledge in the system.
    Each column stores one piece of knowledge or one reasoning strategy
    as a stable activation pattern in its attractor network.
    """

    def __init__(
        self,
        config: Config,
        column_type: str = "knowledge",
        source_text: str = "",
        column_id: Optional[str] = None,
    ):
        self.config = config
        self.id = column_id or str(uuid.uuid4())
        self.metadata = ColumnMetadata(
            column_id=self.id,
            column_type=column_type,
            created_at=time.time(),
            source_text=source_text,
        )
        self.network = AttractorNetwork(
            input_dim=config.attractor_input_dim,
            hidden_dim=config.attractor_hidden_dim,
        ).to(config.get_device())

        self.signature: Optional[torch.Tensor] = None
        self.stored_pattern: Optional[torch.Tensor] = None

    def train_on_pattern(self, target_pattern: torch.Tensor) -> float:
        """
        Train this column's attractor to store the given pattern.

        The pattern becomes a stable fixed point of the encode-decode loop.
        We also train with noisy input to widen the attractor basin,
        enabling pattern completion from partial/corrupted cues.

        Returns:
            Final reconstruction loss.
        """
        self.stored_pattern = target_pattern.detach().clone().cpu()
        device = self.config.get_device()
        target = target_pattern.to(device)
        if target.dim() == 1:
            target = target.unsqueeze(0)  # (1, 384)

        # Set the stored pattern in the network
        self.network.pattern.data = target.squeeze(0).clone()

        # Only train the gate network (pattern is fixed)
        optimizer = torch.optim.Adam(
            self.network.gate.parameters(),
            lr=self.config.attractor_lr,
        )
        bce = nn.BCELoss()

        self.network.train()
        final_loss = 0.0

        for epoch in range(self.config.attractor_train_epochs):
            optimizer.zero_grad()

            # Positive examples: stored pattern + noisy versions
            # Gate should be HIGH (close to 1.0) for these
            pos_batch = [target]
            for _ in range(3):
                noise = torch.randn_like(target) * self.config.attractor_noise_std
                pos_batch.append(target + noise)
            pos_inputs = torch.cat(pos_batch, dim=0)  # (4, 384)
            pos_gates = self.network.gate(pos_inputs)  # (4, 1)
            pos_targets = torch.ones_like(pos_gates)
            pos_loss = bce(pos_gates, pos_targets)

            # Negative examples: random patterns
            # Gate should be LOW (close to 0.0) for these
            neg_inputs = torch.randn(4, target.shape[1], device=device)
            neg_inputs = neg_inputs / neg_inputs.norm(dim=1, keepdim=True)
            neg_gates = self.network.gate(neg_inputs)  # (4, 1)
            neg_targets = torch.zeros_like(neg_gates)
            neg_loss = bce(neg_gates, neg_targets)

            total_loss = pos_loss + neg_loss
            total_loss.backward()
            optimizer.step()
            final_loss = total_loss.item()

        self.network.eval()
        return final_loss

    def activate(self, input_pattern: torch.Tensor) -> torch.Tensor:
        """
        Activate this column with an input pattern.

        Runs attractor dynamics — the input converges toward the stored
        pattern (pattern completion). Returns the converged activation.
        """
        self.metadata.access_count += 1
        device = self.config.get_device()
        x = input_pattern.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            result = self.network.attract(x, num_steps=self.config.attractor_num_steps)

        return result.squeeze(0)

    def reconstruction_quality(self, input_pattern: torch.Tensor) -> float:
        """How well does this column reconstruct the input? Returns cosine similarity (0-1)."""
        activated = self.activate(input_pattern)
        device = self.config.get_device()
        cos_sim = nn.functional.cosine_similarity(
            activated.unsqueeze(0),
            input_pattern.to(device).unsqueeze(0),
        )
        return cos_sim.item()

    def save(self, path: Path):
        """Save column state to disk."""
        torch.save({
            "network_state": self.network.state_dict(),
            "metadata": {
                "column_id": self.metadata.column_id,
                "column_type": self.metadata.column_type,
                "created_at": self.metadata.created_at,
                "access_count": self.metadata.access_count,
                "confidence": self.metadata.confidence,
                "decay_level": self.metadata.decay_level,
                "tags": self.metadata.tags,
                "source_text": self.metadata.source_text,
            },
            "signature": self.signature,
            "stored_pattern": self.stored_pattern,
        }, path)

    @classmethod
    def load(cls, path: Path, config: Config) -> "MemoryColumn":
        """Load column from disk."""
        data = torch.load(path, map_location=config.get_device(), weights_only=False)
        meta = data["metadata"]
        col = cls(
            config,
            column_type=meta["column_type"],
            source_text=meta["source_text"],
            column_id=meta["column_id"],
        )
        col.network.load_state_dict(data["network_state"])
        col.network.eval()
        col.metadata = ColumnMetadata(**meta)
        col.signature = data["signature"]
        col.stored_pattern = data["stored_pattern"]
        return col
