"""Central configuration for Humanity's Last Creation v1."""
import torch
from dataclasses import dataclass, field
from pathlib import Path


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # === Paths ===
    # Default to Google Drive on Colab; override for local development
    project_root: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/HLC"))
    data_dir: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/HLC/data"))
    columns_dir: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/HLC/data/columns"))
    faiss_dir: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/HLC/data/faiss_index"))
    hebbian_path: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/HLC/data/hebbian_graph.json"))
    event_log_path: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/HLC/data/event_log.jsonl"))

    # === Device ===
    device: str = field(default_factory=_default_device)

    # === Attractor Network (Column) ===
    attractor_input_dim: int = 384       # matches all-MiniLM-L6-v2 embedding dim
    attractor_hidden_dim: int = 128      # internal hidden state
    attractor_num_steps: int = 10        # convergence iterations per activation
    attractor_lr: float = 0.01           # learning rate for training new columns
    attractor_train_epochs: int = 100    # epochs to train a new column's attractor
    attractor_noise_std: float = 0.2     # noise injection during training

    # === Embedding Model ===
    embedding_model_name: str = "all-MiniLM-L6-v2"
    signature_dim: int = 384

    # === Language Model ===
    lm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    lm_max_new_tokens: int = 200

    # === Sparse Activation ===
    similarity_threshold: float = 0.35   # minimum cosine similarity to activate
    top_k_columns: int = 7              # max columns returned by ANN query

    # === Working Memory ===
    working_memory_capacity: int = 7     # ~5-9 active columns (Miller's number)

    # === Routing Loop ===
    max_routing_iterations: int = 8      # max loops before forced convergence
    convergence_threshold: float = 0.05  # error below this = converged
    competition_strength: float = 0.5    # how aggressively hypotheses suppress each other
    exploration_rate: float = 0.3        # tendency to activate less-similar columns

    # === Hebbian Links ===
    hebbian_learning_rate: float = 0.1   # link strengthening per co-activation
    hebbian_max_weight: float = 1.0
    hebbian_min_weight: float = 0.01     # below this, link is pruned

    # === Value System Signal Weights ===
    pain_weight: float = 1.0
    joy_weight: float = 0.8
    fear_weight: float = 0.7
    curiosity_weight: float = 0.6
    surprise_weight: float = 0.5

    # === Column Types ===
    COLUMN_TYPE_KNOWLEDGE: str = "knowledge"
    COLUMN_TYPE_STRATEGY: str = "strategy"
    COLUMN_TYPE_CLUSTER: str = "cluster"

    def ensure_dirs(self):
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.columns_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

    def get_device(self) -> torch.device:
        return torch.device(self.device)

    @classmethod
    def local(cls) -> "Config":
        """Config for local development (not Colab)."""
        return cls(
            project_root=Path("/Users/biboahmed/AGI"),
            data_dir=Path("/Users/biboahmed/AGI/data"),
            columns_dir=Path("/Users/biboahmed/AGI/data/columns"),
            faiss_dir=Path("/Users/biboahmed/AGI/data/faiss_index"),
            hebbian_path=Path("/Users/biboahmed/AGI/data/hebbian_graph.json"),
            event_log_path=Path("/Users/biboahmed/AGI/data/event_log.jsonl"),
        )
