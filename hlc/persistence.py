"""Persistence: save/load the mind's state. Columns persisted, working memory ephemeral."""
import json
import time
from pathlib import Path

from hlc.config import Config


class EventLog:
    """
    Append-only event log for recovery and debugging.

    Every significant event (column creation, link update, query)
    is logged with a timestamp. This enables recovery from incomplete
    operations if the system crashes mid-thought.
    """

    def __init__(self, path: Path):
        self.path = path

    def log(self, event_type: str, data: dict):
        """Log an event."""
        entry = {"timestamp": time.time(), "type": event_type, **data}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_recent(self, n: int = 20) -> list:
        """Read the last N events."""
        if not self.path.exists():
            return []
        with open(self.path) as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-n:]]


class PersistenceManager:
    """
    Manages saving and loading the full system state.

    Memory columns are persisted individually as .pt files (saved during creation).
    The Hebbian graph is saved as JSON.
    The FAISS index is saved as numpy arrays.
    Working memory and routing state are ephemeral — lost on crash (like the brain).
    """

    def __init__(self, config: Config):
        self.config = config
        self.event_log = EventLog(config.event_log_path)

    def save_all(self, column_store, hebbian_graph):
        """Full persist: graph + index. Columns are saved individually during creation."""
        hebbian_graph.save(self.config.hebbian_path)
        column_store.index.save()
        self.event_log.log("full_save", {
            "column_count": column_store.column_count(),
            "link_count": hebbian_graph.total_links(),
        })

    def load_all(self, column_store, hebbian_graph):
        """Load persisted state. Columns are loaded on-demand by ColumnStore."""
        hebbian_graph.load(self.config.hebbian_path)
        # Index loads itself in __init__
        # Columns are lazy-loaded by ColumnStore.get_column()
