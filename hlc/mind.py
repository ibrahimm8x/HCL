"""Mind: the top-level orchestrator. One instance. One mind."""
import torch
import numpy as np
from typing import List, Optional

from hlc.config import Config
from hlc.column_store import ColumnStore
from hlc.index import SparseActivation
from hlc.hebbian import HebbianGraph
from hlc.routing import RoutingLoop, RoutingResult
from hlc.working_memory import WorkingMemory
from hlc.value_system import ValueSystem
from hlc.language import LanguageInterface
from hlc.persistence import PersistenceManager


class Mind:
    """
    The singular entity. One instance. One mind.

    Orchestrates the complete processing flow:
    input -> embedding -> sparse activation -> column activation ->
    working memory -> routing loop -> convergence -> response ->
    post-processing (Hebbian strengthening, new column creation) -> persist.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.ensure_dirs()

        # Initialize subsystems
        self.language = LanguageInterface(self.config)
        self.index = SparseActivation(self.config)
        self.graph = HebbianGraph(self.config)
        self.store = ColumnStore(self.config, self.index, self.graph)
        self.wm = WorkingMemory(self.config)
        self.value = ValueSystem(self.config)
        self.routing = RoutingLoop(
            self.config, self.store, self.wm, self.value, self.graph,
        )
        self.persistence = PersistenceManager(self.config)

        # Load persisted state
        self.persistence.load_all(self.store, self.graph)

        # Track last result for diagnostics
        self.last_result: Optional[RoutingResult] = None

    def process(self, text: str) -> str:
        """
        The complete processing flow. Text in, text out.
        Everything in between is the mind working.
        """
        # 1. Encode: text -> embedding vector
        embedding = self.language.encode(text)
        input_tensor = torch.from_numpy(embedding).float()

        # 2. Sparse activation: find relevant columns
        matches = self.store.find_relevant(embedding)

        # 3. Routing loop: reason
        result = self.routing.run(input_tensor, matches, input_text=text)
        self.last_result = result

        # 4. Post-processing: Hebbian strengthening
        active_ids = result.active_column_ids
        for i, id_a in enumerate(active_ids):
            for id_b in active_ids[i + 1:]:
                self.graph.strengthen(id_a, id_b)

        # 5. Should we create a new column?
        novelty_threshold = self.config.similarity_threshold + 0.1
        if not matches or matches[0][1] < novelty_threshold:
            self.store.create_column(
                pattern=input_tensor,
                source_text=text,
                column_type="knowledge",
                signature=embedding,
            )
            self.persistence.event_log.log(
                "column_created", {"text": text[:100]},
            )

        # 6. Generate response
        response = self.language.generate_response(
            result.active_source_texts,
            text,
            value_state=result.value_state,
        )

        # 7. Persist
        self.persistence.save_all(self.store, self.graph)

        return response

    def process_without_llm(self, text: str) -> dict:
        """
        Process input without LLM generation.
        Useful for testing the architecture without loading the large LLM.
        Returns diagnostic info instead of generated text.
        """
        embedding = self.language.encode(text)
        input_tensor = torch.from_numpy(embedding).float()
        matches = self.store.find_relevant(embedding)
        result = self.routing.run(input_tensor, matches, input_text=text)
        self.last_result = result

        # Post-processing
        active_ids = result.active_column_ids
        for i, id_a in enumerate(active_ids):
            for id_b in active_ids[i + 1:]:
                self.graph.strengthen(id_a, id_b)

        novelty_threshold = self.config.similarity_threshold + 0.1
        new_column_created = False
        if not matches or matches[0][1] < novelty_threshold:
            self.store.create_column(
                pattern=input_tensor,
                source_text=text,
                column_type="knowledge",
                signature=embedding,
            )
            new_column_created = True

        self.persistence.save_all(self.store, self.graph)

        return {
            "mode": result.mode,
            "converged": result.converged,
            "iterations": result.iterations,
            "prediction_error": result.prediction_error,
            "active_columns": result.active_column_ids,
            "active_knowledge": result.active_source_texts,
            "value_state": str(result.value_state),
            "matches": [(cid, f"{score:.3f}") for cid, score in matches],
            "new_column_created": new_column_created,
        }

    def seed_knowledge(self, facts: List[str], verbose: bool = True):
        """
        Seed the mind with initial knowledge.
        Each fact becomes a memory column.
        """
        embeddings = self.language.encode_batch(facts)
        for i, (text, emb) in enumerate(zip(facts, embeddings)):
            pattern = torch.from_numpy(emb).float()
            col_type = "strategy" if text.startswith("STRATEGY:") else "knowledge"
            self.store.create_column(
                pattern=pattern,
                source_text=text,
                column_type=col_type,
                signature=emb,
            )
            if verbose and (i + 1) % 10 == 0:
                print(f"  Seeded {i + 1}/{len(facts)} columns...")

        self.persistence.save_all(self.store, self.graph)
        if verbose:
            print(f"  Done. Total columns: {self.store.column_count()}")

    def stats(self) -> dict:
        """Return system statistics."""
        return {
            "total_columns": self.store.column_count(),
            "loaded_in_memory": len(self.store.columns),
            "hebbian_links": self.graph.total_links(),
            "working_memory_active": self.wm.size(),
            "value_state": str(self.value.state),
        }
