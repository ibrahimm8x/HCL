"""Routing Loop: the reasoning engine. Fixed dynamics, no learnable weights."""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from hlc.config import Config
from hlc.working_memory import WorkingMemory
from hlc.value_system import ValueSystem, ValueState
from hlc.hebbian import HebbianGraph
from hlc.column_store import ColumnStore


@dataclass
class RoutingResult:
    """Result of the routing loop."""
    converged: bool
    iterations: int
    final_state: torch.Tensor
    active_column_ids: List[str]
    active_source_texts: List[str]
    prediction_error: float
    value_state: ValueState
    mode: str  # "fast", "light", "slow"


class RoutingLoop:
    """
    The 6-step routing loop. This IS reasoning.

    Fixed dynamical system — no learnable weights. The intelligence
    comes from how activation routes between memory columns.

    Step 1: Column interaction via Hebbian links (spreading activation)
    Step 2: Competition (winner-take-all among hypotheses)
    Step 3: Prediction (current state generates expected pattern)
    Step 4: Error signal (prediction vs reality)
    Step 5: Value adjustment (emit motivational signals)
    Step 6: Convergence check (stable? exit. unstable? loop.)
    """

    def __init__(
        self,
        config: Config,
        column_store: ColumnStore,
        working_memory: WorkingMemory,
        value_system: ValueSystem,
        hebbian_graph: HebbianGraph,
    ):
        self.config = config
        self.store = column_store
        self.wm = working_memory
        self.value = value_system
        self.graph = hebbian_graph

    def determine_mode(self, match_scores: List[Tuple[str, float]]) -> str:
        """
        Decide processing depth based on match quality.

        fast:  direct retrieval, no loop (System 1)
        light: 1-2 iterations, fill small gaps
        slow:  full reasoning, up to max iterations (System 2)
        """
        if not match_scores:
            return "slow"
        best_score = match_scores[0][1]
        if best_score > 0.85:
            return "fast"
        elif best_score > 0.6:
            return "light"
        return "slow"

    def run(
        self,
        input_pattern: torch.Tensor,
        initial_matches: List[Tuple[str, float]],
        input_text: str = "",
    ) -> RoutingResult:
        """
        Execute the routing loop.

        This is the complete reasoning process: activation flows
        between columns, hypotheses compete, predictions are checked,
        and the system iterates until convergence.
        """
        device = self.config.get_device()
        mode = self.determine_mode(initial_matches)
        max_iters = {
            "fast": 0,
            "light": 2,
            "slow": self.config.max_routing_iterations,
        }[mode]

        # Load initial matches into working memory
        self.wm.clear()
        for col_id, score in initial_matches[:self.config.working_memory_capacity]:
            pattern = self.store.activate_column(col_id, input_pattern)
            if pattern is not None:
                self.wm.load_column(col_id, pattern, priority=score)

        # Fast path: direct retrieval, skip the loop entirely
        if mode == "fast":
            state = self.wm.get_combined_state()
            if state is None:
                state = input_pattern.to(device)
            return RoutingResult(
                converged=True,
                iterations=0,
                final_state=state,
                active_column_ids=self.wm.get_active_ids(),
                active_source_texts=self._get_source_texts(),
                prediction_error=0.0,
                value_state=self.value.state,
                mode=mode,
            )

        # Slow / Light path: run the routing loop
        input_on_device = input_pattern.to(device)
        previous_state = None
        prediction_error = 1.0
        novelty = 1.0 - (initial_matches[0][1] if initial_matches else 0.0)

        for iteration in range(max_iters):
            # --- STEP 1: Spreading activation via Hebbian links ---
            active_ids = set(self.wm.get_active_ids())
            spreading = self.graph.get_spreading_activation(active_ids, top_k=3)
            for neighbor_id, link_weight in spreading:
                neighbor_pattern = self.store.activate_column(
                    neighbor_id, input_pattern,
                )
                if neighbor_pattern is not None:
                    self.wm.load_column(
                        neighbor_id, neighbor_pattern,
                        priority=link_weight * 0.5,
                    )

            # --- STEP 2: Competition ---
            # Columns that align with input get amplified; others get suppressed
            patterns = self.wm.get_active_patterns()
            for col_id, pattern in patterns.items():
                sim = F.cosine_similarity(
                    pattern.unsqueeze(0), input_on_device.unsqueeze(0),
                ).item()
                old_priority = self.wm.priorities.get(col_id, 0.5)
                adjustment = self.config.competition_strength * (sim - 0.5)
                new_priority = old_priority * (1.0 + adjustment)
                self.wm.priorities[col_id] = max(0.01, min(1.0, new_priority))

            # Evict over-capacity (weakest get suppressed)
            while self.wm.size() > self.config.working_memory_capacity:
                self.wm._evict_lowest()

            # --- STEP 3: Prediction ---
            current_state = self.wm.get_combined_state()
            if current_state is None:
                current_state = input_on_device

            # --- STEP 4: Error signal ---
            prediction_error = 1.0 - F.cosine_similarity(
                current_state.unsqueeze(0), input_on_device.unsqueeze(0),
            ).item()

            # --- STEP 5: Value evaluation ---
            match_conf = (
                max(s for _, s in initial_matches) if initial_matches else 0.0
            )
            self.value.evaluate(prediction_error, novelty, match_conf)

            # --- STEP 6: Convergence check ---
            if previous_state is not None:
                state_delta = (current_state - previous_state).norm().item()
                if (
                    prediction_error < self.config.convergence_threshold
                    or state_delta < 0.01
                ):
                    return RoutingResult(
                        converged=True,
                        iterations=iteration + 1,
                        final_state=current_state,
                        active_column_ids=self.wm.get_active_ids(),
                        active_source_texts=self._get_source_texts(),
                        prediction_error=prediction_error,
                        value_state=self.value.state,
                        mode=mode,
                    )

            previous_state = current_state.clone()

        # Did not converge within max iterations
        final_state = self.wm.get_combined_state() or input_pattern.to(device)
        return RoutingResult(
            converged=False,
            iterations=max_iters,
            final_state=final_state,
            active_column_ids=self.wm.get_active_ids(),
            active_source_texts=self._get_source_texts(),
            prediction_error=prediction_error,
            value_state=self.value.state,
            mode=mode,
        )

    def _get_source_texts(self) -> List[str]:
        """Get source texts from all active columns in working memory."""
        texts = []
        for col_id in self.wm.get_active_ids():
            col = self.store.get_column(col_id)
            if col and col.metadata.source_text:
                texts.append(col.metadata.source_text)
        return texts
