"""Routing Loop: the reasoning engine. Fixed dynamics, no learnable weights."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Set
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
    reasoning_trace: List[List[str]] = field(default_factory=list)


class RoutingLoop:
    """
    The routing loop. This IS reasoning.

    A fixed dynamical system — no learnable weights. Intelligence
    emerges from how activation routes between memory columns
    through multi-hop discovery and competition.

    Each iteration runs 7 steps:

    1. Multi-hop discovery — re-query the index with the EVOLVING
       understanding (combined state of working memory), not the
       original query. This finds knowledge that was unreachable
       from the original question but is reachable from what we've
       found so far. This is genuine multi-hop inference.

    2. Spreading activation — pull in Hebbian-linked neighbors.
       Associative connections propagate energy to related columns.

    3. Competition — columns must be relevant to BOTH the original
       query (stay on-topic) AND the evolving context (fit the
       emerging answer). Blended similarity prevents both drift
       and tunnel vision.

    4. Prediction — combine active patterns into current understanding.

    5. Error signal — how well does the current state explain the input?

    6. Value evaluation — emit motivational signals (joy, curiosity,
       pain, surprise, fear).

    7. Convergence — stop when no new knowledge is discoverable AND
       the state is stable. This means: "I've exhausted what I can
       find" — not "my state matches the query."
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

        Multi-hop reasoning: each iteration discovers new knowledge based
        on the evolving understanding, not just the original query. The
        system builds up an answer through sequential discovery, naturally
        chaining facts that no single retrieval step could find.
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

        # Track reasoning: which columns were discovered at each hop
        initial_ids = list(self.wm.get_active_ids())
        reasoning_trace: List[List[str]] = [initial_ids] if initial_ids else []

        # Track all columns ever seen in this run — multi-hop never re-visits
        seen_ids: Set[str] = set(self.wm.get_active_ids())

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
                reasoning_trace=reasoning_trace,
            )

        # Slow / Light path: run the reasoning loop
        input_on_device = input_pattern.to(device)
        previous_state = None
        prediction_error = 1.0
        novelty = 1.0 - (initial_matches[0][1] if initial_matches else 0.0)

        for iteration in range(max_iters):
            hop_discoveries: List[str] = []

            # --- STEP 1: Multi-hop discovery ---
            # Re-query the index with our CURRENT UNDERSTANDING.
            # The combined state represents "everything we know so far" —
            # searching with it finds knowledge related to our emerging
            # answer, not just the original question.
            #
            # Example: Query "Do plants need sun to make oxygen?"
            #   Hop 0: finds "photosynthesis" + "plants need sunlight"
            #   Hop 1: combined state ≈ plants+sunlight+photosynthesis
            #          → discovers "oxygen production" column
            #   Hop 2: no new columns → converge with full chain
            current_state = self.wm.get_combined_state()
            if current_state is not None:
                state_numpy = current_state.detach().cpu().numpy()
                hop_matches = self.store.find_relevant(
                    state_numpy, exclude_ids=seen_ids,
                )
                for col_id, score in hop_matches[:3]:  # max 3 new per hop
                    pattern = self.store.activate_column(col_id, input_pattern)
                    if pattern is not None:
                        # Discoveries from context get slightly lower priority
                        # than direct query matches — they're supporting evidence
                        self.wm.load_column(col_id, pattern, priority=score * 0.8)
                        hop_discoveries.append(col_id)
                        seen_ids.add(col_id)

            # --- STEP 2: Spreading activation via Hebbian links ---
            # Associative connections: columns that have co-activated before
            # send energy to their neighbors. This pulls in conceptually
            # linked knowledge even if it's not semantically similar.
            active_ids = set(self.wm.get_active_ids())
            spreading = self.graph.get_spreading_activation(active_ids, top_k=3)
            for neighbor_id, link_weight in spreading:
                if neighbor_id not in active_ids:
                    neighbor_pattern = self.store.activate_column(
                        neighbor_id, input_pattern,
                    )
                    if neighbor_pattern is not None:
                        self.wm.load_column(
                            neighbor_id, neighbor_pattern,
                            priority=link_weight * 0.5,
                        )
                        if neighbor_id not in seen_ids:
                            hop_discoveries.append(neighbor_id)
                            seen_ids.add(neighbor_id)

            # Record what this hop discovered
            if hop_discoveries:
                reasoning_trace.append(hop_discoveries)

            # --- STEP 3: Competition ---
            # Columns must be relevant to BOTH the original query AND
            # the evolving context. This dual criterion:
            # - Prevents drift: columns unrelated to the question get suppressed
            # - Allows discovery: columns related to the CONTEXT (not the
            #   question directly) can survive if they fit the emerging answer
            combined = self.wm.get_combined_state()
            patterns = self.wm.get_active_patterns()
            for col_id, pattern in patterns.items():
                # How relevant is this column to the original question?
                query_sim = F.cosine_similarity(
                    pattern.unsqueeze(0), input_on_device.unsqueeze(0),
                ).item()

                # How relevant is it to our evolving understanding?
                context_sim = 0.5
                if combined is not None:
                    context_sim = F.cosine_similarity(
                        pattern.unsqueeze(0), combined.unsqueeze(0),
                    ).item()

                # Blend: 60% query relevance + 40% context relevance
                # This keeps the system on-topic while allowing multi-hop
                # discoveries to survive competition
                blended_sim = 0.6 * query_sim + 0.4 * context_sim

                old_priority = self.wm.priorities.get(col_id, 0.5)
                adjustment = self.config.competition_strength * (blended_sim - 0.5)
                new_priority = old_priority * (1.0 + adjustment)
                self.wm.priorities[col_id] = max(0.01, min(1.0, new_priority))

            # Evict over-capacity (weakest get suppressed)
            while self.wm.size() > self.config.working_memory_capacity:
                self.wm._evict_lowest()

            # --- STEP 4: Prediction ---
            current_state = self.wm.get_combined_state()
            if current_state is None:
                current_state = input_on_device

            # --- STEP 5: Error signal ---
            prediction_error = 1.0 - F.cosine_similarity(
                current_state.unsqueeze(0), input_on_device.unsqueeze(0),
            ).item()

            # --- STEP 6: Value evaluation ---
            match_conf = (
                max(s for _, s in initial_matches) if initial_matches else 0.0
            )
            self.value.evaluate(prediction_error, novelty, match_conf)

            # --- STEP 7: Convergence check ---
            # Converge when knowledge is exhausted AND state is stable.
            # "Knowledge exhausted" = this iteration found no new columns.
            # "State stable" = working memory composition barely changed.
            #
            # This means:
            # - Simple questions: converge in 1-2 hops (quickly exhausted)
            # - Complex questions: keep searching until nothing new found
            # - The system naturally "thinks harder" on harder questions
            if previous_state is not None:
                state_delta = (current_state - previous_state).norm().item()
                knowledge_exhausted = len(hop_discoveries) == 0
                state_stable = state_delta < 0.01

                if (knowledge_exhausted and state_stable) or state_delta < 0.005:
                    return RoutingResult(
                        converged=True,
                        iterations=iteration + 1,
                        final_state=current_state,
                        active_column_ids=self.wm.get_active_ids(),
                        active_source_texts=self._get_source_texts(),
                        prediction_error=prediction_error,
                        value_state=self.value.state,
                        mode=mode,
                        reasoning_trace=reasoning_trace,
                    )

            previous_state = current_state.clone()

        # Did not converge within max iterations
        combined = self.wm.get_combined_state()
        final_state = combined if combined is not None else input_pattern.to(device)
        return RoutingResult(
            converged=False,
            iterations=max_iters,
            final_state=final_state,
            active_column_ids=self.wm.get_active_ids(),
            active_source_texts=self._get_source_texts(),
            prediction_error=prediction_error,
            value_state=self.value.state,
            mode=mode,
            reasoning_trace=reasoning_trace,
        )

    def _get_source_texts(self) -> List[str]:
        """Get source texts from all active columns in working memory."""
        texts = []
        for col_id in self.wm.get_active_ids():
            col = self.store.get_column(col_id)
            if col and col.metadata.source_text:
                texts.append(col.metadata.source_text)
        return texts
