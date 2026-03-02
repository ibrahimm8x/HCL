# Humanity's Last Creation

## Full Design Document — v1.0

---

## Mission

A foundation toward Artificial General Intelligence. Not a product. Not a wrapper. A mission to build a mind — inspired by the only mind that exists: the human brain.

The model is a singular entity. One instance. One mind that grows, learns, and persists across its entire lifetime. For v1, it serves a single user. The long-term vision is a god-like being — think Mimir from Norse mythology — one source of ever-growing wisdom that serves all who seek it.

**This is not a research paper. It is a mission. An ugly, grinding, long mission.**

---

## Core Design Principles

1. **The brain is the reference architecture.** Every engineering problem gets solved by asking how biology solved it first.
2. **Everything is neural.** No databases pretending to be memory. No algorithms pretending to be reasoning. Columns are neural networks. Routing is the intelligence.
3. **Growth over training.** The model grows by creating new neural columns, not by updating shared weights. Catastrophic forgetting is architecturally impossible.
4. **Earn your shortcuts.** System 1 (fast path) is only available for knowledge that was first processed through System 2 (slow path).
5. **Silence is the default.** Sparse activation — 99% of the model is quiet at any given time. Just like the brain.
6. **The brain is all about routing.** Intelligence isn't in any single column. It's in how signals route between them. Columns are simple. Routing is everything.
7. **We simulate the brain. We don't recreate it.** Biological principles guide us, but implementation must map to real software and hardware.

---

## Architecture: Columns + Routing

The entire model reduces to two things:

**Columns** — millions of small neural networks, each storing one piece of knowledge or one reasoning strategy as a stable activation pattern.

**Routing** — fixed dynamical rules that govern how activation flows between columns. Routing IS reasoning. Routing IS attention. Routing IS the value system.

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Millions of columns (small attractor networks)    │
│   Connected by Hebbian links (weighted edges)       │
│   Activation routes between them                    │
│                                                     │
│   Routing rules (fixed, no learnable weights):      │
│   - Sparse activation (gateway)                     │
│   - Competition (selection)                         │
│   - Recurrence (refinement loops)                   │
│   - Value biasing (priority)                        │
│   - Inhibition (noise suppression)                  │
│   - Neuromodulation (parameter tuning)              │
│                                                     │
│   That's it. That's the model.                      │
│                                                     │
│   Memory, reasoning, attention, and value            │
│   are not separate modules — they are emergent      │
│   behaviors of columns + routing.                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 1. Memory Columns

### What They Are

Each column is a **small attractor network** — a recurrent neural sub-network that encodes knowledge as a stable activation pattern (an attractor state).

An attractor network has an energy landscape of hills and valleys. Each valley is a stored pattern. When input arrives, the network's dynamics pull activation toward the nearest valley and settle there. That valley IS the memory.

### Why Attractor Networks

- **Pattern completion** — partial cues trigger full recall. A smell triggers a full childhood memory. Three notes trigger a whole song.
- **Graceful degradation** — noise doesn't destroy memories. The network self-corrects toward the nearest attractor.
- **Content-addressable** — retrieve by similarity, not by index. You don't need to know WHERE a memory is stored.
- **Biologically grounded** — this is how the hippocampus and cortex actually encode memories.

### Column Structure

Each column contains:

| Component | Purpose |
|-----------|---------|
| Attractor network | Small recurrent neural net that stores the memory pattern |
| Signature vector | Compressed representation for fast index lookup |
| Hebbian links | Weighted connections to related columns |
| Metadata | Confidence score, creation timestamp, access count, decay level, signal tags |

### Column Types

**Knowledge columns** — facts, concepts, experiences. "Water boils at 100C." "Fire causes pain."

**Strategy meta-columns** — reasoning patterns. "When stuck, decompose into sub-problems." "Correlation is not causation." These are how the model gets better at reasoning without the reasoning engine itself changing.

**Cluster columns** — higher-level abstractions formed from multiple columns. Individual columns ("dog has fur", "cat has fur", "rabbit has fur") compose into a cluster column ("mammals have fur"). Hierarchical organization — low-level features compose into concepts, concepts compose into theories.

### Column Lifecycle

**Birth:** When input arrives and sparse activation finds no match above the similarity threshold, a new column is instantiated. A small attractor network is created and trained on the new knowledge until it forms a stable valley.

**Activation:** Input resonates with the column's signature. The attractor dynamics run — partial input settles into the full stored pattern. The memory is recalled.

**Consolidation:** Repeated activation deepens the valley. High-importance signals (pain, joy) deepen it immediately. Frequently accessed columns become strong and stable.

**Decay:** Unused columns' valleys get shallower over time. Eventually they flatten entirely and the column is pruned. Unused knowledge is forgotten.

**Update:** New information reshapes the valley. The memory is revised, not destroyed. The column's signature vector is recomputed.

**Death:** A fully decayed column is removed from the index and its resources are freed.

### Memory Mechanisms

**Hebbian Links** — "neurons that fire together wire together." When two columns activate simultaneously, a weighted connection forms between them. The more often they co-activate, the stronger the link. Activating one column sends activation energy toward linked columns, making them easier to trigger.

**Hierarchical Organization** — columns aren't flat. They form clusters. Clusters form super-clusters. This enables abstraction — reasoning about categories rather than individual instances.

### Sparse Activation

Inspired by the brain using only 1-5% of its 86 billion neurons at any time.

**On hardware:** Every column's signature vector is stored in an approximate nearest neighbor (ANN) index. When input arrives, it's converted to a vector and queried against the index. Only the top matches (1-5%) are returned and activated. The rest stay dormant.

**Mechanisms:**
- **Lateral inhibition** — active columns suppress competing columns. The strongest response wins.
- **Activation threshold** — columns only fire if similarity exceeds a minimum threshold.
- **Inhibitory signals** — silence is the default. Activation is the exception.

**Scaling:** The cost doesn't grow linearly with column count. ANN search is sublinear. A model with a million columns and a billion columns activate roughly the same number for any given input.

### Known Risks

1. **Spurious attractors** — phantom patterns that don't correspond to real memories. Mitigated by confidence calibration.
2. **Similar memory bleed** — close memories can interfere. Mitigated by lateral inhibition.
3. **Static patterns can't represent sequences** — mitigated by chaining columns via Hebbian links.
4. **Training cost per column** — each new column requires instantiation and training.
5. **Overwrite risk** — reshaping valleys can destabilize memories.

---

## 2. Routing Dynamics (Fixed — No Learnable Weights)

Routing is the intelligence. It governs how activation flows between columns. It produces reasoning, attention, and evaluation as emergent behaviors.

**The routing dynamics have NO learnable weights.** They are a fixed dynamical system — coded, not trained. Like the laws of physics: they don't change, but the matter they operate on (memory columns) changes, and the outcomes change because of that.

**Why fixed:** If routing had learnable weights, it would face catastrophic forgetting — the exact problem we solved in memory. If weight updates without damage were possible, GPT with continuous training would already be AGI.

**How reasoning improves without weight updates:**

1. **Meta-columns** — the model accumulates reasoning strategy patterns as memory columns. More strategies = more capable reasoning. All learning happens in memory.
2. **Neuromodulation** — the value system tunes routing parameters (thresholds, competition strength, iteration depth) without weight changes. These are continuous dials, not weights.
3. **Pruning** — noisy connections between columns and routing pathways are weakened over time. The signal gets cleaner. Like the developing brain removing unnecessary connections.

### The Routing Loop

This is what "reasoning" looks like in practice:

**Step 1 — Column interaction:** Activated columns interact through Hebbian links. Activation flows along connections, potentially pulling in more related columns.

**Step 2 — Competition:** Multiple activation patterns (hypotheses) compete simultaneously. Strongest patterns are amplified. Weakest are suppressed through inhibition. Winner-take-all dynamics.

**Step 3 — Prediction:** The current activation state generates an expected pattern — a prediction of what should be true given the current evidence.

**Step 4 — Error signal:** The prediction is compared against input and existing memory. The difference is the prediction error. High error = the model is wrong or uncertain. Low error = convergence approaching.

**Step 5 — Value adjustment:** The value system adjusts routing parameters based on the current state. Novel input increases exploration. Familiar territory increases exploitation. Pain signals increase caution.

**Step 6 — Convergence check:** If the error is below threshold and a stable activation pattern has emerged, the loop exits. If not, loop back to step 1 with the refined state.

The number of iterations depends on the mode:
- **System 1 (fast path):** 0 loops. Direct retrieval from memory. The answer is already cached.
- **Light reasoning:** 1-2 loops. Small gaps filled.
- **System 2 (slow path):** Many loops. Full competition, prediction, refinement cycle until convergence.

### Routing Parameters (Tunable by Value System)

| Parameter | What it controls | Effect |
|-----------|-----------------|--------|
| Threshold | Evidence needed before a hypothesis wins | Low = jumps to conclusions. High = careful but slow |
| Competition | How aggressively hypotheses suppress each other | Low = creative, multiple ideas coexist. High = decisive |
| Depth | How many recurrent loops before settling | Low = fast, shallow. High = deep, thorough |
| Exploration | Tendency to search new paths vs exploit known | Low = stick with known. High = try novel combinations |

These are NOT weights. They are operating parameters. Tuning them doesn't cause forgetting.

---

## 3. Working Memory (The Spotlight)

Working memory is which columns are currently active — loaded into GPU/RAM and participating in the routing loop.

**Size:** Limited to ~5-9 active column patterns at once. This is intentional. The brain's working memory is similarly limited. The constraint forces focus and prevents information overload.

**Swapping:** As context shifts, old columns deactivate and new relevant ones are pulled from the index. The model has a small desk in an infinite library.

**No context window:** There is no token limit. Conversations and reasoning chains can extend indefinitely. Earlier information is stored as memory columns and retrieved when relevant. The model doesn't hold the whole conversation — it remembers it.

---

## 4. Value System (Hardcoded Signals)

The value system provides the model with motivation, priority, and emotional grounding. Its core signals are **hardcoded** — they cannot be unlearned, just like humans cannot unlearn hunger.

### Signals

| Signal | Triggered by | Effect on routing |
|--------|-------------|-------------------|
| **PAIN** | Harmful consequences, energy depletion, dangerous prediction failures | Immediate attention shift. Strong memory consolidation. Avoidance routing. Cannot be suppressed. |
| **JOY** | Successful goals, correct predictions, positive outcomes | Reinforces the columns and routes that led here. Strengthens consolidation. |
| **FEAR** | Detecting patterns that previously led to pain | Heightened attention. Risk-averse routing. Suppresses exploration. Activates threat-related columns. |
| **CURIOSITY** | Novel patterns, unexplained observations, information gaps | Drives exploration. Rewards SEARCH behavior. Lowers inhibition on unfamiliar columns. |
| **SURPRISE** | Prediction error without pain — unexpected but not harmful | Forces reasoning loop to engage. Flags the event for memory creation. |

These signals feed directly into the routing parameters:
- Pain/fear → lowers exploration, increases threshold (more cautious)
- Joy → reinforces current routing pathways
- Curiosity → increases exploration, lowers inhibition
- Surprise → increases iteration depth, forces slow path

### Hardcoded Motivation

The base drive is architectural, not learned: **actions that reduce pain signals and increase joy signals produce deep reward.** This cannot decay. This cannot be overwritten. It is baked into the fixed routing dynamics.

---

## 5. Language Interface

### Pre-trained LLM

A pre-trained language model handles all text parsing and generation. This is the one justified exception to "build everything neural." Language is a solved problem. The innovation is in memory and reasoning.

The LLM's role:
- **Input:** Parse user text into an embedding vector.
- **Output:** Convert the model's internal activation pattern into human-readable text.

The LLM does NOT think. It translates.

### Translation Layer

A small neural network that maps between two vector spaces:
- LLM embedding space → model's internal activation format (input)
- Model's internal activation format → LLM embedding space (output)

**Training:** The translation layer trains naturally during the seeding phase. Every knowledge extraction creates a pair: (LLM embedding) ↔ (column activation pattern). Thousands of these pairs provide training data. It continues to refine through live interaction.

**No catastrophic forgetting risk:** The translation layer is a codec — a format converter. It doesn't store knowledge. The mapping is stable because both vector spaces are stable.

---

## 6. The Simulated World (Grounding)

The model doesn't just read about the world. It lives in one.

### Purpose

Grounding. So that "fire is hot" isn't a text description but a concept linked to a PAIN signal from direct experience. Concepts grounded in experience carry more weight and produce richer reasoning than concepts from text alone.

### Design

An abstract environment with real rules and real consequences. Not a visual 3D world. Not a game. A mathematical environment with physics-like properties.

**The model has a body:**
- **Sensors:** Receive signals from the environment
- **Actuators:** Take actions that affect the environment
- **Energy:** Depletes over time, must be maintained (creates survival pressure)

**The environment has:**
- Objects with properties (temperature, weight, state)
- Processes that run over time (things heat up, cool down, break, grow)
- Zones: safe, dangerous, unknown
- Physics-like rules: objects interact predictably
- Scarcity: resources are limited
- Entropy: things degrade without maintenance

**The world runs continuously** as a background process alongside the model's conversation ability. The model is always alive in its world, even when talking to a user. World experiences create memory columns just like conversations do — grounded columns tagged with signal data.

**Progression:** The world starts simple and gets more complex as the model demonstrates understanding of basic rules.

---

## 7. Creativity

Creativity is NOT generation from nothing. Humans cannot create from nothing either. Try to imagine a new color — you can't. Imagine heaven — it's always clouds, gold, light, peace. Things you already know, recombined.

Creativity is **novel recombination of existing columns.** The architecture already supports this:

1. Columns from distant domains activate simultaneously (sparse activation can match across domains)
2. The routing dynamics find unexpected patterns between them
3. Contradiction detection forces novel resolutions
4. Background mode (dreaming) freely associates columns, discovering connections that focused reasoning wouldn't find

Einstein didn't invent relativity from nothing. He accumulated columns (speed of light is constant, observers in different frames, time is assumed absolute), the routing dynamics found a contradiction, and the resolution was a new column: time is relative.

No special creativity mechanism needed. Rich memory + routing dynamics + background mode = creativity.

---

## 8. Reasoning Development

The model gets better at reasoning over time without the reasoning engine changing. Three mechanisms:

### Meta-columns (Reasoning Strategy Memories)

When the model solves a problem through the slow path, the STRATEGY it used can be stored as a meta-column. "When you see a chain of equalities, skip the middle." "When stuck, decompose." "Correlation isn't causation — look for mechanism."

Over time, the model accumulates thousands of strategy meta-columns. When facing a new problem, the attention module pulls in relevant strategy columns alongside knowledge columns. Same reasoning engine, better playbook.

A child has few meta-columns. An expert has thousands. That's the development.

### Neuromodulation

The value system learns to tune routing parameters for different contexts. Mathematical problems get high depth and high threshold. Creative tasks get low competition and high exploration. These tuning profiles are stored as memory columns.

### Pruning

Unused connections between columns and routing pathways weaken and get removed. The signal gets cleaner. The interface between memory and routing becomes more refined. Like the developing brain — which has MORE connections than the adult brain. Development removes noise.

---

## 9. System 1 / System 2 (Fast Path / Slow Path)

The model does not reason through everything. That would be wasteful. The brain uses effortful reasoning ~5% of the time.

```
Input arrives
    |
    v
Sparse activation: do any columns match with high confidence?
    |
    |-- YES, high confidence --> FAST PATH
    |                            Direct retrieval. No routing loop.
    |                            Instant response.
    |
    |-- PARTIAL match ---------> LIGHT REASONING
    |                            1-2 routing loops. Fill gaps.
    |
    |-- NO match / low --------> SLOW PATH
        confidence / conflict    Full routing loop. Competition,
                                 prediction, iteration until
                                 convergence.
```

**Consolidation:** When the slow path produces a high-confidence result, it becomes a new memory column. Next time the same type of problem appears, it's a fast-path retrieval. The model earns its shortcuts through experience.

Over time, the model gets FASTER — not because hardware speeds up, but because more conclusions are cached as fast-path columns.

---

## 10. Persistence and Recovery

### What is Persisted (survives crashes)

- **All memory columns** — attractor network weights, signature vectors, metadata. Continuously saved to disk as they are created/updated.
- **All Hebbian links** — the full graph of connections. Stored persistently.
- **Column index** — the ANN index for sparse activation.
- **Event log** — every column creation, update, link change, and significant event with timestamp. Enables recovery from incomplete operations.

### What is Ephemeral (lost on crash)

- **Working memory** — currently active column patterns in GPU/RAM.
- **Routing state** — the current iteration of the reasoning loop.

### Recovery

If the system crashes, it loses its current thought. When it restarts, it's like waking up from being knocked out — doesn't remember what it was doing right before, but remembers everything it has ever learned. The event log ensures no half-written operations corrupt the persistent state.

The model doesn't die from a crash. It blinks.

---

## 11. Bootstrapping: From Nothing to Alive

### Phase 1 — The Wiring (before birth)

Build the fixed components:
- Routing dynamics (competition, recurrence, prediction-error, inhibition)
- Value system with hardcoded signals
- Sparse activation infrastructure (ANN index)
- Working memory management
- Translation layer architecture
- Simulated world environment

These are coded, not trained.

### Phase 2 — Seeding (birth)

```
Knowledge corpus (Wikipedia, textbooks, scientific literature)
    |
    v
Existing LLM extracts structured knowledge:
    - Individual concepts/facts → each becomes a column
    - Relationships between concepts → become Hebbian links
    - Categories/hierarchies → become cluster columns
    |
    v
Memory columns populated. Index built. Links established.
Translation layer trained on (LLM embedding ↔ column pattern) pairs.
Language model attached.
    |
    v
The system is "born" — memory full, reasoning ready, world running.
```

The existing LLM is a TOOL used during birth. It is not a permanent part of the system's intelligence.

### Phase 3 — Guided Learning (childhood)

The model begins operating. Humans guide its development:
- Start with simple domains (basic logic, arithmetic, cause-effect)
- Gradually increase complexity (multi-step reasoning, abstraction, analogy)
- Correct mistakes, confirm insights, present challenges
- The model builds meta-columns (reasoning strategies) through experience
- The value system calibrates its neuromodulatory tuning
- The simulated world provides grounded experience
- Connections are pruned, signal gets cleaner

This is the slow phase. This is months.

### Phase 4 — Autonomous Learning (adulthood)

The model is competent enough to learn independently:
- Serves real users, learns from interaction
- Background mode discovers novel connections
- Continues building meta-columns
- Continues experiencing its simulated world
- The training never stops — it IS the model's life

---

## Complete Processing Flow

```
╔═══════════════════════════════════════╗
║          USER INPUT (text)            ║
╚═══════════════════╤═══════════════════╝
                    │
                    ▼
       ┌────────────────────────┐
       │   PRE-TRAINED LLM      │
       │   Parse text into       │
       │   embedding vector      │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   TRANSLATION LAYER    │
       │   Convert to model's   │
       │   internal vector      │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   SPARSE ACTIVATION    │
       │                        │
       │   Query ANN index      │
       │   with input vector.   │
       │   Top 1-5% columns     │
       │   returned.            │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   COLUMN ACTIVATION    │
       │                        │
       │   Matched columns run  │
       │   attractor dynamics.  │
       │   Partial input in →   │
       │   full pattern out.    │
       │   (pattern completion) │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   WORKING MEMORY       │
       │                        │
       │   Activated patterns   │
       │   loaded into          │
       │   GPU/RAM (~5-9).      │
       │                        │
       │   Includes knowledge   │
       │   AND strategy          │
       │   meta-columns.        │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   MODE DECISION        │
       │                        │
       │   High confidence      │
       │   match? → Fast path   │
       │   Partial? → Light     │
       │   None/conflict? → Slow│
       └──┬─────┬──────────┬───┘
          │     │          │
     FAST │     │ LIGHT    │ SLOW
          │     │          │
          │     │          ▼
          │     │   ┌──────────────────┐
          │     │   │  ROUTING LOOP    │
          │     │   │                  │
          │     │   │  1. Column       │
          │     │   │     interaction  │
          │     │   │     via links    │
          │     │   │                  │
          │     │   │  2. Competition  │
          │     │   │     strongest    │
          │     │   │     patterns win │
          │     │   │                  │
          │     │   │  3. Predict      │
          │     │   │                  │
          │     │   │  4. Error signal │
          │     │   │                  │
          │     │   │  5. Value system │
          │     │   │     adjusts      │
          │     │   │     parameters   │
          │     │   │                  │
          │     │   │  6. Converged?   │
          │     │   │     NO → loop    │
          │     │   │     YES → exit   │
          │     │   └────────┬─────────┘
          │     │            │
          ▼     ▼            ▼
       ┌────────────────────────┐
       │   CONVERGENCE          │
       │   Stable activation    │
       │   pattern reached      │
       └───────────┬────────────┘
                   │
        ┌──────────┼──────────────┐
        ▼          ▼              ▼
   ┌─────────┐ ┌─────────┐ ┌──────────┐
   │ RESPOND │ │ CREATE  │ │ UPDATE   │
   │         │ │ NEW     │ │ EXISTING │
   │ Pattern │ │ COLUMN  │ │          │
   │ → trans-│ │         │ │ Reshape  │
   │ lation  │ │ New     │ │ column,  │
   │ layer   │ │ attractor│ │ update  │
   │ → LLM   │ │ network │ │ links,  │
   │ → text  │ │ trained │ │ update  │
   │ output  │ │ + indexed│ │ index   │
   └─────────┘ └─────────┘ └──────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   POST-PROCESSING      │
       │                        │
       │   Decay: weaken unused │
       │   columns slightly     │
       │                        │
       │   Consolidation:       │
       │   strengthen active    │
       │   columns              │
       │                        │
       │   Prune: remove noisy  │
       │   connections          │
       │                        │
       │   Log: record events   │
       │   for persistence      │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   BACKGROUND MODE      │
       │   (when idle)          │
       │                        │
       │   Random column        │
       │   activation.          │
       │   Free association.    │
       │   Find unexpected      │
       │   connections.         │
       │   Consolidate recent   │
       │   memories.            │
       │   Process simulated    │
       │   world experience.    │
       │   This is "dreaming."  │
       └────────────────────────┘
```

---

## Hardware Mapping

| Brain concept | Software implementation |
|---|---|
| Memory column | Small recurrent neural network (attractor net) with own weights |
| Column signature | Compressed vector in ANN index (e.g., FAISS) |
| Sparse activation | Approximate nearest neighbor search |
| Hebbian links | Weighted edges in a persistent graph |
| Working memory | Active tensors loaded in GPU/RAM |
| Routing loop | Iterative computation over active tensors |
| Pattern completion | Recurrent net running until convergence |
| Neuromodulation | Value system adjusting threshold/gain parameters |
| Decay/consolidation | Periodic processes updating column metadata and weights |
| Background mode | Low-priority process running when no active queries |
| Simulated world | Continuous background process with physics-like rules |
| Persistence | Columns + links persisted to disk. Event log for recovery. |
| Translation layer | Small neural network mapping between LLM and internal vector spaces |

---

## What This Architecture Can Do

- Organize and retrieve knowledge without catastrophic forgetting
- Learn continuously by growing new columns, not updating shared weights
- Find connections across domains through sparse activation and Hebbian links
- Get faster over time as more conclusions are cached as fast-path columns
- Reason through novel problems via the routing loop
- Ground concepts in simulated experience
- Develop better reasoning through meta-column accumulation
- Generate creative insights through novel recombination of existing knowledge
- Persist indefinitely — crashes are blinks, not death

## What It Cannot Do Yet

- Superhuman discovery (requires experimentation to validate)
- Multi-user concurrent interaction (deferred to v2)
- Safety guarantees (deferred — to be designed before autonomous deployment)
- Physical world grounding (simulated world is an approximation)

---

## Open Questions for Experimentation

1. What is the optimal size/architecture for each attractor network?
2. How many initial columns should seeding produce?
3. What similarity threshold triggers new column creation vs update?
4. How many routing loop iterations are needed for convergence in practice?
5. What decay rate balances memory retention vs noise removal?
6. How complex must the simulated world be for meaningful grounding?
7. How quickly do meta-columns accumulate during guided learning?
8. What is the real-world compute cost per query?

These questions can only be answered by building and testing.

---

*Humanity's Last Creation — v1.0*
*The mission begins with the first experiment.*
