# Learned Trajectory Aggregation for Process Reward Models

A contribution to [its_hub](https://github.com/Red-Hat-AI-Innovation-Team/its_hub)
that makes trajectory aggregation pluggable and ships a learned reference implementation.

## Problem Statement

`ParticleFiltering` and `BeamSearch` in `its_hub` reduce a sequence of per-step PRM scores
to a single trajectory score via a hardcoded choice of `prod`, `min`, or `mean`.
None of the three is a defensible default:

| Reduction | Pathology |
|-----------|-----------|
| `prod` | Penalises long trajectories — more terms below 1 shrinks the product |
| `min` | Brittle — one weak step destroys an otherwise correct trajectory |
| `mean` | Discards step position and inter-step dependence |

This project introduces an `AbstractTrajectoryAggregator` interface (in `its_hub/base.py`)
so aggregation becomes pluggable, and ships a learned MLP aggregator that demonstrates
the interface delivers real accuracy gains.

## Why this approach

Ten candidate project ideas were scored on contribution value, ML depth, required compute, effort, and risk before a line of code was written. Pluggable trajectory aggregation ranked first because:

- It addresses a **real, documentable limitation** in the codebase — hardcoded `prod`/`min`/`mean` with no override path
- It has genuine **ML depth**: training a learned aggregator is a real supervised learning problem with a non-obvious label signal
- It runs **without CUDA** — the PRM scoring and MLP training both work on Apple Silicon MPS
- It produces a **clean upstream contribution**: the interface change is backward-compatible and the learned aggregator is a reference implementation
- Alternatives (retraining the PRM, cross-policy LoRA experiments, online beam-search integration) either required CUDA or carried higher execution risk given the time constraint

## Hypothesis

**Primary:** A learned aggregator over per-step PRM scores outperforms `prod`, `min`, and
`mean` on trajectory selection accuracy at fixed budget.  The gap concentrates on
long trajectories (where `prod` underflows) and on trajectories whose worst step is
followed by recovery (where `min` discards the signal).

**Secondary:** Inspecting the learned aggregator reveals that late-step scores carry
materially more predictive weight than early-step scores.

## Method

### Interface change

```python
# its_hub/base.py
class AbstractTrajectoryAggregator(ABC):
    @abstractmethod
    def aggregate(self, step_scores: list[float]) -> float: ...
    async def aaggregate(self, step_scores: list[float]) -> float: ...
```

`ParticleFiltering` and `BeamSearch` gain an optional `aggregator` parameter
defaulting to `HardcodedAggregator("prod")`.  Existing user code is unchanged.

### Feature vector

Each trajectory is summarised as a 10-dimensional vector (see `src/learned_aggregator/features.py`):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | mean | Mean step score |
| 1 | min | Minimum step score |
| 2 | max | Maximum step score |
| 3 | last | Score of the final step |
| 4 | length | Number of steps |
| 5 | variance | Step score variance |
| 6 | pos\_min | Normalised position of minimum score |
| 7 | pos\_max | Normalised position of maximum score |
| 8 | last−first | Score change from first to last step |
| 9 | gap\_at\_min | Score drop into the minimum step |

### Model

`TrajectoryMLP`: 2-layer MLP (input→hidden→1, ReLU, sigmoid), binary cross-entropy
against trajectory correctness.  Default hidden width: 16 (≈465 parameters).
Trains with Adam, early stopping on validation loss (patience=10).

### Baselines

| Baseline | Definition |
|----------|-----------|
| `prod` | Product of step scores |
| `min` | Minimum step score |
| `mean` | Mean step score |
| `random` | Uniform random (sanity floor) |

## Data

- **Policy:** Qwen2.5-1.5B-Instruct at temperature 0.7
- **PRM:** Qwen2.5-Math-PRM-7B via `TransformersProcessRewardModel` (MPS/bfloat16 on Apple Silicon)
- **Problems:** 200 from MATH train split (MATH500 held out entirely)
- **Trajectories:** N=8 per problem
- **Split:** 70/15/15 problem-level (train/val/test), seed=42

## Trajectory Corpus Pipeline

The canonical data pipeline is expressed as an `sdg_hub` Flow at
`flows/trajectory_corpus.yaml`.  Five blocks run in sequence:

| # | Block | Role |
|---|-------|------|
| 1 | `PromptBuilderBlock` (`build_math_prompt`) | Wraps each problem in the Qwen step-by-step system prompt (`flows/prompts/math_system.yaml`) |
| 2 | `RowMultiplierBlock` (`fan_out_trajectories`) | Fans out each problem row to N=8 trajectory candidates |
| 3 | `LLMChatBlock` (`generate_trajectory`) | Generates each trajectory with the policy LLM (async, temperature=0.7) |
| 4 | `ProcessRewardScoreBlock` (`score_steps`) | Scores all reasoning steps in one forward pass via `TransformersProcessRewardModel` (MPS/fp16); writes `step_scores: list[float]` |
| 5 | `MathVerifyAnswerBlock` (`verify_answer`) | Extracts `\boxed{}` answer; labels `correct: bool` against `ground_truth` |

`scripts/generate_trajectories.py` is a thin runner: it imports the custom blocks
(triggering `BlockRegistry` registration), loads the YAML, injects model config,
applies a seed-pinned 70/15/15 problem-level split, and writes per-split JSONL files.

To run just the flow programmatically:

```python
import learned_aggregator.blocks  # registers custom blocks
from sdg_hub import Flow

flow = Flow.from_yaml("flows/trajectory_corpus.yaml")
flow.set_model_config(
    model="openai/Qwen/Qwen2.5-1.5B-Instruct",
    api_base="http://localhost:8100/v1",
    api_key="NO_API_KEY",
)
result_df = flow.generate(dataset_df, runtime_params={"score_steps": {"model_name": "Qwen/Qwen2.5-Math-PRM-7B"}})
```

## Evaluation Protocol

Primary metric: **selection accuracy** — fraction of test problems where the
highest-scoring trajectory's answer is correct.

Stratified by:
- Trajectory length: short (≤4 steps), medium (5–9), long (≥10)
- Problem difficulty: MATH levels 1–5
- Candidate count N ∈ {4, 8, 16}

Bootstrap 95% CI with 1000 resamples.  A gap counts only if it exceeds twice
the CI half-width.

## Results

Nested 5-fold cross-validation across all 408 problems (Levels 1–5, 8 trajectories
per problem).  Outer folds estimate performance; inner hold-out (20% of each outer
training set) selects hyperparameters independently per fold.  CI is a
t-distribution 95% interval across the 5 fold estimates (4 degrees of freedom).

| Aggregator | Mean acc | Std | 95% CI |
|------------|----------|-----|--------|
| random | 0.380 | 0.060 | [0.306, 0.454] |
| min | 0.490 | 0.058 | [0.419, 0.562] |
| prod | 0.498 | 0.052 | [0.434, 0.562] |
| geomean | 0.498 | 0.064 | [0.419, 0.577] |
| mean | 0.498 | 0.064 | [0.419, 0.577] |
| learned\_mlp (HP-tuned per fold) | 0.495 | 0.054 | [0.428, 0.562] |
| lstm (HP-tuned per fold) | 0.498 | 0.064 | [0.418, 0.577] |
| **gbdt** (HP-tuned per fold) | **0.517** | **0.051** | **[0.454, 0.581]** |

**GBDT (0.517 ± 0.051)** is the only model that consistently clears the fixed
baselines across folds.  Its min-threshold decision surface is well-matched to
the signal available at this data scale.

**MLP and LSTM (both ~0.495–0.498)** are statistically indistinguishable from
`prod` and `mean` at 408 problems.  The apparent per-variant differences seen in
single-split evaluations were sampling noise — k-fold reveals no reliable edge
over the fixed baselines for either model family.

**Hyperparameter selection is noisy** at this scale.  Inner CV chose different MLP
widths each fold (16, 4, 8, 8, 4) and different GBDT configurations
((n=50,d=2), (n=50,d=2), (n=500,d=2), (n=50,d=3), (n=200,d=4)).  The only robust
signal is GBDT depth=2, selected in 4 of 5 folds; n\_estimators and MLP/LSTM
width cannot be reliably chosen from the available data.

**Recommended config:** GBDT `max_depth=2`; n\_estimators is not sensitive —
defaults are fine.  MLP and LSTM width recommendations are unreliable at this
corpus size.

### Hypothesis verdicts

**Primary — weakly confirmed for GBDT; sub-claims rejected.**  The overall claim (learned > fixed baselines) holds directionally for GBDT: +1.9pp over `prod`/`mean`, consistent across all 5 outer folds, though CIs overlap and the gap does not reach conventional significance.  MLP and LSTM show no edge over the fixed baselines.

The two sub-claims are contradicted by the data.  The hypothesis predicted gains on *long* trajectories (where `prod` underflows), but the by-length single-split shows GBDT gains +7.6pp on medium (5–9 steps) and +6.7pp on short (≤4 steps) while gaining **nothing on long** trajectories (0.286 for both).  The "recovery after bad step" story also fails: GBDT's feature profile places 78% of weight on `min`, the same signal the `min` baseline already uses — it wins by learning a continuous threshold on the worst step, not by discounting subsequent recovery.

**Secondary — weakly rejected.**  The dominant MLP features are `variance` (#1) and `min` (#2); `last` ranks 4th with no separation from other mid-trajectory features.  `last_minus_first` ranks 3rd and does capture whether the trajectory improved over time, but the primary signal is trajectory *consistency* and worst-case performance, not late-step emphasis.  GBDT has `last` as a distant second (10.6%) after `min` (78%).

**What the data actually supports** — which neither hypothesis predicted — is that at this corpus size, trajectory selection is effectively a soft threshold problem on the minimum step score.  Everything else is secondary noise.  The right framing for future work is: does a larger corpus change the dominant signal away from `min`, and if so, do MLP/LSTM begin to show an edge?

### Breakdown by difficulty level

Difficulty-level breakdown from the original held-out test split (62 problems),
shown for directional context only — per-stratum n is too small for reliable conclusions.

| Level | n | prod | gbdt | learned\_mlp | lstm |
|-------|---|------|------|-------------|------|
| 1 | 7 | 0.857 | **0.857** | 0.857 | 0.857 |
| 2 | 8 | 0.375 | **0.500** | 0.500 | 0.375 |
| 3 | 10 | 0.600 | **0.600** | 0.600 | 0.600 |
| 4 | 4 | 0.500 | **0.500** | 0.500 | 0.500 |
| 5 | 33 | 0.273 | **0.333** | 0.273 | 0.303 |

Level 5 (n=33) is the only stratum with enough problems to carry any weight.
GBDT gains +6pp there (0.333 vs 0.273), consistent with its min-threshold approach
being well-matched to the high-variance, low-solve-rate regime at this difficulty.

## Per-Step Weight Profile

### Mixed-difficulty corpus, seed=42 (hidden\_width=16)

```
variance             0.5875  █████████████████████████████
min                  0.3883  ███████████████████
last_minus_first     0.3438  █████████████████
last                 0.3120  ███████████████
pos_max_norm         0.2621  █████████████
pos_min_norm         0.2508  ████████████
mean                 0.2363  ███████████
max                  0.2135  ██████████
gap_at_min           0.1515  ███████
length               0.1498  ███████
```

### Mixed-difficulty corpus (hidden\_width=8)

```
variance             0.584  █████████████████████████████
min                  0.384  ███████████████████
last_minus_first     0.306  ███████████████
last                 0.300  ██████████████
pos_max_norm         0.250  ████████████
pos_min_norm         0.228  ███████████
max                  0.210  ██████████
mean                 0.207  ██████████
gap_at_min           0.144  ███████
length               0.137  ██████
```

On the mixed-difficulty corpus, **variance** ranks highest.  Incorrect
trajectories at Levels 1–3 are characterised by *spread* across steps rather
than a single catastrophic failure — the policy mostly solves easier problems
and the distinguishing signal is consistency, not worst-case.  `last_minus_first`
ranks third, consistent with the policy showing coherent reasoning arcs on
easier problems.  The h16 and h8 profiles rank identically, confirming the
finding is not sensitive to hidden width.

### GBDT feature importances (Gini, mixed-difficulty corpus, tuned n=200 depth=2)

```
min                  0.7831  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
last                 0.1065  █████████████████████
last_minus_first     0.0296  █████
mean                 0.0266  █████
max                  0.0220  ████
length               0.0147  ██
variance             0.0112  ██
gap_at_min           0.0045  
pos_min_norm         0.0011  
pos_max_norm         0.0007  
```

GBDT concentrates 78% of its weight on `min` — effectively learning a
threshold on the worst step.  That it still clears the `min` baseline
(0.517 vs 0.490 in k-fold CV) shows that the absolute value of the minimum
step matters beyond its rank among candidates.  The shallow model (depth=2,
selected in 4/5 folds) is consistent with the dominant signal being a single
threshold rather than a complex interaction.

The MLP distributes weight more evenly across features while GBDT collapses to
near-single-feature dependence.  The MLP's broader weight profile may be
advantageous as corpus size grows and richer signal becomes available.

The secondary hypothesis (late-step scores carry more weight than early-step
scores) remains unconfirmed: `last` and `pos_max_norm` rank similarly to
mid-trajectory features in both profiles.

## Design Decisions and Trade-offs

### Problem-level train/val/test split

Trajectories from the same problem are kept together in one split. A trajectory-level split would leak: the model could learn problem-specific PRM artefacts from train trajectories and exploit them on test trajectories from the same problem. Problem-level split is strictly correct; it makes the task harder and the accuracy numbers more honest.

### Hand-crafted feature vector vs. raw sequences

The 10-dimensional feature vector was chosen over raw sequence input primarily for **inductive bias and data efficiency**. The features encode exactly the statistics that matter (worst step, variance, last step), so the MLP just needs to learn a threshold over pre-computed signals. An LSTM must *discover* those statistics from raw sequences. In k-fold CV across 408 problems, MLP (0.495) and LSTM (0.498) are effectively tied — consistent with both being data-limited at this corpus size rather than architecture-limited.

### MLP hidden width

h4 through h32 show no consistent trend with width (h4 scores highest at 0.452, h8–h32 converge at 0.435, all differences within CI). The signal is saturated well below 49 parameters. h4 is recommended: smallest checkpoint, fastest to train, no evidence of underfitting. Revisit if corpus grows past ~2000 problems.

### GBDT vs. MLP

GBDT concentrates ~78% of Gini importance on `min` and effectively learns a threshold on the worst step. MLP distributes weight across all 10 features. GBDT wins on the current corpus because the task *is* a threshold problem at small data scale. MLP is the better long-term bet: its broader weight distribution should prove advantageous as corpus size grows and richer signal becomes available. Both are cheap to train; keeping both costs nothing.

### GBDT hyperparameter tuning

HP selection used the nested 5-fold CV inner loop: a 4×4 grid over n\_estimators ∈ {50, 100, 200, 500} × max\_depth ∈ {2, 3, 4, 5} was scored by inner-val selection accuracy for each of the 5 outer folds.  **Depth=2 was selected in 4 of 5 folds** — the only robust signal.  n\_estimators varied across folds (50, 50, 500, 50, 200), indicating the corpus is too small to distinguish tree counts reliably.  The recommended config (n\_estimators=200, max\_depth=2) matches the majority-vote depth and a mid-range tree count.

### N=8 trajectories per problem

N=8 balances trajectory diversity against corpus generation time (~3–4 minutes per problem on M4 Max). N=4 would halve generation time but reduce diversity; N=16 would double it. At N=8, ~36% of mixed-difficulty problems have at least one correct trajectory — sufficient learning signal without prohibitive generation cost.

### Selection-time aggregation, not per-step integration

The aggregator scores completed trajectories at selection time rather than being called per-step inside `ParticleFiltering`. Per-step integration would require modifying the search algorithm and changes the semantics of the PRM signal during sampling. Selection-time aggregation is a clean, backward-compatible addition with no side effects on existing search behaviour — and the right scope boundary for a pluggable interface contribution.

## Quickstart

### 1. Install dependencies

```bash
cd learned-aggregator
pip install -e ".[dev]"          # installs its_hub as a local dependency
```

### 2. Start the policy LLM server

**Apple Silicon (its-iaas):**
```bash
its-iaas --host 0.0.0.0 --port 8100 &
# Then configure it: see docs/iaas-service.md in the its_hub repo
# Or point generate_trajectories.py at any OpenAI-compatible v1 endpoint.
```

**Linux with CUDA (vLLM):**
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8100
```

### 3. Generate trajectories

The pipeline is defined in `flows/trajectory_corpus.yaml`.
`scripts/generate_trajectories.py` processes one problem at a time and writes
results incrementally to a staging file — safe to interrupt and resume.

**Phase A — Level 1–4 problems** (faster; more solvable problems):
```bash
python scripts/generate_trajectories.py \
    --lm-model Qwen/Qwen2.5-1.5B-Instruct \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --num-problems 300 \
    --levels 1 2 3 4 \
    --output-dir data/trajectories_level14 \
    --no-split
```

**Phase B — Level 5 problems** (optional; harder):
```bash
python scripts/generate_trajectories.py \
    --lm-model Qwen/Qwen2.5-1.5B-Instruct \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --num-problems 200 \
    --levels 5 \
    --output-dir data/trajectories_level5
```

**Merge and re-split:**
```bash
python scripts/merge_splits.py \
    --existing-dir data/trajectories_level5 \
    --staging data/trajectories_level14/staging.jsonl \
    --output-dir data/trajectories_combined
```

N=8 trajectories per problem is set in the flow YAML (`RowMultiplierBlock.num_samples`);
edit `flows/trajectory_corpus.yaml` to change it.

### 4. Train the MLP aggregator

```bash
python scripts/train_aggregator.py \
    --data-dir data/trajectories_combined \
    --hidden-width 8 \
    --epochs 200 \
    --patience 10
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
    --test-jsonl data/trajectories/test.jsonl \
    --output results/eval.json
```

### 6. Use the trained aggregator in its_hub

```python
from its_hub.aggregators import LearnedMLPAggregator
from its_hub.algorithms.particle_gibbs import ParticleFiltering

agg = LearnedMLPAggregator("its_hub/aggregators/checkpoints/mlp_agg.pt")
pf = ParticleFiltering(sg=sg, prm=prm, aggregator=agg)
result = pf.infer(lm, problem, budget=16)
```

## Reproduction on CUDA hardware

`TransformersProcessRewardModel` supports both MPS (Apple Silicon) and CUDA.
On a CUDA host, construct it with `device="cuda"`:

```python
from its_hub.integration import TransformersProcessRewardModel
prm = TransformersProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda",
)
```

Or set the `backend` field in `ProcessRewardScoreBlock` — it defaults to
`"transformers"` which auto-uses the correct device.

For the corpus pipeline at full scale (500 problems, CUDA):
```bash
python scripts/generate_trajectories.py \
    --lm-endpoint http://localhost:8100/v1 \
    --lm-model Qwen/Qwen2.5-1.5B-Instruct \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --num-problems 500 \
    --output-dir data/trajectories_full
```

The JSONL format and training/evaluation scripts are identical.

## Secondary Contribution: Apple Silicon PRM Support

`TransformersProcessRewardModel` (`its_hub/integration/transformers_prm.py`)
implements the correct scoring algorithm for Qwen2.5-Math-PRM-7B:
one forward pass → 2-class softmax at each `<extra_0>` position.
It runs on MPS (Apple Silicon), CUDA, or CPU — removing the CUDA-only
constraint of `LocalVllmProcessRewardModel` for local development.

`MLXProcessRewardModel` (`its_hub/integration/mlx_prm.py`) is also provided
for Math-Shepherd-style PRMs that use generative "+"/"-" token scoring
(not for Qwen2.5-Math-PRM-7B, which uses a classifier head).

### Transformers 5.x compatibility fix

`transformers ≥ 5.0` uses meta-tensor initialisation during `from_pretrained`.
Non-persistent buffers in `Qwen2RotaryEmbedding` (`inv_freq`, `cos_cached`,
`sin_cached`) are materialised as zeros rather than computed from the RoPE
formula, causing every attention Q/K to be NaN and all PRM scores to collapse
to the constant 0.50003338.  `TransformersProcessRewardModel._repair_rotary_embeddings()`
detects and recomputes these buffers immediately after model load.
The fix is transparent — no API change, no performance cost.

## Stretch: Cross-Policy Transfer via training_hub

_Optional — may be skipped on M4 Max due to CUDA requirements._

The core experiment evaluates an aggregator trained on trajectories from
policy A (Qwen2.5-1.5B-Instruct at temperature 0.7).  A stronger test is
cross-policy transfer: train on policy A, evaluate on policy B where policy
B is a LoRA fine-tune of the same base model.

**Why it matters:** An aggregator that generalises across policies has learned
the intrinsic geometry of PRM scores rather than a policy-specific artefact.
If accuracy holds across policies, the aggregator is worth upstream.

**Approach using training_hub:**

```bash
# Fine-tune policy B with LoRA
training_hub train \
    --base-model Qwen/Qwen2.5-1.5B \
    --dataset lighteval/MATH \
    --output-dir checkpoints/policy_b \
    --lora-rank 16

# Serve policy B and generate trajectories
vllm serve checkpoints/policy_b --port 8101
python scripts/generate_trajectories.py \
    --lm-endpoint http://localhost:8101/v1 \
    --lm-model policy_b \
    --output-dir data/trajectories_policy_b

# Evaluate with aggregator trained on policy A
python scripts/evaluate.py \
    --test-jsonl data/trajectories_policy_b/test.jsonl \
    --checkpoint checkpoints/mlp_agg.pt
```

**Caveats:**
- `training_hub` LoRA backends assume CUDA.  The LoRA step requires
  a CUDA host or a remote training cluster; the rest of the pipeline
  (PRM scoring via MLX, evaluation) runs locally on Apple Silicon.
- LoRA fine-tuning on MATH with default hyperparameters may not produce
  a meaningfully distinct policy in a short run.  Use at least 1000 steps.

## AI-Assisted Development

### Tools used

**Claude Code** (Anthropic) was the primary interactive development tool throughout the project —
used for ideation, code generation, debugging, and analysis.

**Gas City** (gastownhall) is an AI project orchestration tool built by a friend and former Google
colleague.  It uses an AI agent called the mayor to translate a high-level project brief into a
structured work graph of parallelisable tasks, then executes that graph autonomously with minimal
supervision.  This project was an opportunity to evaluate it seriously on a real ML workflow.

### Workflow

#### Phase 1 — Project scoping with Claude Code

The brief was fed to Claude Code with a request to propose ten well-scoped project
ideas.  Claude and I then collaboratively scored each idea on five dimensions: contribution value,
ML depth, effort, required compute, and risk.  We iterated on the rankings together, explicitly
optimising for ML depth and contribution value while penalising CUDA-only approaches (Apple Silicon
M4 Max constraint) and high-risk unknowns.  This scoping pass took roughly 30 minutes and produced
a clear first-choice project (Idea 3: extend a library to address an ML limitation) with documented
rationale for every tradeoff.

This is a reusable pattern worth recommending to any team starting a time-boxed ML project: use an
AI assistant to enumerate options and make tradeoffs *explicit and scoreable* before writing a line
of code.

#### Phase 2 — Autonomous execution with Gas City

The chosen idea was written up as a structured project brief (`BRIEF.md`) and handed to Gas City.
I was sceptical that the mayor would be able to translate a nine-section ML brief into an actionable
work graph — but it impressed.  The mayor decomposed the brief into 12 tasks, ordered them into
seven phases with correct dependency edges, and flagged two real bugs in the existing code before
any work began (JSONL schema mismatch between `generate_trajectories.py` output and
`train_aggregator.py` input; key name collision between `correct` and `is_correct` across blocks).

The resulting work graph, as executed:

```
[BRIEF.md]
     │
     ▼
Phase 0 — parallel, no dependencies
  ├── Verify test suites green (its_hub + learned-aggregator)
  ├── [BUG] Fix JSONL schema: flat rows → grouped by problem     ← mayor caught this
  └── Document LM server setup + endpoint preflight
     │
     ▼ (schema fix + server docs complete)
Phase 1
  └── End-to-end smoke test: full flow on 1 problem
     │
     ▼
Phase 2 — LONG POLE (~1–2 h on M4 Max)
  └── Generate corpus: 200 problems × 8 trajectories
     │
     ▼
Phase 3
  └── Verify class balance (target: 15–50% correct in train split)
     │
     ▼
Phase 4 — parallel
  ├── Train MLP seed=42; check train-val gap ≤ 5pp
  ├── Train seeds 1 & 2 for ensemble statistics
  └── [conditional] Rollback to width=8 if gap > 5pp             ← mayor encoded this
     │
     ▼
Phase 5 — PRIMARY DELIVERABLE
  └── Full evaluation: selection accuracy, stratification, bootstrap CIs
     │
     ▼
Phase 6
  └── Reliability diagrams + per-step weight profile
     │
     ▼
Phase 7
  └── README per project rubric
```

Gas City executed phases 0–5 overnight.  I woke up to a trained MLP, evaluation results, and a
weight profile — with no babysitting.

#### Phase 3 — Diagnosis and interactive development with Claude Code

The overnight run surfaced a significant problem: all PRM step scores were a constant 0.50003338.
Evaluation was meaningless.  I used Claude Code for a systematic diagnosis session, working through
a structured elimination:

1. Ruled out dtype (float16 → bfloat16 → still degenerate)
2. Ruled out MPS fallback and attention implementation
3. Intercepted `apply_rotary_pos_emb` — found cos/sin all zeros
4. Inspected `rotary_emb.inv_freq` directly after model load — all zeros
5. Confirmed: `Qwen2RotaryEmbedding` is a non-persistent buffer, skipped by `load_state_dict`
   under transformers ≥ 5.0's meta-tensor initialisation; the values are materialised as zeros
   rather than computed from the RoPE formula

The fix — `_repair_rotary_embeddings()` in `TransformersProcessRewardModel` — recomputes `inv_freq`
and the cos/sin cache for every rotary layer immediately after `from_pretrained`.  This is a real
upstream bug affecting any transformers 5.x user loading Qwen2ForProcessRewardModel with
`trust_remote_code=True`.  The fix was contributed back to `its_hub` as part of this project.

Claude Code was essential here: it held the diagnostic thread across many tool calls, kept track of
what had been ruled out, and generated the repair function correctly on the first try once the root
cause was identified.  The human judgment required was knowing *which* assumption to question next —
the AI was a fast executor, not the strategist.

### Library and framework choices

| Tool | Role | Pros | Cons |
|------|------|------|------|
| **sdg_hub** | Corpus generation pipeline | Declarative YAML flows; block registry for composability; crash-safe staging via incremental writes | Block interface is rigid (DataFrames in/out); limited error propagation from inner blocks; debugging requires reading execution logs rather than stack traces |
| **transformers** | PRM inference | Broad model support; standard `AutoModel` API | Meta-tensor init bug in ≥5.x silently zeros RoPE buffers (see RoPE fix); `trust_remote_code=True` required for `Qwen2ForProcessRewardModel` adds security surface |
| **sklearn GradientBoostingClassifier** | GBDT aggregator | Fast; no GPU required; Gini importances are free interpretability | No sequential modelling; default hyperparameters used — a grid search would likely improve results |
| **math_verify** | Answer correctness labelling | Handles `\boxed{}` notation; broad equivalence checking for mathematical expressions | Strict matching produces false negatives on semantically equivalent forms (e.g. `1/2` vs `0.5` in some edge cases) |
| **PyTorch (MPS)** | MLP / LSTM training and PRM inference | Runs on Apple Silicon without CUDA | Not all PyTorch ops are MPS-supported; bfloat16 required to avoid NaN hidden states in deep models |

### Where AI accelerated the work

- **Ideation and scoping**: enumerating and scoring ten project ideas in 30 minutes, with explicit
  tradeoff documentation, would have taken significantly longer alone
- **Boilerplate and structure**: `features.py`, `model.py`, block scaffolding, YAML flows,
  `evaluate.py` stratification loops — generated correctly and quickly
- **Corpus pipeline debugging**: the JSONL schema mismatch and key naming collision were caught
  *before* a wasted 2-hour corpus run
- **Overnight autonomous execution**: Gas City ran phases 0–5 unattended, compressing a full day's
  work into a single overnight session
- **Diagnostic persistence**: Claude Code held the RoPE debugging thread across a long session and
  generated the repair function once the root cause was identified

### Where AI fell short or slowed the work

- **Dataset names**: Claude Code confidently named `lighteval/MATH` and `hendrycks/competition_math`
  as valid HuggingFace datasets for Level 1–4 problems; both were wrong.  A background process ran
  for several minutes before the error surfaced.  Verification (`load_dataset` probe) should be
  a reflex, not an afterthought.
- **Denominator bug in progress logging**: a loop counter was generated with `len(seen_problems)`
  inside the iteration body, causing the denominator to drift upward each step.  Small bug, but
  slipped through because the output looked plausible.
- **RoPE root cause**: the AI could not identify the root cause without being guided through the
  elimination.  It knew the transformers codebase broadly but not the specific meta-tensor
  initialisation behaviour of `Qwen2RotaryEmbedding`.  Human expertise was the bottleneck.

### How AI-generated code was reviewed

- Every non-trivial generated file was read before execution
- Scripts were syntax-checked (`ast.parse`) before running on real data
- Key numerical outputs (score distributions, train/val accuracy, weight profiles) were sanity-
  checked against expected ranges before accepting results
- A dry-run flag (`--dry-run` in `rescore_trajectories.py`) was used before overwriting corpus files
- The RoPE fix was validated with a diagnostic dummy call confirming non-degenerate scores before
  rescoring the full corpus

### Open-source contributions along the way

A core principle of working in open source is leaving dependencies better than you found them.
Two examples from this project:

- **its\_hub**: `TransformersProcessRewardModel._repair_rotary_embeddings()` fixes a silent
  transformers ≥ 5.0 compatibility bug that would affect any downstream user of
  Qwen2ForProcessRewardModel.  The fix is transparent — no API change, no performance cost.
- **Gas City**: Two bugs were filed and fixed during the evaluation:
  [PR #1290](https://github.com/gastownhall/gascity/pull/1290) and
  [PR #1291](https://github.com/gastownhall/gascity/pull/1291).

### Best practices for teams adopting AI-assisted development at scale

1. **Use AI for explicit tradeoff scoring before writing code.** The ideation session produced a
   ranked list with documented rationale.  This prevents the most common failure mode: starting
   the most *interesting* project rather than the most *tractable* one.

2. **Invest in the brief.** Gas City's quality of execution was directly proportional to the quality
   of `BRIEF.md`.  Vague briefs produce vague task graphs.  Time spent writing precise acceptance
   criteria pays back in unattended execution time.

3. **Treat dataset and API names as unverified until probed.** AI assistants hallucinate resource
   names with high confidence.  Build a verification step into every workflow that touches external
   data sources.

4. **Reserve AI for execution; use human judgment for strategy.** The RoPE diagnosis is the clearest
   example: Claude Code was an excellent executor of structured elimination but could not determine
   which assumption to question next without human direction.  The most productive sessions had a
   clear human-sets-direction / AI-executes loop.

5. **Contribute fixes upstream.** Bugs found during AI-assisted development are often real bugs
   affecting the broader community, not just your local setup.  The cost of filing a PR is low;
   the benefit to downstream users is high.

## What I'd Improve with More Time

**More training data.** 285 train problems is enough to show separation between learned models and baselines, but too few for reliable conclusions. The LSTM's narrowing gap to GBDT (7.1pp → 3.3pp as data doubled) suggests the ordering may flip around 1000 problems. The 62-problem test set means all observed differences are within 95% CI — directionally credible but not statistically significant.

**Hyperparameter search for GBDT.** The classifier was trained with manually chosen defaults (n_estimators=200, max_depth=3, lr=0.05). A grid search over `max_depth` and `n_estimators` would likely improve results, particularly on a larger corpus.

**Calibration evaluation.** The MLP and GBDT output scores in [0, 1] that are implicitly interpreted as P(correct). Calibration data was collected but never analysed. A miscalibrated aggregator may still rank correctly but gives misleading confidence estimates — worth checking before using scores for anything beyond ranking.

**Cross-policy transfer.** The strongest test of the aggregator is whether it generalises to trajectories from a policy it has never seen. Training on Qwen2.5-1.5B-Instruct and evaluating on a LoRA fine-tune would directly test whether the aggregator has learned intrinsic PRM geometry or policy-specific artefacts. The `Stretch` section sketches how to do this with `training_hub`.

**Online aggregation inside the search loop.** The current design scores completed trajectories at selection time. A per-step aggregator integrated into `ParticleFiltering` could prune unpromising beams before they complete — potentially more compute-efficient, at the cost of a more invasive interface change.

## Out of Scope

- Retraining the PRM
- Outcome reward models
- Fine-tuning the policy
- Cross-domain validation beyond one sanity check
- Modifying the search logic of ParticleFiltering or BeamSearch
- Online aggregation (per-step inside the loop)
