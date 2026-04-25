# Learned Trajectory Aggregation for Process Reward Models

A contribution to [its_hub](https://github.com/Red-Hat-AI-Innovation-Team/its_hub) addressing the open `TODO(GX)` at `its_hub/base.py:104`.

## 1. Problem statement

In its_hub, the algorithms that consume process reward model (PRM) signals (`ParticleFiltering`, `BeamSearch`) reduce a sequence of per-step scores to a single trajectory score using a hardcoded choice of `prod`, `min`, or `mean`. The maintainers have flagged this with an explicit `TODO(GX) deal with aggregation of PRM scores` at `its_hub/base.py:104`.

None of the three reductions is a defensible default. Reading each as a probabilistic claim about trajectory correctness:

`prod` interprets per-step scores as $P(\text{step}_t\ \text{correct})$ under a step-independence assumption. In log-space this is the standard particle-filter accumulation. Its pathology is length: a long, mostly-correct trajectory loses to a short, mostly-correct one because more terms below 1 always shrinks the product. Long-form reasoning is systematically penalized.

`min` reads as a worst-case bound. Its pathology is brittleness: one underweighted step destroys an otherwise correct trajectory, and the choice ignores everything outside the bottleneck.

`mean` is length-invariant but discards both step position and inter-step dependence. It treats setup steps and conclusion steps as equally informative. Whether they are is an open question that section 2 takes up explicitly.

The library forces a global choice among these three. The right choice almost certainly depends on policy, problem domain, trajectory length, and PRM calibration. Worse, the architecture does not even expose aggregation as a pluggable component, so a user who wanted to try their own would need to modify algorithm code.

The contribution targets both layers of the gap. First, introduce an `AbstractTrajectoryAggregator` interface so aggregation becomes pluggable. Second, ship a learned reference implementation that demonstrates the interface buys real accuracy.

## 2. Hypothesis and falsification criteria

**Primary claim.** A learned aggregator over per-step PRM scores outperforms `prod`, `min`, and `mean` on trajectory selection accuracy at fixed budget. The gap concentrates on long trajectories (where `prod` underflows) and on trajectories whose worst step is followed by recovery (where `min` discards the recovery signal).

**Secondary claim.** Inspecting the learned aggregator reveals that late-step scores carry materially more predictive weight than early-step scores, formalizing the intuition that early reasoning steps are mostly setup and that errors near the conclusion matter most.

**Conditions under which the contribution becomes a negative result.** Any of:

1. The three baseline reductions perform within seed variance of each other on the held-out set. This would imply aggregation choice does not matter at the data scale we can afford, defeating the premise.
2. The learned aggregator fails to exceed the best baseline by more than 2x the bootstrap confidence interval half-width.
3. The learned weights are flat with no positional structure.

A negative result is still a meaningful contribution because the interface change stands on its own and the empirical study itself becomes the artifact: a public, controlled benchmark of the three reductions across length and difficulty strata, which the library presently lacks.

## 3. Out of scope

The following are explicitly declined and called out in the README as such:

- Retraining the PRM. The aggregator operates on PRM outputs as given.
- Outcome reward models. Aggregation is meaningful only for step-sequence signals.
- Fine-tuning the policy. Trajectories are produced by a fixed policy.
- Cross-domain validation beyond one sanity check. Math reasoning only.
- Modifying the search logic of `ParticleFiltering` or `BeamSearch` themselves. They become aggregator-aware via a constructor parameter, nothing more.
- Online aggregation that updates per step inside the search loop. Aggregators here are evaluated once per completed trajectory.

## 4. Method and baselines

The aggregator is a function $f(s_1, \dots, s_T) \to \mathbb{R}$ mapping a variable-length sequence of per-step scores to a scalar trajectory score. Variable-length is required because trajectories in the corpus span roughly 3 to 15 steps depending on problem difficulty.

**Baselines.** All three current reductions plus a random floor:

| Baseline | Definition |
|---|---|
| `prod` | $\sum_t \log s_t$ |
| `min` | $\min_t s_t$ |
| `mean` | $T^{-1} \sum_t s_t$ |
| `random` | $U[0,1]$ per trajectory, sanity floor |

**Models considered for the learned aggregator.**

1. *MLP over fixed-length featurization.* Summarize the trajectory by a small feature vector: mean, min, max, last-step score, length, score variance, position of min, position of max, gap between first and last score, gap between adjacent scores at the minimum position. Pass through a 2-layer MLP. Most interpretable, smallest data appetite.
2. *LSTM over the raw step-score sequence.* Final hidden state projects to a scalar. Captures sequential structure but harder to attribute findings to specific positions.
3. *Set or sequence transformer with positional encoding.* Strongest expressivity. Included only if both the MLP and a small LSTM ablation hit the saturation criterion defined in section 7 (validation accuracy stable across MLP capacity scaling, errors concentrated on the trajectory-shape buckets the fixed featurization compresses, and a small LSTM beating the MLP on those buckets). At ~4k trajectories the transformer is unlikely to clear that bar because it needs more data to learn its internal representations than the corpus supports.

**Starting choice.** MLP. The trajectory length distribution is short (median around 6 steps), the labeled data budget is modest, and the interpretability demand favors fixed-length features whose weights are inspectable.

**Training objective.** Binary cross-entropy against trajectory correctness, with the aggregator's output passing through a sigmoid. A pairwise rank loss is a natural alternative since the deployment use is argmax-over-N selection rather than absolute probability assessment. We do not run this ablation in the primary submission for two reasons. First, the output of a pure rank loss is not a meaningful probability, which would invalidate the calibration metrics reported in section 7. Second, in argmax-selection settings with balanced labels and small N, BCE and pairwise rank losses typically perform within seed variance of each other, so the ablation is unlikely to change the contribution's defensibility. We list it as future work.

## 5. Architectural placement

New module: `its_hub/aggregators/`.

New abstract class in `its_hub/base.py` alongside the existing reward model abstractions:

```python
class AbstractTrajectoryAggregator(abc.ABC):
    @abc.abstractmethod
    def aggregate(self, step_scores: list[float]) -> float: ...
    async def aaggregate(self, step_scores: list[float]) -> float: ...
```

Concrete implementations:

- `HardcodedAggregator(reduction: Literal["prod", "min", "mean"])` wrapping the current behavior. This becomes the default and reproduces current numerics exactly.
- `LearnedMLPAggregator(checkpoint_path: str)` loading a trained MLP and running a forward pass.
- `LearnedLSTMAggregator(checkpoint_path: str)` only if added.

`ParticleFiltering` and `BeamSearch` gain an optional `aggregator` parameter that defaults to `HardcodedAggregator("prod")`. Existing user code is unchanged.

Training scaffolding lives outside the library, in `scripts/train_aggregator.py`. Trained checkpoints ship as small files under `its_hub/aggregators/checkpoints/`.

The shape is deliberate. The contribution is the interface plus a reference implementation, not a single trained model. A user who wants their own aggregator subclasses `AbstractTrajectoryAggregator` and changes nothing else.

**Secondary contribution: Apple Silicon support for PRM scoring.** The existing `LocalVllmProcessRewardModel` requires CUDA via vLLM, which excludes Apple Silicon hardware. To run the trajectory-scoring step of section 6 on M4 Max, we add `MLXProcessRewardModel`, a sibling implementation of `AbstractProcessRewardModel` backed by MLX with 4-bit quantized weights. This is not a contribution to the aggregator interface but it is a contribution to the library: it removes a hardware constraint that affects any user trying to develop or test PRM-consuming algorithms locally on a Mac. `LocalVllmProcessRewardModel` continues to work unchanged on its existing CUDA path.

**Trajectory corpus pipeline as an sdg_hub flow.** Trajectory generation, PRM scoring, answer extraction, and correctness verification are expressed as an sdg_hub flow living at `learned-aggregator/flows/trajectory_corpus.yaml` rather than as a one-off Python script. This integrates the second library required by the assignment and makes corpus construction composable, reproducible, and amenable to swapping in different policies, PRMs, or filters without rewriting glue code. The flow uses three built-in blocks (`PromptBuilderBlock`, `RowMultiplierBlock`, `LLMChatBlock`) and adds two project-specific blocks: `MLXProcessRewardScoreBlock` (delegates to the `MLXProcessRewardModel` from above to score per-step) and `MathVerifyAnswerBlock` (extracts `\boxed{...}` and runs `math_verify` against ground truth). The two custom blocks ship in the project repo as a small secondary contribution to the sdg_hub block ecosystem.

## 6. Data plan

**Question source.** Math problems from the MATH dataset's train split. The MATH500 evaluation set is held entirely out of trajectory generation so that section 7's evaluation is uncontaminated.

**Trajectory generation protocol.** Implemented as the sdg_hub flow described in section 5 (`learned-aggregator/flows/trajectory_corpus.yaml`). Block sequence:

1. `PromptBuilderBlock`: wraps each MATH problem in the math system prompt.
2. `RowMultiplierBlock`: fans out to $N = 8$ trajectories per problem.
3. `LLMChatBlock`: rolls out the policy (Qwen2.5-1.5B-Instruct) at temperature 0.7. Step boundary is `"\n\n"`, consistent with its_hub's existing `StepGeneration` convention so trajectories are interoperable with downstream its_hub algorithms.
4. `MLXProcessRewardScoreBlock` (custom): scores each step with Qwen2.5-Math-PRM-7B at 4-bit, delegating to the `MLXProcessRewardModel` from section 5. Per-step scores persist as a list field on the row.
5. `MathVerifyAnswerBlock` (custom): extracts `\boxed{...}` via regex and runs `math_verify` against the problem's ground-truth answer. Emits both the extracted answer and a boolean correctness label.

The output dataset has columns `problem`, `trajectory_text`, `step_scores`, `extracted_answer`, `correct`, plus the original problem metadata (difficulty level). This is the input to aggregator training and to the held-out evaluation in section 7.

**Corpus size.** 200 problems times 8 trajectories per problem yields 1,600 trajectories, with roughly 1,120 in the training split after a 70/15/15 problem-level partition. The reduction from a notional 500-problem corpus is compute-driven, not statistical: PRM scoring on M4 Max via the 4-bit MLX wrapper runs at roughly 0.3 to 0.7 seconds per step, and 200 problems keeps wall-clock for the scoring pass under two hours at the upper end of that range. The MLP is correspondingly reduced to hidden width 16 (about 465 parameters), which keeps the absolute hypothesis space modest at the smaller absolute data scale. The sample-to-parameter ratio is essentially unchanged from a 500-problem regime; both regimes sit well below the classical 10:1 heuristic, and the actual safety mechanisms are not the headline ratio. Three factors carry that load. The input space is a fixed 10-dimensional feature vector rather than a learned representation, so effective capacity is materially below raw parameter count. Training uses early stopping on validation loss, which determines the effective parameter count actually fit to the data. The train-validation gap is reported in section 7, and capacity is rolled back to width 8 or weight decay is added if the gap exceeds 5 percentage points in absolute accuracy.

**Scaling note.** The experiment scales linearly to a full 500-problem corpus given a vLLM-capable host, since the only Apple-Silicon-driven constraint is PRM scoring throughput. The README will document the larger-scale reproduction recipe so reviewers with CUDA hardware can rerun at full size without code changes.

**Class balance.** A 1.5B policy at temperature 0.7 on MATH-train tends to land near 30 percent correct, which is a healthy mix. If yield is heavily skewed during the initial sample, the mitigation is to add easier problems and increase $N$ on hard ones.

**Splits.** 70/15/15 train/validation/test on the problem axis. All trajectories from a given problem stay in one split. Splitting on trajectories rather than problems would leak the question text across splits and inflate metrics.

**What ships in the repo.** Not the trajectories themselves; they are too large. The deterministic generation script ships, with a fixed seed and pinned model versions. A small sample of trajectories ships for unit tests.

## 7. Evaluation protocol

**Primary metric: selection accuracy.** For each held-out problem, score all $N$ candidate trajectories with the aggregator, return the highest-scoring trajectory's final answer, mark correct or incorrect. Report the fraction of problems where the selected answer is correct.

**Why selection accuracy and not aggregator AUC.** The library uses the aggregator to select inside `ParticleFiltering` and `BeamSearch`. An aggregator that ranks well while being poorly calibrated still helps the library; an aggregator that is calibrated but ranks poorly does not. AUC tracks ranking but selection accuracy tracks the decision the algorithm actually makes downstream.

**Stratification.** Three axes:

- Trajectory length, binned: short ($\leq 4$ steps), medium (5 to 9), long ($\geq 10$). Directly tests the `prod`-underflow hypothesis.
- Problem difficulty, MATH levels 1 through 5.
- Candidate count $N \in \{4, 8, 16\}$. Tests whether the aggregator gap widens or narrows as the selection pool grows.

**Statistical treatment.**

- Three random seeds for the learned aggregator's training. Report mean and standard deviation.
- Bootstrap 95 percent confidence intervals on selection accuracy, since per-problem outcomes are Bernoulli and the held-out set is small.
- A claimed gap counts only if it exceeds twice the bootstrap CI half-width.

**Saturation criteria for promoting to a sequence model.** The MLP is the primary model; promoting to LSTM or transformer requires all three of:

1. *Capacity check.* Train MLP at hidden widths 8, 16, 32 with the same regularization (sweeping around the new default of 16). If validation accuracy is statistically indistinguishable across the three (all bootstrap 95% CIs overlap), the MLP is not capacity-limited. If accuracy keeps climbing, scale up the MLP first; capacity, not architecture, is the issue.
2. *Representation diagnostic.* Bucket MLP validation errors along the axes the fixed featurization compresses (trajectory length, score variance, position of min, gap between adjacent step scores). If errors concentrate disproportionately on long or high-variance trajectories where the fixed features lose information, the representation is plausibly the bottleneck. If errors are uniform across buckets, more model will not help.
3. *LSTM ablation.* Train a small LSTM (one layer, hidden size 32 to 64). If it beats the MLP by more than twice the bootstrap CI on the same held-out problems, sequence structure matters. The transformer is reached only if both the MLP and the LSTM saturate by the first two checks above.

**Secondary metrics.**

- Expected calibration error against trajectory correctness, with reliability diagrams comparing the learned aggregator's sigmoid output to the implied probability of `prod`.
- Per-step weight profile. For the MLP, this is the input-layer weights tied to position-indexed features. For the LSTM, gradient-times-input attribution per step.

**Comparison protocol.** The PRM is run once per trajectory; trajectories and step scores are cached on disk. All aggregators consume the same cached inputs deterministically. This isolates aggregator effect from any noise in trajectory sampling, which would otherwise confound the comparison entirely.

## 8. Success criteria, tiered

The project ships at any of three levels. Each tier represents a defensible submission, with later tiers strictly increasing the strength of the contribution.

**Minimum.** Everything required to call the project complete:

- `AbstractTrajectoryAggregator` plus `HardcodedAggregator` wrapping current behavior, with `ParticleFiltering` accepting it as a parameter and the default reproducing current numerics.
- The sdg_hub trajectory corpus flow runnable end-to-end on at least one problem, with the two custom blocks (`MLXProcessRewardScoreBlock`, `MathVerifyAnswerBlock`) implemented and tested.
- One trained `LearnedMLPAggregator` checkpoint.
- Selection accuracy comparing all three baselines plus the learned MLP on at least one stratification slice with confidence intervals.
- README sections required by the assignment rubric.

**Target (the realistic ship goal).** Everything in the minimum, plus:

- Full stratification across length, difficulty, and $N$.
- Calibration measurement with reliability diagrams for the learned aggregator and `prod`.
- Per-step weight profile inspection, with a written interpretation of what the profile says about which steps matter.
- Discussion of where each baseline wins. The point is not that the learned aggregator dominates everywhere; the point is that the right aggregator depends on regime.

**Stretch.** Anything beyond target that strengthens defense without distracting from the core claim:

- LSTM aggregator side-by-side with MLP, conditional on the saturation criteria in section 7.
- Cross-policy transfer using training_hub: produce policy B by LoRA-fine-tuning Qwen2.5-1.5B with training_hub on a small targeted dataset, then evaluate the aggregator (trained on policy A's trajectories) on policy B's. This brings the third library into the contribution. The cost is real: training_hub's backends assume CUDA, so on M4 Max this is high-risk and may need to be skipped if executing it would compromise the core deliverable. We attempt it only after the target tier is shipped.

The plan is to aim at target. Minimum exists so a partially completed project still ships cleanly under unforeseen pressure. Stretch exists so there is somewhere defined to go if data collection runs ahead of schedule.

## 9. Future work

Items beyond the success-criteria stretch tier, flagged as future work in the README:

1. **Cross-domain transfer.** Train aggregator on math, evaluate on GSM8K (the closest reasoning-style sibling). The hypothesis is that learned weights are domain-portable to the extent that step-position structure is universal across chain-of-thought.

2. **Streaming aggregation inside the particle filter.** Currently the aggregator runs once per complete trajectory. A streaming variant could replace the per-step log-weight accumulation inside PF resampling. This is an algorithmic change to PF rather than an aggregator contribution and belongs in a follow-up.

3. **Distilled aggregator-PRM head.** The MLP sits on top of PRM outputs. A natural follow-up trains a single end-to-end scoring head that subsumes both. Lower inference cost, but loses the pluggable aggregator interface, and so cuts against the architectural point of this contribution.

4. **Training-free aggregators.** The MLP needs labeled data. A length-corrected `prod` (the geometric mean, equivalent to length-normalized log-prod) is a single line of code, requires no checkpoint, and may be competitive in some regimes. If competitive, it should ship as a recommended default for users without labeled trajectories. Worth a small ablation.

5. **Pairwise rank loss.** Section 4 discusses why this was deferred. If selection accuracy and ranking metrics diverge in unexpected ways on the held-out evaluation, this becomes higher priority.

None of these block the primary contribution. They are listed so reviewers can see the contribution sits inside a coherent research direction rather than a one-off.
