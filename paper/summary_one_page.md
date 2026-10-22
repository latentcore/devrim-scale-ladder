# One-page summary (draft v0, 18 Oct 2026)

**How much compute do bundled training tricks save for small language models? A preregistered scale ladder on one laptop GPU (0.4M–85M parameters)**

**Question.** Efficiency tricks (new optimizers, schedules, architecture tweaks) are often reported to save 1.4–2× compute, but usually against untuned baselines and at one scale. How much does a *bundle* of such tricks save against a *tuned* baseline, and does the saving change as models grow?

**Setup.** GPT-style models from 0.4M to 85M non-embedding parameters, 20 training tokens per parameter, one RTX 4060 Laptop GPU (8 GB). Baseline: AdamW with a tuned learning rate. Recipe A: canon layers, MuonW, QK-norm, hashed bigram embeddings, warmup-stable-decay schedule. Recipe B: Recipe A with Hyperball in place of MuonW and depth growth (start at half depth). Metric: the compute multiplier *m* = how many times more FLOPs the tuned baseline needs to reach the same validation loss, plus a slope-free alternative (relative loss reduction at equal tokens). Pass/kill rules, sizes and metrics were written to dated, hashed files before the runs; deviations are logged.

**Results.**
- Against the tuned baseline, Recipe B reaches the same loss with *m* ≈ 2.7 (25M, two seeds), 2.7 (49M) and 2.3 (85M) under the baseline's local scaling slope (b = 0.055); under the steeper slope of the small ladder (b = 0.093) *m* ≈ 2.0, 2.0 and 1.8. At 85M the preregistered rule gives *m* = 2.2 ("preserved", threshold 2) with its pre-chosen slope, and "shrinking" with the other; the rule is slope-sensitive (it holds only for b ≤ 0.074).
- The gain does not grow with scale. Relative loss reduction at equal tokens: ≈ 10% (1.3M–3.2M), 5% (7.4M), 4.1% (25M), 3.9% (49M), 3.1% (85M). 49M and 85M are single runs; the 25M seed spread is 0.1 points.
- Tuning only the baseline's learning rate lowers its loss by 2.4% / 1.4% / 1.3% at 25M / 49M / 85M — worth 1.25–1.5× compute, i.e. an untuned baseline overstates any trick's gain by about that much.
- Negative results: of eight variants added to Recipe A, three were killed by their rules (Hyperball alone, a masked hybrid objective, learnability-based example selection), three gave no gain, two passed; of twelve first-generation variants (curricula, multi-token prediction, flat low-LR schedule, data pruning, …) only canon layers and MuonW gained at the largest small size, and they became part of the recipe.
- Synthetic one-seed probes: Recipe A improves out-of-distribution accuracy on length-generalisation tasks by 2–8 points and stores ≈ 1.8× more bits per non-embedding parameter; downstream benchmarks at 85M show no conclusive difference between the arms (at most 3.6 points, n = 1000).

**Limitations (what this does not show).** Models ≤ 85M, one English corpus and tokenizer; learning-rate tuning only (grid-edge optimum at 25M, not re-tuned at 49M/85M; Recipe A's LR never tuned); single runs at the two largest sizes; *m* depends on the assumed slope (1.8–2.3 at 85M); FLOP accounting ignores optimizer iterations and hash-table memory; several pipeline errors (a checkpoint-resume bug, an invalid mixture-of-experts arm, inconsistent slope conventions, wrong run timers) are documented in the appendix. Nothing here shows that gains persist, grow or vanish above 85M.

**Provenance.** Most code and experiment infrastructure were written and run by AI coding agents (Claude) under the author's direction; the author set the goals and constraints, chose which experiments to run, and reviewed results. All numbers are regenerated from raw run files by a script that re-implements the formulas independently (137 checks against internal reports, 0 mismatches).

*Author: Mert Yünlü, independent researcher (Ankara, Türkiye). Code, preregistration files and raw logs: [URL to be added by the author].*
