# How much compute do bundled training tricks save for small language models?

A preregistered scale ladder on one laptop GPU (0.4M–85M non-embedding parameters). **Draft v0, 18 October 2026, not peer reviewed.**
Paper draft: [`paper/main.pdf`](paper/main.pdf) · one-page summary: [`paper/summary_one_page.md`](paper/summary_one_page.md) · Turkish summary: [`README_TR.md`](README_TR.md)

Question: against a *learning-rate-tuned* AdamW baseline, how much compute does a bundle of recent tricks (canon layers, Muon-type
optimizers, QK-norm, hashed bigram embeddings, a warmup-stable-decay schedule, depth growth) save for GPT-style models, and does the
saving change as the model grows?

## Main results

| | 1.3M | 7.4M | 25M | 49M | 85M |
|---|---|---|---|---|---|
| Relative loss reduction at equal tokens, best recipe vs tuned baseline (slope-free) | ≈10% | ≈5% | 4.1% | 3.9% | 3.1% |
| Compute multiplier *m* (baseline's local slope b = 0.0548) | – | – | 2.73 / 2.68 | 2.69 | 2.32 |
| Compute multiplier *m* (steeper small-ladder slope b = 0.0932) | – | – | 1.99 / 1.97 | 1.99 | 1.83 |

* *m* depends on the assumed scaling slope; the preregistered "m ≥ 2" transfer rule at 85M holds only for b ≤ 0.0735.
* The gain does **not** grow with scale. Tuning only the baseline's learning rate is worth about 1.25–1.5× compute.
* All numbers in the paper's tables and figure are regenerated from the raw run files in `results/` by `scripts/make_report.py`
  (137 checks against the pipeline's own reports, 0 mismatches).

## Limitations

* Models ≤ 85M non-embedding parameters; one English web corpus and tokenizer; learning-rate tuning only.
* 49M and 85M are single runs; the baseline learning rate was at the edge of its grid at 25M and not re-tuned at 49M/85M;
  Recipe A's learning rate was never tuned.
* *m* is slope-sensitive (1.8–2.3 at 85M); FLOP accounting ignores optimizer iterations and hash-table memory.
* Several pipeline errors (a checkpoint-resume bug, an invalid mixture-of-experts arm, inconsistent slope conventions, wrong run
  timers) are documented in the paper's appendix.

## What this does not show

That the gains persist, grow or vanish above 85M parameters; that any single component is individually necessary; or that the recipe
helps downstream capability (the only downstream evaluation is five small multiple-choice benchmarks at 85M, with no conclusive difference).

## Layout

| Path | Content |
|---|---|
| `paper/` | `main.tex` (+ `tab_*.tex`, `fig_m_size.*`, `refs.bib`), compiled `main.pdf` |
| `results/` | raw run records: `main_ladder_runs.jsonl` (25M–85M: result, validation log and benchmark measurements per run), `small_ladder_runs.jsonl`, `t1_baseline_tuning.jsonl`, `f1a_results.json`, `f1b_reasoning_results.jsonl`, `f1k_knowledge_results.jsonl`, `queue_status_log.txt` (timestamped queue log, also holds the preregistration hashes); generated: `numbers.json`, `comparison_report_tr.md` (Turkish) |
| `scripts/make_report.py` | recomputes every number, table and the figure from `results/` (re-implements the formulas independently of the pipeline code) |
| `prereg/` | preregistration documents (Turkish originals, unchanged), `SHA256SUMS.txt`, `verify_hashes.py` |
| `code/` | training and measurement code as it was run (queue and orchestration scripts are not included) |

Names in logs and code are Turkish (project history): `taban` = baseline, `paket` = Recipe A, `paket2` = Recipe B, `kanit` = the 85M run with
a preregistered prediction, `buyuyen` = depth growth, `muonw` = MuonW, `muonh` = Hyperball, `S1/S2/S3` = 25M/49M/85M, `g2/g3/g4` = 1.3M/3.2M/7.4M.

## Reproduce the numbers, tables and figure (no GPU needed)

```
pip install numpy matplotlib
python scripts/make_report.py     # prints "KARSILASTIRMA: 137 / 137 tamam"; rewrites paper/tab_*.tex, paper/fig_m_size.*, results/numbers.json
python prereg/verify_hashes.py    # prints "SONUC: TUM HASH'LER ESLESTI"
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The script also writes timestamps into `results/numbers.json` and the report; the `tab_*.tex` files and the figure are otherwise identical between runs.
Do not edit `tab_*.tex` by hand.

## What cannot be reproduced with one command

The training runs. Each 85M run took 14–17 hours on one RTX 4060 Laptop GPU (8 GB). The training code in `code/` expects the ClimbMix shards
(`code/indir_climbmix.py` downloads the shuffled copy `karpathy/climbmix-400b-shuffle` from the Hugging Face Hub) and was written for the
author's Windows machine: paths are absolute and the code is not packaged as a library. Checkpoints are not included. The exact command lines
of the queue are not part of this repository; the recipe is described in the paper's Appendix A.

## Provenance and AI assistance

Most of the code and the experiment infrastructure were written and run by AI coding agents (Claude, Anthropic) under the author's direction;
the author set the goals and constraints, chose which experiments to run or stop, and reviewed the results. See the paper's statement on AI assistance.
Logs and code are published verbatim, including the author's local Windows paths.

## License

Apache-2.0 (see `LICENSE`).
