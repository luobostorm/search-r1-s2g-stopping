# Search-R1 × S2G Stopping

Public reproduction package for:

> **When Should Multi-Round RAG Stop? Candidate Reachability, Policy Risk, and Retrieval Cost**
>
> Weimeng Luo, Independent Researcher

This repository adapts an S2G-style structured judge to a frozen Search-R1
pipeline, trains the judge on Search-R1 states, and evaluates whether retrieval
calls can be reduced while keeping the loss in answer accuracy within the
two-percentage-point range fixed before evaluation.

## Main result

On the 800-question confirmatory test set:

| Policy | Official EM | Mean search calls | Early stops |
|---|---:|---:|---:|
| Native Search-R1 | 0.44875 | 2.60125 | 0 |
| Expanded S2G LoRA policy | 0.44250 | 2.50500 | 69 |

The paired differences (expanded policy minus Native) were:

- Official EM: **-0.00625**, 95% CI **[-0.01250, 0]**. This stays inside the
  pre-specified -0.02 non-inferiority boundary.
- Search calls: **-0.09625**, 95% CI **[-0.12000, -0.07375]**: 77 fewer calls,
  or a 3.70% relative reduction.

This supports a narrow system-level claim: the complete trained-and-thresholded
policy reduced retrieval calls under the stated answer-quality tolerance. It
does **not** show improved answer quality, certified STOP safety, lower total
inference cost, or superiority to every fixed-budget baseline.

## Repository contents

- `kstar/`: official EM/F1 utilities and causal first-stop evaluation logic.
- `deploy/searchr1-v02/`: Qwen3.5 LoRA implementation, judge inference,
  teacher querying, training, and the frozen experiment protocol.
- `scripts/`: split/state preparation, leakage filtering, threshold selection,
  evaluation, and figure generation.
- `results/`: aggregate H0 and confirmatory outputs plus a public training
  summary. No per-question trajectories or labels are included.
- `paper/arxiv/`: English manuscript, Chinese cross-check manuscript,
  bibliography, and reproducible figures.
- `docs/`: frozen experimental protocols and the paper's claim boundary.
- `tests/`: synthetic unit tests that do not require benchmark labels or model
  weights.

## Quick start

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test,model]"
pytest -q
```

Regenerate the paper figures from the frozen aggregate JSON:

```bash
python scripts/generate_s2g_paper_figures.py
```

If [Tectonic](https://tectonic-typesetting.github.io/) is installed, build and
isolation-check the paper with:

```bash
make -C paper/arxiv final-pdf
make -C paper/arxiv check-isolated
```

## External assets not redistributed

The following must be obtained under their own licenses and terms:

- Search-R1 and its upstream retrieval stack;
- HotpotQA train and distractor development data;
- Qwen3.5-2B and Qwen3.6-35B-A3B model weights;
- the trained LoRA adapter (its frozen aggregate SHA-256 is recorded in
  `results/training-summary.json`).

To protect the evaluation firewall and avoid redistributing third-party or
operational material, this repository omits per-question trajectories,
benchmark labels, model weights, credentials, host information, and
infrastructure recovery logs.

## Provenance

This public snapshot was extracted from private research revision
`9f6dbaaa3541bbf01cd9503bf8cad5aa9f469489`. The quantitative figure generator
reads only the aggregate JSON committed here. See `PUBLICATION_SCOPE.md` for the
exact inclusion and exclusion boundary.

## License

No software license has been granted in this initial public snapshot. Public
availability alone does not grant reuse rights. A license can be added in a
later revision after the author chooses one explicitly.

## Contact

Weimeng Luo — [weimengluo@gmail.com](mailto:weimengluo@gmail.com)
