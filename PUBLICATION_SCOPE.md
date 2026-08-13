# Publication scope

## Included

- S2G/Search-R1 judge training, inference, thresholding, and evaluation code.
- Frozen scientific protocols and claim boundaries.
- Aggregate H0, threshold, training, and confirmatory results used by the paper.
- Paper source, bibliography, generated figures, and synthetic unit tests.

## Deliberately excluded

- Raw or per-question trajectories and prompts.
- HotpotQA questions, answers, supporting-fact labels, and derived label files.
- Base-model and LoRA weight files.
- API keys, passwords, access tokens, usernames, IP addresses, and host paths.
- Container orchestration, swap recovery, OOM incident logs, and remote-service
  recovery tooling.

The exclusions do not change the aggregate scientific result. They prevent
third-party data redistribution, avoid publishing operational credentials or
host details, and preserve the documented blind-evaluation boundary.
