# npc-smarts

[![CI](https://github.com/earth-metabolome-initiative/npc-smarts/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/npc-smarts/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/earth-metabolome-initiative/npc-smarts/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/npc-smarts)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19701295.svg)](https://doi.org/10.5281/zenodo.19701295)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`npc-smarts` downloads the distilled `NPClassifier` train/validation/test splits from Zenodo and evolves one-vs-rest SMARTS for NPC pathway, superclass, and class labels with fixed GA defaults. The published train and validation splits are merged into one training pool, and the test split is held out for reporting.

## Run

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

By default, each training/test task set includes all positives and samples up
to 16384 negatives per NPC class. Override negative sampling with
`--max-negatives-per-npc-class`.

Labels with fewer than 50 training examples are filtered out by default. Override
with `--min-train-positives`.

The default GA evaluates 1024 SMARTS per generation for up to 500 generations,
with early stopping after 50 stagnant generations.

Results report both MCC and match coverage scores for the merged training pool
and held-out test split.

Generated SMARTS are restricted to the conservative PubChem-compatible subset
provided by `smarts-evolution`.

Slow SMARTS warnings are logged by default after 30 seconds. Each SMARTS
evaluation also has a cooperative 1 second time limit by default, and SMARTS
length can be capped before evaluation. Use
`--slow-evaluation-log-threshold-millis`, `--match-time-limit-millis`, and
`--max-evaluation-smarts-len` to tune evaluation guardrails. The run writes
these warnings to
`artifacts/slow-smarts.log` without colliding with the progress bars.
