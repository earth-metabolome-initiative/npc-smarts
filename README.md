# npc-smarts

[![CI](https://github.com/earth-metabolome-initiative/npc-smarts/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/npc-smarts/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/earth-metabolome-initiative/npc-smarts/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/npc-smarts)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19701295.svg)](https://doi.org/10.5281/zenodo.19701295)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`npc-smarts` downloads the distilled `NPClassifier` train/validation/test splits from Zenodo and evolves one-vs-rest SMARTS for NPC pathway, superclass, and class labels with aggressive fixed GA defaults. Validation is used to choose the final SMARTS from the evolved leader set.

## Run

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

By default, each train/validation/test task set samples up to 512 positives and
512 negatives per NPC class. Override with `--max-positives-per-npc-class` and
`--max-negatives-per-npc-class`.

Slow SMARTS warnings are logged by default after 30 seconds. The default
evaluation complexity cap is 1536, with a hard upstream SMARTS genome ceiling
of 4096. Use `--slow-evaluation-log-threshold-millis`,
`--max-evaluation-smarts-complexity`, and `--max-evaluation-smarts-len` to
tighten evaluation guardrails. The run writes these warnings to
`artifacts/slow-smarts.log` without colliding with the progress bars.
