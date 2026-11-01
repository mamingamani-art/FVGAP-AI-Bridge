# FVGAP AI Bridge

Private repository for the FVGAP trading bridge and its Phase 5 model package.

## Contents

- `Phase5_Bridge.py` and `phase5_columns.json`: optimized local HTTP bridge for MT4 on port 8000.
- `BEST_Trial257_KPI_0.0075_2026-10-07_23-31-58/`: inference model weights and required preprocessing assets.
- `MQL4/`: multi-trade Expert Advisor source and compiled MT4 binary.
- The original strategy Python source and local speed-validation results.
- `FVGAP-AI-Bridge-source.zip`: a complete copy of the included source and inference assets.

## Run

The VPS executable `Phase5_Bridge.exe` is not stored in Git because it is about 406 MB and exceeds GitHub's 100 MB per-file limit. The Python source and model assets are included. A Windows build can be recreated from `Phase5_Bridge.py` using the original compatible Python/TensorFlow environment; the historical packaged executable is not reproducibly pinned here.

For the prebuilt VPS workflow, see `README_Fast_FA.txt`. Keep the model folder beside the bridge executable (or use `--models PATH`). The bridge binds to `127.0.0.1:8000` and serves `/predict` and `/health`.

This model package currently has BUY category `1`; SELL requests return `WAIT`. Backtest and speed measurements are historical, hardware and broker conditions vary, and they are not a guarantee of future returns. Use a demo account until independently validated.

## Data policy

Training datasets, Optuna databases, backup databases, and out-of-fold caches are excluded. They are not needed for inference and can contain large or non-portable research data.
