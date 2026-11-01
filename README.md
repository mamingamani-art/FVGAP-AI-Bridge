# FVGAP AI Bridge

Private repository for the FVGAP trading bridge and its Phase 5 model package.

## Files

The repository root contains the bridge source (Phase5_Bridge.py), original strategy source, multi-trade MT4 Expert Advisor source, and two ZIP archives.

- [FVGAP-AI-Bridge-source.zip](FVGAP-AI-Bridge-source.zip) contains the bridge source, model inference assets, EA source and binary, README, and validation results.
- [BEST_Trial257_KPI_0.0075_2026-10-07_23-31-58.zip](BEST_Trial257_KPI_0.0075_2026-10-07_23-31-58.zip) contains the complete Trial 257 folder, including model artifacts, reports, charts, Optuna database and backups, and out-of-fold files.

## Run

The VPS executable Phase5_Bridge.exe is not stored in Git because it is about 406 MB and exceeds GitHub's 100 MB per-file limit. The Python source and model inference assets are in the source ZIP. A Windows build can be recreated from Phase5_Bridge.py using the original compatible Python/TensorFlow environment; the historical packaged executable is not reproducibly pinned here.

For the prebuilt VPS workflow, see README_Fast_FA.txt inside the source ZIP. Keep the model folder beside the bridge executable (or use --models PATH). The bridge binds to 127.0.0.1:8000 and serves /predict and /health.

This model package currently has BUY category 1; SELL requests return WAIT. Backtest and speed measurements are historical, hardware and broker conditions vary, and they are not a guarantee of future returns. Use a demo account until independently validated.

## Data policy

The Trial 257 archive includes Optuna databases and backups, along with out-of-fold files. Training datasets are not included.
