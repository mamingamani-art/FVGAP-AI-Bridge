# FVGAP AI Bridge

Private repository for the FVGAP trading bridge and its Phase 5 model package.

## Files

The repository root contains the bridge source (Phase5_Bridge.py), the original strategy source, the multi-trade MT4 Expert Advisor source, and a ZIP archive with the complete source and inference assets.

Download and extract FVGAP-AI-Bridge-source.zip to get the model folder, column definitions, EA source and binary, README, and validation results in their expected directory structure.

## Run

The VPS executable Phase5_Bridge.exe is not stored in Git because it is about 406 MB and exceeds GitHub's 100 MB per-file limit. The Python source and model inference assets are in the ZIP. A Windows build can be recreated from Phase5_Bridge.py using the original compatible Python/TensorFlow environment; the historical packaged executable is not reproducibly pinned here.

For the prebuilt VPS workflow, see README_Fast_FA.txt inside the ZIP. Keep the model folder beside the bridge executable (or use --models PATH). The bridge binds to 127.0.0.1:8000 and serves /predict and /health.

This model package currently has BUY category 1; SELL requests return WAIT. Backtest and speed measurements are historical, hardware and broker conditions vary, and they are not a guarantee of future returns. Use a demo account until independently validated.

## Data policy

Training datasets, Optuna databases, backup databases, and out-of-fold caches are excluded. They are not needed for inference and can contain large or non-portable research data.
