# GWO IDS — Feature Selection & Network Defense Lab

A final-year research application built around the supplied project's **Grey Wolf Optimization feature selection and machine-learning intrusion classification**. It includes a practical Streamlit analyst console, reproducible offline experiments, optional passive packet-flow inference, SQLite event history and human-reviewed incidents.

This update preserves the original NSL-KDD pipeline, UNSW-NB15 loader, GWO/bGWO variants, four classifiers, feature extractor and notebooks. It repairs confirmed defects and extends their integration; it is not an unrelated replacement scaffold. See [CHANGES.md](docs/CHANGES.md) and [VERIFICATION.md](docs/VERIFICATION.md).

**Scope:** local, single-analyst academic software. No automatic traffic blocking, internet attack generation, account system or production-grade detection claim. An ML prediction is not proof of an attack.

## Major-project demo

Start with [the demo guide, feature comparison and measured results](docs/MAJOR_PROJECT_GUIDE.md). The new **Research results** workspace displays measured baseline/selected comparisons, convergence charts, feature lists, confusion matrices and downloadable evidence. **Traffic lab** supports real localhost UDP measurement, bounded PCAP analysis and a separate authorized passive collector. Saved Random Forest observations can be explained during investigation.

To launch the optional full-trained, validation-selected 20-feature model with your existing Windows environment:

```powershell
$env:IDS_MODEL_PROFILE = "compact20"
.\.venv312\Scripts\python.exe -X faulthandler -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

The original model remains the default. Research thresholds, classifier scores and live alerts are distinct. No profile retrains from incoming observations.

## Quick start — Windows PowerShell

Install **Python 3.12 x64** with its Windows launcher. Extract the ZIP, open PowerShell inside `IDS-GWO-FeatureSelection`, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open **http://127.0.0.1:8501**. No React/Node build or separate API server is required: Streamlit is the existing frontend and application server. The delivered, newly trained Random Forest bundle is under `artifacts/deployment/`. If its dependency version differs, retrain using the command below. The application refuses incomplete or incompatible bundles instead of silently using them.

If you already created **`.venv312`**, use `.\.venv312\Scripts\python.exe` in place of `.\.venv\Scripts\python.exe` in these commands. The explicit executable determines the interpreter, even if the prompt still says `(.venv)`. Keep the dashboard terminal running. The reported Windows access violation used Python 3.14 with pandas 2.2.3; use the tested Python 3.12 setup for these pinned packages. No database reset is needed.

For a reproducible new model and persistent demo observations:

```powershell
.\.venv\Scripts\python.exe train.py --quick
.\.venv\Scripts\python.exe replay.py --limit 100
.\.venv\Scripts\python.exe realtime.py --selftest
.\.venv\Scripts\python.exe -m pytest -q
```

Stop any running collector before retraining. The dashboard remains usable without a trained model; inference explains how to train one. A failed training attempt may require retraining before inference resumes. The writer lock is released by the operating system when training exits, including process crashes. Its small lock file remains in place; do not delete it while processes are running.

Convenience wrappers are also provided:

```powershell
.\scripts\Setup.ps1
.\scripts\Train.ps1 -Quick
.\scripts\Replay.ps1 -Limit 100
.\scripts\Test.ps1
.\scripts\Start.ps1
```

If local execution policy blocks scripts, use the direct Python commands above. No policy or administrator changes are needed for offline analysis.

## Linux / macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python train.py --quick
.venv/bin/python replay.py --limit 100
.venv/bin/python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Core requirements are pinned. `requirements-lock.txt` records the exact clean Linux verification environment; Windows may add platform-specific dependencies. Jupyter is optional for the original notebooks and is not needed to run the application.

## What is connected

| Workspace | Real implementation |
|---|---|
| Overview | SQLite totals, recent alert table, activity chart, collector heartbeat and source labels |
| Events & investigations | Parameterized filters, pagination, feature evidence, incident transitions, notes, audit history and CSV export |
| CSV analysis | Validated feature CSV → persisted encoders/scaler → GWO subset → trained classifier → downloadable predictions → optional idempotent event save |
| Model lab | Real held-out metrics, confusion matrix, false-positive/negative rates, selected features, feature importance, experiment history and quick training |
| Research results | Both measured studies, per-class metrics, convergence, evidence downloads, calibration trade-offs and attack-family audits |
| Traffic lab | Real measured UDP loopback traffic, classic PCAP upload, passive capture instructions and independently labelled event history |
| Setup & methodology | Actual commands, paths, permission requirements and research limitations |

No synthetic live counters are displayed. Every event is labelled `live`, `loopback_probe`, `pcap_replay`, `dataset_replay`, `csv_import`, `selftest`, or `legacy`. CSV analysis performs real inference, even when using the supplied benchmark sample. Synthetic self-tests validate plumbing and **do not evaluate detection accuracy**.

## Experiments

### Matched research benchmark (does not replace the dashboard model)

```powershell
.\.venv312\Scripts\python.exe benchmark.py --fit-limit 4800 --validation-limit 1200 --wolves 10 --iterations 20 --seeds 42 43 44 --jobs 2
# Or the wrapper, which selects .venv312 first and verifies Python 3.12:
.\scripts\Benchmark.ps1
```

This compares **42 offline features** against the **35 runtime features** on identical rows. Each pool has an all-feature RF baseline and three bGWO seed results. GWO uses 4,800 training-fit and 1,200 training-validation rows; final RF models fit the full cleaned training split and evaluate the same remaining held-out test rows. Every selected mask is frozen before final-test evaluation. The existing fitness and RF settings are reused, not retuned to force a favorable result.

The command creates a new directory under `outputs/benchmarks/` with the frozen protocol, input/code hashes, split counts, selection records, run log, per-class metrics, confusion matrices, CSV and chart. Existing output directories are refused. It never writes the deployment bundle, event database, or dashboard experiment history. These research results are therefore **not** the metrics of the active quick-trained dashboard model.

See [MATCHED_BENCHMARK.md](docs/MATCHED_BENCHMARK.md) for the measured results, methodology and limits. An exact reproduction of the friend's `NIDS-UNSW` experiment still needs that source/configuration; matching the dataset name and iteration count is insufficient.

### Existing deployment training

```powershell
# UNSW-NB15 binary deployment pipeline; complete supplied presplit dataset:
.\.venv\Scripts\python.exe train.py
# Bounded experiments with the original four classifier options:
.\.venv\Scripts\python.exe train.py --quick --classifier random_forest
.\.venv\Scripts\python.exe train.py --quick --classifier decision_tree
.\.venv\Scripts\python.exe train.py --quick --classifier knn
.\.venv\Scripts\python.exe train.py --quick --classifier svm
# Original NSL-KDD multiclass research pipeline (does not replace the deployment bundle):
.\.venv\Scripts\python.exe main.py --quick
```

Each UNSW training command updates the deployment model; restore Random Forest by training it last if desired. Model lab retains completed experiment reports. Do not select a winner from repeatedly inspected final-test scores and then claim that score is an unbiased estimate. Use nested validation or a new untouched test set for comparative research.

Quick UNSW mode uses 3,000 stratified training records and up to 1,000 test records, 4 wolves, 3 iterations, 1 run and 40 Random Forest trees. NSL quick mode retains at least ten available real training records per rare class, so its count is approximately 3,000. Full runs use `config.py` settings and can take substantially longer.

The objective remains the original `alpha * accuracy_percent + beta / selected_feature_count`; accuracy dominates because it is expressed in percentage points. This is explicitly documented, not presented as a newly validated optimal objective. GWO does not guarantee either dimensionality reduction or improved accuracy.

## Data and artifacts

The package includes the original presplit UNSW-NB15 and NSL-KDD CSVs, the feature-definition CSV, and a 50-row feature-only sample. It excludes the redundant ~586 MB four-part UNSW raw release, processed copies, caches, old unverified pickles and local databases. These remain in the original upload; no original data was erased. To use full raw mode, copy the four original files to `dataset/raw_unsw_full/` and set `IDS_DATA_SOURCE=raw_full`.

Data lineage and redistribution considerations are described in [DATA.md](docs/DATA.md). Do not substitute private production traffic without authorization and a retention policy. Fresh model hashes and the scikit-learn version are checked before loading. Hashes detect corruption or mixed artifacts; they do not authenticate a malicious model. Only load trusted, locally trained joblib files.

Outputs:

- `artifacts/deployment/`: model, encoders, scaler, selected features, schema, version and measured metrics.
- `outputs/reports/runs/`: completed experiment JSON reports.
- `outputs/graphs/`: regenerated confusion matrices and feature charts.
- `logs/alerts.db`: runtime observations, reviews, review audit and collector status; created on demand.

## Optional live capture

```powershell
.\.venv\Scripts\python.exe realtime.py --list-ifaces
.\.venv\Scripts\python.exe realtime.py --iface "Wi-Fi"
```

On Windows, install Npcap and use appropriate capture privileges. On Linux, use an authorized capture-capable account. **Keep the dashboard as a normal user and bound to localhost.** A single collector is supported per database. Monitor only an interface/network you own or are authorized to inspect.

The collector handles IPv4 TCP/UDP. Flows close on FIN/RST, idle timeout or active timeout. Memory is bounded; dropped packets at capacity are surfaced in collector status. Timing units are corrected to milliseconds where the supplied UNSW definition requires them. Several features still approximate Argus definitions; offline accuracy does **not** prove live-network accuracy. See [RUNTIME_FEATURE_AUDIT.md](docs/RUNTIME_FEATURE_AUDIT.md).

## Configuration

Environment values are read by `config.py`; no secret is required. A `.env` file is not automatically loaded.

| Variable | Default | Purpose |
|---|---|---|
| `IDS_DB_PATH` | `<project>/logs/alerts.db` | SQLite location; parent directory created automatically |
| `IDS_MODEL_PROFILE` | `deployment` | Select deployment or the separate compact20 profile |
| `IDS_MODEL_DIR` | `<project>/artifacts/deployment` | Trusted model bundle location |
| `IDS_DATA_SOURCE` | `presplit` | `presplit` or optional `raw_full` training input |
| `IDS_INTERFACE` | `eth0`, with detected fallback | Capture interface; prefer explicit `--iface` |

Example: `$env:IDS_DB_PATH = "$PWD\logs\demo.db"`. Restart relevant processes after changing environment values. Other hyperparameters remain in `config.py`. Existing SQLite events are preserved through additive columns and tables; do not delete a database to upgrade.

## Verification and submission materials

- [Actual verification results and limitations](docs/VERIFICATION.md)
- [Architecture, data flow and schema](docs/ARCHITECTURE.md)
- [Demo walkthrough and report outline](docs/PROJECT_REPORT.md)
- [Changes and preserved features](docs/CHANGES.md)
- [Runtime feature audit](docs/RUNTIME_FEATURE_AUDIT.md)
- [Original documentation, retained as historical reference](docs/original/README-original.md)

The original documentation may contain outdated paths and claims; this README and the current verification report take precedence. The system is demonstration-ready for the verified offline workflow, not certified, production-ready or guaranteed to detect every attack.
