# Assessment and changes

## Major-project release — 1 November 2026

See [MAJOR_PROJECT_GUIDE.md](MAJOR_PROJECT_GUIDE.md) for the feature-parity audit, real input modes, experiments, model comparison and Windows guide. The existing default model is preserved; the optional compact20 profile is separately trained and versioned. The original project remains a local investigation tool. New event columns are additive. See update-from-v4.patch for source deltas from the previous ZIP; source-changes.patch remains the earlier change record.

## Matched benchmark and Python 3.12 dashboard verification

- `benchmark.py`: a separate reproducible offline comparison of 42-feature and 35-feature RF/bGWO experiments. Reuses existing preprocessing, classifier factory and optimizer without modifying deployment training or inference. Shared data, grouped validation, fixed settings, three optimizer seeds, input/code hashes, measured reports and a chart; no model promotion or event writes.
- `scripts/Benchmark.ps1`: optional Python 3.12 runner; prefers the user's `.venv312` environment, with an explicit interpreter override.
- `tests/test_benchmark.py`: eight cases covering a real small end-to-end benchmark, label-independent overlap rejection, no final-test influence on selection, preservation of model/database files, existing-output refusal and invalid settings.
- `tests/test_dashboard.py`: populated Overview regression covering the datetime conversion path, repeated timestamps, refresh and navigation.
- `scripts/browser_smoke.cjs`: persist real CSV predictions, revisit populated Overview, open two sessions, exercise refresh, verify health and retain existing mobile checks.
- `README.md`, `docs/MATCHED_BENCHMARK.md`, `docs/BENCHMARK_RESULTS.md`, `docs/VERIFICATION.md`, `docs/ARCHITECTURE.md`, `docs/PROJECT_REPORT.md`: current commands, measured results, comparison limits, research/runtime separation and Windows verification boundary.
- Package hashes and cumulative source patch refreshed. No application code, requirements, configured fitness, deployment artifacts or database schemas changed in this stage.

## SQLite cleanup follow-up

- `src/runtime/alerting.py`: close initialization connections explicitly and close connections if foreign-key setup fails. Preserve existing transaction and schema behavior.
- `tests/test_store.py`: five connection-lifecycle regression cases; close the historical-schema fixture connection.
- `docs/VERIFICATION.md`: record the reproduced failures and actual 52-test result.
- Refresh `docs/source-changes.patch` and `FILE_MANIFEST.sha256` for the updated package.

## Inspected baseline

The supplied ZIP already contained a Streamlit dashboard, NSL-KDD and UNSW-NB15 pipelines, standard/binary GWO, Random Forest/Decision Tree/SVM/kNN, packet-flow inference and SQLite logging. No test suite was present. The original code compiled, but database initialization failed with `OperationalError: unable to open database file`. A controlled optimizer initialization test made zero objective calls and returned a zero score.

No crash screenshots were needed to reproduce these specific defects. They are confirmed problems, not a claim that every earlier crash had the same cause.

## Core repairs

- Portable database location with parent creation, parameterized queries and additive schema upgrades.
- GWO evaluates starting masks and every changed candidate rather than borrowing another mask's score; seeded random streams and candidate memoization.
- Feature selection uses internal validation. Final test labels do not enter the selector; a regression test flips final-test labels and verifies the selected features remain unchanged.
- Categorical encoders use a reserved unknown bucket fitted without test vocabulary. Runtime and offline transforms agree.
- Numeric inputs reject missing, infinite and negative values. Deployment inference requires binary models and probability support.
- Model versions, checksums and dependency compatibility are inspected before loading trusted bundles. OS-backed writer locks coordinate training and inference and release on process exit.
- Locked flow updates, bounded flow/packet memory, FIN/RST flag handling and millisecond timing corrections.
- Synthetic self-test failures return a failing process status; collector heartbeat is separate from replay observations.
- The NSL quick experiment retains sufficient real rare-class records for validation; its original pipeline and outputs remain available.

## Enhancements

- Five connected Streamlit workspaces with consistent navy/teal styling and responsive layout.
- Bounded CSV inference, sample download, predictions export and atomic idempotent event persistence.
- Explicit live/replay/import/self-test provenance and model-version evidence.
- Incident filtering, pagination, validated transitions, reasoned notes and append-only review history at application level.
- Held-out classification report, confusion matrix, FPR/FNR, feature importance, feature selection and retained experiment history.
- Quick training and replay commands, PowerShell wrappers, pinned dependencies, isolated automated tests and current research documentation.

## Preserved

Original GWO/bGWO equations and accuracy/parsimony objective; four classifiers; NSL-KDD experiment; UNSW presplit and optional full-raw loaders; configured runtime feature set; packet-capture workflow; notebooks; supplied source datasets; historical documentation under `docs/original/`.

## Deliberate compatibility changes

`IDS_DB_PATH` replaces the Linux-only hardcoded location. The default data source is the included presplit files. Fresh models use `artifacts/deployment`; old pickles are not silently loaded. New encoders require retraining, which was performed for the delivered bundle. SVM probabilities are enabled to support the runtime confidence threshold. SQLite upgrades add columns and tables without removing existing events. No original dataset or database was reset.

The compact package omits redundant giant raw partitions, processed copies, original unverified pickles, runtime databases and environments. The original upload retains them. See `DATA.md`.

## Exact changed / added source and documentation files

- `.gitignore`
- `.streamlit/config.toml`
- `README.md`
- `README_UNSW_REALTIME.md`
- `config.py`
- `dashboard/app.py`
- `docs/ARCHITECTURE.md`
- `docs/BENCHMARK_RESULTS.md`
- `docs/CHANGES.md`
- `docs/DATA.md`
- `docs/PROJECT_REPORT.md`
- `docs/RUNTIME_FEATURE_AUDIT.md`
- `docs/VERIFICATION.md`
- `docs/original/README-UNSW-original.md`
- `docs/original/README-original.md`
- `docs/original/RUNTIME_FEATURE_AUDIT-original.md`
- `examples/flows.csv`
- `main.py`
- `pytest.ini`
- `realtime.py`
- `replay.py`
- `requirements-dev.txt`
- `requirements-lock.txt`
- `requirements.txt`
- `scripts/Replay.ps1`
- `scripts/Setup.ps1`
- `scripts/Start.ps1`
- `scripts/Test.ps1`
- `scripts/Train.ps1`
- `scripts/browser_smoke.cjs`
- `scripts/browser_verify.py`
- `src/gwo/feature_selector.py`
- `src/gwo/fitness.py`
- `src/gwo/gwo_algorithm.py`
- `src/preprocessing/encoding.py`
- `src/preprocessing/unsw_preprocessing.py`
- `src/runtime/alerting.py`
- `src/runtime/artifacts.py`
- `src/runtime/batch.py`
- `src/runtime/feature_extraction.py`
- `src/runtime/flow_tracker.py`
- `src/runtime/inference.py`
- `src/runtime/training_lock.py`
- `tests/conftest.py`
- `tests/test_dashboard.py`
- `tests/test_flows.py`
- `tests/test_gwo.py`
- `tests/test_pipeline.py`
- `tests/test_store.py`
- `train.py`

Generated deployment artifacts and experiment results are included separately. Original-source changes can be reviewed in `source-changes.patch`.
