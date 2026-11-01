# Matched UNSW-NB15 benchmark — 31 October 2026

## What was executed

The 42-feature offline research pool and the existing 35-feature runtime pool were compared using identical data rows, preprocessing rules, RF settings and search budgets. This is a matched comparison **within our repository**, not an exact reproduction of the friend's `NIDS-UNSW` project. That project's source, objective, encoding and full RF settings were not supplied with the media.

Executed from the project root on Linux / Python 3.12.14:

```bash
.verify-venv/bin/python -u benchmark.py --output outputs/benchmarks/matched-20261031 --fit-limit 4800 --validation-limit 1200 --wolves 10 --iterations 20 --seeds 42 43 44 --jobs 2
```

Exit status **0**; measured elapsed time **377.1 seconds** on this environment. Runtime on another machine will differ.

For Windows, using the existing Python 3.12 environment:

```powershell
.\.venv312\Scripts\python.exe benchmark.py --fit-limit 4800 --validation-limit 1200 --wolves 10 --iterations 20 --seeds 42 43 44 --jobs 2
```

Omitting `--output` creates a new timestamped directory. Completed output directories are never overwritten. The optional `scripts/Benchmark.ps1` wrapper prefers `.venv312` and checks Python 3.12; PowerShell execution was not available in the Linux verification environment.

## Frozen protocol

- Raw supplied files: **175,341 training rows / 82,332 test rows**. File and source-code SHA-256 hashes are in `protocol.json`.
- Existing `clean_unsw` cleaning: **107,740 training rows / 55,945 test rows** after dropping the row ID, duplicates and missing values as implemented.
- Remove **2,030** test rows whose 35-feature input hash occurs in training, without using labels in that overlap key. Both feature pools receive the same remaining **53,915** test rows: **33,666 normal / 20,249 attack**.
- Group the training data by the shared 35-feature input hashes before the 80/20 selection split, so identical inputs cannot straddle selection fit and validation. Bound the stratified selection samples to **4,800 fit / 1,200 validation** rows.
- Fit the existing rare-protocol handling, categorical encoders and numeric scaler on selection-fit data only. Validation does not fit transforms.
- Use existing **bGWO**, **10 wolves**, **20 iterations**, optimizer seeds **42, 43, 44**. Keep the data split and RF seed fixed at **42**.
- Preserve the RF factory settings: **150 trees**, with `n_jobs=2` for bounded parallel work. Use the same settings for each candidate, baseline and final selected model.
- Preserve the objective exactly: **0.6 × accuracy_percentage + 0.4 / selected_feature_count**.
- Freeze all six selected masks before final-test evaluation. Then refit preprocessing and each final RF on **all 107,740 cleaned training rows**. Evaluate all eight models on the same held-out records.

This is full cleaned-split **final fitting/evaluation with sampled GWO selection**, not GWO search on the entire training set. Grouping by feature hashes also does not establish independence of capture sessions or networks.

## Measured results

Precision, recall and F1 below are for the attack class, on a 0–1 scale. FPR is the fraction of normal records predicted as attacks. These are classifier decisions, not the separate 0.6-confidence alert policy.

| Pool | Model / optimizer seed | Features | Accuracy | Attack precision | Attack recall | Attack F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Offline 42 | All-feature baseline | 42 | 82.89% | 0.6942 | 0.9731 | 0.8103 | 25.78% |
| Offline 42 | bGWO 42 | 41 | 83.00% | 0.6959 | 0.9723 | 0.8112 | 25.55% |
| Offline 42 | bGWO 43 | 32 | 82.17% | 0.6853 | 0.9714 | 0.8037 | 26.83% |
| Offline 42 | bGWO 44 | 41 | 83.00% | 0.6959 | 0.9723 | 0.8112 | 25.55% |
| Runtime 35 | All-feature baseline | 35 | 82.83% | 0.6932 | 0.9740 | 0.8099 | 25.93% |
| Runtime 35 | bGWO 42 | 28 | 82.59% | 0.6900 | 0.9739 | 0.8077 | 26.32% |
| Runtime 35 | bGWO 43 | 32 | 81.95% | 0.6824 | 0.9715 | 0.8017 | 27.19% |
| Runtime 35 | bGWO 44 | 27 | 83.10% | 0.6975 | 0.9713 | 0.8120 | 25.33% |

Across the three optimizer seeds:

- **Offline 42:** mean accuracy **82.73%**, sample standard deviation **0.48 percentage points**; 32–41 retained features, or **2.38–23.81% reduction**.
- **Runtime 35:** mean accuracy **82.55%**, sample standard deviation **0.58 percentage points**; 27–32 retained features, or **8.57–22.86% reduction**.

These standard deviations describe optimizer variability on one fixed split, not independent dataset replications or a confidence interval. Both mean accuracies are below their all-feature baselines. Some individual seeds slightly improve accuracy, but selecting one after seeing test scores would overstate the evidence. All results are retained and no winning model was promoted.

The **25–27% false-positive rates remain substantial**. High attack recall does not justify automatic blocking or a production-detection claim. More data alone did not eliminate this limitation.

The objective strongly favors accuracy: at 90% accuracy its first term is 54, whereas the feature term is only 0.02 for 20 features. More iterations therefore do not guarantee a large feature reduction. A future objective study should be preregistered and selected using training validation, with a fresh evaluation set, rather than tuned against these test results.

## Comparison with the friend's recording

Their recorded experiment used **82,332 training / 175,341 test rows**, 42 features, 20 GWO iterations, and reported **90.07% baseline / 88.34% selected accuracy**, retaining **20 features**. The train/test direction differs from this benchmark, and their cleaning, selection validation, fitness and model parameters cannot be established from the screenshots alone. Those figures are not a direct head-to-head comparison. This run does not claim to beat or reproduce them.

The new application still offers CSV analysis, replay, event history, human review and a dashboard around the shared GWO research concept. The original optimization code, runtime feature extraction, database, training configuration and active deployment bundle were not changed by this experiment.

## Evidence files

Under `outputs/benchmarks/matched-20261031/`:

- `protocol.json`: settings frozen before evaluation, environment versions, input and source-code hashes, limitations.
- `splits.json`: raw/cleaned counts, overlap removal, selection class counts and row-index fingerprints.
- `selection.json`: every seed's feature mask, validation accuracy, fitness, unique candidate count and search duration.
- `run.log`: actual optimization iterations and final fitting/evaluation log.
- `results.json`: completed report, all per-class reports, confusion matrices, ROC-AUC, FPR/FNR, timings and feature names.
- `comparison.csv`: flat measured comparison table.
- `comparison.png`: accuracy and feature-count figure generated from those measurements.

The dashboard deliberately continues to show its **existing quick-trained model's** 81.5% result. These separate research numbers are not substituted into the Model lab or deployed without a model-training/promotion decision.

## Regression verification and local use

The updated complete suite ran: **61 passed, 0 failed, 0 skipped**, including the actual small benchmark workflow, test-label independence, file preservation and populated dashboard tests. Browser verification passed for all five pages, 50 real predictions saved to temporary history, two simultaneous sessions, automatic refresh, health HTTP 200, zero JavaScript errors and no mobile page overflow. See `VERIFICATION.md` for exact commands and the corrected browser-test locator failure.

All ten pre-existing deployment/database files checked by SHA-256 remained unchanged. No dependencies were added and no database or Docker configuration was altered. Windows-specific stability and live capture remain unverified here; the user's reported Windows suite result was 47 passed under Python 3.12.10.

For an existing local installation, keep `.venv312`, `logs/` and `artifacts/deployment/`. Copy the new `benchmark.py`, `scripts/Benchmark.ps1`, the complete updated `tests/` directory, `scripts/browser_smoke.cjs`, README and documentation from the ZIP. Copying the full test directory includes the earlier five SQLite cleanup tests that may be absent from the user's 47-test installation. If the earlier SQLite fix has not been applied, also copy `src/runtime/alerting.py` from the ZIP; that is the previously documented cleanup fix, not a new database/schema change. The included `outputs/benchmarks/matched-20261031/` is the measured report, not a replacement model. Run `.\.venv312\Scripts\python.exe -m pytest -q`, then start the dashboard with that same interpreter.
