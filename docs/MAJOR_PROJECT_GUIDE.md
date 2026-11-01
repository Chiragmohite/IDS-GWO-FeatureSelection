# GWO IDS: research and live-data demonstration

**Question:** Can a trained Random Forest classify network flows using fewer measurements, and what happens to false alarms and missed attacks? GWO chooses columns during training. The saved classifier predicts on new flows without retraining. Human review determines what an alert means.

## Start and demonstrate (Windows)

Use Python 3.12 and the existing requirements. If this is a fresh extraction, run `scripts/Setup.ps1` or set up a 3.12 environment as described in the README. If you already have a working installation, stop running project processes and back up your folder. Keep your local .venv312, logs/alerts.db and custom model bundles. The release does not contain an event database. Schema upgrades preserve existing records. Start commands below assume PowerShell in the project root.

```powershell
$env:IDS_MODEL_PROFILE = "compact20"
.\.venv312\Scripts\python.exe -X faulthandler -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. Keep the terminal running. The optional compact profile is already trained. The original quick-trained model remains the default if IDS_MODEL_PROFILE is unset. An explicit IDS_MODEL_DIR overrides profile selection. To return to the original model, stop the server, set IDS_MODEL_PROFILE to deployment and restart. The profile is shared by CSV, PCAP, probe and collector processes **started in that environment**; set it in each new PowerShell terminal. Quick training is disabled for the packaged compact bundle.

1. **Research results:** choose the fixed-budget experiment. Compare baseline, random subset and ranked GWO across all seeds. Inspect class counts, per-class metrics, both confusion matrices and GWO convergence. Download the Markdown/CSV comparison or complete evidence ZIP. The matched unconstrained experiment is also available.
2. **Traffic lab → Local UDP exchange:** click *Measure local traffic*. Two temporary localhost sockets actually send/receive UDP datagrams. Inspect measured flows, source, model version and predictions; save to history. These are controlled benign exchanges, not passive monitoring or attack examples. Socket receive time, sender TTL and estimated IPv4+UDP length differ from a packet sniffer's wire-level readings.
3. **Traffic lab → Recorded PCAP:** upload an authorized classic capture, or select the clearly labelled synthetic teaching fixture. Analyze, inspect and save. Identical capture bytes plus model version are idempotent. Captures require chronological timestamps, Ethernet (up to two VLAN tags) or raw IPv4, TCP/UDP only. Fragments and unsupported/truncated packets are counted as skipped. Payloads are not persisted. Limits: 10 MiB, 20,000 packets, 1,000 flows. PCAPNG and compressed files are unsupported.
4. **Events & investigations:** filter by input source. Inspect an observation and click *Explain this prediction*. For a matching saved Random Forest, signed tree-path contributions reconstruct the event's attack score from its base score. These are model-specific explanations, not causal evidence. A different model version cannot explain a historical event. Alerts retain the existing review and audit workflow.
5. **Overview:** compare source totals and activity. Dataset replay, captured files and the loopback probe never mark the passive collector healthy. Observation time and storage time are distinct. The activity chart uses standard-library UTC parsing to avoid the pandas native datetime call that appeared in the Windows crash trace.
6. **Passive capture, optional:** list adapters, then run a bounded verification on an interface you own or are allowed to monitor. It uses an isolated temporary database and reports success only when the process exits cleanly and stores actual live flows. For your normal dashboard history, run the regular collector in a separate terminal. Generate ordinary authorized traffic on that interface while it runs.

```powershell
.\.venv312\Scripts\python.exe realtime.py --list-ifaces
.\.venv312\Scripts\python.exe verify_capture.py --iface "Wi-Fi" --seconds 30
.\.venv312\Scripts\python.exe realtime.py --iface "Wi-Fi" --duration 60
```

Windows passive capture requires a working Npcap installation and appropriate permissions. Use the actual interface name printed. Keep the dashboard as a normal user. The probe, PCAP, CSV and replay paths do not need an administrator.

## Comparison with the reference demonstration

| Visible reference capability | Where it is in this release |
|---|---|
| UNSW-NB15 loading; train/test shapes and class balance | Research results → Source data and class balance; protocol and splits in evidence pack |
| Random Forest on all 42 features | Offline-42 baseline in both research studies |
| GWO search for feature subsets; fitness by iteration | Original GWO/bGWO retained; iteration chart, CSV and PNG exported |
| Reduced-feature RF, selected feature names and counts | Model inspection; selected_features.txt and selected_features.csv |
| Normal/attack precision, recall, F1 and support | Per-class report for each model, comparison CSV and report JSON |
| Baseline and selected confusion matrices | On-screen numeric matrices and individual PNGs in the evidence pack |
| CSV and Markdown comparison, reproducible outputs | Report downloads and CLI evidence.py; timing fields where recorded |
| GWO-selected 20 of 42 in the reference run | New fixed-budget comparison enforces exactly 20 of 42 or 20 of 35; selecting 20 is a **specified constraint**, not a discovered optimum |

This maps what was visible in the friend's recording. No claim of identical private code, identical split, unobserved features, or superior accuracy on their exact experiment is made.

## New fixed-budget experiment

Before held-out scoring, the run specified 20 features, two candidate pools, 8 wolves, 20 iterations, optimizer seeds 42/43/44 and a 5% calibration false-positive-rate target. Continuous GWO ranks positions to decode exactly K columns; the existing unconstrained optimizer remains available. Each seeded GWO subset is compared to a seeded, uniformly random subset of the same size and an all-feature baseline. Search uses 32 trees for bounded computation, final models use 150. The objective and splits are fixed across candidate pools.

Raw supplied rows: 175,341 train / 82,332 test. The shared test set contains 53,915 rows after cleaning and overlap removal by the 35 input features without labels. GWO uses 4,800 training-fit and 1,200 validation rows. A separate group-disjoint calibration partition of 10,020 rows is excluded from final fitting; final classifiers fit 97,720 training records. Test outcomes do not choose the seed, features or alert cutoff. This test dataset had been examined in earlier research; it is not a fresh external validation set.

| Pool | All-feature accuracy | Mean random-20 accuracy | Mean GWO-20 accuracy |
|---|---:|---:|---:|
| Offline 42 | 82.77% | 82.34% | 82.60% |
| Runtime 35 | 82.77% | 82.47% | 82.16% |

Means vary only optimizer/random-subset seed on a fixed split. Both GWO means are below the all-feature baseline. GWO beats the random control on average for the 42-feature pool but not for the runtime pool. Results for every seed, including negative comparisons, are in outputs/benchmarks/budget-20261101/. The 5% calibration target did **not** transfer to the test set: runtime GWO models averaged 15.69% test false alerts and 90.51% attack-alert recall at the calibrated cutoff. A lower alert rate can cost recall. No live detection-performance claim follows.

## Reference-direction run: comparison with the photographed setup

An additional run trained on the 82,332-row file and tested on the 175,341-row file using 42 candidate features, matching the **visible file roles and row counts**. It preserves raw row weighting for that comparison. Search used 4,800 fit and 1,200 separate validation rows from training, eight wolves, 20 iterations and seed 42. Both selected models and the baseline used the same final 150-tree RF settings. All selected masks were frozen before final-test scores.

| Model | Features | Raw test accuracy (175,341 rows) | Strict-view accuracy (99,869 rows) |
|---|---:|---:|---:|
| All-feature RF | 42 | 90.08% | 85.07% |
| Original bGWO, unconstrained | 39 | 90.04% | 85.01% |
| Ranked GWO with fixed budget | 20 | 89.65% | 85.35% |

10,279 raw test rows had feature inputs also found in training. The stricter view excluded training-overlapping inputs and kept one instance per remaining input/label pair. It uses the same trained models with no additional fitting. The friend's screenshots report **90.07% baseline, 88.34% selected, 20 features**. Our raw 20-feature result is numerically higher on the same split sizes, but their exact preprocessing, code, selection objective and RF settings are unknown. The strict numbers are **not** available for their project. No proof of universal or live superiority follows from this comparison.

Inspect the raw and strict views in Research results. To repeat independently, run `python reference_study.py`; it creates a new directory rather than overwriting the delivered run.

## What is actually shipped and scored

The optional compact20 profile was selected by highest final **training-validation fitness** among runtime GWO runs, with smallest seed as tie-breaker. That selects seed 42; it does not select the highest held-out accuracy. The model, encoder, scaler, feature schema, calibrated threshold and hashes are packaged separately under artifacts/compact20. The rebuild checked the frozen study's accuracy, calibration threshold and held-out alert confusion matrix. The existing artifacts/deployment bundle is unchanged.

Both saved bundles were then scored directly on the same held-out 53,915 rows, using their own preprocessing and alert policies:

| Saved bundle | Selected features | Binary accuracy | Normal flows alerted | Attacks alerted |
|---|---:|---:|---:|---:|
| Original quick-trained model | 33/35 | 80.26% | 23.23% | 91.22% |
| Optional compact model | 20/35 | 82.66% | 15.98% | 92.32% |

Binary accuracy uses the classifier's normal/attack prediction. An alert also requires a predicted attack with confidence at least the profile cutoff: 0.60 for the original bundle and 0.67333 for compact20. Therefore classification accuracy is **not** the accuracy of a thresholded alert/no-alert decision. Compact20 alerted on 5,380 of 33,666 normal rows and on 18,694 of 20,249 attack rows. The original quick-run Model lab number of 81.50% used a different, 1,000-row quick test; it should not replace the comparable 80.26% audit above.

The model, amount of training data, selected features and alert policy all differ across these bundles. Do not attribute their difference to GWO alone. Attack-family tables use held-out reference labels for evaluation; the classifier predicts only normal or attack. Live feature definitions approximate several dataset features.

## Reproduce or inspect

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe budget_study.py
.\.venv312\Scripts\python.exe evidence.py outputs/benchmarks/budget-20261101
.\.venv312\Scripts\python.exe build_candidate.py --output artifacts/my-new-compact-model
.\.venv312\Scripts\python.exe audit_model.py
.\.venv312\Scripts\python.exe traffic.py --probe 5 --save
.\.venv312\Scripts\python.exe traffic.py --pcap examples/teaching_capture.pcap --save
```

The build command defaults to the delivered completed study; use --study with a new run to rebuild from it. Output bundle directories must be new. Run audit_model.py with the desired IDS_MODEL_PROFILE. To inspect the export without running experiments, open either research_evidence.zip in outputs/benchmarks/. The dated budget run includes experiment_source.zip containing the source bytes recorded at protocol freeze. Runtime/UI work completed later does not change these recorded metrics.

See VERIFICATION.md for commands, actual results, failures fixed and remaining platform checks.
