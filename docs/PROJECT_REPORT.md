# Final-year report and demonstration guide


For current research results, the seven-workspace demonstration and the limits of the live check, use [MAJOR_PROJECT_GUIDE.md](MAJOR_PROJECT_GUIDE.md). The previous quick benchmark below is historical.

## Suggested title

**Intrusion Detection with Grey Wolf Optimized Feature Selection and an Explainable Analyst Workbench**

## Abstract

This project evaluates Grey Wolf Optimization for selecting network-flow features used by machine-learning intrusion classifiers. The original GWO and binary-GWO search, four classifier implementations and benchmark loaders are combined with separated fitting, validation and test partitions. A practical analyst interface exposes measured classifier performance, selected features and persisted observations. Optional passive packet-flow inference shares the offline preprocessing bundle, while CSV analysis enables a repeatable demonstration without capture privileges. Incident review preserves notes and state transitions. Results must be read with the reported false-positive rates, benchmark sampling limits and live feature-definition differences; they do not establish production intrusion-detection accuracy.

## Objectives

1. Preserve and verify GWO-based feature selection and classifier comparison.
2. Prevent final-test leakage and train/runtime category mismatch.
3. Make experiments reproducible and expose performance/reduction trade-offs.
4. Connect inference, persistence, investigation and audit in a usable frontend.
5. Provide a reliable offline demo and optional authorized passive monitoring.

## Methodology and report structure

1. Introduction: network-flow classification, dimensionality, false-positive cost.
2. Related work: GWO, bGWO, Random Forest, Decision Tree, kNN, SVM; cite primary papers and dataset publications.
3. Dataset and ethics: supplied benchmark lineage, labels, split construction, privacy and authorization.
4. Architecture: use `ARCHITECTURE.md`; distinguish research pipeline, shared inference and analyst UI.
5. Optimization: original accuracy/parsimony objective, seed, population, iterations, candidate caching and validation-only selection.
6. Evaluation: report all-feature versus selected-feature accuracy, precision, recall, F1, FPR, FNR, ROC-AUC where available, confusion matrix, feature count and runtime.
7. Implementation: unknown-category handling, flow state locks/limits, atomic imports, incident transition audit, portable configuration.
8. Testing: use actual counts and limitations from `VERIFICATION.md`; do not claim capture accuracy from a plumbing self-test.
9. Results and discussion: reproduce values from `outputs/reports/runs` and `artifacts/deployment/training_config.json`. Discuss negative results as well as improvements.
10. Conclusion and future work: nested repeated validation, more recent authorized traffic, calibrated thresholds, proper authentication for shared use, drift monitoring, richer protocol extraction and retention controls.

## Demonstration (about 8–10 minutes)

1. Start the console and show honest empty/live-status states.
2. Open Model lab; explain train/validation/test counts, selected features and confusion matrix.
3. Run CSV analysis with the bundled benchmark sample; inspect predictions and download results.
4. Save once, then save again to demonstrate idempotency.
5. Open Overview; identify `dataset_replay` provenance and why it is not live attack activity.
6. Open an alert; inspect raw feature evidence and model version.
7. Move it to investigating with a reason, then resolve or mark false positive after reviewing evidence.
8. Show the review audit and export the filtered event page.
9. Run `python realtime.py --selftest`; explain that it tests flow-to-model-to-storage connectivity, not detection performance.
10. Optionally demonstrate authorized capture on the target machine after installing capture support. Explain any limitations honestly.

Suggested screenshots: overview with provenance; CSV predictions; feature importance and metrics; incident detail/audit; normal/attack confusion matrix. Browser screenshots are not bundled in this delivery.

## Conclusions supported by this work

The project now exposes a connected offline IDS research workflow and retains its original optimization basis. Reduced feature count does not necessarily increase held-out accuracy. In particular, the recorded quick Random Forest/bGWO run underperformed its all-feature baseline. A separate matched 42-versus-35-feature benchmark with three optimizer seeds is documented in `MATCHED_BENCHMARK.md`; it must be presented with its bounded selection subset, fixed data split and full confusion-matrix results. Independent data-split replication and representative live validation remain future work. Do not substitute a generic high-accuracy claim or present research metrics as those of the unchanged dashboard model.
