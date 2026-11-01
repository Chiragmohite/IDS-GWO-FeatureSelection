# Executed quick benchmark results

These historical quick results remain the metrics of the packaged dashboard model. The larger **separate** 42-versus-35-feature benchmark is documented in [MATCHED_BENCHMARK.md](MATCHED_BENCHMARK.md). It does not replace this deployment bundle or establish live-network accuracy.

These are small supplied-dataset experiments, not production or full-dataset accuracy. All runs used seed 42, bGWO, four wolves, three iterations and one run. Test evaluation was not used for GWO selection. Comparing these reports must not be used to tune against the final test split.

| Classifier | All-feature accuracy | GWO accuracy | Selected / 35 | Attack precision | Attack recall | Attack F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_forest | 83.00% | 81.50% | 33 | 0.690 | 0.951 | 0.800 | 27.17% | 4.88% |
| decision_tree | 82.00% | 81.90% | 34 | 0.711 | 0.902 | 0.795 | 23.40% | 9.77% |
| knn | 77.50% | 77.80% | 15 | 0.649 | 0.936 | 0.766 | 32.24% | 6.43% |
| svm | 76.50% | 76.20% | 15 | 0.624 | 0.977 | 0.762 | 37.48% | 2.31% |

The delivered default bundle is Random Forest. Its 27.17% false-positive rate on this quick test split is substantial. The project is an academic research baseline and analyst workflow; these results do not justify operational blocking.

The original NSL-KDD quick pipeline also completed: all-feature accuracy 77.8%, selected-feature accuracy 77.6%, 22/41 features. Rare-class performance was weak, including zero U2R recall in that run. Full reports must accompany any headline accuracy.
