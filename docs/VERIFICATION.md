# Verification actually executed

## Major-project release — November 2026

The earlier sections below describe historical runs. This release was based on the delivered v4 ZIP, not a new scaffold. Isolated project dependencies were installed from requirements-dev.txt in a Python 3.12.14 environment using the matching system numerical libraries (pandas 2.2.3, NumPy 2.3.5, sklearn 1.8.0). No additional application packages.

Commands from the project root (shell interpreter used here: ../../verify-env/bin/python):

```bash
../../verify-env/bin/python -m pytest -q -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning
../../verify-env/bin/python budget_study.py --output outputs/benchmarks/budget-20261101
../../verify-env/bin/python reference_study.py --output outputs/benchmarks/reference-20261101
../../verify-env/bin/python build_candidate.py --output artifacts/compact20
../../verify-env/bin/python audit_model.py
IDS_MODEL_PROFILE=compact20 ../../verify-env/bin/python audit_model.py
IDS_DB_PATH=/tmp/gwo-verification-20261101.db ../../verify-env/bin/python traffic.py --probe 6 --save
IDS_DB_PATH=/tmp/gwo-verification-20261101.db ../../verify-env/bin/python traffic.py --pcap examples/teaching_capture.pcap --save
../../verify-env/bin/python verify_capture.py --iface lo --seconds 3
../../verify-env/bin/python -m pip check
```

The baseline was **61 passed in 8.55s**. The first full post-change run exposed eight failures in the single-event SQL insert (a missing column/placeholder for observation time). Fixed the insert; the next full run was **75 passed in 7.29s**. Subsequent tests exercised compact-model selection, alert threshold propagation, model directory protection and concurrent additive schema initialization. See the final validation record in RELEASE_CHECKS.md. Historical Windows tests are not counted as tests of this release.

- Default package: six genuine local UDP exchanges (12 received datagrams) persisted six instrumented observations. The synthetic teaching PCAP yielded two flows from six supported packets; one non-IPv4 frame was skipped. Compact20 was also run on five real local exchanges and the same PCAP. Source totals remain distinct. No payload capture is necessary for the probe.
- The Scapy-based offline PCAP prototype failed under adapter enumeration restrictions. Replaced only offline parsing with a bounded standard-library header reader; the live path still uses Scapy. Original and compact bundle predictions were exercised with the actual saved preprocessing; CSV batch inference was checked against direct fitted-encoder/estimator results.
- The passive adapter verifier on Linux **failed** with an OS permission error (no raw socket access). No passive live row was claimed. Windows/Npcap capture and Windows PowerShell wrappers were not run here. Use the isolated verify_capture.py check on the intended Windows PC.
- The original candidate-independent benchmark and the new budget study evidence archives include real per-seed convergence, per-class results, individual heatmaps, selected names and CSV/Markdown comparisons. The compact model reproduced its corresponding study confusion matrix; alert-level recall is reported separately from binary-classifier accuracy.
- Real browser checks for both model profiles used an isolated event database, seven workspaces, saved CSV/probe/PCAP predictions, two simultaneous browser sessions, automatic refresh, a research ZIP download, local RF explanation, and a mobile viewport. Chromium 153 could not start in this host; the checks passed with the compatible Chromium 131 runtime. No change to application dependencies was needed.
- Compilation completed and pip check reported no broken requirements. Windows crash stability is not claimed; the native pandas datetime conversion seen in the trace has been removed from Overview, and the startup scripts now resolve a 3.12 interpreter.


## Matched benchmark follow-up — 31 October 2026

Environment: Linux, Python **3.12.14**, pandas **2.2.3**, NumPy **2.3.5**, scikit-learn **1.8.0**, Streamlit **1.55.0**. The user's separate Windows run reported **47 passed in 18.66s** under `.venv312`; that is user-supplied evidence, not a Windows run performed here.

Executed from the project root:

```bash
# Before this stage:
.verify-venv/bin/python -m pytest -q -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning
# 52 passed in 5.54s; 0 failed, 0 skipped.

# Focused benchmark and dashboard regression tests:
.verify-venv/bin/python -m pytest tests/test_benchmark.py tests/test_dashboard.py -q -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning
# 12 passed in 5.98s; 0 failed, 0 skipped.

# Full suite after changes:
.verify-venv/bin/python -m pytest -q -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning
# 61 passed in 8.83s; 0 failed, 0 skipped.
.verify-venv/bin/python -m pip check
# No broken requirements found.
.verify-venv/bin/python -m py_compile benchmark.py tests/test_benchmark.py tests/test_dashboard.py
node --check scripts/browser_smoke.cjs
# Both exited 0.

IDS_CHROMIUM_EXECUTABLE=/tmp/ids-chromium IDS_BROWSER_OUTPUT=/tmp/ids-benchmark-browser .verify-venv/bin/python scripts/browser_verify.py
```

The browser run exercised all five workspaces, analyzed 50 actual benchmark flows, saved 50 predictions to an isolated database, opened populated Overview in **two concurrent sessions**, enabled the five-second refresh, navigated away/back, checked HTTP health **200**, and checked a 390px mobile viewport. Final output: zero page errors and no document-level horizontal overflow. This was a short regression exercise, not a long-duration soak test or Windows live-capture test.

The first browser attempt failed because Playwright tried to click Streamlit's visually hidden checkbox input. The test was corrected to click its visible label and assert the checked state; the application was unchanged. The subsequent complete run passed.

The populated AppTest creates 100 explicitly synthetic self-test events with repeated timestamps and verifies counts **100 / 36 / 64 / 60**, repeated rendering, refresh setting and navigation. This directly exercises the `pd.to_datetime(...).dt.floor('h')` path from the reported native Windows crash. The crash itself has not been reproduced or verified fixed on Windows here.

The new benchmark tests execute the real selector and classifiers against bounded copies of the supplied data, flip final-test labels, and confirm identical selection masks and fitness, disjoint input groups, accurate confusion-matrix totals, preserved model/database bytes, and refusal to overwrite output. Temporary paths are used throughout.

The measured larger benchmark and exact command are recorded in [MATCHED_BENCHMARK.md](MATCHED_BENCHMARK.md). No final-test scores were used to change the optimizer, objective, feature pools or deployed model.

### Remaining Windows check

Use the existing `.venv312` environment. Keep this command running:

```powershell
.\.venv312\Scripts\python.exe -X faulthandler -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open Overview with stored observations, enable refresh, visit Events and return to Overview. If the process exits, preserve the terminal traceback. Windows stability and Npcap live capture remain unverified by this Linux test run. No database deletion is required.

## SQLite connection cleanup follow-up

A reported Windows `ResourceWarning: unclosed database` led to a confirmed leak in `init_db`: a SQLite connection's transaction context manager commits or rolls back but does not close the connection. A second leak was confirmed when `_connection` setup failed before entering its `try/finally` block. No warning suppression, dependency change, database reset or schema change was used.

Changes: `src/runtime/alerting.py` explicitly closes the initialization connection while retaining its transaction context, and puts connection setup inside the existing cleanup scope. `tests/test_store.py` adds five real-SQLite regression cases and closes the historical-schema test's own connection.

Executed in Linux/Python 3.12.14 from the project root:

```bash
.verify-venv/bin/python -m pytest -q
# Before the change: 47 passed in 4.62s.
.verify-venv/bin/python -m pytest tests/test_store.py -q
# New regression tests against unfixed code: 3 failed, 13 passed in 0.24s.
.verify-venv/bin/python -m pytest -q -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning
# After the fix: 52 passed in 4.95s; 0 failed, 0 skipped.
.verify-venv/bin/python -m py_compile src/runtime/alerting.py tests/test_store.py
.verify-venv/bin/python -m pip check
git diff --check
```

Compilation and whitespace checks exited 0; dependency check reported `No broken requirements found`. Tests verify explicit closure after initialization success/failure and connection setup/body failure, plus commit/rollback behavior. Closure assertions retain real connections and attempt a query after exit, so they do not depend on garbage collection or whether a Python version emits this warning.

Windows was not available here. The supplied warning's scikit-learn line is where it surfaced; it does not identify where the connection was opened. These changes fix the reproduced project leaks, but a local rerun is needed to confirm the reported warning no longer occurs on the user's machine.

For an existing installation, stop running project processes, back up `src/runtime/alerting.py`, and replace only that file with the corrected version. Restart the same dashboard command. No retraining, package reinstall, or database deletion is required. To run the new regression tests locally, also copy the updated `tests/test_store.py` from the updated ZIP, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error::ResourceWarning -W error::pytest.PytestUnraisableExceptionWarning
```

The original verification record below remains historical; the SQLite-only follow-up had 52 tests. The latest suite result is recorded at the top.

Environment: Linux, Python 3.12.14. Two environments were used: an initial environment with the available numerical libraries, then a **clean virtual environment** installed from `requirements-dev.txt`. No original uploaded pickle was loaded.

## Final automated result

**47 passed, 0 failed, 0 skipped** (`4.78s` in the final source-tree run). There were no existing tests in the uploaded project. The 47 tests were added for this project; these counts are unrelated to the earlier AegisLend project.

Executed from the project root:

```bash
python -m venv .verify-venv
.verify-venv/bin/python -m pip install -r requirements-dev.txt
.verify-venv/bin/python -m pip check
.verify-venv/bin/python -m pytest -q
.verify-venv/bin/python -m compileall -q config.py main.py train.py realtime.py replay.py src dashboard tests scripts
git diff --check
```

Installation completed; `pip check` reported `No broken requirements found`; compilation and diff whitespace checks returned exit status 0. The exact installed environment is recorded in `requirements-lock.txt`.

Coverage includes: actual objective initialization, both GWO variants, reproducible seeds, invalid optimizer inputs, mask caching, unknown categories, no final-test-label influence on selected features, artifact integrity, training locks, nonfinite inputs, CSV validation, atomic batch rollback, concurrent idempotent imports, SQLite upgrade preservation, parameterized filters, incident transitions, concurrent review conflicts, bounded/thread-safe flows, timing units, four model probability outputs, all five UI pages, CSV-to-history-to-review integration and a busy-training UI error state.

## Real benchmark commands

```bash
.venv/bin/python train.py --quick --classifier random_forest
.venv/bin/python train.py --quick --classifier decision_tree
.venv/bin/python train.py --quick --classifier knn
.venv/bin/python train.py --quick --classifier svm
.venv/bin/python main.py --quick
```

All completed after fixes. Random Forest was trained last to restore the delivered default bundle. See `BENCHMARK_RESULTS.md` and completed JSON reports. The default bundle used 2,400 fitting rows, 600 validation rows and 1,000 final test rows. It selected 33/35 features and achieved 81.5% accuracy versus the 83.0% all-feature baseline. FPR was 27.17%; this is **not** a production-quality accuracy claim.

## Runtime integration commands

```bash
IDS_DB_PATH=/tmp/ids-final-replay.db .verify-venv/bin/python replay.py --limit 100
IDS_DB_PATH=/tmp/ids-final-selftest.db .verify-venv/bin/python realtime.py --selftest
```

Replay: 100 rows stored, 64 predicted attack, 61 thresholded alerts, source `dataset_replay`, `live_capture=false`. These are model predictions on supplied benchmark rows, not confirmed attacks.

Self-test: isolated database count 0 → 1, successful flow extraction → saved model inference → SQLite persistence, exit status 0. The synthetic flow was predicted attack with approximately 0.68 confidence; this is a plumbing test, **not evidence of correct detection**.

## Browser and frontend

The optional reproducible harness is `scripts/browser_verify.py` with `scripts/browser_smoke.cjs`. It starts an isolated local server on a free port and a temporary database. Node/Playwright and a compatible browser are test-only prerequisites, not application dependencies.

Actual successful command:

```bash
IDS_CHROMIUM_EXECUTABLE=/tmp/ids-chromium IDS_BROWSER_OUTPUT=/tmp/ids-ui-final \
  .verify-venv/bin/python scripts/browser_verify.py
```

Result: HTTP health `200 ok`; five workspaces opened; 50 bundled rows analyzed by the real model; zero JavaScript page errors; no page-level horizontal overflow at 390×844; reduced-motion mode enabled; 1440×1000 desktop and mobile screenshots visually inspected. The isolated Python UI tests also exercised persistence and incident review. Screenshots are internal verification artifacts, not bundled user data.

There is no JavaScript production build step in this existing Streamlit application. App startup, real browser behavior and Python compilation were verified instead. PowerShell wrappers were inspected but could not be executed because PowerShell is unavailable here.

## Initial failures and resolutions

- Original database initialization failed on `/home/osboxes/alerts.db`: replaced by portable configurable storage and directory creation.
- Original GWO initialization never evaluated the objective: corrected and regression-tested.
- NSL quick sampling left only one rare-class record and failed stratified validation: minimum real rare-class sampling added; rerun completed.
- The initial browser setup lacked a browser executable; its standard download failed. A separate test-only Chromium binary enabled the final executed browser checks. No browser package was added to the application requirements.
- Early browser automation raced reactive form updates / sidebar resizing. CSV analysis now submits one form, and the harness waits for the responsive sidebar transition. Final browser checks passed.
- Existence-only training locks could remain after interrupted work. OS-backed locks now release on process exit, and model deserialization holds the same lock.

## Checks not performed / limits

- Windows PowerShell execution, Windows Npcap installation and actual live packet capture.
- Full-size raw UNSW training, exhaustive hyperparameter tuning, nested cross-validation, independent recent-network evaluation or production throughput testing.
- Comprehensive accessibility audit, all browsers/tablets, formal penetration testing or multi-user deployment security.
- Multiclass UNSW deployment inference: intentionally rejected by the binary live-alert pipeline. Original NSL multiclass offline research remains available and its quick pipeline ran.

No tests were marked skipped to hide these items; these are separate unexecuted integration/evaluation activities. The delivered status is a verified offline academic workflow with explicitly unverified live/Windows integration.

## Archive verification

The completed ZIP was extracted into `/tmp/ids-package-check/IDS-GWO-FeatureSelection`. From that extracted project root, the clean verification interpreter ran `python -m pytest -q`: **47 passed, 0 failed, 0 skipped in 4.54s**. An initial invocation from its parent directory failed to import `config`; rerunning from the documented project root resolved the command-location issue. ZIP CRC checks and all packaged SHA-256 manifest entries passed. Current documentation links were checked: zero missing local targets.
