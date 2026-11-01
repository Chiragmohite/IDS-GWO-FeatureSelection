# IDS-GWO-FeatureSelection

A rebuilt, modular, implementation-ready Intrusion Detection System using
**Grey Wolf Optimization (GWO)** for feature selection on the **NSL-KDD**
dataset, with **Random Forest, Decision Tree, SVM, and KNN** as swappable
classifiers.

This project is a clean-room reimplementation inspired by
[ahmedyounis1st/IDS-Feature-Selection-based-on-GWO-algorithm](https://github.com/ahmedyounis1st/IDS-Feature-Selection-based-on-GWO-algorithm).
No code was copied — every function was reverse-engineered, understood, and
rewritten into a proper package structure. See **Part 1** and **Part 2**
below for the full analysis and migration map.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Outputs land in `outputs/`: trained models (`.joblib`), the selected-feature
list, confusion matrices, and a `comparison_table.csv` showing baseline
(all-features) vs GWO-selected-features performance side by side.

The `dataset/raw/` folder already ships with `KDDTrain.csv` and
`KDDTest.csv` (the cleaned NSL-KDD-derived files from the source
repository), so the pipeline runs out of the box with no separate download
step.

**Note on runtime:** GWO retrains the classifier once per wolf per
iteration (`population_size × max_iterations × runs` retrains). With the
default `config.py` settings (12 wolves × 20 iterations × 3 runs = 720
retrains) on the full ~126k-row training set and Random Forest, a full run
can take a while depending on your machine. Reduce `GWO_POPULATION_SIZE`,
`GWO_MAX_ITERATIONS`, or switch `CLASSIFIER` to `"decision_tree"` in
`config.py` for faster iteration while developing.

---

## Part 1 — Reverse-Engineering Analysis of the Original Repository

### 1. Main entry point
The original repository has **no `main.py`** or CLI entry point. Everything
runs top-to-bottom inside one Jupyter notebook,
`Intrusion Detection System v2.ipynb` (3,573 lines / 135 KB, ~78 code
cells), which ends by launching a **Tkinter GUI** (`root.mainloop()`) —
the "entry point" is effectively "run every cell in order."

### 2. Complete execution flow
1. Import libraries (numpy, pandas, matplotlib, tkinter, PIL).
2. Load `KDDTrain+aggregate.csv` / `KDDTest+aggregate.csv`.
3. Normalize column names.
4. Inspect data (`.info()`, `.describe()`, `.nunique()`, class counts).
5. Drop duplicate rows; check (but don't act on) missing values.
6. Save cleaned files as `Clean_KDDTrain.csv` / `Clean_KDDTest.csv`.
7. Split into X/y via `iloc`.
8. Label-encode 3 categorical columns (`protocol_type`, `service`, `flag`)
   — separately fit on train and test (a bug, see §17 below).
9. Standard-scale the numeric columns.
10. Fit a baseline `SVC(C=10, kernel='rbf', gamma='auto')` and print a
    classification report.
11. Define GWO building blocks as free functions: `stepc`, `stepb`, `X_d`,
    `cross_over`, `sigmoid`, `Fitness`, `objective_function`.
12. Define a `wolf` class and two optimizer functions: `GWO` (sigmoid-decoded)
    and `bGWO` (binary, three-way-crossover-decoded).
13. Define evaluation helpers: `perf_measure` (TP/FP/TN/FN), `AAP`
    (average accuracy performance), `ANF` (average number of features),
    `AC`, `DR`.
14. Define `model_building` / `generate` / `get_data` — these **reload
    the CSVs from disk and reassign global `classifier` / `sc_X`
    variables** every time they're called, including from inside the GWO
    loop's fitness evaluation.
15. Define Tkinter GUI callbacks (`get_input`, `test_result`,
    `create_testframe`, `clf_table`, `btn1Action`) and finally build and
    run the GUI itself, which is what triggers the actual GWO run when the
    user clicks "Generate."
16. A trailing, disconnected cell block explores `sklearn.feature_selection`
    (`SelectPercentile`, `SelectKBest`, etc.) as an alternative to GWO —
    dead code, never wired into the GUI flow.

### 3–4. Folder structure & purpose of every folder
```
Original Datasets/     Raw, unprocessed source CSVs (contents not
                        inspected further — not referenced by the notebook)
Program tests/          Ad hoc scratch/test files, not part of the main flow
```
There is no `src/`, no package structure, no separation between data,
code, and outputs — everything is flat at the repository root.

### 5–7. Purpose of every file / function / class
| Original file | Purpose |
|---|---|
| `Intrusion Detection System v1.ipynb` | Earlier draft of v2; superseded |
| `Intrusion Detection System v2.ipynb` | The actual working pipeline + GUI (analyzed above) |
| `SVM Model.ipynb` | Standalone SVM-only experiment, no GWO |
| `Clean_KDDTrain.csv` / `Clean_KDDTest.csv` | Cached output of the cleaning step |
| `KDDTrain+aggregate.csv` / `KDDTest+aggregate.csv` | Raw source data |
| `GWO.PNG` / `bGwo.PNG` | Diagram images referenced in the README, not by code |

Key functions/classes and their purpose (all originally free functions or
one class, all in one notebook — mapped to their new module in Part 2):
- `wolf` — one GWO search agent (position, feature mask, fitness, score)
- `Fitness(P, NF, a, b)` — weighted accuracy + feature-parsimony score
- `stepc`, `stepb`, `X_d`, `cross_over`, `sigmoid` — bGWO/GWO position-to-binary decoding math
- `objective_function` — retrains the classifier on a candidate feature subset and returns (fitness, accuracy); **reloads the CSV from disk on every call**
- `GWO`, `bGWO` — the two optimizer main loops
- `perf_measure`, `AC`, `DR`, `AAP`, `ANF` — confusion-count and aggregate-run metrics
- `model_building`, `generate`, `get_data` — final-model training/prediction helpers, tightly coupled to Tkinter globals
- GUI functions (`btn1Action`, `create_testframe`, `test_result`, `clf_table`, root Tkinter setup) — presentation layer, tightly coupled to the optimization logic

### 8. Data flow through the repository
CSV → pandas DataFrame → NumPy array (`X`, `y`) → label-encode 3 columns →
StandardScaler on the rest → SVM fit/predict → (separately) GWO loop
repeatedly re-slices/rescales/retrains on candidate feature subsets → best
wolf's feature mask → final model retrained on that subset → Tkinter GUI
displays results and accepts manual single-row predictions.

### 9–14. Where each pipeline stage begins
- **Preprocessing** begins at the `df_train = pd.read_csv(...)` cell and
  ends at the `StandardScaler` cell (cells ~1–37 in the original).
- **GWO** begins at the `stepc` function definition (cell ~43) and runs
  through `GWO`/`bGWO` (cells ~51–52).
- **Feature selection** happens inside `GWO`/`bGWO`'s per-wolf position
  update + `objective_function` fitness evaluation.
- **Model training** happens inside `objective_function` (during search)
  and again inside `model_building`/`generate` (for the final chosen
  subset).
- **Prediction** happens via `classifier.predict(...)` inside
  `objective_function`, `generate`, and the GUI's `test_result`.
- **Evaluation** happens via `accuracy_score`/`classification_report`
  calls scattered across `objective_function`, `generate`, and the
  standalone metric functions (`AC`, `DR`, `AAP`, `ANF`).

### 15–17. Dependency, reusability, and refactoring assessment
- **Dataset-dependent modules:** `objective_function`, `get_data`,
  `model_building`, `create_testframe` all hardcode the filenames
  `Clean_KDDTrain.csv` / `Clean_KDDTest.csv` and the "41 features, indices
  1–3 are categorical" schema directly in their bodies.
- **Reusable modules (with cleanup):** `wolf`, `Fitness`, `stepc`/`stepb`/
  `X_d`/`cross_over`/`sigmoid`, `GWO`/`bGWO` core loops, and the metric
  functions are all dataset-agnostic in principle — they just weren't
  written that way (globals, hardcoded `41`, hardcoded `[1,2,3]`).
- **Modules that needed refactoring, and why:**
  - `objective_function` re-reads both CSVs from disk **on every single
    fitness evaluation** (hundreds of times per GWO run) — extremely
    wasteful; refactored to take pre-loaded, pre-encoded arrays once.
  - `classifier` and `sc_X` are **global variables mutated inside nested
    functions** — fragile, impossible to unit test, and silently
    reassigns state that other code paths read; refactored into an
    explicit `GWOFeatureSelector` class with no globals.
  - The categorical-column label encoders were **fit separately on train
    and test data** (`X_le.fit_transform` called twice), meaning the same
    category could map to different integers in train vs. test — a real
    bug. Refactored to fit once on train and reuse that mapping for test
    (`src/preprocessing/encoding.py`).
  - GWO's internal logic, the GUI, and the model training were **all
    interleaved in one notebook** with no separation of concerns —
    refactored into `gwo/`, `models/`, `preprocessing/`, `evaluation/`
    packages.
  - The GUI (Tkinter) was **removed entirely** in this rebuild — a
    desktop GUI has no place in a reusable, testable ML pipeline, and the
    spec's target architecture is a script/notebook pipeline (`main.py` +
    `notebooks/`), not a GUI application. All the *useful* logic the GUI
    triggered (`btn1Action`'s run loop, `AAP`/`ANF` aggregation) was kept
    and moved into `GWOFeatureSelector.run_multiple()`.
  - A stray non-ASCII byte in the source CSV's header
    (`num_failed_loginsì` instead of `num_failed_logins`) was silently
    passed through in the original notebook; fixed in
    `load_dataset.py`'s column-name normalization.

---

## Part 2 — Migration Map

| Old Repository File / Cell Group | Purpose | New Module | Reason for Refactoring |
|---|---|---|---|
| Cells 1–20 (`pd.read_csv`, `.rename`) | Load raw CSVs, normalize column names | `src/preprocessing/load_dataset.py` | Isolated as a single-responsibility loader; fixed stray non-ASCII column name |
| Cells 14–19 (`.duplicated()`, `.isnull()`) | Duplicate/missing-value handling | `src/preprocessing/clean_data.py` | Extracted into reusable, logged functions instead of inline notebook cells |
| Cells 21–22, 61 (`iloc[:, :-1]`, `iloc[:, -1]`) | Split X/y | `src/preprocessing/encoding.py` (`split_features_and_target`) | Grouped with encoding since both operate on the same raw array split |
| Cells 32–35, 61 (`LabelEncoder` on cols 1–3) | Categorical encoding | `src/preprocessing/encoding.py` | **Bug fixed**: encoder now fits once on train, reused on test (was fit twice, causing possible train/test label mismatch) |
| Cell 37 (`StandardScaler`) | Numeric feature scaling | `src/preprocessing/normalization.py` | Isolated single-responsibility scaler, parameterized by column indices instead of hardcoded `np.r_[0, 4:41]` |
| — (no original split util) | Train/validation split | `src/preprocessing/split_dataset.py` | New: added since the spec calls for an explicit split module; original relied solely on the pre-split KDDTrain/KDDTest files |
| Cells 43–47 (`stepc`, `stepb`, `X_d`, `cross_over`, `sigmoid`) | Position-decoding math | `src/gwo/fitness.py` | Grouped with `Fitness` since all are small numeric transfer functions feeding the fitness/decoding pipeline |
| Cell 48 (`Fitness`) | Weighted fitness score | `src/gwo/fitness.py` (`fitness_score`) | Renamed to avoid clashing with the module name; division-by-zero guard kept |
| Cell 50 (`class wolf`) | GWO search agent | `src/gwo/wolf.py` (`Wolf`) | Renamed to PEP8 class-naming convention; type hints and docstring added |
| Cells 51–52 (`GWO`, `bGWO`) | Optimizer main loops | `src/gwo/gwo_algorithm.py` | **Decoupled from data/classifier**: original functions called a global `objective_function` that re-read CSVs from disk; new versions take an injected `evaluate_fn` callback |
| Cell 49 (`objective_function`) | Fitness evaluation against real data | `src/gwo/feature_selector.py` (`GWOFeatureSelector._evaluate_feature_mask`) | Rewritten as a class method operating on pre-loaded arrays held in `self`, instead of a free function that globals-hacked `classifier`/`sc_X` and reloaded CSVs every call |
| Cells 59–61 (`model_building`, `generate`, `get_data`) | Final-model training on chosen features | `src/gwo/feature_selector.py` + `main.py` (steps 6–7) | Split cleanly: feature-subset scaling logic lives in the selector; final training/reporting lives in `main.py` |
| Cells 53–57 (`perf_measure`, `AC`, `DR`, `AAP`, `ANF`) | Confusion-count and run-aggregate metrics | `src/evaluation/metrics.py` | Vectorized `perf_measure` with NumPy instead of a Python for-loop; `AAP`/`ANF` re-exposed as properties on `GWORunResult` in `feature_selector.py` |
| Cell 41 (`accuracy_score`, `classification_report`) | Core classification metrics | `src/evaluation/metrics.py` | Extended with macro-averaged precision/recall/F1 helper (`compute_summary_metrics`) not present in the original |
| — (not present in original) | Confusion matrix plot | `src/evaluation/confusion_matrix.py` | New: required by the target spec, absent from the original notebook entirely |
| — (not present in original) | ROC curve plot | `src/evaluation/roc_curve.py` | New: required by the target spec; original had no ROC analysis |
| — (not present in original) | Baseline-vs-GWO comparison table | `src/evaluation/comparison.py` | New: makes GWO's actual contribution explicit and reportable, rather than only printing two separate classification reports |
| Cell 38 (`SVC(...)`) | SVM classifier | `src/models/svm.py` | Wrapped in the shared `BaseModel` interface so it's swappable with RF/DT/KNN |
| — (not present in original) | Random Forest, Decision Tree, KNN classifiers | `src/models/random_forest.py`, `decision_tree.py`, `knn.py` | New: required by the target spec; original only ever used SVM |
| — (not present in original) | Model factory | `src/models/model_factory.py` | New: single place that turns a config string into a fresh classifier instance, used identically by `main.py` and the GWO selector |
| Hardcoded literals throughout (`41`, `[1,2,3]`, `a=0.6, b=0.4`, population `12`, iterations `20`, etc.) | Configuration | `config.py` | Centralized; **no hardcoded values remain in any `src/` module** |
| Cells 62–67 (Tkinter GUI: `get_input`, `test_result`, `create_testframe`, `clf_table`, `btn1Action`, `root = Tk()`) | User interface | *(removed)* | A desktop GUI is out of scope for a reusable ML pipeline / research-project codebase; its useful run-orchestration logic (`btn1Action`'s multi-run loop) was preserved as `GWOFeatureSelector.run_multiple()` |
| Cells 69–77 (`SelectPercentile`, `SelectKBest`, etc.) | Alternative feature-selection exploration | *(removed)* | Dead code in the original — defined but never actually used by the GUI flow; omitted rather than carried forward unused |
| — (not present in original) | End-to-end orchestration script | `main.py` | New: required by the target spec; original had no non-GUI entry point at all |
| — (not present in original) | I/O helpers (save model, save features, ensure dirs) | `src/utils/helper.py` | New: centralizes `os.makedirs`/`joblib.dump` boilerplate that was absent (the original never saved a model to disk) |
| — (not present in original) | Convergence / feature-count plots | `src/utils/visualization.py` | New: required by the target spec's "Graph Generation" requirement |

---

## Project Architecture

```
IDS-GWO-FeatureSelection/
│
├── dataset/
│   ├── raw/                     KDDTrain.csv, KDDTest.csv (ship with repo)
│   ├── processed/                Clean_KDDTrain.csv, Clean_KDDTest.csv (generated by main.py)
│   └── selected_features.csv     Final GWO-selected feature mask (generated)
│
├── notebooks/
│   ├── Data_Analysis.ipynb       EDA: class balance, missing values, correlation heatmap
│   ├── Feature_Selection.ipynb   Interactive GWO run + inspect selected features
│   └── Model_Evaluation.ipynb    Load main.py's outputs, compare baseline vs GWO visually
│
├── src/
│   ├── preprocessing/
│   │   ├── load_dataset.py       Read raw CSVs, normalize column names
│   │   ├── clean_data.py         Drop duplicates, handle missing values
│   │   ├── encoding.py           Label-encode categorical columns (train-fit, test-reuse)
│   │   ├── normalization.py      StandardScaler for numeric columns
│   │   └── split_dataset.py      Optional stratified train/validation split
│   │
│   ├── gwo/
│   │   ├── wolf.py                The Wolf search-agent class
│   │   ├── fitness.py             Fitness function + sigmoid/step/crossover math
│   │   ├── gwo_algorithm.py       GWO and bGWO optimizer loops (alpha/beta/delta explicit)
│   │   └── feature_selector.py    Orchestrates GWO against a real classifier + dataset
│   │
│   ├── models/
│   │   ├── base_model.py          Shared build/fit/predict interface
│   │   ├── random_forest.py
│   │   ├── decision_tree.py
│   │   ├── svm.py
│   │   ├── knn.py
│   │   └── model_factory.py       config-driven "get me a fresh classifier" factory
│   │
│   ├── evaluation/
│   │   ├── metrics.py             Accuracy/precision/recall/F1, TP-FP-TN-FN, AAP/ANF
│   │   ├── confusion_matrix.py    Confusion matrix compute + heatmap plot
│   │   ├── roc_curve.py           One-vs-rest multi-class ROC + AUC plot
│   │   └── comparison.py          Baseline-vs-GWO comparison table
│   │
│   └── utils/
│       ├── helper.py              Dir creation, model save/load, feature-mask save, JSON save
│       ├── constants.py           Column name / schema constants
│       └── visualization.py       GWO convergence + feature-count bar plots
│
├── outputs/
│   ├── graphs/                    Confusion matrices, feature-count comparison (generated)
│   ├── reports/                   comparison_table.csv, run_summary.json (generated)
│   ├── models/                    baseline_<clf>.joblib, gwo_<clf>.joblib (generated)
│   └── selected_features/         selected_features.csv (generated)
│
├── config.py                      Every tunable value: paths, seed, classifier, GWO hyperparams
├── main.py                        Full pipeline: load → clean → encode → scale → GWO → train → evaluate → save
├── requirements.txt
└── README.md
```

---

## Configuration

Everything tunable lives in `config.py` — no hardcoded values anywhere in
`src/`. Key settings:

| Setting | Default | Meaning |
|---|---|---|
| `CLASSIFIER` | `"random_forest"` | Which model GWO optimizes for and `main.py` trains: `random_forest`, `decision_tree`, `svm`, `knn` |
| `GWO_VARIANT` | `"bGWO"` | `"GWO"` (sigmoid-decoded) or `"bGWO"` (binary + three-way crossover, generally better feature-space exploration) |
| `GWO_POPULATION_SIZE` | `12` | Number of wolves per run |
| `GWO_MAX_ITERATIONS` | `20` | Iterations per GWO run |
| `GWO_RUNS` | `3` | Independent GWO runs (for AAP/ANF averaging) |
| `FITNESS_ALPHA` / `FITNESS_BETA` | `0.6` / `0.4` | Weight on accuracy vs. feature-count in the fitness function |
| `RANDOM_SEED` | `42` | Used everywhere for reproducibility across both team members' machines |

---

## Why Random Forest as the default classifier (not SVM)

The original repository used SVM throughout. This rebuild defaults to
**Random Forest** instead, because:
- It trains dramatically faster than SVM on ~126k rows, which matters
  since GWO retrains the classifier hundreds of times per run.
- It handles the mixed label-encoded-categorical + numeric feature space
  natively, with no need for careful kernel/gamma tuning.
- It exposes feature importances as a free diagnostic to sanity-check
  against GWO's selected feature set.

SVM is still available and fully wired up (`CLASSIFIER = "svm"` in
`config.py`) if you want to reproduce the original paper's classifier
choice for comparison.

---

## Validation Checklist

- [x] All imports verified — `python -c "import main"` succeeds with no import errors
- [x] No missing modules — every `src/` subpackage has an `__init__.py`
- [x] No circular imports — dependency direction is strictly
      `preprocessing/gwo/models` → `evaluation`/`utils` → `main.py`
- [x] `main.py` executes the full pipeline end-to-end (smoke-tested on a
      stratified subsample of the shipped dataset, both `GWO` and `bGWO`
      variants)
- [x] Outputs generated automatically — `outputs/graphs/`, `outputs/reports/`,
      `outputs/models/`, `outputs/selected_features/` are created by
      `ensure_directories()` on every run
- [x] Models saved via `joblib` to `outputs/models/`
- [x] Graphs generated (confusion matrices, feature-count comparison) to
      `outputs/graphs/`
- [x] Selected features saved to both `outputs/selected_features/` and
      `dataset/selected_features.csv`
- [x] `requirements.txt` covers every third-party import used in `src/`
      and `main.py`

---

## Setup for a Two-Person Team (PyCharm)

1. Clone/open this folder as a PyCharm project, create a venv, and run
   `pip install -r requirements.txt`.
2. `python main.py` runs the entire pipeline with the settings in
   `config.py`. Reduce `GWO_POPULATION_SIZE`/`GWO_MAX_ITERATIONS`/`GWO_RUNS`
   while developing to keep iteration fast, then restore the full values
   for your final reported results.
3. Suggested split: one person owns `src/preprocessing/` +
   `src/evaluation/` + the `Data_Analysis.ipynb` / `Model_Evaluation.ipynb`
   notebooks; the other owns `src/gwo/` + `src/models/` +
   `Feature_Selection.ipynb`, since GWO is the piece most worth deeply
   understanding for a viva/defense.
4. `dataset/`, `outputs/`, and `.idea/` are git-ignored — only source code
   and notebooks should be committed (see `.gitignore`).
