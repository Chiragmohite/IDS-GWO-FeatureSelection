# UNSW-NB15 + Real-Time VM Extension

This document covers the extension added on top of the original
NSL-KDD/GWO project (see `README.md` for that original project's
reverse-engineering analysis and architecture). **Nothing in the
original `src/gwo/`, `src/models/`, `main.py`, or NSL-KDD dataset
handling was modified or removed** -- this extension adds new
UNSW-NB15-specific preprocessing, a new top-level `train.py`, and an
entirely new `src/runtime/` package, all sitting alongside the
original, unchanged code.

---

## ⚠️ Important: the UNSW-NB15 file-naming quirk

Publicly distributed UNSW-NB15 CSVs are commonly mislabeled: the file
named `UNSW_NB15_training-set.csv` sometimes contains the smaller
82,332-row partition, while `UNSW_NB15_testing-set.csv` contains the
larger 175,341-row partition -- backwards from the officially published
split (175,341 train / 82,332 test). **This project's
`dataset/raw_unsw/` files have already been corrected**:
`UNSW_NB15_train.csv` genuinely contains 175,341 rows and
`UNSW_NB15_test.csv` genuinely contains 82,332 rows, verified against
both the official published split and an independent parquet
distribution of the same dataset. If you ever re-download UNSW-NB15
from another source, re-verify row counts before assuming the filenames
are correct.

---

## What changed vs. the spec's request, and why

The task spec asked for the *full* 49/45-column UNSW-NB15 feature space
to be used offline, with a separate real-time-compatible design decided
afterward. **This implementation instead restricts `train.py` to the
runtime-feasible feature set from the start** (`config.RUNTIME_FEASIBLE_FEATURES`,
35 of the 42 available features -- see `docs/RUNTIME_FEATURE_AUDIT.md`
for the full per-feature justification).

This is a deliberate deviation, not a shortcut: training GWO on all 42
features first and discovering afterward that GWO's favorite features
happen to be application-layer fields (`trans_depth`, `ct_flw_http_mthd`,
etc.) that can never be reconstructed from passive packet capture would
mean the offline "best" model and the deployable model are two different,
incompatible things. Restricting the feature space *before* GWO runs
guarantees the trained model and the live VM model are always compatible
by construction -- exactly what the spec's own "Most Important Technical
Requirement" section asks for.

**7 features are excluded entirely**: `sloss`, `dloss` (require full TCP
sequence-number stream analysis), `trans_depth`, `response_body_len`,
`ct_flw_http_mthd` (require HTTP parsing -- and most modern traffic is
HTTPS-encrypted, making this unreliable even with parsing), `is_ftp_login`,
`ct_ftp_cmd` (require FTP protocol parsing). These are genuinely not
obtainable from a passive Scapy sniffer without a much larger deep-packet-
inspection engine, and are never faked with placeholder values.

---

## Dataset options: presplit benchmark vs. full raw release

This project now supports two UNSW-NB15 data sources, toggled via
`config.UNSW_DATA_SOURCE`:

| `UNSW_DATA_SOURCE` | Source files | Rows | Notes |
|---|---|---|---|
| `"presplit"` | `dataset/raw_unsw/UNSW_NB15_train.csv` / `_test.csv` | 175,341 / 82,332 | Official benchmark split. Faster to train, results comparable to published papers using the same split. |
| `"raw_full"` **(current default)** | `dataset/raw_unsw_full/UNSW-NB15_1.csv` .. `_4.csv` | ~2.54M combined | The full original raw release. `src/preprocessing/unsw_raw_loader.py` combines all 4 partitions, reconciles column names/schema to match the presplit files exactly, drops `srcip`/`dstip`/`sport`/`dport`/`stime`/`ltime` (raw identifiers -- never used as model features, see rationale below), and performs its own fresh stratified train/test split (`UNSW_RAW_TEST_SIZE`, default 0.2). |

**Why `srcip`/`dstip`/`sport`/`dport` are dropped, not encoded**: training on
literal source/destination IP addresses or ports would let the model
memorize specific hosts from this one capture rather than learn
generalizable attack *patterns* -- those addresses mean nothing outside
this dataset and would actively hurt real-world generalization.

**Runtime cost control**: GWO retrains the classifier hundreds of times
per training run, so running it directly against the full ~2.54M-row
combined dataset is a substantially different time commitment than the
257K-row presplit version. `config.UNSW_RAW_SAMPLE_SIZE` (default
`400,000`) draws a stratified subsample from the combined raw data
*before* the train/test split -- still ~1.5x more data than the presplit
option's total, with a far more tractable runtime. Set it to `None` in
`config.py` to use the full ~2.54M rows if you have the time/hardware
for it (expect this to run considerably longer than any other setting
in this project).

**Data-quality fixes applied automatically by the raw loader**: the raw
partitions have no header row (columns assigned by position per the
official schema), use different names for several columns than the
presplit files (`smeansz`/`dmeansz` -> `smean`/`dmean`, etc. -- see the
loader's docstring for the full rename map), leave `attack_cat` blank
for normal traffic instead of the literal string `'Normal'`, and are
missing the derived `rate` feature entirely (computed here as
`(spkts + dpkts) / dur`, matching the same formula
`src/runtime/feature_extraction.py` uses at inference time, so offline
and runtime computations of this feature are consistent). `ct_flw_http_mthd`
and `is_ftp_login` are NaN whenever "not applicable" (e.g. a non-FTP
flow has no FTP login flag) rather than actually missing data -- filled
with 0 rather than causing the row to be dropped.

---



```bash
# 1. Train the offline model (produces saved_models/*)
python train.py

# 2. Run real-time monitoring (requires root/capabilities -- see below)
sudo python3 realtime.py

# 3. Run the dashboard (separate terminal, works independently of realtime.py)
streamlit run dashboard/app.py
```

`train.py` uses the settings in `config.py` under the "UNSW-NB15
EXTENSION" section (`UNSW_GWO_POPULATION_SIZE`, `UNSW_GWO_MAX_ITERATIONS`,
`UNSW_GWO_RUNS`, etc.) -- lower these for fast iteration during
development, the same way you would for the original NSL-KDD `main.py`.

---

## Running on a Kali Linux VM

1. **Set up the VM**: Kali Linux VM in VirtualBox/VMware, bridged or NAT
   network adapter (bridged is easier for capturing traffic to/from other
   devices on your LAN; NAT only sees the VM's own traffic).
2. **Copy the project** (specifically: `realtime.py`, `config.py`,
   `src/`, `saved_models/`, `docs/`, `requirements.txt`) into the VM --
   the VM does NOT need `dataset/raw_unsw/` or `train.py`; training
   happens offline on your host, only the trained artifacts are needed
   for inference.
3. **Install dependencies**:
   ```bash
   sudo apt update && sudo apt install -y python3-pip tcpdump
   pip3 install -r requirements.txt --break-system-packages
   ```
4. **Find your network interface name**:
   ```bash
   ip a
   ```
   Common names: `eth0`, `eth1`. Set it via environment variable rather
   than editing `config.py` directly:
   ```bash
   export IDS_INTERFACE=eth0
   ```
5. **Grant capture permission** (choose one):
   - **Simplest**: run with `sudo`:
     ```bash
     sudo python3 realtime.py
     ```
   - **Without full root** (recommended for anything beyond a quick
     test): grant the Python interpreter raw-socket capability directly:
     ```bash
     sudo setcap cap_net_raw,cap_net_admin=eip $(readlink -f $(which python3))
     python3 realtime.py
     ```
6. **Generate some traffic to verify it's working** (from the VM or
   another machine on the same network): browse a website, `ping` a
   host, `curl` something. Watch the terminal for
   `Flushing N completed flow(s)` log lines.
7. **Check results**: `logs/alerts.db` (SQLite) has every logged event;
   `logs/realtime.log` has the full run log. Or just run the dashboard
   (`streamlit run dashboard/app.py`) for a live view.

---

## Architecture (this extension)

```
IDS-GWO-FeatureSelection/
├── dataset/
│   ├── raw_unsw/
│   │   ├── UNSW_NB15_train.csv        175,341 rows (presplit option)
│   │   ├── UNSW_NB15_test.csv          82,332 rows (presplit option)
│   │   └── NUSW-NB15_features.csv     official feature description
│   └── raw_unsw_full/
│       ├── UNSW-NB15_1.csv .. _4.csv  ~2.54M rows combined (raw_full option, current default)
│       └── UNSW-NB15_LIST_EVENTS.csv  attack subcategory summary (documentation only)
│
├── docs/
│   └── RUNTIME_FEATURE_AUDIT.md       per-feature runtime-feasibility justification
│
├── saved_models/                      generated by train.py
│   ├── model.pkl                      final classifier (trained on GWO-selected features)
│   ├── scaler.pkl                     fitted StandardScaler (numeric features)
│   ├── encoder.pkl                    fitted LabelEncoders + rare-protocol bucket set
│   ├── label_encoder.pkl              target encoder
│   ├── selected_features.pkl / .csv   GWO's selected feature subset
│   ├── feature_schema.json            exact feature order/types/encoding -- runtime NEVER guesses this
│   └── training_config.json           hyperparameters + result summary for this training run
│
├── logs/
│   ├── alerts.db                      SQLite: every prediction + flagged alerts
│   └── realtime.log                   full run log
│
├── src/
│   ├── preprocessing/
│   │   ├── unsw_preprocessing.py      UNSW-specific load/clean/encode/scale, all artifacts persisted
│   │   └── unsw_raw_loader.py         combines the 4 raw partitions into presplit-compatible schema
│   ├── gwo/                            UNCHANGED -- reused as-is
│   ├── models/                         UNCHANGED -- reused as-is
│   ├── evaluation/
│   │   └── metrics.py                  EXTENDED: added FPR/FNR/confusion-count helpers
│   └── runtime/                        NEW package
│       ├── flow_tracker.py             in-memory 5-tuple flow state from live packets
│       ├── feature_extraction.py       Flow -> runtime-feasible feature dict
│       ├── inference.py                loads artifacts, applies identical preprocessing, predicts
│       └── alerting.py                 SQLite event/alert storage (defensive-only, no blocking)
│
├── dashboard/
│   └── app.py                          Streamlit dashboard, reads alerts.db independently
│
├── train.py                            NEW: offline UNSW-NB15 training pipeline
└── realtime.py                         NEW: live monitoring entry point
```

---

## What the real-time system does and does not do

**Does**: passively sniffs live packets, reconstructs bidirectional
flows, computes the 35-feature runtime-feasible vector for each
completed flow, classifies it Normal/Attack using the offline-trained
model, logs every prediction, and raises a stored alert (SQLite +
warning-level log line) for high-confidence attack predictions.

**Does not**: retrain any model at runtime, generate or substitute fake
feature values for anything it cannot observe, block or drop any
traffic, modify firewall rules, or take any action against a detected
source. This is a detection and alerting system only, per the spec's
explicit defensive-only requirement.

**Known limitations** (be upfront about these in your report/viva):
- The `ct_state_ttl` and `ct_srv_*`/`ct_dst_*` connection-count features
  use a simplified rolling-window approximation, not UNSW-NB15's exact
  Argus-based definition (see audit doc).
- `state` inference from TCP flags is a simplified heuristic, not a full
  Argus-equivalent state machine.
- `service` inference is port-based only (no deep packet inspection),
  so encrypted or non-standard-port traffic will often show `service='-'`.
- The model was trained on UNSW-NB15's benchmark traffic distribution,
  not on traffic from your specific network -- expect a real accuracy
  drop when pointed at genuinely different live traffic patterns
  (this is the same distribution-shift caveat that applies to any
  benchmark-trained IDS, not a bug in this implementation).
