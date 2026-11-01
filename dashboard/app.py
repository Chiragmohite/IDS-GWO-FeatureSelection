"""Local analyst workstation for the existing GWO IDS pipeline."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import os
import logging
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import config
from src.runtime.alerting import (init_db, get_summary_stats, query_events, review_event,
    review_history, REVIEW_TRANSITIONS, collector_status, EVENT_SOURCES, source_totals)
from dashboard.labs import research, traffic
from src.runtime.artifacts import inspect_bundle
from src.runtime.batch import read_csv_bytes, analyze_frame, store_batch
from src.runtime.inference import RuntimeInferenceEngine

st.set_page_config(page_title="GWO IDS | Research console", page_icon="◈", layout="wide")
st.markdown("""<style>
:root{--ink:#162b43;--muted:#5d6e82;--teal:#087c80}
.stApp{background:#f3f6fa;color:var(--ink)}
.block-container{padding-top:2.3rem;max-width:1380px}
h1,h2,h3{color:var(--ink);letter-spacing:-.025em}
[data-testid="stSidebar"]{background:#13273f}
[data-testid="stSidebar"] *{color:#eef6ff}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#13273f}
[data-testid="stMetric"]{background:white;border:1px solid #dde5ee;border-radius:14px;padding:20px}
[data-testid="stMetricLabel"]{color:#5d6e82}
[data-testid="stMetricValue"]{color:#13273f}
.stButton>button{border-radius:8px;min-height:42px}
.hero{background:#fff;border:1px solid #dce5ee;border-left:4px solid #087c80;border-radius:12px;padding:24px 28px;margin:8px 0 24px}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.15em;color:#087c80;text-transform:uppercase}
.hero h2{margin:7px 0 8px;font-size:30px}.hero p{margin:0;color:#52667d;max-width:850px}
.footer{font-size:12px;color:#6c7b8e;border-top:1px solid #dce5ee;padding-top:18px;margin-top:32px}
@media(max-width:640px){.block-container{padding:1rem}.hero{padding:18px}.hero h2{font-size:24px}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>""", unsafe_allow_html=True)  # Static styling only; never interpolates user data.

with st.sidebar:
    st.markdown("## ◈ GWO IDS")
    st.caption("NETWORK DEFENSE / RESEARCH LAB")
    page = st.radio("Workspace", ["Overview", "Events & investigations", "CSV analysis", "Model lab", "Research results", "Traffic lab", "Setup & methodology"], label_visibility="collapsed")
    st.divider()
    st.caption("LOCAL ANALYST WORKSTATION")
    st.write("Grey Wolf Optimization + machine learning")
    st.caption("Passive detection. Human-reviewed alerts. No automatic blocking.")

try:
    init_db()
except Exception:
    st.error("The event database cannot be opened. Check IDS_DB_PATH and directory permissions.")
    st.stop()


def hero(kicker, title, text):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h2>{title}</h2><p>{text}</p></div>', unsafe_allow_html=True)


def model_report():
    try:
        return inspect_bundle()
    except (OSError, ValueError, KeyError) as exc:
        st.info(str(exc))
        return None


def show_events(rows):
    if not rows:
        st.info("No observations match this view. Analyze a CSV, replay the dataset, or start the collector.")
        return
    columns = ["id", "timestamp", "prediction", "confidence", "is_alert", "source", "proto", "service", "src_ip", "dst_ip", "review_state"]
    st.dataframe(pd.DataFrame(rows)[columns], hide_index=True, width="stretch")


if page == "Overview":
    hero("Security operations", "A clearer view of network activity.",
         "Inspect real observations, investigate model alerts and compare feature-selection experiments from one local workspace.")
    refresh = st.checkbox("Refresh observations every 5 seconds", value=False)

    @st.fragment(run_every="5s" if refresh else None)
    def overview():
        stats = get_summary_stats()
        cols = st.columns(4)
        for col, key, title in zip(cols, stats, ["Observed flows", "Predicted normal", "Predicted attack", "Model alerts"]):
            col.metric(title, stats[key])
        st.caption("Totals include live capture, replay and imported rows. Predictions are model estimates, not confirmed attacks.")
        health = collector_status()
        if not health:
            st.info("Collector has not reported a heartbeat. Offline analysis is available.")
        else:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(health["updated_at"])).total_seconds()
            if health["status"] == "running" and age < 30:
                st.success("Collector heartbeat received · " + health["details"])
            else:
                st.warning(f"Collector: {health['status']} · last heartbeat {int(age)} seconds ago. No live status is inferred from replay data.")
        rows, count = query_events(limit=500)
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Recent activity")
            if rows:
                # Avoid pandas native datetime conversion on concurrent Windows reruns.
                from src.runtime.alerting import hourly_counts
                buckets, invalid = hourly_counts(rows)
                if buckets:
                    trend = pd.DataFrame.from_dict(buckets,orient="index").fillna(0).sort_index()
                    st.bar_chart(trend,color=["#db765d","#0b8a87"] if len(trend.columns)==2 else "#0b8a87")
                if invalid:
                    st.caption(f"{invalid} records had an unreadable timestamp and were omitted from the chart.")
                st.caption(f"Recent {len(rows)} of {count} records; grouped by UTC hour.")
            else:
                st.info("Your activity chart will appear after observations are stored.")
        with right:
            st.subheader("Observation provenance")
            totals = source_totals()
            if totals:
                st.dataframe(pd.Series(totals,name="All stored flows"),width="stretch")
            st.caption("Dataset replay is an offline demonstration; it never means an attack is happening on this machine.")
        st.subheader("Latest alerts")
        alerts, _ = query_events(limit=10, only_alerts=True)
        show_events(alerts)
    overview()

elif page == "Events & investigations":
    hero("Investigation desk", "Evidence first. Decisions recorded.",
         "Filter observations, inspect features and record a review. A model alert alone is not proof of an intrusion.")
    c1, c2, c3 = st.columns(3)
    prediction = c1.selectbox("Prediction", ["all", "attack", "normal"])
    source = c2.selectbox("Source", ["all", *EVENT_SOURCES])
    state = c3.selectbox("Review state", ["all", *REVIEW_TRANSITIONS])
    only_alerts = st.checkbox("Only alerts", value=True)
    page_number = st.number_input("Page (50 records per page)", min_value=1, value=1, step=1)
    rows, total = query_events(limit=50, offset=(page_number - 1) * 50, prediction=prediction,
                              source=source, only_alerts=only_alerts, review_state=state)
    st.caption(f"{total} matching observations")
    show_events(rows)
    if rows:
        # Export fixed, non-personal columns only, preventing spreadsheet formula injection.
        export = pd.DataFrame(rows)[["id", "timestamp", "prediction", "confidence", "is_alert", "source", "review_state"]]
        st.download_button("Export this page", export.to_csv(index=False), "observations.csv", "text/csv")
        chosen = st.selectbox("Inspect observation", [r["id"] for r in rows])
        record = next(r for r in rows if r["id"] == chosen)
        st.subheader(f"Observation #{chosen}")
        st.write({"Prediction": record["prediction"], "Confidence": record["confidence"],
                  "Source": record["source"], "Observed at (UTC)": record.get("observed_at"), "Saved at (UTC)": record["timestamp"], "Batch": record.get("batch_id"), "Model version": record["model_version"]})
        with st.expander("Feature evidence"):
            st.json(json.loads(record["features_json"] or "{}"))
        if record["is_alert"]:
            with st.form("review"):
                action = st.selectbox("Next state", sorted(REVIEW_TRANSITIONS[record["review_state"]]))
                note = st.text_area("Investigation note / reason", max_chars=2000)
                submit = st.form_submit_button("Record review", type="primary")
            if submit:
                try:
                    review_event(chosen, action, note, record["review_state"])
                    st.success("Review and audit entry saved. Refresh the page to update filters.")
                except (ValueError, sqlite3.Error) as exc:
                    st.error(str(exc))
            st.subheader("Review history")
            history = review_history(chosen)
            if history:
                st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
            else:
                st.caption("No review has been recorded yet.")
        else:
            st.caption("This observation did not cross the configured alert threshold.")
        if st.button("Explain this prediction",key=f"explain_{chosen}"):
            from src.runtime.explain import explain_observation
            try:
                detail=explain_observation(RuntimeInferenceEngine(),record)
                st.write(f"Forest base attack score: {detail['base']:.1%} → this flow: {detail['score']:.1%}.")
                st.dataframe(pd.DataFrame(detail['contributions']),hide_index=True,width="stretch")
                st.caption("Signed changes along the trained trees' decision paths. They describe this model's score; they do not establish a cause or confirm an intrusion.")
            except (ValueError,RuntimeError,OSError,KeyError) as exc:st.info(str(exc))

elif page == "CSV analysis":
    hero("Offline analysis", "From flow data to inspectable predictions.",
         "Analyze a CSV using the same saved preprocessing, GWO feature subset and classifier as the live collector. No packet capture is needed.")
    st.caption("Up to 1,000 rows / 10 MB. Use UNSW-compatible flow features, not raw packet CSVs. Extra columns such as labels are excluded from inference.")
    sample_path = ROOT / "examples" / "flows.csv"
    if sample_path.is_file():
        st.download_button("Download sample flow CSV", sample_path.read_bytes(), "flows.csv", "text/csv")
    with st.form("csv_analysis"):
        uploaded = st.file_uploader("Flow CSV", type=["csv"])
        use_sample = st.checkbox("Use bundled benchmark sample", value=False)
        analyze = st.form_submit_button("Analyze flows", type="primary")
    if analyze:
        st.session_state.pop("analysis", None)
        try:
            if not uploaded and not use_sample:
                raise ValueError("Choose a CSV or select the bundled sample.")
            data = sample_path.read_bytes() if use_sample else uploaded.getvalue()
            frame = read_csv_bytes(data)
            with st.spinner("Applying the trained model…"):
                engine = RuntimeInferenceEngine()
                results, features = analyze_frame(frame, engine)
            st.session_state.analysis = (results, features, str(uuid.uuid4()),
                "dataset_replay" if use_sample else "csv_import", engine.schema.get("model_version"))
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            st.error(str(exc))
    if "analysis" in st.session_state:
        results, features, batch_id, origin, version = st.session_state.analysis
        st.success(f"Analyzed {len(results)} flows · {int(results.alert.sum())} model alerts · {origin}")
        st.dataframe(results, hide_index=True, width="stretch")
        st.download_button("Download predictions", results.to_csv(index=False), "predictions.csv", "text/csv")
        if st.button("Save observations to event history"):
            try:
                count = store_batch(results, features, batch_id, origin, version)
                st.success(f"Saved {count} observations." if count else "This analysis was already saved; no duplicate records added.")
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc))

elif page == "Model lab":
    hero("Experiment workspace", "Measure the model. Inspect the trade-offs.",
         "Compare all-feature and GWO-selected models on an independent final test split. Feature selection uses training-only validation.")
    report = model_report()
    if report:
        if report.get("quick_mode"):
            st.warning("Quick benchmark run: a small stratified sample, not a full-dataset or production evaluation.")
        cols = st.columns(4)
        cols[0].metric("All-feature test accuracy", f"{report['baseline_accuracy']:.2f}%")
        cols[1].metric("GWO test accuracy", f"{report['gwo_selected_accuracy']:.2f}%")
        cols[2].metric("Selected features", f"{report['n_features_selected']} / {report['n_features_total_runtime_feasible']}")
        cols[3].metric("Training time", f"{report['duration_seconds']:.1f}s")
        st.caption(f"{report['classifier']} · {report['gwo_variant']} · seed {report['random_seed']} · trained {report['trained_at_utc']}")
        a, b = st.columns(2)
        with a:
            st.subheader("Held-out classification report")
            rows = {k:v for k,v in report["selected_report"].items() if isinstance(v, dict)}
            st.dataframe(pd.DataFrame(rows).T.round(4), width="stretch")
            metrics = report.get("binary_metrics", {})
            if metrics:
                st.write(f"False-positive rate: {metrics['false_positive_rate']:.2%} · False-negative rate: {metrics['false_negative_rate']:.2%}")
                st.caption("These rates describe this test split. They do not establish live-network accuracy.")
        with b:
            st.subheader("Confusion matrix")
            st.dataframe(pd.DataFrame(report["confusion_matrix"], index=["Actual normal", "Actual attack"],
                                      columns=["Predicted normal", "Predicted attack"]), width="stretch")
            st.write({k:report[k] for k in ["fit_rows", "validation_rows", "test_rows", "removed_test_overlaps"]})
        st.subheader("Selected feature importance")
        importance = report.get("feature_importance", {})
        if importance:
            st.bar_chart(pd.Series(importance, name="importance").sort_values(ascending=False).head(15), color="#087c80")
            st.caption("Impurity-based model importance, not causal explanations. Correlated features can share importance.")
        st.write(", ".join(report["selected_features"]))
        st.download_button("Download experiment report", json.dumps(report, indent=2), "experiment.json", "application/json")
    st.subheader("Run a reproducible experiment")
    st.caption("Quick training uses 3,000 training / up to 1,000 independent test rows, 4 wolves and 3 iterations. Stop the collector before retraining.")
    if report and report.get("read_only_candidate"):
        st.caption(f"Compact profile · {report['train_rows']:,} training rows · {report['calibration_rows']:,} calibration rows · alert threshold {report['calibration_alert_threshold']:.3f}.")
    if st.button("Train quick GWO model", type="primary",disabled=bool(report and report.get("read_only_candidate"))):
        try:
            from train import run_training
            with st.spinner("Training and evaluating on the supplied benchmark…"):
                run_training(quick=True)
            st.success("Training completed. Reopen Model lab to view the new report.")
        except Exception as exc:
            logging.getLogger(__name__).exception("Training failed")
            st.error(f"Training failed: {type(exc).__name__}. Check the terminal output and dataset paths.")
    paths = sorted((Path(config.REPORTS_DIR) / "runs").glob("*.json"))
    if paths:
        st.subheader("Completed experiment history")
        run_rows = []
        for path in paths:
            r = json.loads(path.read_text())
            run_rows.append({k:r.get(k) for k in ["trained_at_utc", "classifier", "gwo_variant", "gwo_selected_accuracy", "baseline_accuracy", "n_features_selected", "quick_mode"]})
        st.dataframe(pd.DataFrame(run_rows), hide_index=True, width="stretch")
        st.caption("Repeated experiments on the same final split must not be used to select a winning model and then claim an unbiased test result.")

elif page == "Research results":
    research(ROOT)
elif page == "Traffic lab":
    traffic()
else:
    hero("Project guide", "A research project you can explain.",
         "The original GWO feature-selection and classification pipeline remains the core. This console adds reproducibility, investigation and a practical demonstration workflow.")
    st.subheader("Start here")
    st.code("python train.py --quick\npython replay.py --limit 100\npython -m streamlit run dashboard/app.py --server.address 127.0.0.1", language="powershell")
    st.subheader("Optional live capture")
    st.code('python realtime.py --list-ifaces\npython realtime.py --iface "Wi-Fi"', language="powershell")
    st.write("Windows capture requires Npcap and appropriate privileges. Capture only networks you own or have permission to monitor. CSV analysis and replay do not need either.")
    st.subheader("Scope and limits")
    st.write("This is a local, single-analyst academic console. It has no multi-user authentication and must remain bound to localhost. No traffic is blocked and no external attack traffic is generated.")
    st.write("Live IPv4 TCP/UDP feature extraction approximates several UNSW definitions; matching column names does not prove distribution equivalence. Confidence is an uncalibrated classifier probability. Evaluate representative authorized traffic before making operational claims.")
    st.write("Only locally trained, trusted model bundles are loaded. Never replace joblib/pickle files with untrusted downloads. Model checksums detect accidental mismatches; they are not cryptographic signatures.")
    st.caption(f"Database: {config.ALERTS_DB_PATH}")
    st.caption(f"Model directory: {config.SAVED_MODELS_DIR}")

st.markdown('<div class="footer">GWO IDS · Academic research console · All timestamps UTC · Human review required</div>', unsafe_allow_html=True)
