"""Research evidence and traffic workspaces for the existing local Streamlit console."""
import json
from pathlib import Path
import sqlite3
import zipfile

import pandas as pd
import streamlit as st
import config
from evidence import comparison, convergence, read_run
from src.runtime.alerting import collector_status, source_totals
from src.runtime.inference import RuntimeInferenceEngine
from src.runtime.traffic_lab import analyze_pcap, loopback_probe, persist


def research(root):
    st.title("Research results")
    st.write("Compare feature selection on the same held-out network flows. These experiments do not replace the active dashboard model.")
    base=root/"outputs"/"benchmarks"
    runs=sorted((p for p in base.glob('*') if p.is_dir() and not p.is_symlink() and (p/'results.json').is_file()),reverse=True)
    runs.sort(key=lambda p:not p.name.startswith("budget-"))  # current demo study first
    if not runs:
        st.info("No completed research runs yet. Run benchmark.py or budget_study.py.");return
    run=st.selectbox("Completed experiment",runs,format_func=lambda p:p.name)
    try:
        report=read_run(run);table=comparison(report);curves=convergence(run)
    except (ValueError,KeyError,OSError) as exc:
        st.error(f"Could not read the completed experiment: {exc}");return
    sp=report['splits']; pro=report['protocol']
    a,b,c=st.columns(3);a.metric("Final training flows",f"{sp['final_train_rows']:,}");b.metric("Held-out flows",f"{sp['final_test_rows']:,}")
    c.metric("GWO iterations",pro['iterations'])
    if pro.get("study")=="reference_direction_v1":
        st.warning(f"Screenshot split direction: raw test includes {sp['raw_test_overlaps']:,} overlapping rows. Inspect the deduplicated sensitivity view ({sp['strict_test_rows']:,} flows) before interpreting accuracy.")
        st.dataframe(pd.DataFrame([{"Pool":r["pool"],"Method":r["method"],"Raw accuracy (%)":r["accuracy_pct"],"Deduplicated accuracy (%)":r["strict_test"]["accuracy_pct"]} for r in report["results"]]).round(2),hide_index=True)
    else:st.caption(f"Fit/validation for selection: {sp['selection_fit_rows']:,} / {sp['selection_validation_rows']:,} · overlapping test inputs removed: {sp['removed_test_overlaps_on_35_features']:,}.")
    if 'calibration_rows' in sp:st.caption(f"Separate calibration flows: {sp['calibration_rows']:,}. Only calibration labels set the experimental alert threshold.")
    st.subheader("Baseline and selected subsets")
    st.dataframe(table[['pool','method','seed','feature_count','accuracy_pct','normal_f1-score','attack_f1-score','false_positive_rate_pct']].round(3),hide_index=True,width='stretch')
    st.download_button("Download comparison CSV",table.to_csv(index=False),"comparison_table.csv","text/csv")
    st.subheader("Inspect a model")
    rows=report['results'];i=st.selectbox("Model and seed",range(len(rows)),format_func=lambda j:
        f"{rows[j]['pool']} · {rows[j]['method']}"+(f" · seed {rows[j]['seed']}" if rows[j]['seed'] is not None else ""))
    row=rows[i];a,b=st.columns(2)
    with a:
        st.write("**Confusion matrix**")
        st.dataframe(pd.DataFrame(row['confusion_matrix'],index=['Actual normal','Actual attack'],columns=['Predicted normal','Predicted attack']),width='stretch')
    with b:
        st.write("**Per-class results**")
        st.dataframe(pd.DataFrame({k:v for k,v in row['classification_report'].items() if isinstance(v,dict)}).T,width='stretch')
    st.write(f"**{row['feature_count']} of {row['candidate_features']} features**: "+", ".join(row['features']))
    if row['method'] in ('ranked GWO','bGWO'):
        sub=curves[(curves.pool==row['pool'])&(curves.seed==row['seed'])].set_index('iteration')
        st.subheader("GWO convergence");st.line_chart(sub[['fitness']],color='#087c80')
        st.caption("Fitness uses training-validation records. A flat section means the best score did not improve.")
    if 'calibrated_test' in row:
        m=row['calibrated_test'];st.subheader("False alarms versus missed attacks")
        st.dataframe(pd.DataFrame([{"Policy":"Default classifier","False-positive rate (%)":row['false_positive_rate_pct'],"Attack recall (%)":100*row['attack_recall']},
            {"Policy":f"Calibration threshold {m['threshold']:.3f}","False-positive rate (%)":m['fpr_pct'],"Attack recall (%)":m['attack_recall_pct']}]).round(2),hide_index=True)
        st.caption("The 5% calibration target is not a guarantee on new data. The default profile is unchanged; only the optional compact model uses its own calibration threshold.")
    pack=run/'research_evidence.zip'
    if pack.is_file():
        with zipfile.ZipFile(pack) as files:
            st.download_button("Download comparison Markdown",files.read('comparison_table.md'),"comparison_table.md","text/markdown")
            st.download_button("Download convergence PNG",files.read('gwo_convergence.png'),"gwo_convergence.png","image/png")
            if 'data_profile.csv' in files.namelist():
                with st.expander('Source data and class balance'):
                    with files.open('data_profile.csv') as f:st.dataframe(pd.read_csv(f),hide_index=True)
        st.download_button("Download all evidence",pack.read_bytes(),pack.name,"application/zip")
    else:st.info(f"Generate downloads locally: python evidence.py outputs/benchmarks/{run.name}")


    st.subheader("Attack families in the labelled test data")
    audits=sorted((root/'outputs'/'audits').glob('*.json'))
    if audits:
        audit_path=st.selectbox('Audited saved model',audits,format_func=lambda p:p.stem)
        audit=json.loads(audit_path.read_text())
        st.caption(f"Saved bundle {audit['model_version']} · {audit['feature_count']} selected features · {audit['test_rows']:,} held-out records. This audit is separate from the research models above.")
        st.dataframe(pd.DataFrame(audit['groups']).round(2),hide_index=True,width='stretch')
        st.caption("Family names are reference labels used only for evaluation. The classifier predicts normal/attack, not the name of an attack. On normal rows, predicted_attack_pct is the false-positive rate.")
    else:st.caption("Run python audit_model.py to evaluate the saved bundle by labelled attack family.")


def traffic():
    st.title("Traffic lab")
    st.write("Choose where the observations come from. All three paths use the existing trained model. Only the passive collector monitors an adapter.")
    mode=st.radio("Traffic source",["Local UDP exchange","Recorded PCAP","Passive interface"],horizontal=True)
    source={"Local UDP exchange":"loopback_probe","Recorded PCAP":"pcap_replay","Passive interface":"live"}[mode]
    if source=="live":
        st.write("Run the collector in a separate terminal on a network you own or may monitor. Leave the dashboard running as a normal user.")
        st.code(r'.\.venv312\Scripts\python.exe realtime.py --list-ifaces',language='powershell')
        st.code(r'.\.venv312\Scripts\python.exe realtime.py --iface "Wi-Fi" --duration 60',language='powershell')
        st.caption("Use the exact adapter name printed by the first command. Windows requires Npcap. Supported traffic is IPv4 TCP/UDP.")
        health=collector_status()
        if health:st.write({"Last reported status":health['status'],"UTC heartbeat":health['updated_at'],"Details":health['details']})
        else:st.info("No collector heartbeat recorded.")
        st.write("Persisted live adapter flows:",source_totals().get("live",0));return
    inbox="traffic_result_"+source
    if source=="loopback_probe":
        st.write("Send and receive benign UDP datagrams between two temporary sockets on this computer. Measures actual traffic without Npcap; not a passive capture or an attack test.")
        count=st.number_input("Number of exchanges",min_value=1,max_value=25,value=4)
        start=st.button("Measure local traffic",type="primary")
    else:
        st.write("Analyze an authorized, time-ordered classic PCAP. Ethernet or raw IPv4; TCP/UDP only. Up to 10 MiB, 20,000 packets and 1,000 flows. Payloads are not saved.")
        uploaded=st.file_uploader("Packet capture",type=['pcap','cap'],key='pcap_file')
        sample=Path(config.BASE_DIR)/"examples"/"teaching_capture.pcap"
        use_example=st.checkbox("Use the synthetic teaching capture",value=False)
        if sample.exists():st.download_button("Download teaching PCAP",sample.read_bytes(),sample.name)
        start=st.button("Analyze recorded capture",type="primary")
    if start:
        st.session_state.pop(inbox,None)
        try:
            if source=='pcap_replay' and not use_example and not uploaded:raise ValueError("Choose a capture file first.")
            with st.spinner("Extracting flow measurements and classifying…"):
                engine=RuntimeInferenceEngine()
                batch=loopback_probe(engine,int(count)) if source=='loopback_probe' else analyze_pcap(sample.read_bytes() if use_example else uploaded.getvalue(),engine)
                st.session_state[inbox]=batch
        except (OSError,ValueError,RuntimeError,KeyError) as exc:st.error(str(exc))
    if inbox in st.session_state:
        batch=st.session_state[inbox];st.write({"Source":source,"Model":batch['version'],**batch['stats']})
        st.dataframe(pd.concat([pd.DataFrame(batch['metadata']),batch['results']],axis=1),hide_index=True,width='stretch')
        if st.button("Save measured flows",key="save_"+source):
            try:
                n=persist(batch);st.success(f"Saved {n} observations." if n else "Already saved; no duplicate observations.")
            except (OSError,ValueError,sqlite3.Error) as exc:st.error(str(exc))
        st.caption("Predicted attack is a model output. These flows are not labelled attack examples.")
