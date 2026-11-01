import json
from pathlib import Path
import struct
import uuid
import zipfile

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
from budget_study import threshold_for_fpr, policy_metrics, calibration_partition
from evidence import comparison, convergence, read_run
from src.gwo.gwo_algorithm import grey_wolf_optimizer
from src.runtime.alerting import query_events, hourly_counts, source_totals
from src.runtime.explain import explain_observation
from src.runtime.packet_headers import decode_frame
from src.runtime.traffic_lab import analyze_pcap, persist, loopback_probe
from scripts.example_capture import make_capture, frame

ROOT=Path(__file__).resolve().parents[1]


def test_exact_budget_is_enforced_for_every_scored_mask():
    seen=[];trace=[]
    def score(mask):
        seen.append(mask.copy());s=sum(i*x for i,x in enumerate(mask));return s,s
    wolf=grey_wolf_optimizer(score,12,5,5,0,1,42,feature_budget=4,on_iteration=lambda *row:trace.append(row))
    assert all(sum(m)==4 for m in seen)
    assert sum(wolf.features)==4 and len(trace)==5
    assert [r[1] for r in trace]==sorted(r[1] for r in trace)
    with pytest.raises(ValueError):grey_wolf_optimizer(score,12,5,5,0,1,42,feature_budget=13)


def test_calibration_disjoint_and_threshold_uses_benign_ties():
    import benchmark as bm
    train=pd.read_csv(config.UNSW_TRAIN_PATH,nrows=600);a,b=train.iloc[:50],train.iloc[50:70]
    train,cal=calibration_partition(train,a,b,42)
    keys=lambda d:set(bm.fingerprints(d,config.RUNTIME_FEASIBLE_FEATURES))
    assert not keys(cal)&(keys(train)|keys(a)|keys(b))
    labels=np.array([0]*10+[1]*4);scores=np.array([.5]*8+[.9]*2+[.9,1.,1.,1.])
    cut=threshold_for_fpr(labels,scores,.05);actual=policy_metrics(labels,scores,cut)
    assert actual['fpr_pct']==0 and actual['attack_recall_pct']==75
    assert threshold_for_fpr(labels,np.ones(14),.05)>1


def test_pcap_observations_follow_capture_clock_and_are_idempotent(engine,db):
    data=make_capture();b=analyze_pcap(data,engine)
    assert (b['stats']['packets'],b['stats']['accepted_packets'],b['stats']['flows'])==(7,6,2)
    assert b['stats']['skipped']=={'non_ipv4':1}
    assert persist(b,db)==2 and persist(analyze_pcap(data,engine),db)==0
    rows,n=query_events(db_path=db,source='pcap_replay')
    assert n==2 and all(r['observed_at'].startswith('2023-11-14') and r['batch_id']==b['batch_id'] for r in rows)
    assert all(set(json.loads(r['features_json']))==set(config.RUNTIME_FEASIBLE_FEATURES) for r in rows)
    assert source_totals(db)=={'pcap_replay':2}


@pytest.mark.parametrize("kind",['bad_magic','truncated','backward_time','unsupported_link'])
def test_corrupt_pcap_is_rejected_without_writes(engine,db,kind):
    b=bytearray(make_capture())
    if kind=='bad_magic': b[:4]=b'bad!'
    elif kind=='truncated':del b[-1:]
    elif kind=='backward_time':
        second=24+16+struct.unpack_from('<I',b,32)[0];struct.pack_into('<I',b,second,123)
    else:struct.pack_into('<I',b,20,9999)
    with pytest.raises(ValueError):analyze_pcap(bytes(b),engine)
    assert query_events(db_path=db)[1]==0


def test_raw_ipv4_vlan_fragment_and_truncation():
    ethernet=frame('192.0.2.1','198.51.100.2',17,1000,53,b'q')
    raw=ethernet[14:];fields,_=decode_frame(raw,101,10)
    assert fields['length']==len(raw) and fields['src_port']==1000
    vlan=ethernet[:12]+b'\x81\x00\x00\x01\x08\x00'+raw
    assert decode_frame(vlan,1,10)[0]['ttl']==64
    fragment=bytearray(raw);fragment[6]=0x20
    assert decode_frame(fragment,101,10)[1]=='fragment'
    assert decode_frame(raw[:-1],101,10)[1]=='truncated'


def test_actual_udp_probe_and_local_explanation(engine,db):
    b=loopback_probe(engine,3)
    assert b['stats']['datagrams_received']==6
    assert all(x['spkts']==1 and x['dpkts']==1 for x in b['features'])
    assert persist(b,db)==3 and persist(b,db)==0
    rows,n=query_events(db_path=db,source='loopback_probe')
    assert n==3 and source_totals(db).get('live',0)==0
    for r in rows:
        d=explain_observation(engine,r)
        assert d['base']+sum(x['Change in attack score (points)'] for x in d['contributions'])/100==pytest.approx(d['score'])
        with pytest.raises(ValueError,match='differs'):explain_observation(engine,{**r,'model_version':'old'})
    with pytest.raises(ValueError):loopback_probe(engine,26)


def test_batched_predictions_match_frozen_encoder_and_estimator(engine):
    df=pd.read_csv(config.UNSW_TEST_PATH).head(12);df.loc[0,'service']='unseen-xyz'
    preds,scores=engine.predict_many(df)
    for i,row in enumerate(df.to_dict('records')):
        f=dict(row);f['proto']='other' if f['proto'] in engine.rare_proto_values else f['proto']
        for k,enc in engine.categorical_encoders.items():
            v=str(f[k]);f[k]=enc.transform([v if v in enc.classes_ else '__unknown__'])[0]
        f.update(zip(engine.numeric_features,engine.scaler.transform(pd.DataFrame([f])[engine.numeric_features])[0]))
        v=np.array([[f[n] for n in engine.selected_features]])
        label=engine.model.predict(v)[0]
        assert preds[i]==label and scores[i]==pytest.approx(engine.model.predict_proba(v)[0,list(engine.model.classes_).index(label)])


def test_timezone_bucket_invalid_and_legacy_rows():
    rows=[{'timestamp':v,'prediction':'normal'} for v in ['2026-01-01T03:20:00+03:00','2026-01-01T00:10:00','bad']]
    assert hourly_counts(rows)==({'2026-01-01 00:00Z':{'normal':2}},1)


@pytest.mark.parametrize("name",['matched-20261031','budget-20261101','reference-20261101'])
def test_evidence_bundle_corresponds_to_completed_run(name):
    folder=ROOT/'outputs'/'benchmarks'/name;report=read_run(folder);table=comparison(report)
    ref=name.startswith('reference-')
    assert len(convergence(folder))==(40 if ref else 120)
    with zipfile.ZipFile(folder/'research_evidence.zip') as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith('confusion_matrices/')])==len(table)
        assert {'comparison_table.md','data_profile.csv','gwo_convergence.png','selected_features.csv'}.issubset(z.namelist())
        with z.open('comparison_table.csv') as f:
            csv=pd.read_csv(f);assert np.allclose(csv.accuracy_pct,table.accuracy_pct)
        assert sum(r['candidate_features']==42 for r in report['results'])==(len(table) if ref else len(table)//2)
    if ref:
        for row in report['results']:
            assert sum(map(sum,row['strict_test']['confusion_matrix']))==report['splits']['strict_test_rows']


def test_workspaces_research_and_measured_flow_save(trained_bundle):
    app=AppTest.from_file(str(ROOT/'dashboard'/'app.py'),default_timeout=30).run()
    nav=lambda name:app.sidebar.radio[0].set_value(name).run()
    nav('Research results');assert not app.exception
    assert any('feature' in str(x.value).lower() for x in app.markdown)
    next(b for b in app.selectbox if b.label=='Completed experiment').set_value(ROOT/'outputs'/'benchmarks'/'reference-20261101').run()
    assert not app.exception and any('overlapping rows' in m.value for m in app.warning)
    nav('Traffic lab')
    next(b for b in app.button if b.label=='Measure local traffic').click().run()
    assert not app.exception
    next(b for b in app.button if b.label=='Save measured flows').click().run()
    assert app.success and query_events(source='loopback_probe')[1]>=4
    app.main.radio[0].set_value('Recorded PCAP').run();app.checkbox[0].check().run()
    next(b for b in app.button if b.label=='Analyze recorded capture').click().run()
    assert not app.exception and app.session_state['traffic_result_pcap_replay']['stats']['flows']==2
    app.main.radio[0].set_value('Passive interface').run();assert not app.exception


def test_compact_bundle_matches_validation_rule_and_alert_matrix(monkeypatch):
    from src.runtime.inference import RuntimeInferenceEngine
    folder=ROOT/'outputs/benchmarks/budget-20261101'
    bundle=ROOT/'artifacts/compact20'
    report=json.loads((bundle/'training_config.json').read_text())
    trace=convergence(folder)
    best=trace[(trace.pool=='runtime_35')&(trace.iteration==20)].sort_values(['fitness','seed'],ascending=[False,True]).iloc[0]
    assert report['random_seed']==best.seed
    monkeypatch.setattr(config,'SAVED_MODELS_DIR',str(bundle))
    for attr,n in {'MODEL_PATH':'model.pkl','ENCODER_PATH':'encoder.pkl','SCALER_PATH':'scaler.pkl',
                   'LABEL_ENCODER_PATH':'label_encoder.pkl','FEATURE_SCHEMA_PATH':'feature_schema.json'}.items():
        monkeypatch.setattr(config,attr,str(bundle/n))
    engine=RuntimeInferenceEngine();assert len(engine.selected_features)==20
    assert engine.alert_threshold==report['calibration_alert_threshold']
    assert report['model_version']==engine.schema['model_version']
    audit=json.loads((ROOT/'outputs/audits'/(engine.schema['model_version']+'.json')).read_text())
    normal=[g for g in audit['groups'] if not g['actual_label']]
    attacks=[g for g in audit['groups'] if g['actual_label']]
    m=report['held_out_alert_metrics']['confusion_matrix']
    assert m[0][1]==sum(g['model_alerts'] for g in normal)
    assert m[1][1]==sum(g['model_alerts'] for g in attacks)
    assert sum(map(sum,m))==audit['test_rows']==53915


def test_runtime_alert_threshold_and_record_time_are_preserved(db):
    from src.runtime.alerting import record_event
    for thresh in [.6,.8]:
        record_event(1,.72,{'proto':'tcp'},{'dur':1},db_path=db,source='live',
                     observed_at='2026-01-01T00:00:00+00:00',alert_threshold=thresh)
    rows,total=query_events(source='live',db_path=db)
    assert total==2 and [r['is_alert'] for r in rows]==[0,1]
    assert all(r['observed_at'].startswith('2026-01-01') for r in rows)


def test_compact_profile_training_is_guarded(monkeypatch):
    import train
    monkeypatch.setattr(config,'SAVED_MODELS_DIR',str(ROOT/'artifacts/compact20'))
    with pytest.raises(ValueError,match='read-only'):train.run_training(quick=True)


def test_parallel_schema_initialization_preserves_old_events(tmp_path):
    import sqlite3
    from contextlib import closing
    from concurrent.futures import ThreadPoolExecutor
    from src.runtime.alerting import _SCHEMA,init_db
    db=str(tmp_path/'legacy.db')
    with closing(sqlite3.connect(db)) as conn,conn:
        conn.executescript(_SCHEMA);conn.execute("INSERT INTO events(timestamp,prediction,confidence) VALUES ('2025-01-01','normal',.9)")
    with ThreadPoolExecutor(max_workers=3) as threads:
        list(threads.map(lambda _:init_db(db),range(3)))
    rows,n=query_events(source='legacy',db_path=db)
    assert n==1 and rows[0]['observed_at'] is None and rows[0]['batch_id'] is None
