"""Build an optional runtime bundle from a completed study, selecting only by validation fitness."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import benchmark as bm
from budget_study import calibration_partition, threshold_for_fpr, policy_metrics
import config
from src.models.model_factory import get_model_factory
from src.evaluation.metrics import compute_binary_confusion_counts
from src.preprocessing.unsw_preprocessing import load_unsw_raw,bucket_rare_protocols,encode_categoricals,scale_numeric


def build(run, dest):
    if dest.exists():raise FileExistsError("Choose a new model directory; existing bundles are preserved.")
    start=time.perf_counter();prot=json.loads((run/'protocol.json').read_text())
    if prot.get('study')!='budget_v1':raise ValueError('Expected a fixed-budget study.')
    trace=pd.read_csv(run/'convergence.csv')
    last=trace[(trace.pool=='runtime_35') & (trace.iteration==prot['iterations'])].sort_values(['fitness','seed'],ascending=[False,True])
    if set(last.seed)!=set(prot['gwo_seeds']) or len(last)!=len(prot['gwo_seeds']):raise ValueError('Incomplete validation evidence.')
    seed=int(last.iloc[0].seed)
    masks=json.loads((run/'masks.json').read_text())
    selected=next(m[3] for m in masks if m[:3]==['runtime_35','ranked GWO',seed])
    names=list(config.RUNTIME_FEASIBLE_FEATURES)
    if len(set(selected))!=prot['feature_budget'] or not set(selected)<=set(names):raise ValueError('Invalid selected names.')
    paths=[Path(config.UNSW_TRAIN_PATH),Path(config.UNSW_TEST_PATH)]
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=prot['data_sha256'][p.name] for p in paths):raise ValueError('Data hashes changed.')
    tr,te,fit,val,_,sp=bm.prepare_splits(*[load_unsw_raw(str(p)) for p in paths],fit_limit=4800,validation_limit=1200,seed=prot['split_seed'])
    tr,cal=calibration_partition(tr,fit,val,prot['calibration_group_seed'])
    if set(np.unique(tr.label))!={0,1}:raise ValueError("Binary labels required.")
    numeric=[n for n in names if n not in config.UNSW_CATEGORICAL_COLUMNS]
    a,b,rare=bucket_rare_protocols(tr[names],pd.concat([cal[names],te[names]],ignore_index=True),threshold=config.UNSW_PROTO_RARE_THRESHOLD)
    a,b,enc=encode_categoricals(a,b,config.UNSW_CATEGORICAL_COLUMNS)
    a,b,scaler=scale_numeric(a,b,numeric)
    clf=get_model_factory('random_forest')().set_params(**prot['final_rf'])
    clf.fit(a[selected].to_numpy(),tr.label)
    prob=clf.predict_proba(b[selected].to_numpy())[:,list(clf.classes_).index(1)]
    cut=threshold_for_fpr(cal.label,prob[:len(cal)],prot['validation_fpr_target'])
    if cut<.5 or cut>1:raise ValueError("Calibration policy cannot be represented by the binary alert gate.")
    predicted=clf.predict(b[selected].to_numpy()[len(cal):]);matrix=policy_metrics(te.label,prob[len(cal):],cut)
    # Only now inspect final-test evidence; never use its scores to choose a seed.
    study=json.loads((run/'results.json').read_text())
    rows=study['results'];matched=next(r for r in rows if (r['pool'],r['method'],r['seed'])==('runtime_35','ranked GWO',seed))
    base=next(r for r in rows if (r['pool'],r['method'])==('runtime_35','baseline'))
    if not np.isclose(cut,matched['calibration']['threshold']) or matrix['confusion_matrix']!=matched['calibrated_test']['confusion_matrix'] or not np.isclose(100*np.mean(predicted==te.label),matched['accuracy_pct']):
        raise ValueError("Rebuilt candidate does not reproduce the frozen study.")
    dest.mkdir(parents=True,exist_ok=False)
    joblib.dump(clf,dest/'model.pkl',compress=3)
    joblib.dump({'categorical_encoders':enc,'rare_proto_values':rare},dest/'encoder.pkl',compress=3)
    joblib.dump(scaler,dest/'scaler.pkl',compress=3);joblib.dump(LabelEncoder().fit(tr.label),dest/'label_encoder.pkl')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    ver="compact20-"+sha(dest/'model.pkl')[:12]
    schema=dict(model_version=ver,runtime_feasible_features_in_order=names,categorical_features=config.UNSW_CATEGORICAL_COLUMNS,
        numeric_features=numeric,selected_features_in_order=selected,classification_mode="binary",alert_threshold=cut,
        threshold_source="separate training calibration groups",excluded_features=config.UNSW_EXCLUDED_FEATURES)
    bm.write_json(dest/'feature_schema.json',schema)
    pd.DataFrame({'feature':selected}).to_csv(dest/'selected_features.csv',index=False)
    report=dict(model_version=ver,read_only_candidate=True,quick_mode=False,trained_at_utc=datetime.now(timezone.utc).isoformat(),
        duration_seconds=round(time.perf_counter()-start,2),classifier='random_forest',gwo_variant='ranked GWO (20 features)',
        random_seed=seed,selection_rule='highest final validation fitness, then smallest seed',baseline_accuracy=base['accuracy_pct'],
        gwo_selected_accuracy=matched['accuracy_pct'],fit_rows=len(fit),validation_rows=len(val),train_rows=len(tr),calibration_rows=len(cal),
        test_rows=len(te),removed_test_overlaps=sp['removed_test_overlaps_on_35_features'],selected_features=selected,
        n_features_selected=len(selected),n_features_total_runtime_feasible=len(names),sklearn_version=sklearn.__version__,
        binary_metrics=compute_binary_confusion_counts(te.label,predicted),selected_report=classification_report(te.label,predicted,output_dict=True),
        confusion_matrix=matched['confusion_matrix'],feature_importance=dict(zip(selected,map(float,clf.feature_importances_))),
        calibration_alert_threshold=cut,held_out_alert_metrics=matrix,study_run=run.name,
        artifact_sha256={p.name:sha(p) for p in dest.iterdir() if p.name in {'model.pkl','encoder.pkl','scaler.pkl','label_encoder.pkl','feature_schema.json'}})
    bm.write_json(dest/'training_config.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('outputs/benchmarks/budget-20261101'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    r=build(args.study,args.output);print(r['model_version'],r['random_seed'],r['calibration_alert_threshold'],r['held_out_alert_metrics'])
