"""Evaluate the actual saved binary model by labelled attack family; no retraining."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import config
import benchmark as bm
from src.preprocessing.unsw_preprocessing import load_unsw_raw
from src.runtime.inference import RuntimeInferenceEngine


def audit():
    engine=RuntimeInferenceEngine()
    _,te,_,_,_,splits=bm.prepare_splits(load_unsw_raw(config.UNSW_TRAIN_PATH),load_unsw_raw(config.UNSW_TEST_PATH),
                                          fit_limit=4800,validation_limit=1200,seed=42)
    labels=[];scores=[]
    for i in range(0,len(te),1000):
        y,c=engine.predict_many(te.iloc[i:i+1000]);labels.extend(y);scores.extend(c)
    pred=np.asarray(labels);alerts=(pred==1)&(np.asarray(scores)>=engine.alert_threshold)
    data=pd.DataFrame({"family":te.attack_cat.astype(str).str.strip().to_numpy(),"truth":te.label.to_numpy(),
                       "flagged":pred==1,"alerted":alerts})
    groups=[]
    for (family,label),g in data.groupby(["family","truth"]):
        groups.append(dict(family=family,actual_label=int(label),flows=len(g),
            predicted_attack=int(g.flagged.sum()),model_alerts=int(g.alerted.sum()),
            predicted_attack_pct=100*g.flagged.mean(),alert_pct=100*g.alerted.mean()))
    return dict(model_version=engine.schema["model_version"],feature_count=len(engine.selected_features),
        test_rows=len(te),test_row_index_sha256=splits["row_index_sha256"]["test"],
        classifier_accuracy_pct=float(np.mean(pred==te.label.to_numpy())*100),
        confusion_matrix=confusion_matrix(te.label,pred,labels=[0,1]).tolist(),
        configured_alert_threshold=engine.alert_threshold,groups=groups)


if __name__=="__main__":
    report=audit();out=Path(config.OUTPUT_DIR)/'audits'/(report["model_version"]+".json")
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(out,report["test_rows"],report["classifier_accuracy_pct"])
