"""Local, additive tree-path contributions for a saved binary RF prediction."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def explain_observation(engine, event):
    if event.get("model_version") != engine.schema.get("model_version"):
        raise ValueError("The active model differs from the model that recorded this event.")
    if not isinstance(engine.model,RandomForestClassifier):
        raise ValueError("Decision-path explanations are available for Random Forest models.")
    import json
    raw=json.loads(event["features_json"]);x=engine._apply_preprocessing(raw)[:,engine.selected_idx].astype(np.float32)
    model=engine.model;pos=list(model.classes_).index(1)
    deltas=np.zeros(len(engine.selected_idx));base=0.
    for tree in model.estimators_:
        t=tree.tree_
        def attack(node):
            v=t.value[node,0];return float(v[pos]/v.sum())
        node=0;base+=attack(0)
        while t.children_left[node]!=t.children_right[node]:
            feat=t.feature[node]
            next_node=t.children_left[node] if x[0,feat]<=t.threshold[node] else t.children_right[node]
            deltas[feat]+=attack(next_node)-attack(node);node=next_node
    base/=len(model.estimators_);deltas/=len(model.estimators_)
    actual=float(model.predict_proba(x)[0,pos])
    if not np.isclose(base+deltas.sum(),actual,atol=1e-8):
        raise ValueError("Decision-path reconstruction did not match the forest.")
    stored=float(event["confidence"]) if event['prediction']=='attack' else 1-float(event['confidence'])
    if not np.isclose(actual,stored,atol=1e-7):raise ValueError("The recorded score could not be reproduced.")
    terms=[{"Feature":name,"Observed value":str(raw[name]),"Change in attack score (points)":float(100*delta)}
           for name,delta in zip(engine.selected_features,deltas) if abs(delta)>1e-12]
    return {"base":base,"score":actual,"contributions":sorted(terms,key=lambda v:-abs(v["Change in attack score (points)"]))}
