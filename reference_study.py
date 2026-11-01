"""Reference screenshot split direction. Report literal raw scores AND a deduplicated view."""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import config
import benchmark as bm
from src.gwo.feature_selector import GWOFeatureSelector
from src.gwo.gwo_algorithm import grey_wolf_optimizer,binary_grey_wolf_optimizer
from src.models.model_factory import get_model_factory
from src.preprocessing.unsw_preprocessing import load_unsw_raw
from evidence import write_pack


def run(out):
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    # Match the visible sizes/roles, not the names printed on a downloaded CSV.
    paths=[Path(config.UNSW_TEST_PATH),Path(config.UNSW_TRAIN_PATH)]
    prot=dict(study="reference_direction_v1",train_file=paths[0].name,test_file=paths[1].name,iterations=20,
        wolves=8,gwo_seeds=[42],search_trees=32,split_seed=42,final_rf={**config.RANDOM_FOREST_PARAMS,"n_jobs":2},
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=["Raw test is duplicate-weighted and may overlap training. Strict-view results supplied separately.",
                     "Matches visible dataset direction and dimensions, not the friend's private algorithm or RF settings."])
    bm.write_json(out/'protocol.json',prot)
    tr,te=[load_unsw_raw(str(p)) for p in paths]
    features=[n for n in tr if n not in ('id','label','attack_cat')]
    if len(features)!=42 or len(tr)!=82332 or len(te)!=175341 or tr[features].isna().any().any() or te[features].isna().any().any():
        raise ValueError("Expected the intact shipped presplit UNSW files.")
    keys=bm.fingerprints(tr,features)
    ia,ib=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(tr,groups=keys))
    fit=bm.bounded_sample(tr.iloc[ia],4800,42);val=bm.bounded_sample(tr.iloc[ib],1200,42)
    test_keys=bm.fingerprints(te,features);overlaps=test_keys.isin(set(keys))
    # Count each distinct input/label pair once in the sensitivity view.
    first=~te.duplicated(features+['label']);strict=(~overlaps & first).to_numpy()
    split=dict(raw_rows=dict(train=len(tr),test=len(te)),final_train_rows=len(tr),final_test_rows=len(te),
        selection_fit_rows=len(fit),selection_validation_rows=len(val),raw_test_overlaps=int(overlaps.sum()),
        strict_test_rows=int(strict.sum()),strict_policy="exclude training input hashes, then deduplicate inputs+label")
    bm.write_json(out/'splits.json',split)
    xf,xv=bm.transform_pair(fit,val,features);factory=get_model_factory('random_forest')
    trace=[];masks={}
    for pool,alg,budget in [('reference_auto',binary_grey_wolf_optimizer,None),
                            ('reference_fixed20',grey_wolf_optimizer,20)]:
        selector=GWOFeatureSelector(lambda:factory().set_params(n_jobs=2,n_estimators=32),xf,fit.label.to_numpy(),xv,val.label.to_numpy(),
             list(range(42)),alpha=config.FITNESS_ALPHA,beta=config.FITNESS_BETA)
        def step(i,fitness,score,n):trace.append(dict(pool=pool,seed=42,iteration=i,fitness=fitness,validation_accuracy_pct=score,features=n))
        kw=dict(on_iteration=step)
        if budget is not None:kw['feature_budget']=budget
        win=alg(selector._evaluate_feature_mask,42,8,20,0,1,42,**kw)
        masks[pool]=[f for f,b in zip(features,win.features) if b]
        print(pool,len(masks[pool]),win.score,flush=True)
    bm.write_json(out/'selection.json',masks);pd.DataFrame(trace).to_csv(out/'convergence.csv',index=False)
    xt,xe=bm.transform_pair(tr,te,features);cache={};results=[]
    for pool,subset in masks.items():
        for method,names in [('baseline',features),('bGWO' if pool=='reference_auto' else 'ranked GWO',subset)]:
            key=tuple(names)
            if key not in cache:
                idx=[features.index(f) for f in names];m=factory().set_params(**prot['final_rf'])
                start=time.perf_counter();m.fit(xt[:,idx],tr.label.to_numpy());elapsed=time.perf_counter()-start
                raw=bm.evaluate_model(m,xe[:,idx],te.label.to_numpy())
                narrowed=bm.evaluate_model(m,xe[strict][:,idx],te.label.to_numpy()[strict])
                cache[key]=dict(**raw,strict_test=narrowed,fit_seconds=elapsed)
            results.append(dict(pool=pool,method=method,seed=None if method=='baseline' else 42,feature_count=len(names),
                candidate_features=42,features=names,**cache[key]))
            print(pool,method,results[-1]['accuracy_pct'],results[-1]['strict_test']['accuracy_pct'],flush=True)
    bm.write_json(out/'results.json',dict(status='completed',protocol=prot,splits=split,results=results,duration_seconds=time.perf_counter()-started))
    write_pack(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/benchmarks/reference-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run(p.parse_args().output)
