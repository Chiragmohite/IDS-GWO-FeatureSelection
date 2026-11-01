"""Research-only fixed-budget GWO, random-subset controls and validation-set alarm thresholds."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GroupShuffleSplit

import benchmark as bm
import config
from src.gwo.gwo_algorithm import grey_wolf_optimizer
from src.gwo.feature_selector import GWOFeatureSelector
from src.models.model_factory import get_model_factory
from src.preprocessing.unsw_preprocessing import load_unsw_raw


def threshold_for_fpr(y, scores, ceiling):
    if not 0 < ceiling < 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("A calibration set with both labels and an FPR target in (0,1) are required.")
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    feasible = np.flatnonzero(fpr <= ceiling)
    # Highest recall, then lowest FPR, then highest threshold among ties.
    j = max(feasible, key=lambda i: (tpr[i], -fpr[i], thresholds[i]))
    return float(thresholds[j]) if np.isfinite(thresholds[j]) else float(np.nextafter(1., 2.))


def policy_metrics(y, scores, threshold):
    m = confusion_matrix(y, scores >= threshold, labels=[0, 1])
    tn, fp, fn, tp = map(int, m.ravel())
    return {"threshold": threshold, "fpr_pct": 100*fp/(tn+fp), "attack_recall_pct": 100*tp/(tp+fn),
            "accuracy_pct": 100*(tp+tn)/len(y), "confusion_matrix": m.tolist()}


def calibration_partition(train, fit, validation, seed):
    keys = bm.fingerprints(train, config.RUNTIME_FEASIBLE_FEATURES)
    used = set(bm.fingerprints(fit, config.RUNTIME_FEASIBLE_FEATURES)) | set(bm.fingerprints(validation, config.RUNTIME_FEASIBLE_FEATURES))
    rest = train.loc[~keys.isin(used)]
    if len(rest) < 20: raise ValueError("Too few independent groups for calibration.")
    groups = keys.loc[rest.index]
    _, held = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=seed).split(rest, groups=groups))
    is_cal = keys.isin(set(groups.iloc[held]))
    return train.loc[~is_cal], train.loc[is_cal]


def run(output, *, budget=20, seeds=(42,43,44), wolves=8, iterations=20, jobs=2):
    start = time.perf_counter()
    out = Path(output)
    if not 1 <= budget <= 35 or not seeds or len(set(seeds)) != len(seeds) or wolves < 3 or iterations < 1 or jobs < 1:
        raise ValueError("Invalid study settings.")
    out.mkdir(parents=True, exist_ok=False)
    pools = {"offline_42": None, "runtime_35": list(config.RUNTIME_FEASIBLE_FEATURES)}
    paths = [Path(config.UNSW_TRAIN_PATH), Path(config.UNSW_TEST_PATH)]
    root = Path(__file__).resolve().parent
    src = [Path(__file__), root/'benchmark.py', root/'config.py', *sorted((root/'src').rglob('*.py'))]
    protocol = {"study":"budget_v1","gwo_seeds":list(seeds),"split_seed":42,"feature_budget":budget,
        "wolves":wolves,"iterations":iterations,"search_trees":32,"final_rf":dict(config.RANDOM_FOREST_PARAMS,n_jobs=jobs),
        "validation_fpr_target":.05,"calibration_group_seed":20261101,"selector":"continuous GWO with top-K rank decoder",
        "data_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "code_sha256":{str(p.resolve().relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src}}
    bm.write_json(out/'protocol.json',protocol)
    tr,te,fit,val,features,split=bm.prepare_splits(*[load_unsw_raw(str(p)) for p in paths],fit_limit=4800,validation_limit=1200,seed=42)
    tr,cal=calibration_partition(tr,fit,val,protocol['calibration_group_seed'])
    pools['offline_42']=features
    split.update(final_train_rows=len(tr),calibration_rows=len(cal),calibration_index_sha256=hashlib.sha256(np.asarray(cal.index,dtype='<i8').tobytes()).hexdigest())
    bm.write_json(out/'splits.json',split)
    xf,xv=bm.transform_pair(fit,val,features)
    base=get_model_factory("random_forest")
    candidates=[]; trace=[]
    for pool,names in pools.items():
        indices=[features.index(n) for n in names]
        candidates.append((pool,'baseline',None,list(names)))
        for seed in seeds:
            selector=GWOFeatureSelector(lambda:base().set_params(n_jobs=jobs,n_estimators=32),xf[:,indices],fit.label.to_numpy(),
                xv[:,indices],val.label.to_numpy(),list(range(len(names))),alpha=config.FITNESS_ALPHA,beta=config.FITNESS_BETA)
            def step(i,fitness,score,n):
                trace.append(dict(pool=pool,seed=seed,iteration=i,fitness=fitness,validation_accuracy_pct=score,features=n))
            wolf=grey_wolf_optimizer(selector._evaluate_feature_mask,len(names),wolves,iterations,0.,1.,seed,feature_budget=budget,on_iteration=step)
            chosen=[n for n,bit in zip(names,wolf.features) if bit]
            assert len(chosen)==budget
            candidates.append((pool,'ranked GWO',seed,chosen))
            rng=np.random.default_rng(seed)
            candidates.append((pool,'random subset',seed,[names[i] for i in sorted(rng.choice(len(names),budget,replace=False))]))
            print(f"{pool} seed {seed}: {budget} selected, validation {wolf.score:.2f}%",flush=True)
    bm.write_json(out/'masks.json',candidates)
    pd.DataFrame(trace).to_csv(out/'convergence.csv',index=False)
    # Calibration and test transforms fit only on the final training rows.
    xt,others=bm.transform_pair(tr,pd.concat([cal,te],ignore_index=True),features)
    xc,xe=others[:len(cal)],others[len(cal):]
    rows=[]
    for pool,method,seed,chosen in candidates:
        indices=[features.index(n) for n in chosen];m=base().set_params(n_jobs=jobs)
        m.fit(xt[:,indices],tr.label.to_numpy())
        pos=list(m.classes_).index(1)
        sc=m.predict_proba(xc[:,indices])[:,pos]
        threshold=threshold_for_fpr(cal.label,sc,protocol['validation_fpr_target'])
        st=m.predict_proba(xe[:,indices])[:,pos]
        row=dict(pool=pool,method=method,seed=seed,feature_count=len(chosen),features=chosen,candidate_features=len(pools[pool]),
            **bm.evaluate_model(m,xe[:,indices],te.label.to_numpy()),calibration=policy_metrics(cal.label,sc,threshold),
            calibrated_test=policy_metrics(te.label,st,threshold))
        rows.append(row)
        print(pool,method,seed,f"test accuracy {row['accuracy_pct']:.2f}%",flush=True)
    report=dict(status="completed",protocol=protocol,splits=split,results=rows,duration_seconds=round(time.perf_counter()-start,2))
    bm.write_json(out/'results.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/benchmarks/budget-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=p.parse_args();run(args.output)
    from evidence import write_pack
    print(write_pack(Path(args.output)))
