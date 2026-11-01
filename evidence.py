"""Build a reviewable research evidence pack exclusively from completed run records."""
import argparse
import hashlib
import config
import io
import json
import re
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_run(folder):
    report=json.loads((folder/'results.json').read_text(encoding="utf-8"))
    if report.get("status")!="completed" or not report.get("results"):raise ValueError("No completed evaluation.")
    for row in report['results']:
        m=np.asarray(row['confusion_matrix'])
        if m.shape!=(2,2) or np.any(m<0) or int(m.sum())!=report['splits']['final_test_rows']:
            raise ValueError("Inconsistent held-out confusion matrix.")
    return report


def convergence(folder):
    if (folder/'convergence.csv').exists():
        return pd.read_csv(folder/'convergence.csv')
    pool=None;pending=[];records=[]
    for line in (folder/'run.log').read_text(encoding="utf-8").splitlines():
        if hit:=re.search(r"Starting selection: (\w+)",line):pool=hit[1]
        if hit:=re.search(r"Iteration (\d+)/\d+ \| best fitness=([.\d]+) score=([.\d]+) n_features=(\d+)",line):
            pending.append(dict(pool=pool,iteration=int(hit[1]),fitness=float(hit[2]),validation_accuracy_pct=float(hit[3]),features=int(hit[4])))
        if hit:=re.search(r"Seed (\d+): selected",line):
            records.extend(dict(seed=int(hit[1]),**r) for r in pending);pending=[]
    if pending or not records:raise ValueError("Convergence evidence incomplete.")
    return pd.DataFrame(records)


def comparison(report):
    flat=[]
    for r in report["results"]:
        row={k:r[k] for k in ["pool","method","seed","feature_count","candidate_features","accuracy_pct","false_positive_rate_pct","false_negative_rate_pct","roc_auc"]}
        for cls in ("normal","attack"):
            for metric in ("precision","recall","f1-score","support"):
                row[cls+"_"+metric]=r["classification_report"][cls][metric]
        row.update({k:r[k] for k in ("fit_seconds","predict_seconds","selection_seconds") if k in r})
        flat.append(row)
    return pd.DataFrame(flat)


def png(fig):
    out=io.BytesIO()
    try: fig.savefig(out,format="png",dpi=135,bbox_inches="tight")
    finally: plt.close(fig)
    return out.getvalue()


def write_pack(folder):
    report=read_run(folder); curves=convergence(folder);table=comparison(report)
    expected={(r['pool'],r['seed']) for r in report['results'] if r['method'] in {'ranked GWO','bGWO'}}
    iterations=report['protocol']['iterations']
    if set(curves.groupby(['pool','seed']).groups)!=expected or any(sorted(g.iteration)!=list(range(1,iterations+1)) for _,g in curves.groupby(['pool','seed'])):
        raise ValueError("Missing or duplicate search iterations.")
    output=folder/'research_evidence.zip'
    temporary=output.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(folder/'results.json','results.json')
        profile=[]
        data_paths=[Path(config.UNSW_TRAIN_PATH),Path(config.UNSW_TEST_PATH)]
        if report["protocol"].get("study")=="reference_direction_v1":data_paths.reverse()
        for role,path in zip(["raw_train","raw_test"],data_paths):
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==report['protocol']['data_sha256'].get(path.name):
                labels=pd.read_csv(path,usecols=['label']).label
                profile.append(dict(split=role,rows=len(labels),normal=int((labels==0).sum()),attack=int((labels==1).sum()),attack_pct=100*labels.mean()))
        if profile:z.writestr("data_profile.csv",pd.DataFrame(profile).to_csv(index=False))
        for n in ['protocol.json','splits.json','selection.json','masks.json','run.log']:
            if (folder/n).is_file():z.write(folder/n,n)
        z.writestr('comparison_table.csv',table.to_csv(index=False))
        z.writestr('gwo_convergence.csv',curves.to_csv(index=False))
        md=['# UNSW-NB15 experiment comparison','', 'Rows are evaluated on the same held-out flows in this experiment. Seeds vary the optimizer; results are not independent datasets.','',
            '| Pool | Method | Seed | Features | Accuracy | Normal P/R/F1 | Attack P/R/F1 | FPR |','|---|---|---:|---:|---:|---|---|---:|']
        features=[]
        for i,r in enumerate(report['results']):
            m=np.asarray(r['confusion_matrix']);key=f"{i+1:02d}_{r['pool']}"
            desc=f"{r['pool']} / {r['method']} / seed {r['seed'] if r['seed'] is not None else '—'}"
            md.append(f"| {r['pool']} | {r['method']} | {r['seed'] if r['seed'] is not None else '—'} | {r['feature_count']} | {r['accuracy_pct']:.2f}% | {r['classification_report']['normal']['precision']:.3f} / {r['classification_report']['normal']['recall']:.3f} / {r['classification_report']['normal']['f1-score']:.3f} | {r['attack_precision']:.3f} / {r['attack_recall']:.3f} / {r['attack_f1']:.3f} | {r['false_positive_rate_pct']:.2f}% |")
            features.append(desc+"\n"+", ".join(r['features'])+"\n")
            fig,ax=plt.subplots(figsize=(5,4));ax.imshow(m,cmap='Blues')
            for y in range(2):
                for x in range(2):ax.text(x,y,f'{m[y,x]:,}',ha='center',va='center',color='white' if m[y,x]>m.max()*.55 else 'black')
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Attack'],yticklabels=['Normal','Attack'],
                   xlabel='Predicted class',ylabel='Actual class',title=desc)
            z.writestr('confusion_matrices/'+key+'.png',png(fig))

        z.writestr('selected_features.csv',pd.DataFrame([{"pool":r["pool"],"method":r["method"],"seed":r["seed"],"feature":f} for r in report["results"] for f in r["features"]]).to_csv(index=False))
        md.extend(["", f"Experiment duration: {report['duration_seconds']:.1f} seconds."])
        z.writestr('comparison_table.md',"\n".join(md)+"\n")
        z.writestr('selected_features.txt',"\n".join(features))
        fig,axes=plt.subplots(2,1,figsize=(8,6),layout='constrained')
        for ax,(pool,g) in zip(axes,curves.groupby('pool')):
            for seed,s in g.groupby('seed'):
                ax.plot(s.iteration,s.fitness,label=f'seed {int(seed)}')
            ax.set(xlabel='GWO iteration',ylabel='Best validation fitness',title=pool);ax.legend();ax.grid(alpha=.2)
        z.writestr('gwo_convergence.png',png(fig))
        strict=[dict(pool=r["pool"],method=r["method"],raw_accuracy_pct=r["accuracy_pct"],strict_accuracy_pct=r["strict_test"]["accuracy_pct"],
                     strict_fpr_pct=r["strict_test"]["false_positive_rate_pct"]) for r in report["results"] if "strict_test" in r]
        if strict:z.writestr("raw_and_deduplicated_test.csv",pd.DataFrame(strict).to_csv(index=False))
        policies=[{"pool":r['pool'],"method":r['method'],"seed":r['seed'],**{k:v for k,v in r['calibrated_test'].items() if k!='confusion_matrix'}} for r in report['results'] if 'calibrated_test' in r]
        if policies:z.writestr('validation_threshold_test_results.csv',pd.DataFrame(policies).to_csv(index=False))
    temporary.replace(output)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path)
    print(write_pack(p.parse_args().run))
