"""Classify a bounded local UDP probe or authorized classic PCAP with the saved model."""
import argparse
import json
from pathlib import Path
from src.runtime.alerting import init_db
from src.runtime.batch import MAX_BYTES
from src.runtime.inference import RuntimeInferenceEngine
from src.runtime.traffic_lab import loopback_probe, analyze_pcap, persist

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--probe',type=int,metavar="EXCHANGES");mode.add_argument('--pcap',type=Path)
    p.add_argument('--save',action='store_true',help='persist observations to the configured event database')
    a=p.parse_args();engine=RuntimeInferenceEngine()
    if a.pcap:
        with a.pcap.open('rb') as f:batch=analyze_pcap(f.read(MAX_BYTES+1),engine)
    else:batch=loopback_probe(engine,a.probe)
    summary={"source":batch["source"],"model":batch["version"],**batch["stats"],
             "predictions":batch["results"].to_dict("records"),"stored":0}
    if a.save:init_db();summary["stored"]=persist(batch)
    print(json.dumps(summary,indent=2))
