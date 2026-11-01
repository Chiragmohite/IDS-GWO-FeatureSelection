"""Bounded passive-collector verification on an explicitly chosen, authorized interface."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing

ROOT=Path(__file__).resolve().parent

def verify(iface, seconds=30):
    if not iface.strip() or not 1 <= seconds <= 300: raise ValueError("Choose an interface and 1–300 seconds.")
    with tempfile.TemporaryDirectory(prefix="gwo-capture-") as directory:
        db=Path(directory)/"capture.db"
        env={**os.environ,"IDS_DB_PATH":str(db)}
        cmd=[sys.executable,"-u","realtime.py","--iface",iface,"--duration",str(seconds)]
        try:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,errors="replace",timeout=seconds+40)
        except subprocess.TimeoutExpired:
            return {"status":"timed_out","passive_capture_verified":False,"live_rows":0}
        rows=0;heartbeat=None
        if db.exists():
            with closing(sqlite3.connect(db)) as conn:
                rows=conn.execute("SELECT COUNT(*) FROM events WHERE source='live' AND observed_at IS NOT NULL").fetchone()[0]
                heartbeat=conn.execute("SELECT status FROM collector_health WHERE id=1").fetchone()
        ok=proc.returncode==0 and rows>0 and heartbeat==('stopped',)
        return {"status":"verified" if ok else ("collector_failed" if proc.returncode else "no_verified_flows"),
                "passive_capture_verified":ok,"live_rows":rows,"exit_code":proc.returncode,
                "last_heartbeat":heartbeat[0] if heartbeat else None,
                "diagnostic":None if proc.returncode==0 else "\n".join(proc.stdout.splitlines()[-4:])}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iface',required=True);p.add_argument('--seconds',type=int,default=30)
    a=p.parse_args();result=verify(a.iface,a.seconds);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passive_capture_verified'] else 1)
