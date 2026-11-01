"""Optional browser verification. Requires Node + Playwright installed separately.
Use IDS_CHROMIUM_EXECUTABLE to point at an already installed compatible browser.
Runs a temporary local server/database and never writes to the user's event DB.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import socket
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
with socket.socket() as listener:
    listener.bind(("127.0.0.1",0))
    port = listener.getsockname()[1]
url=f"http://127.0.0.1:{port}"
with tempfile.TemporaryDirectory(prefix="ids-browser-") as folder:
    env = {**os.environ, "IDS_DB_PATH": str(Path(folder)/"browser.db"),
           "IDS_BROWSER_OUTPUT": os.environ.get("IDS_BROWSER_OUTPUT", folder), "IDS_BROWSER_URL": url}
    Path(env["IDS_BROWSER_OUTPUT"]).mkdir(parents=True,exist_ok=True)
    with (Path(folder)/"server.log").open("w") as log:
        server = subprocess.Popen([sys.executable,"-m","streamlit","run","dashboard/app.py",
            "--server.headless","true","--server.address","127.0.0.1","--server.port",str(port)], stdout=log,stderr=log,env=env)
        try:
            for _ in range(50):
                if server.poll() is not None:
                    raise RuntimeError("Browser test server failed to start. Check port 8501.")
                try:
                    response=urllib.request.urlopen(url+"/_stcore/health",timeout=1)
                    print("HTTP health:",response.status,response.read().decode())
                    break
                except OSError:time.sleep(.2)
            else:raise RuntimeError("Browser test server did not become healthy.")
            subprocess.run(["node","scripts/browser_smoke.cjs"],check=True,env=env)
        finally:
            server.terminate()
            server.wait(timeout=10)
