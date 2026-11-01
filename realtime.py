"""
realtime.py  (patched: diagnosable capture + guaranteed DB write path)

Pipeline:
    Scapy capture -> FlowTracker -> extract_features -> RuntimeInferenceEngine
    -> record_event() -> logs/alerts.db (SQLite, table `events`)

What changed vs. the original (all about NOT failing silently):
  * Validates the capture interface up-front and lists the real ones.
    The default "eth0" only exists on Linux; on Windows it never matches.
  * Surfaces errors from Scapy's background sniffer thread. AsyncSniffer
    swallows them (e.g. wrong interface, no Npcap, no privileges) and the
    old main loop just slept forever printing nothing.
  * Prints a heartbeat every few seconds: packets seen, active flows,
    rows stored in the DB.
  * Warns clearly if no packets arrive, with the likely causes.
  * Flushes long-lived flows (active timeout) so they show up in the DB.
  * --selftest writes one synthetic flow through the whole pipeline so you
    can prove the DB side works independent of capture.

Usage:
    python realtime.py --list-ifaces
    python realtime.py --selftest
    python realtime.py --iface "Wi-Fi"        # Windows (run as Administrator)
    sudo python3 realtime.py --iface eth0     # Linux / Kali
"""

from datetime import datetime, timezone
import argparse
import logging
import os
import sys
import time

import config
from src.runtime.alerting import get_summary_stats, init_db, record_event, heartbeat
from src.runtime.feature_extraction import FlowHistory, extract_features
from src.runtime.flow_tracker import Flow, FlowTracker
from src.runtime.inference import RuntimeInferenceEngine
from src.utils.helper import ensure_directories

os.makedirs(config.LOGS_DIR, exist_ok=True)  # FileHandler below needs it

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.RUNTIME_LOG_PATH),
    ],
)
logger = logging.getLogger("realtime")

# Long flows are flushed after this long even if still active.
FLOW_MAX_AGE_SECONDS = getattr(config, "FLOW_MAX_AGE_SECONDS", 30)

# Simple counters shared between the sniffer thread and the main loop.
STATS = {"packets": 0, "ip_tcp_udp": 0, "skipped_non_ip": 0,
         "skipped_other_proto": 0, "flows_processed": 0, "events_stored": 0,
         "flows_failed": 0}


# ----------------------------------------------------------------------
# Scapy helpers
# ----------------------------------------------------------------------
def _require_scapy():
    try:
        from scapy.all import IP, TCP, UDP, AsyncSniffer, conf
        return IP, TCP, UDP, AsyncSniffer, conf
    except ImportError as e:
        raise ImportError(
            "Scapy is required for live capture: pip install scapy "
            "(Windows also needs Npcap from https://npcap.com, installed "
            "with 'WinPcap API-compatible mode')."
        ) from e


def list_interfaces():
    """Return [(name, description, ip)] for every interface Scapy can see."""
    _, _, _, _, conf = _require_scapy()
    rows = []
    for iface in conf.ifaces.values():
        rows.append((
            str(getattr(iface, "name", iface)),
            str(getattr(iface, "description", "") or ""),
            str(getattr(iface, "ip", "") or ""),
        ))
    return rows


def print_interfaces():
    rows = list_interfaces()
    print("\nInterfaces Scapy can capture on:\n")
    print(f"{'NAME (use this with --iface)':<34} {'IP':<16} DESCRIPTION")
    print("-" * 90)
    for name, desc, ip in rows:
        print(f"{name:<34} {ip:<16} {desc}")
    print()


def resolve_interface(requested, explicit):
    """
    Map the requested interface name onto a real one.
    explicit=True  -> user chose it (CLI/env): fail loudly if it's wrong.
    explicit=False -> it's just the 'eth0' default: fall back to Scapy's
                      default-route interface with a loud warning.
    """
    _, _, _, _, conf = _require_scapy()
    rows = list_interfaces()
    key = (requested or "").lower()
    for name, desc, _ip in rows:
        if key and (key == name.lower() or key == desc.lower()):
            return name

    if explicit:
        print_interfaces()
        raise SystemExit(
            f"ERROR: interface '{requested}' does not exist on this machine. "
            f"Pick one from the list above and pass it with --iface."
        )

    fallback = str(getattr(conf.iface, "name", conf.iface))
    logger.warning(
        "Default interface '%s' (from config.py) was not found on this machine. "
        "Falling back to Scapy's default-route interface '%s'. "
        "Use --list-ifaces and --iface to choose explicitly.",
        requested, fallback,
    )
    return fallback


def check_privileges(conf):
    if sys.platform.startswith("win"):
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                logger.warning("Not running as Administrator - Windows capture may fail "
                               "or see no packets. Open the terminal as Administrator.")
        except Exception:
            pass
        if not getattr(conf, "use_pcap", True):
            logger.warning("Npcap/libpcap not detected by Scapy - live capture will not work. "
                           "Install Npcap (https://npcap.com) with WinPcap-compatible mode.")
    elif hasattr(os, "geteuid") and os.geteuid() != 0:
        logger.warning("Not running as root - raw capture will probably fail. "
                       "Use an authorized capture session or analyze an existing PCAP.")


def _build_packet_handler(tracker, IP, TCP, UDP):
    from src.runtime.traffic_lab import packet_fields
    def handler(pkt):
        STATS["packets"] += 1
        fields, reason = packet_fields(pkt)
        if fields is None:
            STATS["skipped_non_ip" if reason == "non_ipv4" else "skipped_other_proto"] += 1
            return
        STATS["ip_tcp_udp"] += 1
        tracker.on_packet(**fields)
    return handler


# ----------------------------------------------------------------------
# Per-flow processing
# ----------------------------------------------------------------------
def process_completed_flow(flow: Flow, history: FlowHistory, engine: RuntimeInferenceEngine, source="live") -> bool:
    """feature extraction -> prediction -> DB write. Returns True if a row was stored."""
    try:
        features = extract_features(flow, history)
        prediction, confidence = engine.predict(features)
    except Exception:
        STATS["flows_failed"] += 1
        logger.exception("Flow %s:%s -> %s:%s skipped (extraction/prediction failed)",
                         flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port)
        return False
    finally:
        history.record({**flow.flow_key_summary(), "state": features.get("state", "unknown")} if "features" in locals() else flow.flow_key_summary())

    flow_summary = {
        "src_ip": flow.src_ip, "dst_ip": flow.dst_ip,
        "src_port": flow.src_port, "dst_port": flow.dst_port,
        "proto": flow.proto, "service": features["service"],
    }
    try:
        record_event(prediction, confidence, flow_summary, features, source=source,
                     model_version=engine.schema.get("model_version", "unversioned"),
                     observed_at=datetime.fromtimestamp(flow.last_seen, timezone.utc).isoformat(),
                     alert_threshold=engine.alert_threshold)
    except Exception:
        STATS["flows_failed"] += 1
        logger.exception("DATABASE WRITE FAILED for %s (db=%s)", flow_summary, config.ALERTS_DB_PATH)
        return False

    STATS["flows_processed"] += 1
    STATS["events_stored"] += 1
    return True


def run_selftest() -> None:
    """Push one synthetic TCP flow through the whole pipeline into the DB (no capture)."""
    ensure_directories(config.ALL_UNSW_OUTPUT_DIRS + [config.LOGS_DIR])
    init_db()
    before = get_summary_stats()["total_observations"]
    engine = RuntimeInferenceEngine()
    tracker, history = FlowTracker(15), FlowHistory(config.FLOW_HISTORY_WINDOW)
    t = time.time()
    seq = [("S", 0), ("SA", 1), ("A", 0), ("PA", 0), ("PA", 1), ("FA", 0), ("A", 1)]
    for i, (flags, reverse) in enumerate(seq):
        a = ("192.168.1.10", 50000, "192.168.1.20", 80)
        s, sp, d, dp = (a[2], a[3], a[0], a[1]) if reverse else a
        tracker.on_packet("tcp", s, sp, d, dp, 60 + i, 64, t + i * 0.01, flags, 1000 + i, 8192)
    for flow in tracker.pop_completed_flows():
        process_completed_flow(flow, history, engine, source="selftest")
    after = get_summary_stats()["total_observations"]
    print(f"\nSELFTEST: events in DB {before} -> {after}   (db: {config.ALERTS_DB_PATH})")
    print("PASS - model + DB path work. If live capture still stores nothing, the problem is capture."
          if after > before else "FAIL - see errors above.")
    if after <= before:
        raise SystemExit(1)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def run_monitor(iface_arg=None, duration=None) -> None:
    if duration is not None and not 1 <= duration <= 86400:
        raise ValueError("Duration must be 1–86400 seconds.")
    IP, TCP, UDP, AsyncSniffer, conf = _require_scapy()
    ensure_directories(config.ALL_UNSW_OUTPUT_DIRS + [config.LOGS_DIR])
    init_db()

    explicit = bool(iface_arg or os.environ.get("IDS_INTERFACE"))
    iface = resolve_interface(iface_arg or config.RUNTIME_INTERFACE, explicit)
    check_privileges(conf)

    logger.info("Loading trained model and preprocessing artifacts...")
    engine = RuntimeInferenceEngine()

    tracker = FlowTracker(config.FLOW_IDLE_TIMEOUT_SECONDS, max_flow_age_seconds=FLOW_MAX_AGE_SECONDS)
    history = FlowHistory(window_size=config.FLOW_HISTORY_WINDOW)
    handler = _build_packet_handler(tracker, IP, TCP, UDP)

    logger.info("Capturing on '%s'. DB: %s. Flows are stored when they finish (FIN/RST), "
                "go idle for %ds, or reach %ds of age. Ctrl+C to stop.",
                iface, config.ALERTS_DB_PATH, config.FLOW_IDLE_TIMEOUT_SECONDS, FLOW_MAX_AGE_SECONDS)

    sniffer = AsyncSniffer(iface=iface, prn=handler, store=False)
    sniffer.start()
    started = time.time()
    warned_silence = False
    collector_failed = False

    try:
        deadline = time.monotonic() + duration if duration is not None else None
        while deadline is None or time.monotonic() < deadline:
            pause = config.MONITORING_INTERVAL_SECONDS if deadline is None else min(config.MONITORING_INTERVAL_SECONDS, max(0, deadline-time.monotonic()))
            time.sleep(pause)

            # AsyncSniffer swallows startup errors inside its thread. Check.
            thread = getattr(sniffer, "thread", None)
            exc = getattr(sniffer, "exception", None)
            if exc is not None or (thread is not None and not thread.is_alive()):
                raise RuntimeError(
                    f"Packet capture on '{iface}' died: {exc!r}. Likely causes: wrong "
                    f"interface, missing Npcap (Windows), or not root/Administrator."
                )

            heartbeat("running", f"Interface={iface}; packets={STATS['packets']}; dropped_at_capacity={tracker.dropped_packets}; failed_flows={STATS['flows_failed']}")
            completed = tracker.pop_completed_flows()
            stored = sum(process_completed_flow(f, history, engine) for f in completed)

            logger.info("[heartbeat] packets=%d (tcp/udp=%d, non-IPv4=%d, icmp/other=%d) | "
                        "active_flows=%d | flushed=%d | rows_stored_this_run=%d",
                        STATS["packets"], STATS["ip_tcp_udp"], STATS["skipped_non_ip"],
                        STATS["skipped_other_proto"], tracker.active_flow_count(),
                        len(completed), STATS["events_stored"])

            if STATS["packets"] == 0 and not warned_silence and time.time() - started > 10:
                warned_silence = True
                logger.warning("0 packets captured after 10s on '%s'. Check: right interface "
                               "(--list-ifaces), root/Administrator, Npcap installed, and that "
                               "traffic is actually flowing on that adapter.", iface)
    except KeyboardInterrupt:
        logger.info("Stopping capture (Ctrl+C)...")
    except Exception:
        collector_failed = True
        heartbeat("error", "Capture failed. See local collector log.")
        raise
    finally:
        if not collector_failed:
            heartbeat("stopped", "Collector stopped.")
        try:
            sniffer.stop()
        except Exception as e:  # stop() re-raises a stored startup error
            logger.error("Sniffer reported: %s", e)
        for flow in tracker.pop_completed_flows(flush=True):   # include partially observed open flows on shutdown
            process_completed_flow(flow, history, engine)
        logger.info("Stopped. %d rows stored this run. Total in %s: %s",
                    STATS["events_stored"], config.ALERTS_DB_PATH, get_summary_stats())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real-time IDS monitor")
    ap.add_argument("--duration",type=int,help="stop after this many seconds; flush remaining flows")
    ap.add_argument("--iface", help="interface name (see --list-ifaces)")
    ap.add_argument("--list-ifaces", action="store_true", help="show capturable interfaces and exit")
    ap.add_argument("--selftest", action="store_true", help="push a synthetic flow into the DB and exit")
    args = ap.parse_args()

    if args.list_ifaces:
        print_interfaces()
    elif args.selftest:
        run_selftest()
    else:
        run_monitor(args.iface,args.duration)
