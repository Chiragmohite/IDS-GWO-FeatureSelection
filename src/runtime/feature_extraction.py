"""
src/runtime/feature_extraction.py

Converts a completed src.runtime.flow_tracker.Flow into a feature
dictionary matching, feature-for-feature, config.RUNTIME_FEASIBLE_FEATURES
-- the exact same schema train.py trained the deployment model on. No
feature outside that list is ever produced here, and no feature inside
that list is ever faked; if a value cannot be computed for a given flow
(e.g. TCP timing fields on a UDP flow), it is set to 0, matching how
UNSW-NB15 itself encodes "not applicable" for such fields.

The `ct_*` connection-count features are computed from a rolling window
of recently completed flows (FlowHistory below) rather than UNSW-NB15's
original Argus-based 100-connection/time-window definition -- see
docs/RUNTIME_FEATURE_AUDIT.md, category "Approximate", for the exact
caveat that applies to each of these.
"""

import logging
from collections import deque
from typing import Deque, Dict, List

import numpy as np

from src.runtime.flow_tracker import Flow, infer_service

logger = logging.getLogger(__name__)


def _infer_state(flow: Flow) -> str:
    """
    Best-effort mapping of observed TCP flags (or lack thereof, for
    UDP/ICMP) onto UNSW-NB15's simplified state vocabulary. This is
    explicitly an approximation -- see docs/RUNTIME_FEATURE_AUDIT.md,
    feature #4 (`state`).
    """
    flags = flow.tcp_state_flags_seen
    if flow.proto != "tcp":
        return "CON" if len(flow.packets) > 1 else "INT"

    if any("R" in f for f in flags):
        return "RST"
    if any("F" in f for f in flags):
        return "FIN"
    if any(f == "S" for f in flags) and not any("A" in f for f in flags if f != "S"):
        return "REQ"
    if len(flow.packets) > 2:
        return "CON"
    return "INT"


def _tcp_handshake_timing(flow: Flow):
    """
    Extract SYN / SYN-ACK / ACK timestamps if present, to compute
    synack, ackdat, tcprtt. Returns zeros for non-TCP or incomplete
    handshakes (never fabricated).
    """
    if flow.proto != "tcp":
        return 0.0, 0.0, 0.0

    syn_t = synack_t = ack_t = None
    for pkt in flow.packets:
        f = pkt.tcp_flags or ""
        if f == "S" and syn_t is None:
            syn_t = pkt.timestamp
        elif f == "SA" and synack_t is None:
            synack_t = pkt.timestamp
        elif f == "A" and synack_t is not None and ack_t is None and pkt.timestamp > synack_t:
            ack_t = pkt.timestamp

    synack = max(synack_t - syn_t, 0.0) if (syn_t is not None and synack_t is not None) else 0.0
    ackdat = max(ack_t - synack_t, 0.0) if (synack_t is not None and ack_t is not None) else 0.0
    return synack, ackdat, synack + ackdat


class FlowHistory:
    """
    Rolling window of the most recently completed flows' summaries, used
    to compute the `ct_*` "connections matching X in the last N flows"
    family of approximate features. See docs/RUNTIME_FEATURE_AUDIT.md.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._history: Deque[Dict] = deque(maxlen=window_size)

    def record(self, flow_summary: Dict) -> None:
        self._history.append(flow_summary)

    def count_matching(self, **filters) -> int:
        """Count how many recent flows match all given key=value filters."""
        return sum(
            1 for h in self._history
            if all(h.get(k) == v for k, v in filters.items())
        )


def extract_features(flow: Flow, history: FlowHistory) -> Dict:
    """
    Compute the full runtime-feasible feature dictionary for one
    completed flow. Keys match config.RUNTIME_FEASIBLE_FEATURES exactly.

    Parameters
    ----------
    flow : Flow
        A completed flow popped from FlowTracker.
    history : FlowHistory
        Rolling window of recently completed flows (for ct_* features).
        The caller is responsible for calling history.record(...) with
        this flow's summary AFTER extraction, so a flow never counts
        itself.

    Returns
    -------
    Dict
        Feature name -> value, ready to be handed to
        src/runtime/inference.py for preprocessing + prediction.
    """
    src_pkts = [p for p in flow.packets if p.direction == "src_to_dst"]
    dst_pkts = [p for p in flow.packets if p.direction == "dst_to_src"]

    spkts = len(src_pkts)
    dpkts = len(dst_pkts)
    sbytes = sum(p.length for p in src_pkts)
    dbytes = sum(p.length for p in dst_pkts)
    dur = flow.duration

    sttl = src_pkts[0].ttl if src_pkts else 0
    dttl = dst_pkts[0].ttl if dst_pkts else 0
    swin = (src_pkts[0].tcp_window or 0) if src_pkts else 0
    dwin = (dst_pkts[0].tcp_window or 0) if dst_pkts else 0
    stcpb = (src_pkts[0].tcp_seq or 0) if src_pkts else 0
    dtcpb = (dst_pkts[0].tcp_seq or 0) if dst_pkts else 0

    def _interpacket_stats(pkts: List) -> "tuple[float, float]":
        if len(pkts) < 2:
            return 0.0, 0.0
        gaps = np.diff([p.timestamp for p in pkts]) * 1000.0  # UNSW definition: milliseconds
        return float(np.mean(gaps)), float(np.std(gaps))

    sinpkt, sjit = _interpacket_stats(src_pkts)
    dinpkt, djit = _interpacket_stats(dst_pkts)

    synack, ackdat, tcprtt = _tcp_handshake_timing(flow)

    rate = (spkts + dpkts) / dur if dur > 0 else 0.0
    sload = (sbytes * 8) / dur if dur > 0 else 0.0
    dload = (dbytes * 8) / dur if dur > 0 else 0.0
    smean = sbytes / spkts if spkts > 0 else 0
    dmean = dbytes / dpkts if dpkts > 0 else 0

    service = infer_service(flow.src_port, flow.dst_port)
    state = _infer_state(flow)
    is_sm_ips_ports = int(flow.src_ip == flow.dst_ip and flow.src_port == flow.dst_port)

    ct_srv_src = history.count_matching(service=service, src_ip=flow.src_ip)
    ct_srv_dst = history.count_matching(service=service, dst_ip=flow.dst_ip)
    ct_dst_ltm = history.count_matching(dst_ip=flow.dst_ip)
    ct_src_ltm = history.count_matching(src_ip=flow.src_ip)
    ct_src_dport_ltm = history.count_matching(src_ip=flow.src_ip, dst_port=flow.dst_port)
    ct_dst_sport_ltm = history.count_matching(dst_ip=flow.dst_ip, src_port=flow.src_port)
    ct_dst_src_ltm = history.count_matching(src_ip=flow.src_ip, dst_ip=flow.dst_ip)
    # ct_state_ttl: simplified bucket count by (state, sttl) pair, in place
    # of UNSW's original TTL-range binning table (see audit doc).
    ct_state_ttl = history.count_matching(state=state)

    features = {
        "dur": dur, "proto": flow.proto, "service": service, "state": state,
        "spkts": spkts, "dpkts": dpkts, "sbytes": sbytes, "dbytes": dbytes,
        "rate": rate, "sttl": sttl, "dttl": dttl, "sload": sload, "dload": dload,
        "sinpkt": sinpkt, "dinpkt": dinpkt, "sjit": sjit, "djit": djit,
        "swin": swin, "stcpb": stcpb, "dtcpb": dtcpb, "dwin": dwin,
        "tcprtt": tcprtt, "synack": synack, "ackdat": ackdat,
        "smean": smean, "dmean": dmean,
        "ct_srv_src": ct_srv_src, "ct_state_ttl": ct_state_ttl,
        "ct_dst_ltm": ct_dst_ltm, "ct_src_dport_ltm": ct_src_dport_ltm,
        "ct_dst_sport_ltm": ct_dst_sport_ltm, "ct_dst_src_ltm": ct_dst_src_ltm,
        "ct_src_ltm": ct_src_ltm, "ct_srv_dst": ct_srv_dst,
        "is_sm_ips_ports": is_sm_ips_ports,
    }
    return features
