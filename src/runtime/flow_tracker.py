"""
src/runtime/flow_tracker.py

Maintains in-memory network flow state built from live packets captured
by Scapy. A "flow" here is a bidirectional 5-tuple session (protocol,
src IP, src port, dst IP, dst port), matching the granularity UNSW-NB15
was originally computed at. Flows are closed (pushed for feature
extraction) either on a TCP FIN/RST or after being idle for
config.FLOW_IDLE_TIMEOUT_SECONDS.

This module ONLY tracks what a passive packet sniffer can actually see.
It does not invent or approximate any value beyond what's documented as
obtainable/approximate in docs/RUNTIME_FEATURE_AUDIT.md.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimal, well-known port -> service map used for the `service` feature.
# UNSW-NB15's `service` column only distinguishes a handful of services
# and uses '-' for everything else -- this mirrors that granularity
# rather than attempting full protocol fingerprinting.
PORT_SERVICE_MAP = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "dns", 67: "dhcp", 68: "dhcp", 80: "http", 110: "pop3",
    143: "imap", 161: "snmp", 194: "irc", 443: "ssl", 993: "imap",
    995: "pop3",
}


def infer_service(sport: int, dport: int) -> str:
    """Best-effort service inference from well-known port numbers."""
    return PORT_SERVICE_MAP.get(dport, PORT_SERVICE_MAP.get(sport, "-"))


@dataclass
class PacketRecord:
    """A single observed packet's relevant fields, tagged with its flow direction."""
    timestamp: float
    direction: str  # "src_to_dst" or "dst_to_src"
    length: int
    ttl: int
    tcp_flags: Optional[str] = None
    tcp_seq: Optional[int] = None
    tcp_window: Optional[int] = None


@dataclass
class Flow:
    """
    Aggregated state for one bidirectional network flow, built up as
    packets arrive. `to_raw_stats()` turns this into the intermediate
    values feature_extraction.py needs to compute the final feature
    vector.
    """
    proto: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    first_seen: float
    last_seen: float = 0.0
    packets: List[PacketRecord] = field(default_factory=list)
    tcp_state_flags_seen: set = field(default_factory=set)

    def add_packet(self, pkt: PacketRecord) -> None:
        self.packets.append(pkt)
        self.last_seen = pkt.timestamp
        if pkt.tcp_flags:
            self.tcp_state_flags_seen.add(pkt.tcp_flags)

    @property
    def duration(self) -> float:
        return max(self.last_seen - self.first_seen, 0.0)

    @property
    def is_closed_by_flags(self) -> bool:
        """A TCP flow is considered complete once FIN or RST is observed."""
        return any("F" in flags or "R" in flags for flags in self.tcp_state_flags_seen)

    def flow_key_summary(self) -> Dict:
        """Small summary used for the rolling flow-history 'ct_*' features."""
        return {
            "proto": self.proto,
            "service": infer_service(self.src_port, self.dst_port),
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
        }


class FlowTracker:
    """
    Ingests packets one at a time (via `on_packet`, wired to Scapy's
    sniff callback in realtime.py) and maintains a dict of active flows.
    `pop_completed_flows()` returns and removes flows that are finished
    (FIN/RST seen) or have gone idle past the configured timeout, so the
    caller can run feature extraction + prediction on them.
    """

    def __init__(self, idle_timeout_seconds: float, max_flow_age_seconds: Optional[float] = None):
        self.idle_timeout_seconds = idle_timeout_seconds
        # Active-flow timeout: a long-lived flow (streaming, SSH session,
        # big download) is flushed after this many seconds even though it
        # has not gone idle or sent FIN/RST. Without this, such flows never
        # reach the database until they end. None = old behaviour.
        self.max_flow_age_seconds = max_flow_age_seconds
        self._flows: Dict[Tuple, Flow] = {}
        self._lock = threading.RLock()
        self.dropped_packets = 0
        self.max_flows = 10000
        self.max_packets_per_flow = 10000

    @staticmethod
    def _canonical_key(proto: str, ip_a: str, port_a: int, ip_b: str, port_b: int) -> Tuple:
        """
        Build a direction-independent flow key so packets from either
        direction of the same conversation map to the same flow.
        """
        if (ip_a, port_a) <= (ip_b, port_b):
            return (proto, ip_a, port_a, ip_b, port_b)
        return (proto, ip_b, port_b, ip_a, port_a)

    def on_packet(
        self,
        proto: str,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        length: int,
        ttl: int,
        timestamp: float,
        tcp_flags: Optional[str] = None,
        tcp_seq: Optional[int] = None,
        tcp_window: Optional[int] = None,
    ) -> None:
        """Register one observed packet against its flow, creating the flow if new."""
        with self._lock:
            key = self._canonical_key(proto, src_ip, src_port, dst_ip, dst_port)

            if key not in self._flows and len(self._flows) >= self.max_flows:
                self.dropped_packets += 1
                return
            if key not in self._flows:
                self._flows[key] = Flow(
                    proto=proto, src_ip=src_ip, src_port=src_port,
                    dst_ip=dst_ip, dst_port=dst_port, first_seen=timestamp,
                )

            flow = self._flows[key]
            direction = "src_to_dst" if (src_ip, src_port) == (flow.src_ip, flow.src_port) else "dst_to_src"
            if len(flow.packets) >= self.max_packets_per_flow:
                self.dropped_packets += 1
                return
            flow.add_packet(PacketRecord(
                timestamp=timestamp, direction=direction, length=length, ttl=ttl,
                tcp_flags=tcp_flags, tcp_seq=tcp_seq, tcp_window=tcp_window,
            ))

    def pop_completed_flows(self, now=None, *, flush=False) -> List[Flow]:
        """
        Remove and return every flow that is either explicitly closed
        (FIN/RST seen) or idle past the timeout, leaving all other
        still-active flows in place for future packets.
        """
        with self._lock:
            now = time.time() if now is None else now
            completed = []
            for key in list(self._flows.keys()):
                flow = self._flows[key]
                is_idle = (now - flow.last_seen) > self.idle_timeout_seconds
                is_too_old = (
                    self.max_flow_age_seconds is not None
                    and (now - flow.first_seen) > self.max_flow_age_seconds
                )
                if flush or flow.is_closed_by_flags or is_idle or is_too_old:
                    completed.append(flow)
                    del self._flows[key]
            return completed

    def active_flow_count(self) -> int:
        with self._lock:
            return len(self._flows)
