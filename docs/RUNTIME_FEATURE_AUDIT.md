# Runtime Feature-Availability Audit — UNSW-NB15

This document is the single source of truth for which UNSW-NB15 features
are actually reconstructable from live traffic captured via Scapy on a
Kali VM, and which are not. `config.py`'s `RUNTIME_FEASIBLE_FEATURES`
list is derived directly from the "Obtainable" rows below. **The training
pipeline (`train.py`) trains its deployment model using ONLY the
"Obtainable" feature set** — GWO then selects a subset of *those*, so the
offline-trained model and the real-time VM model are configured to use the same column schema; several values remain approximations. No feature is ever faked, guessed, or filled with a
placeholder at runtime.

Legend: **✅ Obtainable** = computed directly from packet/flow data via
Scapy in `src/runtime/feature_extraction.py`. **⚠️ Approximate** =
computable but using a simplified definition that will not numerically
match UNSW-NB15's original Argus-based computation exactly. **❌
Excluded** = not implemented; reason given.

| # | Feature | UNSW-NB15 Definition | Runtime Status | Notes |
|---|---|---|---|---|
| 1 | `dur` | Record total duration | ✅ Obtainable | `last_packet_time - first_packet_time` per flow |
| 2 | `proto` | Transaction protocol | ✅ Obtainable | Read directly from IP header (`tcp`/`udp`/`icmp` etc via Scapy) |
| 3 | `service` | Inferred service (http, dns, ssh...) | ✅ Obtainable | Inferred from well-known destination port number (best-effort port→service map, not deep protocol inspection) |
| 4 | `state` | Connection state (FIN, CON, INT...) | ⚠️ Approximate | Simplified TCP state machine tracked from SYN/FIN/RST flags seen in-flow; UDP/ICMP mapped to a fixed placeholder state (`'CON'`/`'INT'`) rather than Argus's full state model |
| 5 | `spkts` | Source→dest packet count | ✅ Obtainable | Running counter per flow |
| 6 | `dpkts` | Dest→source packet count | ✅ Obtainable | Running counter per flow |
| 7 | `sbytes` | Source→dest bytes | ✅ Obtainable | Sum of packet lengths per direction |
| 8 | `dbytes` | Dest→source bytes | ✅ Obtainable | Sum of packet lengths per direction |
| 9 | `rate` | Packet rate | ✅ Obtainable | `(spkts + dpkts) / dur` |
| 10 | `sttl` | Source→dest TTL | ✅ Obtainable | TTL field of first source-side packet |
| 11 | `dttl` | Dest→source TTL | ✅ Obtainable | TTL field of first dest-side packet |
| 12 | `sload` | Source bits/sec | ✅ Obtainable | `sbytes * 8 / dur` |
| 13 | `dload` | Dest bits/sec | ✅ Obtainable | `dbytes * 8 / dur` |
| 14 | `sloss` | Source retransmit/drop count | ❌ Excluded | Requires sequence-number gap analysis; excluded entirely from the deployed feature vector |
| 15 | `dloss` | Dest retransmit/drop count | ❌ Excluded | Same as `sloss` |
| 16 | `sinpkt` | Source interpacket arrival time | ✅ Obtainable | Mean gap between consecutive source-side packet timestamps, converted to milliseconds |
| 17 | `dinpkt` | Dest interpacket arrival time | ✅ Obtainable | Mean gap between consecutive dest-side packet timestamps, converted to milliseconds |
| 18 | `sjit` | Source jitter | ⚠️ Approximate | Standard deviation of source interpacket gaps in milliseconds; approximation, not verified against the original extractor |
| 19 | `djit` | Dest jitter | ⚠️ Approximate | Same approach as `sjit` |
| 20 | `swin` | Source TCP window | ✅ Obtainable | TCP window field of first source-side packet (TCP flows only; 0 for UDP/ICMP) |
| 21 | `stcpb` | Source TCP base seq number | ✅ Obtainable | TCP sequence number of first source-side packet (TCP only) |
| 22 | `dtcpb` | Dest TCP base seq number | ✅ Obtainable | TCP sequence number of first dest-side packet (TCP only) |
| 23 | `dwin` | Dest TCP window | ✅ Obtainable | TCP window field of first dest-side packet (TCP only) |
| 24 | `tcprtt` | TCP setup round-trip time | ✅ Obtainable | `synack + ackdat` (TCP only; 0 for UDP/ICMP) |
| 25 | `synack` | Time between SYN and SYN-ACK | ✅ Obtainable | Timestamp difference (TCP only) |
| 26 | `ackdat` | Time between SYN-ACK and ACK | ✅ Obtainable | Timestamp difference (TCP only) |
| 27 | `smean` | Mean source packet size | ✅ Obtainable | `sbytes / spkts` |
| 28 | `dmean` | Mean dest packet size | ✅ Obtainable | `dbytes / dpkts` |
| 29 | `trans_depth` | HTTP request/response pipeline depth | ❌ Excluded | Requires full HTTP protocol parsing (not just port-based service inference); out of scope for a passive Scapy sniffer without an HTTP proxy/parser layer |
| 30 | `response_body_len` | HTTP response body size | ❌ Excluded | Same reason as `trans_depth` |
| 31 | `ct_srv_src` | Connections w/ same service+src in last 100 | ⚠️ Approximate | Computed from an in-memory rolling window of the last N completed flows (`FLOW_HISTORY_WINDOW` in config), not Argus's exact 100-connection/time-based definition |
| 32 | `ct_state_ttl` | Count by state+TTL range bucket | ⚠️ Approximate | Count of matching state in the completed-flow window; TTL ranges are not reconstructed, so this remains approximate |
| 33 | `ct_dst_ltm` | Connections to same dest in last 100 | ⚠️ Approximate | Same rolling-window approach as `ct_srv_src` |
| 34 | `ct_src_dport_ltm` | Connections w/ same src+dport in last 100 | ⚠️ Approximate | Same rolling-window approach |
| 35 | `ct_dst_sport_ltm` | Connections w/ same dst+sport in last 100 | ⚠️ Approximate | Same rolling-window approach |
| 36 | `ct_dst_src_ltm` | Connections w/ same src+dst in last 100 | ⚠️ Approximate | Same rolling-window approach |
| 37 | `is_ftp_login` | FTP session authenticated flag | ❌ Excluded | Requires FTP application-layer parsing |
| 38 | `ct_ftp_cmd` | Count of FTP commands in flow | ❌ Excluded | Same reason as `is_ftp_login` |
| 39 | `ct_flw_http_mthd` | Count of HTTP GET/POST methods | ❌ Excluded | Requires HTTP parsing; a naive payload string-search for "GET "/"POST " was considered but rejected as unreliable (fails on encrypted HTTPS, which is most modern traffic) |
| 40 | `ct_src_ltm` | Connections from same src in last 100 | ⚠️ Approximate | Same rolling-window approach |
| 41 | `ct_srv_dst` | Connections w/ same service+dst in last 100 | ⚠️ Approximate | Same rolling-window approach |
| 42 | `is_sm_ips_ports` | src IP+port == dst IP+port flag | ✅ Obtainable | Direct comparison, trivial to compute |

## Corrected summary and operational limits

The configured runtime schema has **35 columns**, with **7 excluded offline-only columns** (`sloss`, `dloss`, `trans_depth`, `response_body_len`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`). The original document's 33/9 totals were inconsistent and are corrected here.

Matching column names does **not** guarantee matching live/benchmark distributions. Port-based service inference, TCP state mapping, byte accounting, jitter and completed-flow history differ from the original Argus extractor. Millisecond timing units now follow the supplied `dataset/raw_unsw/NUSW-NB15_features.csv`; no live numerical equivalence claim is made.

The actual capture handler accepts **IPv4 TCP and UDP**, not all protocols named by the benchmark. IPv6, ICMP, fragmentation/reassembly, encrypted application semantics and packet-loss/retransmission reconstruction are not implemented. Missing TCP timings are zero where a handshake is not observed, which does not prove that a handshake did not occur.

The tracker locks packet updates and completed-flow removal. It allows at most 10,000 active flows and 10,000 retained packets per flow; overflow increments a dropped-packet counter exposed in collector status. These are protective bounds, not evidence of high-throughput capability. No throughput stress benchmark or live capture was executed during this update.

GWO may retain approximate features. Offline reports must not be represented as live intrusion-detection accuracy. A future study should compare extracted vectors against a trusted flow exporter on the same authorized packet capture and evaluate a newer representative dataset.
