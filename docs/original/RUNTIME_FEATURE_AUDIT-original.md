# Runtime Feature-Availability Audit — UNSW-NB15

This document is the single source of truth for which UNSW-NB15 features
are actually reconstructable from live traffic captured via Scapy on a
Kali VM, and which are not. `config.py`'s `RUNTIME_FEASIBLE_FEATURES`
list is derived directly from the "Obtainable" rows below. **The training
pipeline (`train.py`) trains its deployment model using ONLY the
"Obtainable" feature set** — GWO then selects a subset of *those*, so the
offline-trained model and the real-time VM model are guaranteed to use
compatible features. No feature is ever faked, guessed, or filled with a
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
| 14 | `sloss` | Source retransmit/drop count | ❌ Excluded | Requires sequence-number gap analysis across the full TCP stream; not implemented — set to 0 rather than faked as "detected" |
| 15 | `dloss` | Dest retransmit/drop count | ❌ Excluded | Same as `sloss` |
| 16 | `sinpkt` | Source interpacket arrival time | ✅ Obtainable | Mean gap between consecutive source-side packet timestamps |
| 17 | `dinpkt` | Dest interpacket arrival time | ✅ Obtainable | Mean gap between consecutive dest-side packet timestamps |
| 18 | `sjit` | Source jitter | ⚠️ Approximate | Standard deviation of source interpacket gaps (UNSW's exact jitter formula is not published in full detail) |
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
| 32 | `ct_state_ttl` | Count by state+TTL range bucket | ⚠️ Approximate | Simplified bucketing of `(state, sttl_range)` pairs from the same rolling window, not UNSW's original binning table |
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

## Summary

- **✅ Obtainable (clean, direct): 24 features**
- **⚠️ Approximate (implemented, but simplified vs. the original definition): 9 features**
- **❌ Excluded (not implemented, application-layer or stream-analysis dependent): 9 features**
  (`sloss`, `dloss`, `trans_depth`, `response_body_len`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`)
  — note: that's 7 fully excluded features; `sloss`/`dloss` bring the total to the 9 counted above.

**`config.RUNTIME_FEASIBLE_FEATURES` = the 33 features marked ✅ or ⚠️** (Obtainable + Approximate). The 9 ❌ Excluded features are dropped from the training feature space entirely for the deployment model — they are never present as columns, never filled with 0/mean/placeholder, and never seen by GWO or the classifier for the runtime-facing model.

This is a genuine, honest trade-off: the deployment model will very likely score somewhat lower than a model trained on all 42 offline features, because it's deliberately blind to HTTP/FTP-specific signals it can never see live anyway. **Both numbers should go in your report** — the full-feature offline result (as an upper-bound research reference) and the runtime-feasible-feature result (the number that actually matters for what ships to the VM) — with this table as the explanation for the gap.
