"""Measured local UDP exchanges and bounded classic-PCAP analysis. No payload is persisted."""
import hashlib
import socket
import struct
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import config
from src.runtime.batch import MAX_BYTES, MAX_ROWS, analyze_frame, store_batch
from src.runtime.feature_extraction import extract_features, FlowHistory
from src.runtime.flow_tracker import FlowTracker


def tracker():
    return FlowTracker(config.FLOW_IDLE_TIMEOUT_SECONDS, getattr(config,"FLOW_MAX_AGE_SECONDS",30))


def packet_fields(packet):
    from scapy.layers.inet import IP
    from src.runtime.packet_headers import decode_ipv4
    if IP not in packet:return None,"non_ipv4"
    return decode_ipv4(bytes(packet[IP]),len(packet),float(packet.time))


def predict_flows(flows, engine, source, token, stats):
    if not flows: raise ValueError("No complete supported IPv4 TCP/UDP flows to analyze.")
    history=FlowHistory(config.FLOW_HISTORY_WINDOW); features=[]; meta=[]
    for flow in flows:
        values=extract_features(flow,history);features.append(values)
        history.record({**flow.flow_key_summary(),"state":values["state"]})
        meta.append({**flow.flow_key_summary(),"observed_at":datetime.fromtimestamp(flow.last_seen,timezone.utc).isoformat()})
    results, features=analyze_frame(pd.DataFrame(features),engine)
    return dict(results=results,features=features,metadata=meta,batch_id=token,source=source,
                version=engine.schema["model_version"],stats=stats)


def persist(batch, db_path=None):
    return store_batch(batch["results"],batch["features"],batch["batch_id"],batch["source"],
                       batch["version"],db_path,metadata=batch["metadata"])


def analyze_pcap(data,engine):
    from src.runtime.packet_headers import decode_frame
    formats={b"\xd4\xc3\xb2\xa1":("<",1e6),b"\xa1\xb2\xc3\xd4":(">",1e6),
             b"\x4d\x3c\xb2\xa1":("<",1e9),b"\xa1\xb2\x3c\x4d":(">",1e9)}
    if not 24 <= len(data) <= MAX_BYTES or data[:4] not in formats:
        raise ValueError("Provide classic PCAP, uncompressed, up to 10 MiB (not PCAPNG).")
    endian, unit=formats[data[:4]]
    major,minor,_,_,snap,kind=struct.unpack_from(endian+"HHiIII",data,4)
    if (major,minor)!=(2,4) or kind not in {1,101} or not 1<=snap<=262144:
        raise ValueError("Expected PCAP 2.4 with Ethernet or raw IP.")
    flow_tracker=tracker(); flows=[]; offset=24; previous=-1.; packets=0; skipped={};accepted=0
    while offset<len(data):
        packets+=1
        if packets>20000 or offset+16>len(data):raise ValueError("Too many packets or incomplete PCAP record.")
        sec,fraction,captured,original=struct.unpack_from(endian+"IIII",data,offset);offset+=16
        if fraction>=unit or not 0<captured<=min(original,snap,len(data)-offset):raise ValueError("Invalid PCAP record length/time.")
        ts=sec+fraction/unit
        if ts<previous:raise ValueError("PCAP timestamps must be chronological.")
        previous=ts
        payload=data[offset:offset+captured];offset+=captured
        flows.extend(flow_tracker.pop_completed_flows(now=ts))
        if len(flows)+flow_tracker.active_flow_count()>MAX_ROWS:raise ValueError("Maximum 1,000 flows per import.")
        if captured!=original:reason="truncated";fields=None
        else:fields,reason=decode_frame(payload,kind,ts)
        if fields is None:
            skipped[reason]=skipped.get(reason,0)+1;continue
        flow_tracker.on_packet(**fields);accepted+=1
    flows.extend(flow_tracker.pop_completed_flows(now=previous,flush=True))
    if len(flows)>MAX_ROWS or flow_tracker.dropped_packets:raise ValueError("Flow capacity exceeded; use a smaller capture.")
    # Same file analyzed with the same trained bundle cannot be imported twice.
    key=hashlib.sha256(data).hexdigest()
    token=str(uuid.uuid5(uuid.NAMESPACE_URL, "ids-pcap-v1:"+key+":"+engine.schema["model_version"]))
    return predict_flows(flows,engine,"pcap_replay",token,dict(packets=packets,accepted_packets=accepted,
                         skipped=skipped,flows=len(flows),file_sha256=key))


def loopback_probe(engine, count=4):
    """Only sends to ephemeral localhost sockets. Instrumented test traffic, never adapter capture."""
    if not 1 <= count <= 25: raise ValueError("Choose 1–25 exchanges.")
    flows=[]
    for i in range(count):
        t=tracker()
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as client,socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as server:
            for s in (client,server):
                s.bind(("127.0.0.1",0));s.settimeout(2.)
            for sender,receiver,size in [(client,server,40+i),(server,client,32+i)]:
                payload=bytes(size)
                sender.sendto(payload,receiver.getsockname())
                actual,origin=receiver.recvfrom(2048)
                if actual!=payload or origin!=sender.getsockname():raise OSError("Unexpected local datagram.")
                # Socket-observed receive time, sender TTL and IPv4+UDP byte estimate;
                # no link-layer measurement or payload-based inference.
                dest=receiver.getsockname()
                t.on_packet("udp",origin[0],origin[1],dest[0],dest[1],len(actual)+28,
                            sender.getsockopt(socket.IPPROTO_IP,socket.IP_TTL),time.time())
        flows.extend(t.pop_completed_flows(flush=True))
    return predict_flows(flows,engine,"loopback_probe",str(uuid.uuid4()),dict(flows=count,
        datagrams_received=2*count,network="IPv4 UDP localhost; socket instrumented",attack_labels=False))
