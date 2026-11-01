"""Read only IPv4/transport metadata. Never retain payloads or reassemble fragments."""
import socket
import struct


def decode_ipv4(ip, wire_length, ts):
    if not ip or ip[0] >> 4 != 4:return None,"non_ipv4"
    if len(ip)<20:return None,"truncated"
    ihl=(ip[0]&15)*4
    total,frag=struct.unpack_from("!H",ip,2)[0],struct.unpack_from("!H",ip,6)[0]
    if not 20<=ihl<=total:return None,"malformed"
    if total>len(ip):return None,"truncated"
    if frag&0x3fff:return None,"fragment"
    payload=ip[ihl:total];proto=ip[9]
    if proto not in (6,17):return None,"unsupported_transport"
    if len(payload)<(20 if proto==6 else 8):return None,"truncated"
    sport,dport=struct.unpack_from("!HH",payload)
    flags=seq=window=None
    if proto==6:
        hdr=(payload[12]>>4)*4
        if not 20<=hdr<=len(payload):return None,"malformed"
        flags="".join(c for bit,c in enumerate("FSRPAUEC") if payload[13]&(1<<bit))
        seq=struct.unpack_from("!I",payload,4)[0];window=struct.unpack_from("!H",payload,14)[0]
    elif not 8<=struct.unpack_from("!H",payload,4)[0]<=len(payload):return None,"malformed"
    return dict(proto="tcp" if proto==6 else "udp",src_ip=socket.inet_ntoa(ip[12:16]),src_port=sport,
        dst_ip=socket.inet_ntoa(ip[16:20]),dst_port=dport,length=wire_length,ttl=ip[8],timestamp=ts,
        tcp_flags=flags,tcp_seq=seq,tcp_window=window),None


def decode_frame(data, linktype, ts):
    if linktype==101:return decode_ipv4(data,len(data),ts)
    if linktype!=1:return None,"unsupported_link"
    if len(data)<14:return None,"truncated"
    off=14;etype=int.from_bytes(data[12:14],"big")
    for _ in range(2):
        if etype not in (0x8100,0x88a8):break
        if off+4>len(data):return None,"truncated"
        etype=int.from_bytes(data[off+2:off+4],"big");off+=4
    if etype!=0x800:return None,"non_ipv4"
    return decode_ipv4(data[off:],len(data),ts)
