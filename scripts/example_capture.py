"""Generate a classic PCAP fixture offline. No sockets or packet capture required."""
from pathlib import Path
import socket
import struct


def checksum(data):
    if len(data)%2:data+=bytes(1)
    n=sum(struct.unpack("!%dH"%(len(data)//2),data))
    while n>>16:n=(n&65535)+(n>>16)
    return (~n)&65535


def frame(source,destination,proto,sport,dport,payload=b"",flags=0):
    src,dst=map(socket.inet_aton,(source,destination))
    if proto==17:transport=struct.pack("!HHHH",sport,dport,8+len(payload),0)+payload
    else:transport=struct.pack("!HHIIBBHHH",sport,dport,1,1,0x50,flags,4096,0,0)+payload
    c=checksum(src+dst+struct.pack("!BBH",0,proto,len(transport))+transport)
    off=6 if proto==17 else 16
    transport=transport[:off]+struct.pack("!H",c or 65535)+transport[off+2:]
    ip=struct.pack("!BBHHHBBH4s4s",0x45,0,20+len(transport),1,0,64,proto,0,src,dst)
    ip=ip[:10]+struct.pack("!H",checksum(ip))+ip[12:]
    return bytes.fromhex("0200000000020200000000010800")+ip+transport


def make_capture():
    a,b="192.0.2.10","198.51.100.20"
    packets=[frame(a,b,17,50001,53,b"fixture query"),frame(b,a,17,53,50001,b"fixture reply"),
             frame(a,b,6,50002,443,flags=2),frame(b,a,6,443,50002,flags=18),
             frame(a,b,6,50002,443,flags=16),frame(a,b,6,50002,443,flags=17)]
    # One explicit non-IPv4 frame exercises the skipped-frame counter.
    packets.append(bytes.fromhex("0200000000020200000000010806")+bytes(28))
    out=struct.pack("<IHHIIII",0xa1b2c3d4,2,4,0,0,65535,1)
    for i,p in enumerate(packets):out+=struct.pack("<IIII",1700000000,i*50000,len(p),len(p))+p
    return out


if __name__=="__main__":
    (Path(__file__).resolve().parents[1]/"examples"/"teaching_capture.pcap").write_bytes(make_capture())
