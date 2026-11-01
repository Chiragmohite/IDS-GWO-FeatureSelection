import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import config
from src.runtime.flow_tracker import FlowTracker
from src.runtime.feature_extraction import FlowHistory, extract_features


def send(tracker, ts, reverse=False, flags='A'):
    a=('192.0.2.1',10000,'192.0.2.2',80)
    if reverse:a=(a[2],a[3],a[0],a[1])
    tracker.on_packet('tcp',*a,60,64,ts,flags,1,1024)


def test_bidirectional_handshake_units_and_schema():
    tracker=FlowTracker(15);now=time.time()
    send(tracker,now,flags='S');send(tracker,now+.01,True,'SA')
    send(tracker,now+.02);send(tracker,now+.03,flags='FA')
    flow=tracker.pop_completed_flows()[0]
    features=extract_features(flow,FlowHistory(100))
    assert set(features)==set(config.RUNTIME_FEASIBLE_FEATURES)
    assert features['spkts']==3 and features['dpkts']==1
    assert features['sinpkt']==pytest.approx(15,abs=.001)
    assert features['tcprtt']==pytest.approx(.02,abs=.00001)
    assert tracker.active_flow_count()==0


def test_flow_capacity_expiry_and_flag_variants():
    t=FlowTracker(1);t.max_flows=1;t.max_packets_per_flow=2
    send(t,time.time()-5,flags='FPA')
    send(t,time.time()-4);send(t,time.time()-3)
    assert t.dropped_packets==1
    assert len(t.pop_completed_flows())==1
    assert t.active_flow_count()==0


def test_packet_ingestion_and_expiry_are_race_safe():
    t=FlowTracker(0)
    def produce(_):
        for i in range(100):send(t,time.time()-1)
    def drain(_):
        for i in range(100):t.pop_completed_flows()
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(fn,0) for fn in [produce,drain,produce,drain]]
        for task in tasks: task.result()
    t.pop_completed_flows()
    assert t.active_flow_count()==0


def test_history_is_bounded_and_counts_only_prior_flows():
    h=FlowHistory(2)
    for n in range(4):h.record({'src_ip':'192.0.2.1'})
    assert h.count_matching(src_ip='192.0.2.1')==2
