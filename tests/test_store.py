from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3
import pytest
from src.runtime import alerting
from src.runtime.alerting import (init_db, record_event, query_events, review_event,
    review_history, get_summary_stats, collector_status, heartbeat)


def add(db, label=1, confidence=.95):
    return record_event(label,confidence,{"proto":"tcp","service":"http"},{"dur":1},db,source="selftest")


def test_portable_database_and_provenance(db):
    assert add(db)
    assert not add(db,0,.9)
    rows,count=query_events(db_path=db)
    assert count==2 and {r['source'] for r in rows}=={'selftest'}
    assert get_summary_stats(db)['alerts']==1


def test_additive_schema_preserves_old_records(tmp_path):
    from src.runtime.alerting import _SCHEMA
    path=str(tmp_path/'old.db')
    with closing(sqlite3.connect(path)) as c, c:
        c.executescript(_SCHEMA)
        c.execute("INSERT INTO events(timestamp,prediction,confidence) VALUES('2020','normal',.5)")
    init_db(path);init_db(path)
    rows,count=query_events(db_path=path)
    assert count==1 and rows[0]['source']=='legacy'


@pytest.mark.parametrize('label,prob',[(2,.5),(1,-1),(1,1.1),(1,float('nan'))])
def test_invalid_event_rejected(db,label,prob):
    with pytest.raises(ValueError):add(db,label,prob)
    assert get_summary_stats(db)['total_observations']==0


def test_review_state_machine_and_audit(db):
    add(db)
    review_event(1,'investigating','Check flow evidence',db_path=db)
    review_event(1,'false_positive','Verified authorized test traffic','investigating',db)
    rows,_=query_events(review_state='false_positive',db_path=db)
    assert len(rows)==1
    assert [r['new_state'] for r in review_history(1,db)]==['investigating','false_positive']
    with pytest.raises(ValueError):review_event(1,'resolved','Stale UI action','open',db)
    with pytest.raises(ValueError):review_event(1,'resolved','No valid transition','false_positive',db)
    assert len(review_history(1,db))==2


def test_concurrent_review_only_one_wins(db):
    add(db)
    def update(state):
        try:review_event(1,state,'Concurrent review reason',db_path=db);return True
        except ValueError:return False
    with ThreadPoolExecutor(max_workers=2) as pool: outcomes=list(pool.map(update,['resolved','false_positive']))
    assert sum(outcomes)==1 and len(review_history(1,db))==1


def test_parameterization_and_normal_review_rejection(db):
    add(db,0)
    with pytest.raises(ValueError):review_event(1,'resolved','Not an alert',db_path=db)
    with pytest.raises(ValueError):query_events(prediction="';DROP TABLE events;--",db_path=db)
    assert get_summary_stats(db)['total_observations']==1


def test_heartbeat_not_inferred_from_replay(db):
    add(db)
    assert collector_status(db) is None
    heartbeat('running','Synthetic test heartbeat',db)
    assert collector_status(db)['status']=='running'
    heartbeat('stopped','Stopped',db)
    assert collector_status(db)['status']=='stopped'


def test_review_reason_and_nonexistent_event_rejected(db):
    add(db)
    with pytest.raises(ValueError):review_event(1,'resolved','x',db_path=db)
    with pytest.raises(ValueError):review_event(999,'resolved','Not a real event',db_path=db)
    assert review_history(1,db)==[]


@pytest.fixture
def tracked_connections(monkeypatch):
    """Retain real connections so closure is checked without relying on GC."""
    connect = sqlite3.connect
    connections = []

    def tracked_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(alerting.sqlite3, "connect", tracked_connect)
    yield connections
    for conn in connections:
        conn.close()


@pytest.mark.parametrize("invalid_schema", [False, True])
def test_init_db_closes_connection(tmp_path, monkeypatch, tracked_connections, invalid_schema):
    path = str(tmp_path / "init.db")
    if invalid_schema:
        monkeypatch.setattr(alerting, "_SCHEMA", "INVALID SQL;")
        with pytest.raises(sqlite3.OperationalError):
            init_db(path)
    else:
        init_db(path)
    assert len(tracked_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        tracked_connections[0].execute("SELECT 1")


@pytest.mark.parametrize("body_fails", [False, True])
def test_connection_closes_and_preserves_transactions(db, tracked_connections, body_fails):
    def write():
        with alerting._connection(db) as conn:
            conn.execute("INSERT INTO events(timestamp,prediction,confidence) VALUES('2020','normal',.5)")
            if body_fails:
                raise ValueError("Simulated write failure")

    if body_fails:
        with pytest.raises(ValueError, match="Simulated write failure"):
            write()
    else:
        write()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        tracked_connections[0].execute("SELECT 1")
    assert get_summary_stats(db)["total_observations"] == (0 if body_fails else 1)


def test_connection_closes_when_setup_fails(tmp_path, monkeypatch):
    class FailingSetupConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA foreign_keys=ON":
                raise sqlite3.OperationalError("Simulated setup failure")
            return super().execute(sql, *args, **kwargs)

    conn = sqlite3.connect(tmp_path / "setup.db", factory=FailingSetupConnection)
    monkeypatch.setattr(alerting.sqlite3, "connect", lambda *args, **kwargs: conn)
    try:
        with pytest.raises(sqlite3.OperationalError, match="Simulated setup failure"):
            with alerting._connection():
                pytest.fail("Failed setup must not yield a connection")
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    finally:
        conn.close()
