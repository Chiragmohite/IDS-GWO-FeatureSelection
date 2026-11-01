"""
src/runtime/alerting.py

Defensive-only event storage: every prediction (normal or attack) is
logged, and predictions above config.ALERT_CONFIDENCE_THRESHOLD that are
classified as an attack additionally raise a stored "alert" record. This
module NEVER takes any blocking/enforcement action against the source of
traffic -- it only records and surfaces information, per the spec's
explicit defensive-only requirement.
"""

import logging
import sqlite3
import math
from pathlib import Path
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from typing import Dict, Optional

import config

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prediction TEXT NOT NULL,          -- 'normal' or 'attack'
    confidence REAL NOT NULL,
    src_ip TEXT,
    dst_ip TEXT,
    src_port INTEGER,
    dst_port INTEGER,
    proto TEXT,
    service TEXT,
    is_alert INTEGER NOT NULL DEFAULT 0,
    features_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_is_alert ON events(is_alert);
"""


EVENT_SOURCES = ("live", "loopback_probe", "pcap_replay", "dataset_replay", "csv_import", "selftest", "legacy")


def init_db(db_path: str = None) -> None:
    """Create the SQLite database and schema if they don't already exist."""
    db_path = db_path or config.ALERTS_DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # SQLite's transaction context does not close the connection on exit.
    with closing(sqlite3.connect(db_path, timeout=10)) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.execute("BEGIN IMMEDIATE")  # Serialize discovery and additive ALTERs across sessions.
        # Additive migration: original event rows remain intact.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        if "source" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy'")
        if "model_version" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN model_version TEXT")
        if "observed_at" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN observed_at TEXT")
        if "batch_id" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN batch_id TEXT")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS incident_reviews (
            event_id INTEGER PRIMARY KEY REFERENCES events(id),
            state TEXT NOT NULL, note TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS review_audit (
            id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL REFERENCES events(id),
            previous_state TEXT NOT NULL, new_state TEXT NOT NULL,
            note TEXT NOT NULL, timestamp TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_review_event ON review_audit(event_id);
        CREATE TABLE IF NOT EXISTS import_batches (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, created_at TEXT NOT NULL,
            row_count INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS collector_health (
            id INTEGER PRIMARY KEY CHECK(id=1), status TEXT NOT NULL,
            details TEXT NOT NULL, updated_at TEXT NOT NULL);
        """)
    logger.info("Alert database ready at %s", db_path)


@contextmanager
def _connection(db_path: str = None):
    db_path = db_path or config.ALERTS_DB_PATH
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_event(
    prediction: int,
    confidence: float,
    flow_summary: Dict,
    features: Dict,
    db_path: str = None,
    source: str = "live",
    model_version: str = None,
    observed_at: str = None,
    alert_threshold: float = None,
) -> bool:
    """
    Store one prediction event. If it's an attack prediction with
    confidence >= config.ALERT_CONFIDENCE_THRESHOLD, it is also flagged
    as an alert (is_alert=1).

    Parameters
    ----------
    prediction : int
        0 (normal) or 1 (attack).
    confidence : float
        Model's confidence in the predicted class, in [0, 1].
    flow_summary : Dict
        Small dict with src_ip/dst_ip/src_port/dst_port/proto/service,
        for populating the human-readable columns.
    features : Dict
        The full raw feature dict (for the JSON audit trail column).
    db_path : Optional[str]
        Override for the database path (defaults to config.ALERTS_DB_PATH).

    Returns
    -------
    bool
        True if this event was flagged as an alert, else False.
    """
    import json as _json

    if prediction not in (0, 1) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Prediction must be binary and confidence finite in [0, 1].")
    if source not in EVENT_SOURCES[:-1]:
        raise ValueError("Invalid source.")
    prediction_label = "attack" if prediction == 1 else "normal"
    cutoff = config.ALERT_CONFIDENCE_THRESHOLD if alert_threshold is None else alert_threshold
    if not math.isfinite(cutoff) or not .5 <= cutoff <= 1:raise ValueError("Invalid alert threshold.")
    is_alert = int(prediction == 1 and confidence >= cutoff)

    with _connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO events
                (timestamp, prediction, confidence, src_ip, dst_ip, src_port,
                 dst_port, proto, service, is_alert, features_json, source, model_version, observed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                prediction_label,
                confidence,
                flow_summary.get("src_ip"),
                flow_summary.get("dst_ip"),
                flow_summary.get("src_port"),
                flow_summary.get("dst_port"),
                flow_summary.get("proto"),
                flow_summary.get("service"),
                is_alert,
                _json.dumps(features, default=str, allow_nan=False),
                source, model_version, observed_at,
            ),
        )

    if is_alert:
        logger.warning(
            "ALERT: attack detected %s:%s -> %s:%s (%s, confidence=%.2f)",
            flow_summary.get("src_ip"), flow_summary.get("src_port"),
            flow_summary.get("dst_ip"), flow_summary.get("dst_port"),
            flow_summary.get("proto"), confidence,
        )
    return bool(is_alert)


def get_summary_stats(db_path: str = None) -> Dict:
    """Aggregate stats for the dashboard: totals, normal/attack counts, alert count."""
    with _connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        normal = conn.execute("SELECT COUNT(*) AS n FROM events WHERE prediction='normal'").fetchone()["n"]
        attack = conn.execute("SELECT COUNT(*) AS n FROM events WHERE prediction='attack'").fetchone()["n"]
        alerts = conn.execute("SELECT COUNT(*) AS n FROM events WHERE is_alert=1").fetchone()["n"]
    return {"total_observations": total, "normal_observations": normal, "attack_observations": attack, "alerts": alerts}


def get_recent_events(limit: int = 50, only_alerts: bool = False, db_path: str = None):
    """Fetch the most recent events (or alerts only) for the dashboard/log view."""
    query = "SELECT * FROM events"
    if only_alerts:
        query += " WHERE is_alert=1"
    query += " ORDER BY id DESC LIMIT ?"

    with _connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (limit,)).fetchall()
    return [dict(r) for r in rows]


REVIEW_TRANSITIONS = {
    "open": {"investigating", "resolved", "false_positive"},
    "investigating": {"resolved", "false_positive"},
    "resolved": {"investigating"},
    "false_positive": {"investigating"},
}


def query_events(limit=50, offset=0, prediction="all", source="all", only_alerts=False,
                 review_state="all", db_path=None):
    """Parameterized, bounded event search for the local analyst console."""
    if not 1 <= limit <= 500 or offset < 0:
        raise ValueError("Invalid pagination.")
    if prediction not in {"all", "normal", "attack"}:
        raise ValueError("Invalid prediction filter.")
    if source not in {"all", *EVENT_SOURCES}:
        raise ValueError("Invalid event source.")
    if review_state not in {"all", *REVIEW_TRANSITIONS}:
        raise ValueError("Invalid review state.")
    clauses, args = [], []
    for column, value in [("e.prediction", prediction), ("e.source", source),
                           ("COALESCE(r.state, 'open')", review_state)]:
        if value != "all":
            clauses.append(column + " = ?")
            args.append(value)
    if only_alerts:
        clauses.append("e.is_alert = 1")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    base = " FROM events e LEFT JOIN incident_reviews r ON e.id = r.event_id"
    with _connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        count = conn.execute("SELECT COUNT(*)" + base + where, args).fetchone()[0]
        rows = conn.execute("SELECT e.*, COALESCE(r.state, 'open') AS review_state, r.note" +
                            base + where + " ORDER BY e.id DESC LIMIT ? OFFSET ?",
                            args + [limit, offset]).fetchall()
    return [dict(row) for row in rows], count


def review_event(event_id, new_state, note, expected_state="open", db_path=None):
    """Validated optimistic transition and audit entry in one transaction.

    This is a local, single-analyst tool; it does not claim authenticated identity.
    No network blocking or automatic account restrictions are performed.
    """
    if not isinstance(note, str) or not 5 <= len(note.strip()) <= 2000:
        raise ValueError("Provide a review reason between 5 and 2000 characters.")
    with _connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        event = conn.execute("SELECT is_alert FROM events WHERE id=?", (event_id,)).fetchone()
        if not event or not event[0]:
            raise ValueError("Only an existing alert can be reviewed.")
        existing = conn.execute("SELECT state FROM incident_reviews WHERE event_id=?", (event_id,)).fetchone()
        current = existing[0] if existing else "open"
        if current != expected_state:
            raise ValueError("Review changed in another session. Refresh before retrying.")
        if new_state not in REVIEW_TRANSITIONS[current]:
            raise ValueError("Invalid incident transition.")
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""INSERT INTO incident_reviews(event_id,state,note,updated_at) VALUES(?,?,?,?)
            ON CONFLICT(event_id) DO UPDATE SET state=excluded.state,note=excluded.note,updated_at=excluded.updated_at""",
            (event_id, new_state, note.strip(), now))
        conn.execute("INSERT INTO review_audit(event_id,previous_state,new_state,note,timestamp) VALUES(?,?,?,?,?)",
                     (event_id, current, new_state, note.strip(), now))


def review_history(event_id, db_path=None):
    with _connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM review_audit WHERE event_id=? ORDER BY id", (event_id,))]


def heartbeat(status, details, db_path=None):
    if status not in {"running", "stopped", "error"}:
        raise ValueError("Invalid collector status.")
    with _connection(db_path) as conn:
        conn.execute("""INSERT INTO collector_health(id,status,details,updated_at) VALUES(1,?,?,?)
            ON CONFLICT(id) DO UPDATE SET status=excluded.status,details=excluded.details,updated_at=excluded.updated_at""",
            (status, details[:400], datetime.now(timezone.utc).isoformat()))


def collector_status(db_path=None):
    with _connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM collector_health WHERE id=1").fetchone()
        return dict(row) if row else None


def source_totals(db_path=None):
    """All stored rows grouped by input source, including historical records."""
    with _connection(db_path) as conn:
        return dict(conn.execute("SELECT source, COUNT(*) FROM events GROUP BY source"))


def hourly_counts(rows):
    """UTC buckets on ISO timestamp strings, allowing historical naive UTC records."""
    buckets = {}; invalid = 0
    for row in rows:
        try:
            stamp = datetime.fromisoformat(row["timestamp"])
            stamp = (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
            hour = stamp.strftime("%Y-%m-%d %H:00Z")
        except (ValueError, TypeError, OverflowError):
            invalid += 1
            continue
        counts = buckets.setdefault(hour, {})
        label = row["prediction"]; counts[label] = counts.get(label, 0) + 1
    return buckets, invalid
