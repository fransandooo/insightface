"""P3 - Function creep through the access log (flows 8 and 9, zone Z4).

Privacy issue: ``Storage.log_recognition`` writes every recognition event to
``recognition_logs`` with a timestamp, the predicted person, the similarity and
an arbitrary JSON payload. Nothing ever deletes these rows, and the payload is
stored as given, so it can contain embeddings or face crops. A permanent,
per-employee timeline of entries is exactly what is needed for attendance or
productivity monitoring, which is a different purpose from door access.

GDPR: Art. 5(1)(b) purpose limitation, Art. 5(1)(c) data minimisation,
Art. 5(1)(e) storage limitation, Art. 88 (processing in the employment context).
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

RETENTION_DAYS = 30  # retention period proposed in our recommendations


def _log_access(storage, person_id, payload=None):
    return storage.log_recognition(
        source_type="camera",
        source_path="rtsp://door-1",
        predicted_person_id=person_id,
        similarity=0.71,
        threshold=0.45,
        status="matched",
        payload=payload,
    )


def _backdate(db_path, log_id, days):
    old = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE recognition_logs SET created_at=? WHERE id=?", (old, log_id))


def test_access_events_older_than_retention_period_are_purged(gallery, query_one):
    storage, db_path = gallery
    person_id = storage.add_person("Alice")
    old_event = _log_access(storage, person_id)
    recent_event = _log_access(storage, person_id)
    _backdate(db_path, old_event, days=400)

    purge = getattr(storage, "purge_expired_records", None)
    assert callable(purge), (
        "P3 OPEN: Storage has no retention mechanism; access events are kept "
        "forever and build a complete attendance history of every employee."
    )
    purge(retention_days=RETENTION_DAYS)

    assert query_one(db_path, "SELECT COUNT(*) FROM recognition_logs WHERE id=?", (old_event,)) == 0
    assert query_one(db_path, "SELECT COUNT(*) FROM recognition_logs WHERE id=?", (recent_event,)) == 1


def test_access_log_never_stores_biometric_data(gallery, employee_embedding, query_one):
    storage, db_path = gallery
    person_id = storage.add_person("Alice")
    log_id = _log_access(
        storage,
        person_id,
        payload={"embedding": employee_embedding.tolist(), "bbox": [10, 20, 110, 140], "door": "door-1"},
    )

    payload = json.loads(query_one(db_path, "SELECT payload_json FROM recognition_logs WHERE id=?", (log_id,)))
    leaked = sorted({"embedding", "normed_embedding", "reid_feature", "kps", "thumbnail"} & set(payload))
    assert not leaked, (
        f"P3 OPEN: the access log accepted biometric fields {leaked} in its payload; "
        f"the audit trail must only hold who/when/which door."
    )
