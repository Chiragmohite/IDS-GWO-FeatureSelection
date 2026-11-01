from pathlib import Path
from streamlit.testing.v1 import AppTest
import config
from src.runtime.alerting import init_db, record_event, query_events

APP=str(Path(__file__).resolve().parents[1]/'dashboard'/'app.py')


def test_all_pages_render_without_exceptions(trained_bundle):
    app=AppTest.from_file(APP,default_timeout=30).run()
    assert not app.exception
    for name in ['Events & investigations','CSV analysis','Model lab','Setup & methodology','Overview']:
        app.radio[0].set_value(name).run()
        assert not app.exception, name


def test_csv_to_history_and_incident_review(trained_bundle):
    app=AppTest.from_file(APP,default_timeout=30).run()
    app.radio[0].set_value('CSV analysis').run()
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception and app.success
    save=next(b for b in app.button if b.label=='Save observations to event history')
    save.click().run()
    assert not app.exception
    rows,count=query_events()
    assert count>=50
    # Controlled known alert tests triage, not detector accuracy.
    record_event(1,.99,{'proto':'tcp'},{'dur':1},source='selftest')
    app.radio[0].set_value('Events & investigations').run()
    app.text_area[0].set_value('Synthetic alert verified by test operator')
    app.button[0].click().run()
    assert not app.exception and any('saved' in s.value for s in app.success)


def test_csv_reports_training_busy_without_crashing(trained_bundle):
    from src.runtime.training_lock import training_lock
    app=AppTest.from_file(APP,default_timeout=30).run()
    app.radio[0].set_value('CSV analysis').run()
    app.checkbox[0].check().run()
    with training_lock(config.SAVED_MODELS_DIR):
        app.button[0].click().run()
    assert not app.exception
    assert app.error and 'already active' in app.error[0].value


def test_populated_overview_repeated_timestamps_refresh_and_navigation(db, monkeypatch):
    """Exercise the pandas datetime path that crashed in the Windows 3.14 environment."""
    import sqlite3
    from contextlib import closing
    monkeypatch.setattr(config, 'ALERTS_DB_PATH', db)
    for i in range(100):
        record_event(int(i >= 36), .55 if i >= 96 else .99,
                     {'proto': 'tcp'}, {'dur': 1}, source='selftest', db_path=db)
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("UPDATE events SET timestamp=?", ('2026-10-30T17:30:00+00:00',))
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert [m.value for m in app.metric] == ['100', '36', '64', '60']
    assert any('Recent 100 of 100 records' in c.value for c in app.caption)
    app.checkbox[0].check().run()
    for _ in range(2):
        app.run()
        assert not app.exception
        app.radio[0].set_value('Events & investigations').run()
        assert not app.exception
        app.radio[0].set_value('Overview').run()
        assert not app.exception
        assert [m.value for m in app.metric] == ['100', '36', '64', '60']
