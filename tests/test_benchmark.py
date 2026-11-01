import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import benchmark
import config


@pytest.fixture
def benchmark_data():
    train = pd.read_csv(config.UNSW_TRAIN_PATH).sample(300, random_state=11)
    test = pd.read_csv(config.UNSW_TEST_PATH).sample(160, random_state=12)
    return train, test


def test_shared_splits_remove_overlap_even_when_test_label_differs(benchmark_data):
    train, test = benchmark_data
    duplicate = train.iloc[[0]].copy()
    duplicate['label'] = 1 - duplicate.label
    test = pd.concat([test, duplicate], ignore_index=True)
    final_train, final_test, fit, validation, features, meta = benchmark.prepare_splits(
        train, test, fit_limit=100, validation_limit=40, seed=42)
    hashes = lambda frame: set(benchmark.fingerprints(frame, config.RUNTIME_FEASIBLE_FEATURES))
    assert not hashes(final_train) & hashes(final_test)
    assert not hashes(fit) & hashes(validation)
    assert meta['removed_test_overlaps_on_35_features'] >= 1
    assert len(features) == 42 and 'label' not in features and 'attack_cat' not in features


def test_complete_benchmark_keeps_models_and_database_and_ignores_test_labels(tmp_path, monkeypatch, benchmark_data):
    train, test = benchmark_data
    train_path, test_path = tmp_path / 'train.csv', tmp_path / 'test.csv'
    train.to_csv(train_path, index=False); test.to_csv(test_path, index=False)
    monkeypatch.setattr(config, 'UNSW_TRAIN_PATH', str(train_path))
    monkeypatch.setattr(config, 'UNSW_TEST_PATH', str(test_path))
    monkeypatch.setattr(config, 'RANDOM_FOREST_PARAMS', {'n_estimators': 3, 'random_state': 42, 'n_jobs': 1})
    model_dir = tmp_path / 'deployment'; model_dir.mkdir()
    db_dir = tmp_path / 'events'; db_dir.mkdir()
    model = model_dir / 'model.pkl'; model.write_bytes(b'preserved model; never load')
    db = db_dir / 'events.db'; db.write_bytes(b'preserved database; never open')
    monkeypatch.setattr(config, 'SAVED_MODELS_DIR', str(model_dir))
    monkeypatch.setattr(config, 'ALERTS_DB_PATH', str(db))
    original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [model, db]}
    reports = []
    selections = []
    for index in range(2):
        out = tmp_path / f'run-{index}'
        args = benchmark.parse_args(['--output', str(out), '--fit-limit', '100',
            '--validation-limit', '40', '--wolves', '3', '--iterations', '1', '--seeds', '7', '--jobs', '1'])
        reports.append(benchmark.run_benchmark(args))
        selections.append(json.loads((out / 'selection.json').read_text()))
        assert (out / 'comparison.csv').is_file() and (out / 'comparison.png').is_file()
        assert not (out / 'results.partial.json').exists()
        assert reports[-1]['status'] == 'completed'
        assert len(reports[-1]['results']) == 4
        for row in reports[-1]['results']:
            assert sum(map(sum, row['confusion_matrix'])) == reports[-1]['splits']['final_test_rows']
            assert row['feature_count'] == len(row['features'])
        test['label'] = 1 - test.label
        test.to_csv(test_path, index=False)
    for pool in ['offline_42', 'runtime_35']:
        assert selections[0][pool][0]['mask'] == selections[1][pool][0]['mask']
        assert selections[0][pool][0]['fitness'] == selections[1][pool][0]['fitness']
    assert reports[0]['splits']['row_index_sha256'] == reports[1]['splits']['row_index_sha256']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in original.items())
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(args)
    args.output = str(model_dir / 'must-not-create')
    with pytest.raises(ValueError, match='separate'):
        benchmark.run_benchmark(args)
    assert not Path(args.output).exists()


@pytest.mark.parametrize('arguments', [
    ['--wolves', '2'], ['--iterations', '0'], ['--seeds', '42', '42'],
    ['--seeds', '-1'], ['--fit-limit', '1'], ['--jobs', '0'],
])
def test_invalid_benchmark_settings_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark.parse_args(arguments)
