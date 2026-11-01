import json
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import pytest
import config
from src.preprocessing.unsw_preprocessing import encode_categoricals
from src.preprocessing.encoding import encode_categorical_features
from src.runtime.artifacts import inspect_bundle
from src.runtime.batch import analyze_frame, read_csv_bytes, store_batch
from src.runtime.alerting import query_events


def test_unknown_category_no_test_vocabulary_leak():
    train=pd.DataFrame({'proto':['tcp','udp']})
    test=pd.DataFrame({'proto':['secret_test_category','tcp']})
    a,b,enc=encode_categoricals(train,test,['proto'])
    assert 'secret_test_category' not in enc['proto'].classes_
    assert b.proto.iloc[0]==enc['proto'].transform(['__unknown__'])[0]
    assert b.proto.iloc[1]==a.proto.iloc[0]
    x,y,e=encode_categorical_features(train.values,test.values,[0])
    assert y[0,0]==e[0].transform(['__unknown__'])[0]


def test_fresh_training_has_independent_validation_and_integrity(trained_bundle):
    report=inspect_bundle()
    assert report['fit_rows']+report['validation_rows']==report['train_rows']
    assert report['test_rows']<=120
    assert 'final test excluded' in report['selection_data']
    assert report['n_features_selected']>0
    assert sum(sum(r) for r in report['confusion_matrix'])==report['test_rows']


def test_runtime_and_offline_encoding_match_unknown(engine):
    row=pd.read_csv(config.UNSW_TEST_PATH).iloc[0].to_dict()
    row['service']='unknown-testing-service'
    vector=engine._apply_preprocessing(row)
    index=engine.feature_order.index('service')
    assert vector[0,index]==engine.categorical_encoders['service'].transform(['__unknown__'])[0]
    prediction,prob=engine.predict(row)
    assert prediction in (0,1) and 0<=prob<=1


def test_incomplete_or_nonfinite_features_fail(engine):
    with pytest.raises(ValueError):engine.predict({'dur':0})
    row=pd.read_csv(config.UNSW_TEST_PATH).iloc[0].to_dict();row['dur']=float('inf')
    with pytest.raises(ValueError):engine.predict(row)


def test_bundle_tampering_detected_before_deserialization(trained_bundle):
    path=Path(config.MODEL_PATH);original=path.read_bytes()
    try:
        path.write_bytes(b'not a pickle')
        with pytest.raises(ValueError,match='integrity'):inspect_bundle()
    finally:path.write_bytes(original)


def test_batch_prediction_persistence_and_idempotency(engine,db):
    f=pd.read_csv(config.UNSW_TEST_PATH).head(12)
    results,features=analyze_frame(f,engine)
    batch=str(uuid.uuid4())
    assert store_batch(results,features,batch,db_path=db)==12
    assert store_batch(results,features,batch,db_path=db)==0
    rows,total=query_events(db_path=db)
    assert total==12 and all(r['source']=='csv_import' for r in rows)
    assert all('label' not in json.loads(r['features_json']) for r in rows)


@pytest.mark.parametrize('data',[b'',b'foo\n',b'\xff',b'a\n'+b'1\n'*1001])
def test_bad_csv_rejected(data):
    with pytest.raises(ValueError):read_csv_bytes(data)


def test_csv_missing_negative_and_infinite_fields(engine):
    with pytest.raises(ValueError):analyze_frame(pd.DataFrame({'bad':[1]}),engine)
    f=pd.read_csv(config.UNSW_TEST_PATH).head(1)
    for value in [-1,float('inf'),None,'not-a-number']:
        bad=f.copy();bad['dur']=value
        with pytest.raises(ValueError):analyze_frame(bad,engine)


def test_test_labels_cannot_influence_selection(trained_bundle):
    import train
    original=pd.read_csv(config.UNSW_TEST_PATH)
    first=json.loads(Path(config.FEATURE_SCHEMA_PATH).read_text())['selected_features_in_order']
    changed=original.copy();changed['label']=1-changed.label
    try:
        changed.to_csv(config.UNSW_TEST_PATH,index=False)
        train.run_training()
        second=json.loads(Path(config.FEATURE_SCHEMA_PATH).read_text())['selected_features_in_order']
        assert first==second
    finally:
        original.to_csv(config.UNSW_TEST_PATH,index=False)
        train.run_training()


def test_training_lock_rejects_parallel_writer_and_inference(trained_bundle):
    import train
    from src.runtime.training_lock import training_lock, training_in_progress
    with training_lock(config.SAVED_MODELS_DIR):
        assert training_in_progress(config.SAVED_MODELS_DIR)
        with pytest.raises(RuntimeError,match='already active'):train.run_training()
        with pytest.raises(ValueError,match='active'):inspect_bundle()
    assert not training_in_progress(config.SAVED_MODELS_DIR)
    inspect_bundle()


def test_invalid_late_row_rolls_back_whole_batch(db):
    results=pd.DataFrame([{'prediction':'attack','confidence':.9,'alert':True}]*2)
    features=[{'proto':'tcp','service':'http'},{'proto':'tcp','service':'http','bad':float('nan')}]
    with pytest.raises(ValueError):store_batch(results,features,str(uuid.uuid4()),db_path=db)
    assert query_events(db_path=db)[1]==0


def test_concurrent_batch_token_is_idempotent(db):
    from concurrent.futures import ThreadPoolExecutor
    token=str(uuid.uuid4())
    results=pd.DataFrame([{'prediction':'normal','confidence':.9,'alert':False}])
    features=[{'proto':'tcp','service':'http'}]
    def save(_):return store_batch(results,features,token,db_path=db)
    with ThreadPoolExecutor(max_workers=2) as pool:counts=list(pool.map(save,range(2)))
    assert sum(counts)==1 and query_events(db_path=db)[1]==1


@pytest.mark.parametrize('name',['random_forest','decision_tree','knn','svm'])
def test_all_preserved_classifiers_produce_valid_probabilities(name):
    from src.models.model_factory import get_model_factory
    X=np.array([[i%2, i/30] for i in range(30)],dtype=float)
    y=np.array([i%2 for i in range(30)])
    model=get_model_factory(name)();model.fit(X,y)
    probabilities=model.predict_proba(X[:3])
    assert probabilities.shape==(3,2)
    assert np.allclose(probabilities.sum(axis=1),1)
