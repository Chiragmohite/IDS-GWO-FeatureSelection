import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from src.gwo.gwo_algorithm import run_gwo
from src.gwo.feature_selector import GWOFeatureSelector
from src.gwo.fitness import sigmoid_transfer, _step_probability


@pytest.mark.parametrize("variant", ["GWO","bGWO"])
def test_initial_masks_are_evaluated_and_score_matches_mask(variant):
    seen=[]
    def evaluate(mask):
        seen.append(tuple(mask))
        score = sum((i+1)*v for i,v in enumerate(mask))
        return score, score
    wolf=run_gwo(variant,evaluate,6,4,0,0,1)
    assert len(seen)==4
    assert wolf.score == sum((i+1)*v for i,v in enumerate(wolf.features))
    assert wolf.score>0


@pytest.mark.parametrize("variant", ["GWO","bGWO"])
def test_seed_reproducibility_and_final_score_integrity(variant):
    def evaluate(mask):
        score=sum((i%3-1)*v for i,v in enumerate(mask))
        return score,score
    a=run_gwo(variant,evaluate,9,5,4,0,1,seed=61)
    b=run_gwo(variant,evaluate,9,5,4,0,1,seed=61)
    assert (a.features,a.score,a.position)==(b.features,b.score,b.position)
    assert a.score==evaluate(a.features)[1]


@pytest.mark.parametrize("dim,wolves,iterations,lo,hi", [(0,3,1,0,1),(5,2,1,0,1),(5,3,-1,0,1),(5,3,1,1,0)])
def test_invalid_optimizer_parameters(dim,wolves,iterations,lo,hi):
    with pytest.raises(ValueError):run_gwo("bGWO",lambda m:(1,1),dim,wolves,iterations,lo,hi)


def test_mask_cache_and_empty_feature_guard():
    X=np.array([[0,1],[1,0],[0,2],[2,0],[0,3],[3,0]],dtype=float)
    y=np.array([0,1,0,1,0,1])
    selector=GWOFeatureSelector(lambda:DecisionTreeClassifier(random_state=42),X,y,X,y,[])
    assert selector._evaluate_feature_mask([0,0])==(float('-inf'),0)
    first=selector._evaluate_feature_mask([1,0])
    selector.classifier_factory=lambda:(_ for _ in ()).throw(AssertionError('should use cache'))
    assert selector._evaluate_feature_mask([1,0])==first
    with pytest.raises(ValueError):selector._evaluate_feature_mask([1])
    with pytest.raises(ValueError):selector.run_multiple('bGWO',0,3,1)


def test_extreme_numeric_transfer_stable():
    assert sigmoid_transfer(-1e9)==0 and sigmoid_transfer(1e9)==1
    assert 0 <= _step_probability(1e9,1e9) <= 1
    assert 0 <= _step_probability(-1e9,1e9) <= 1
