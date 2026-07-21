from food_affordability_ai.evaluation.metrics import mae,rmse,smape

def test_metrics_zero():
    a=[1,2,3]
    assert mae(a,a)==0 and rmse(a,a)==0 and smape(a,a)==0
