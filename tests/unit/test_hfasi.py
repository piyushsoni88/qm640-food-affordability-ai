import pandas as pd
from food_affordability_ai.affordability.hfasi import calculate_hfasi

def test_hfasi():
    assert calculate_hfasi(pd.Series({"rice":10,"wheat":0}),pd.Series({"rice":0.5,"wheat":0.5}))==105
