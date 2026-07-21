import pandas as pd

def calculate_hfasi(price_changes: pd.Series,expenditure_weights: pd.Series,baseline=100.0):
    expenditure_weights=expenditure_weights.reindex(price_changes.index)
    if expenditure_weights.isna().any(): raise ValueError("Missing expenditure weights")
    weights=expenditure_weights/expenditure_weights.sum()
    return baseline*(1+float((price_changes*weights).sum())/100)
