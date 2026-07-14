"""Descending average-tie ranking for predicted-vs-observed rank-correlation scatters.

Figures: rank-correlation scatters (Fig 3, Fig 4; Fig S14, S19; AL panels Fig S32-S45, S54-S67).
"""
import numpy as np
import pandas as pd


def rank_descending_average(values):
    """Rank 1 = highest value; ties share the mean rank; returned as float (never astype(int))."""
    return pd.Series(np.asarray(values, dtype=float)).rank(
        ascending=False, method='average').to_numpy()


def snap_floor(values, atol=1e-7):
    """Collapse values within atol of the data-derived minimum to one float so a ULP-split detection floor ranks as one tie."""
    v = np.asarray(values, dtype=float).copy()
    fv = float(v.min())
    v[np.abs(v - fv) < atol] = fv
    return v
