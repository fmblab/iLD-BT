"""End-to-end two-stage prediction: classify active/inactive, then rank only the
predicted-active variants with the regressor.

Stage 1 (classifier) and stage 2 (regressor) are tuned and validated independently in
`evaluation.py`. This module composes the SELECTED fitted models into the cascade that
produces the navigation / rank-correlation panels: the regressor scores ONLY the
variants the classifier predicts active, and predicted-inactive variants rank last.

The two stages may use different encodings (the published pipeline uses an ESM-Partial
classifier and a ProtParam regressor), so classifier and regressor feature matrices are
passed separately - same rows, same order.

Figures: two-stage navigation and rank correlation (Fig 4; UGTSL2 transfer Fig 7).
"""

import ast
import glob
import os

import numpy as np
import pandas as pd

from .models import CLASSIFIERS, REGRESSORS
from .evaluation import _build_model

# Validation-table conventions written by evaluation.run_classification / run_regression.
_VAL_GLOB = {'classification': 'Class_Val_*.xlsx', 'regression': 'Reg_Val_*.xlsx'}
_DEFAULT_METRIC = {'classification': 'Avg_Precision', 'regression': 'Spearman_r'}
_LOWER_IS_BETTER = {'MSE', 'RMSE', 'MAE', 'Spearman_p'}


def _registry(stage):
    if stage == 'classification':
        return CLASSIFIERS
    if stage == 'regression':
        return REGRESSORS
    raise ValueError("stage must be 'classification' or 'regression', got %r" % stage)


def _load_val(results_dir, stage):
    files = sorted(glob.glob(os.path.join(results_dir, _VAL_GLOB[stage])))
    if not files:
        raise FileNotFoundError(
            'no %s validation tables in %s - run the stage first' % (stage, results_dir))
    return pd.concat([pd.read_excel(f) for f in files], ignore_index=True)


def rank_models(results_dir, stage, metric=None):
    """Rank models best-first by mean validation score from the *_Val_*.xlsx tables.

    metric defaults to Avg_Precision (classification) / Spearman_r (regression);
    error metrics (MSE/RMSE/MAE) sort ascending. Returns a DataFrame [Model, mean_<metric>].
    """
    _registry(stage)
    metric = metric or _DEFAULT_METRIC[stage]
    df = _load_val(results_dir, stage)
    if metric not in df.columns:
        raise KeyError('metric %r not in %s tables; available: %s'
                       % (metric, stage, sorted(c for c in df.columns if df[c].dtype != object)))
    asc = metric in _LOWER_IS_BETTER
    agg = df.groupby('Model')[metric].mean().sort_values(ascending=asc)
    return agg.rename('mean_%s' % metric).reset_index()


def select_model(results_dir, stage, metric=None, model_name=None):
    """The model name for a stage: `model_name` if given (validated against the
    registry), otherwise the top-ranked model by `metric` (see rank_models)."""
    reg = _registry(stage)
    if model_name is not None:
        if model_name not in reg:
            raise KeyError('%r not a %s model; choose from %s'
                           % (model_name, stage, list(reg)))
        return model_name
    return rank_models(results_dir, stage, metric).iloc[0]['Model']


def fit_stage_model(results_dir, stage, model_name, X, y, metric=None, random_state=0):
    """Fit one stage's model using its best stored hyperparameters.

    Reads the top-scoring parameter set for `model_name` from the validation tables; if
    no tables/rows exist, falls back to the model's default hyperparameters.
    """
    reg = _registry(stage)
    if model_name not in reg:
        raise KeyError('%r not a %s model; choose from %s' % (model_name, stage, list(reg)))
    metric = metric or _DEFAULT_METRIC[stage]
    params = {}
    try:
        rows = _load_val(results_dir, stage)
        rows = rows[rows['Model'] == model_name]
        if len(rows) and metric in rows.columns:
            best = rows.sort_values(metric, ascending=(metric in _LOWER_IS_BETTER)).iloc[0]
            if isinstance(best.get('Params'), str):
                params = ast.literal_eval(best['Params'])
    except FileNotFoundError:
        pass  # no tables -> default hyperparameters
    model = _build_model(reg[model_name], params, random_state=random_state)
    model.fit(X, y)
    return model


def two_stage_predict(clf, reg, X_clf, X_reg, ids=None, active_label=1):
    """Cascade a fitted classifier and regressor.

    clf, reg     : fitted stage-1 classifier and stage-2 regressor.
    X_clf, X_reg : feature matrices for the two stages (same rows, same order).
    ids          : optional row identifiers for the returned table.
    active_label : classifier label that means "active" (default 1).

    Returns a DataFrame with:
      predicted_active : bool, the stage-1 call.
      reg_score        : stage-2 regressor score; NaN where predicted inactive.
      rank             : 1 = best predicted-active (descending, average ties);
                         predicted-inactive variants all share the last rank.
    """
    X_clf = np.asarray(X_clf)
    X_reg = np.asarray(X_reg)
    if len(X_clf) != len(X_reg):
        raise ValueError('X_clf and X_reg must have the same number of rows')

    active = np.asarray(clf.predict(X_clf)) == active_label
    score = np.full(len(active), np.nan, dtype=float)
    if active.any():
        score[active] = reg.predict(X_reg[active])

    # Rank predicted-actives by descending score (average ties); NaN scores (the
    # predicted-inactives) are left out by pandas and then parked at the last rank.
    rank = pd.Series(score).rank(ascending=False, method='average').to_numpy()
    rank[~active] = int(active.sum()) + 1

    out = pd.DataFrame({'predicted_active': active, 'reg_score': score, 'rank': rank})
    if ids is not None:
        out.insert(0, 'id', list(ids))
    return out
