"""RandomizedSearchCV loops and L2-validation functions for classification and regression.

Figures: two-stage prediction pipeline (Fig 3, Fig 4; UGTSL2 transfer Fig 7).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              precision_score, roc_auc_score,
                              average_precision_score, confusion_matrix,
                              mean_squared_error, mean_absolute_error,
                              r2_score, ndcg_score, make_scorer)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr



def _ndcg(y_true, y_pred):
    yt = np.array(y_true).reshape(1, -1)
    yp = np.array(y_pred).reshape(1, -1)
    return ndcg_score(yt, yp)

ndcg_scorer = make_scorer(_ndcg, greater_is_better=True)


def _build_model(cfg, params, random_state=None):
    """Instantiate a model (or Pipeline) from a CLASSIFIERS/REGRESSORS entry."""
    ModelClass = cfg['model']
    extra      = cfg['extra_init']
    scale      = cfg['needs_scaling']

    try:   # random_state only if the class accepts it
        m = ModelClass(**params, **extra, random_state=random_state)
    except TypeError:
        m = ModelClass(**params, **extra)

    if scale:
        return Pipeline([('scaler', StandardScaler()), ('model', m)])
    return m


def _strip_pipe_prefix(params):
    """Remove 'model__' prefix from piped parameter keys."""
    return {k.replace('model__', ''): v for k, v in params.items()}


def run_classification(X_train, y_train, X_val, y_val,
                       classifiers, scoring_metrics,
                       encoding, scope_label, train_filter, val_data,
                       output_dir, n_iter=100):
    """RandomizedSearchCV (StratifiedKFold 5) per (classifier, metric), then 5-seed re-fit + validation; saves CV and val xlsx."""
    os.makedirs(output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    tag = f'{encoding}_{scope_label}_{train_filter}_{val_data}'

    for scoring in scoring_metrics:
        for clf_name, cfg in classifiers.items():
            print(f'  [{clf_name}] {scoring} ...')

            if cfg['needs_scaling']:
                base   = Pipeline([('scaler', StandardScaler()),
                                   ('model', cfg['model']())])
                pgrid  = {f'model__{k}': v for k, v in cfg['params'].items()}
            else:
                base   = cfg['model'](**cfg['extra_init'])
                pgrid  = cfg['params']

            search = RandomizedSearchCV(base, pgrid, n_iter=n_iter,
                                        cv=skf, scoring=scoring,
                                        random_state=0, n_jobs=-1)
            search.fit(X_train, y_train)
            cv_df = pd.DataFrame(search.cv_results_)

            cv_path = os.path.join(
                output_dir,
                f'Class_CV_{clf_name}_{scoring}_{tag}.xlsx')
            cv_df.to_excel(cv_path, index=False)

            best_rows = cv_df[cv_df['rank_test_score'] == 1][
                ['params', 'mean_test_score']].reset_index()

            val_records = []
            for _, row in best_rows.iterrows():
                raw = _strip_pipe_prefix(row['params']) if cfg['needs_scaling'] \
                    else row['params']
                for rs in range(5):
                    m = _build_model(cfg, raw, random_state=rs)
                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_val)
                    y_prob = (m.predict_proba(X_val)[:, 1]
                              if hasattr(m, 'predict_proba') else
                              y_pred.astype(float))
                    bi = len(np.unique(y_val)) > 1
                    try:
                        tn, fp, fn, tp = confusion_matrix(
                            y_val, y_pred, labels=[0, 1]).ravel()
                    except Exception:
                        tn = fp = fn = tp = np.nan

                    val_records.append({
                        'Model':         clf_name,
                        'Scoring':       scoring,
                        'Scope':         scope_label,
                        'Random_state':  rs,
                        'Accuracy':      accuracy_score(y_val, y_pred),
                        'F1':            f1_score(y_val, y_pred, zero_division=0),
                        'Recall':        recall_score(y_val, y_pred, zero_division=0),
                        'Precision':     precision_score(y_val, y_pred, zero_division=0),
                        'ROC_AUC':       roc_auc_score(y_val, y_prob) if bi else np.nan,
                        'Avg_Precision': average_precision_score(y_val, y_prob) if bi else np.nan,
                        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                        'CV_mean_score': row['mean_test_score'],
                        'Params':        str(raw),
                    })

            val_path = os.path.join(
                output_dir,
                f'Class_Val_{clf_name}_{scoring}_{tag}.xlsx')
            pd.DataFrame(val_records).to_excel(val_path, index=False)
            print(f'    saved: {val_path}')



REG_SCORING_MAP = {
    'mae':      'neg_mean_absolute_error',
    'mse':      'neg_mean_squared_error',
    'rmse':     'neg_root_mean_squared_error',
    'r2':       'r2',
    'ndcg':     ndcg_scorer,
}


def run_regression(X_train, y_train, X_val, y_val,
                   regressors, scoring_metrics,
                   encoding, scope_label, train_filter, val_data,
                   output_dir, n_iter=100):
    """RandomizedSearchCV (KFold 5) per (regressor, metric), then 5-seed re-fit + validation; scoring_metrics are REG_SCORING_MAP keys."""
    os.makedirs(output_dir, exist_ok=True)
    kf  = KFold(n_splits=5, shuffle=True, random_state=0)
    tag = f'{encoding}_{scope_label}_{train_filter}_{val_data}'

    for metric_key in scoring_metrics:
        scoring = REG_SCORING_MAP.get(metric_key, metric_key)
        for reg_name, cfg in regressors.items():
            print(f'  [{reg_name}] {metric_key} ...')

            if cfg['needs_scaling']:
                base  = Pipeline([('scaler', StandardScaler()),
                                  ('model', cfg['model']())])
                pgrid = {f'model__{k}': v for k, v in cfg['params'].items()}
            else:
                base  = cfg['model'](**cfg['extra_init'])
                pgrid = cfg['params']

            search = RandomizedSearchCV(base, pgrid, n_iter=n_iter,
                                        cv=kf, scoring=scoring,
                                        random_state=0, n_jobs=-1)
            search.fit(X_train, y_train)
            cv_df = pd.DataFrame(search.cv_results_)

            cv_path = os.path.join(
                output_dir,
                f'Reg_CV_{reg_name}_{metric_key}_{tag}.xlsx')
            cv_df.to_excel(cv_path, index=False)

            best_rows = cv_df[cv_df['rank_test_score'] == 1][
                ['params', 'mean_test_score']].reset_index()

            val_records = []
            for _, row in best_rows.iterrows():
                raw = _strip_pipe_prefix(row['params']) if cfg['needs_scaling'] \
                    else row['params']
                for rs in range(5):
                    m = _build_model(cfg, raw, random_state=rs)
                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_val)
                    sp_r, sp_p = spearmanr(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)

                    val_records.append({
                        'Model':              reg_name,
                        'Scoring':            metric_key,
                        'Scope':              scope_label,
                        'Random_state':       rs,
                        'MSE':                mse,
                        'RMSE':               mse ** 0.5,
                        'MAE':                mean_absolute_error(y_val, y_pred),
                        'R2':                 r2_score(y_val, y_pred),
                        'NDCG':               _ndcg(y_val, y_pred),
                        'Spearman_r':         sp_r,
                        'Spearman_p':         sp_p,
                        'OOB_Score':          getattr(m, 'oob_score_', np.nan),
                        'CV_mean_score':      row['mean_test_score'],
                        'Params':             str(raw),
                    })

            val_path = os.path.join(
                output_dir,
                f'Reg_Val_{reg_name}_{metric_key}_{tag}.xlsx')
            pd.DataFrame(val_records).to_excel(val_path, index=False)
            print(f'    saved: {val_path}')
