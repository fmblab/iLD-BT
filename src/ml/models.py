"""Classifier and regressor definitions with hyperparameter grids; entries: model, params, needs_scaling, extra_init (unsearched kwargs).

Figures: two-stage prediction pipeline (Fig 3, Fig 4; UGTSL2 transfer Fig 7).
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


CLASSIFIERS = {
    'RandomForest': {
        'model': RandomForestClassifier,
        'params': {
            'max_features':      ['sqrt', 'log2', None],
            'class_weight':      ['balanced'],
            'criterion':         ['gini', 'entropy', 'log_loss'],
            'n_estimators':      [10, 50, 100, 500, 1000, 2000],
            'min_samples_split': [2, 5, 10, 50],
            'min_samples_leaf':  [1, 2, 5, 10, 50],
            'bootstrap':         [True],
        },
        'needs_scaling': False,
        'extra_init':    {'oob_score': True},
    },
    'LogisticRegression': {
        'model': LogisticRegression,
        'params': {
            'C':            [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty':      ['l1', 'l2'],
            'solver':       ['liblinear', 'saga'],
            'max_iter':     [5000],
            'class_weight': ['balanced'],
        },
        'needs_scaling': True,
        'extra_init':    {},
    },
    'SVC': {
        'model': SVC,
        'params': {
            'C':            [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel':       ['rbf', 'linear', 'poly'],
            'gamma':        ['scale', 'auto'],
            'class_weight': ['balanced'],
            'probability':  [True],
        },
        'needs_scaling': True,
        'extra_init':    {},
    },
    'XGBoost': {
        'model': XGBClassifier,
        'params': {
            'n_estimators':     [10, 50, 100, 500, 1000],
            'max_depth':        [3, 5, 7, 10],
            'learning_rate':    [0.01, 0.05, 0.1, 0.3],
            'subsample':        [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        },
        'needs_scaling': False,
        'extra_init':    {'eval_metric': 'logloss', 'verbosity': 0},
    },
}


REGRESSORS = {
    'RandomForest': {
        'model': RandomForestRegressor,
        'params': {
            'max_features':      ['sqrt', 'log2', None],
            'criterion':         ['squared_error', 'absolute_error',
                                  'friedman_mse', 'poisson'],
            'n_estimators':      [10, 50, 100, 500, 1000, 2000],
            'min_samples_split': [2, 5, 10, 50],
            'min_samples_leaf':  [1, 2, 5, 10, 50],
            'bootstrap':         [True],
            'max_depth':         [None, 5, 10, 50, 100],
        },
        'needs_scaling': False,
        'extra_init':    {'oob_score': True},
    },
    'LinearRegression': {
        'model': LinearRegression,
        'params': {
            'fit_intercept': [True, False],
        },
        'needs_scaling': True,
        'extra_init':    {},
    },
    'Ridge': {
        'model': Ridge,
        'params': {
            'alpha':         [0.01, 0.1, 1, 10, 100, 1000],
            'fit_intercept': [True, False],
        },
        'needs_scaling': True,
        'extra_init':    {},
    },
    'Lasso': {
        'model': Lasso,
        'params': {
            'alpha':         [0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
            'max_iter':      [5000],
        },
        'needs_scaling': True,
        'extra_init':    {},
    },
    'SVR': {
        'model': SVR,
        'params': {
            'C':       [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel':  ['rbf', 'linear', 'poly'],
            'gamma':   ['scale', 'auto'],
            'epsilon': [0.01, 0.1, 0.5, 1.0],
        },
        'needs_scaling': True,
        'extra_init':    {},
    },
    'XGBoost': {
        'model': XGBRegressor,
        'params': {
            'n_estimators':     [10, 50, 100, 500, 1000],
            'max_depth':        [3, 5, 7, 10],
            'learning_rate':    [0.01, 0.05, 0.1, 0.3],
            'subsample':        [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        },
        'needs_scaling': False,
        'extra_init':    {'verbosity': 0},
    },
}
