"""Active-learning acquisition: GP-UCB and DNN Thompson-sampling.

The two acquisition strategies benchmarked across the active-learning rounds:

  * GP-UCB - a Gaussian-process regressor (Matern nu=5/2 + WhiteKernel) scored by
    the upper confidence bound mu + beta*sigma (beta = 2).
  * DNN-TS - a 5-member bootstrap ensemble of MLP regressors; the acquisition score
    is a single Thompson draw (one random ensemble member), with the ensemble mean
    and spread available for the exploitation/uncertainty views.

`rank_correlation` evaluates a round by Spearman rho between the acquisition ranking
and the observed-activity ranking, using the project's descending average-tie
ranking with detection-floor snapping (see `rankcorr`).

Features are produced upstream (one-hot / ProtParam via `ml.feature_engineering`,
ESM-Partial via `structure`, ESM-2 separately). Per Yang et al. 2025 (ALDE), ESM-2
is too high-dimensional for the GP and is used with the DNN only. The per-round
prediction caches behind the published panels are deposited on Zenodo, not shipped.

The ProtParam panels use the FULL-chain encoding (`get_features(..., scope='full')`,
per-residue physicochemical properties over the whole protein) to match the deployed
Stage-2 regressor. The full-length encoding is dominated by residues that are constant
across variants (the fixed scaffold flanking the mutated helix); a raw MLP is numerically
degenerate on it, so `fit_dnn_ensemble(..., standardize=True)` standardises features before
the DNN (as the deployed ProtParam pipeline does). The GP needs no scaling: its Matern
kernel is invariant to the constant features, so GP-UCB is identical for the full and
helix-only ProtParam encodings.

Figures: AL rank-correlation panels across encoding x acquisition x enzyme
(Fig S32-S45 SrUGT76G1; Fig S54-S67 UGTSL2).
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from rankcorr import rank_descending_average, snap_floor

GP_BETA = 2.0
N_ENSEMBLE = 5
BOOTSTRAP_FRAC = 0.90
TS_SEED = 42


def fit_gp(X, y, random_state=42):
    """Fit a Matern(nu=5/2) + WhiteKernel Gaussian process with normalized targets."""
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 10.0)))
    gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=5,
                                  normalize_y=True, random_state=random_state)
    gp.fit(X, y)
    return gp


def gp_ucb(gp, X, beta=GP_BETA):
    """GP posterior mean, std, and UCB acquisition score mu + beta*sigma."""
    mu, sigma = gp.predict(X, return_std=True)
    return mu, sigma, mu + beta * sigma


def fit_dnn_ensemble(X, y, n_ensemble=N_ENSEMBLE, bootstrap_frac=BOOTSTRAP_FRAC,
                     standardize=False):
    """Fit a bootstrap ensemble of MLP regressors. Returns None if n < 3.

    Set ``standardize=True`` for the full-chain ProtParam encoding: it fits a
    per-member StandardScaler (train-only) ahead of the MLP, matching the deployed
    ProtParam pipeline and preventing the numerical degeneracy a raw MLP shows on the
    constant-flank-dominated full-length features. Each returned model is then a
    (StandardScaler -> MLPRegressor) pipeline, so ``dnn_thompson`` works unchanged.
    Leave ``standardize=False`` (default) for the low-dimensional encodings (one-hot,
    ProtParam-helix, ESM), preserving the published raw-feature behaviour.
    """
    n = len(y)
    if n < 3:
        return None
    n_b = max(2, int(n * bootstrap_frac))
    models = []
    for i in range(n_ensemble):
        Xb, yb = resample(X, y, n_samples=n_b, replace=False, random_state=i)
        mlp = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', alpha=0.01,
                           max_iter=2000, random_state=i,
                           early_stopping=(n_b > 10),
                           validation_fraction=0.1 if n_b > 10 else 0.0,
                           n_iter_no_change=20, tol=1e-4)
        m = make_pipeline(StandardScaler(), mlp) if standardize else mlp
        m.fit(Xb, yb)
        models.append(m)
    return models


def dnn_thompson(models, X, ts_seed=TS_SEED):
    """Ensemble predictions: mean, spread, and a single Thompson-sampling draw.

    The Thompson score is the prediction of one randomly chosen ensemble member
    (fixed seed for reproducibility) - the exploratory acquisition signal.
    """
    preds = np.column_stack([m.predict(X) for m in models])
    rng = np.random.default_rng(ts_seed)
    ts = preds[:, rng.integers(0, len(models))]
    return preds.mean(axis=1), preds.std(axis=1), ts


def rank_correlation(pred_scores, y_observed):
    """Spearman rho between an acquisition ranking and the observed-activity ranking.

    Both are ranked descending with average ties; the observed floor group is snapped
    first so ULP-level float differences do not split the tied detection-floor block.
    Returns (rho, p, pred_rank, obs_rank); rho is nan when scores are all equal.
    """
    pred_scores = np.asarray(pred_scores, dtype=float)
    if len(np.unique(pred_scores)) <= 1:
        ones = np.ones(len(pred_scores))
        return np.nan, np.nan, ones, ones
    pred_rank = rank_descending_average(pred_scores)
    obs_rank = rank_descending_average(snap_floor(y_observed))
    rho, p = spearmanr(pred_rank, obs_rank)
    return rho, p, pred_rank, obs_rank
