"""Sequence-space landscape: one-hot encoding, t-SNE embedding, and Nadaraya-Watson activity terrain.

Figures: sequence-space landscapes and activity terrains (Fig 3, Fig 5, Fig 7; Fig S13, S16).
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from utils import load_data, WT_SEQ, ROUND_ORDER

AAS        = list('ACDEFGHIKLMNPQRSTVWY')
TSNE_PERP  = 25
TSNE_ITER  = 2000
TSNE_SEED  = 42
NW_BW      = 1.0
NW_GRID_N  = 200
SEQ_NW_BW  = 2.0    # one-hot Euclidean units; half-max weight at ~2 AA changes
DATA_FILE  = 'SrUGT76G1_variant_library.xlsx'


def one_hot(seq):
    arr = np.zeros(len(seq) * len(AAS))
    for i, aa in enumerate(str(seq)):
        if aa in AAS:
            arr[i * len(AAS) + AAS.index(aa)] = 1
    return arr


def load_and_encode(path=None):
    """Load variant library, apply Cycle 1.4 filter, one-hot encode; returns (all_seqs, X, acts, cycles, ids, wt_idx)."""
    if path is None:
        path = os.path.join(_here, '..', 'data', DATA_FILE)
    df = load_data(path)
    variants = df[df['Type'] == 'Variant'].copy()

    # Cycle 1.4: retain only model-predicted actives (Class_Predictions == 1)
    mask4    = (variants['DBTL Cycle'] == 'LD-BT 3') & (variants['Class_Predictions'] == 1)
    mask_oth = variants['DBTL Cycle'] != 'LD-BT 3'
    variants = variants[mask_oth | mask4].copy()

    wt_row = pd.DataFrame([{
        'ID': 'WT',
        'Candidate_Sequence': WT_SEQ,
        'DBTL Cycle': 'WT',
        'RebA Relative Specific Activity to WT': 1.0,
        'Class_Predictions': 1.0,
    }])
    all_seqs = pd.concat([variants, wt_row], ignore_index=True)

    seqs   = all_seqs['Candidate_Sequence'].fillna(WT_SEQ).tolist()
    X      = np.array([one_hot(s) for s in seqs])
    acts   = all_seqs['RebA Relative Specific Activity to WT'].values.astype(float)
    cycles = all_seqs['DBTL Cycle'].values
    ids    = all_seqs['ID'].values
    wt_idx = int(np.where(cycles == 'WT')[0][0])
    return all_seqs, X, acts, cycles, ids, wt_idx


def run_tsne(X):
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERP,
        random_state=TSNE_SEED,
        n_iter=TSNE_ITER,
        learning_rate='auto',
        init='pca',
        metric='euclidean',
    )
    Z = tsne.fit_transform(X)
    print(f't-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}')
    return Z


def compute_nw_terrain(Z, acts, wt_idx, idx=None):
    """Nadaraya-Watson activity estimate on a 200x200 grid in t-SNE space; idx selects rows (default all non-WT plus WT)."""
    xs = np.linspace(Z[:, 0].min() - 2, Z[:, 0].max() + 2, NW_GRID_N)
    ys = np.linspace(Z[:, 1].min() - 2, Z[:, 1].max() + 2, NW_GRID_N)
    XX, YY = np.meshgrid(xs, ys)
    grid   = np.column_stack([XX.ravel(), YY.ravel()])

    if idx is None:
        idx = np.where(np.arange(len(acts)) != wt_idx)[0]
    use  = np.unique(np.append(idx, wt_idx))
    diff = grid[:, None, :] - Z[use][None, :, :]
    W    = np.exp(-np.sum(diff ** 2, axis=2) / (2 * NW_BW ** 2))
    ZZ   = ((W * acts[use][None, :]).sum(1) / W.sum(1)).reshape(XX.shape)
    return XX, YY, ZZ, xs, ys


def compute_seq_nw(X, acts, bw=SEQ_NW_BW):
    """Per-variant leave-one-out NW activity estimate in one-hot sequence space; bw in Euclidean units."""
    n = len(acts)
    estimates = np.zeros(n)
    for i in range(n):
        d2    = np.sum((X - X[i]) ** 2, axis=1)
        d2[i] = np.inf           # leave-one-out
        w     = np.exp(-d2 / (2 * bw ** 2))
        estimates[i] = (w * acts).sum() / w.sum()
    return estimates


if __name__ == '__main__':
    print('Loading and encoding data...')
    all_seqs, X, acts, cycles, ids, wt_idx = load_and_encode()
    print(f'  {len(all_seqs)} sequences  |  {X.shape[1]} one-hot features')
    for r in ROUND_ORDER + ['WT']:
        print(f'  {r}: n={np.sum(cycles == r)}')

    print('\nRunning t-SNE...')
    Z = run_tsne(X)
    print(f'  2-D range: x=[{Z[:,0].min():.1f}, {Z[:,0].max():.1f}]  '
          f'y=[{Z[:,1].min():.1f}, {Z[:,1].max():.1f}]')
    print(f'  WT position: ({Z[wt_idx,0]:.2f}, {Z[wt_idx,1]:.2f})')

    print('\nComputing sequence-space NW estimates (leave-one-out, BW=%.1f)...' % SEQ_NW_BW)
    seq_nw = compute_seq_nw(X, acts)
    print(f'  Range: {seq_nw.min():.3f} to {seq_nw.max():.3f} xWT')

    print('\nComputing NW terrain grid (BW=%.1f, %d×%d)...' % (NW_BW, NW_GRID_N, NW_GRID_N))
    XX, YY, ZZ, xs, ys = compute_nw_terrain(Z, acts, wt_idx)
    print(f'  Terrain range: {ZZ.min():.3f} to {ZZ.max():.3f} xWT')

    coord_path = os.path.join(_here, 'SrUGT76G1_tsne_coordinates.csv')
    coord_df = pd.DataFrame({
        'Variant_ID':            ids,
        'DBTL_Cycle':            cycles,
        'Candidate_Sequence':    all_seqs['Candidate_Sequence'].fillna(WT_SEQ).values,
        'RebA_RSA':              acts,
        'tSNE_1':                Z[:, 0],
        'tSNE_2':                Z[:, 1],
        'Seq_NW_Estimate':       seq_nw,
    })
    coord_df.to_csv(coord_path, index=False)
    print(f'\nSaved: {coord_path}')

    terrain_path = os.path.join(_here, 'SrUGT76G1_nw_terrain.csv')
    terrain_df = pd.DataFrame({
        'tSNE_1':               XX.ravel(),
        'tSNE_2':               YY.ravel(),
        'NW_Activity_Estimate': ZZ.ravel(),
    })
    terrain_df.to_csv(terrain_path, index=False)
    print(f'Saved: {terrain_path}')
    print('\nDone.')
