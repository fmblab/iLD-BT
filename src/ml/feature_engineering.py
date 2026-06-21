"""The four feature-encoding methods (one-hot, ProtParams, AAindex, ESMFold) for the UGT76G1 A1-helix ML pipeline.

Figures: two-stage prediction pipeline (Fig 3, Fig 4; UGTSL2 transfer Fig 7).
"""

import os
import numpy as np
import pandas as pd
from itertools import product as iproduct

_OH_PARTIAL_START = 'POS_19_RES_A'
_OH_PARTIAL_END   = 'POS_40_RES_Y'
_AMINO_ACIDS      = list('ACDEFGHIKLMNPQRSTVWY')

_PP_PARTIAL_REGEX = r'_(19|2[0-9]|3[0-9]|40)$'   # residues 19-40 (A1 helix)

CATALYTIC = ['25', '124', '146', '147', '155', '380', '381']
POCKET    = ['85', '88', '90', '195', '196', '199', '200', '203', '204']


def one_hot_encode(df_in):
    """Return full one-hot DataFrame for all sequences in df_in."""
    seq0    = df_in.iloc[0]['Full_Sequence'].rstrip('*')
    seq_len = len(seq0)
    cols    = [f'POS_{p}_RES_{a}'
               for p, a in iproduct(range(1, seq_len + 1), _AMINO_ACIDS)]
    rows = []
    for seq in df_in['Full_Sequence']:
        seq     = seq.rstrip('*')
        present = {f'POS_{i+1}_RES_{seq[i]}' for i in range(len(seq))}
        rows.append({c: (1 if c in present else 0) for c in cols})
    return pd.DataFrame(rows, columns=cols)


def _one_hot_partial(df_full):
    return df_full.loc[:, _OH_PARTIAL_START:_OH_PARTIAL_END].reset_index(drop=True)


def protparams_encode(df_in):
    """Per-residue physicochemical properties (hydrophobicity, molar mass, pI, aromaticity) via Biopython ProteinAnalysis."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    records = []
    for seq in df_in['Full_Sequence']:
        seq = seq.rstrip('*')
        row = {}
        for i, aa in enumerate(seq, start=1):
            try:
                pa = ProteinAnalysis(aa)
                row[f'Residue Hydrophobicity_{i}'] = pa.gravy()
                row[f'Residue MolarMass_{i}']       = pa.molecular_weight()
                row[f'Residue pI_{i}']              = pa.isoelectric_point()
                row[f'Residue Aromaticity_{i}']     = pa.aromaticity()
            except Exception:
                row[f'Residue Hydrophobicity_{i}'] = np.nan
                row[f'Residue MolarMass_{i}']       = np.nan
                row[f'Residue pI_{i}']              = np.nan
                row[f'Residue Aromaticity_{i}']     = np.nan
        records.append(row)
    return pd.DataFrame(records)


def _protparams_partial(df_full):
    return df_full.filter(regex=_PP_PARTIAL_REGEX).reset_index(drop=True)


def aaindex_encode(df_in, feat_dict, start=19, end=40, partial=True):
    """Embed sequences using a pre-loaded AAindex feature dict; start/end are 1-based residue bounds for partial encoding."""
    final_output = []
    for idx in range(len(df_in)):
        seq = str(df_in.iloc[idx]['Full_Sequence']).rstrip('*')
        if partial:
            seq_region  = seq[start - 1:end]
            pos_offset  = start
        else:
            seq_region  = seq
            pos_offset  = 1

        row = {}
        for feature, aa_map in feat_dict.items():
            for j, aa in enumerate(seq_region):
                col = f'{feature}_{j + pos_offset}'
                row[col] = aa_map.get(aa, 0.0)
        final_output.append(row)

        if idx % 50 == 0:
            print(f'  AAindex: {idx + 1}/{len(df_in)}')

    return pd.DataFrame(final_output).reset_index(drop=True)


def load_aaindex(data_dir):
    """Load AAIndex_dimensionality_reduced.xlsx from data_dir."""
    fpath = os.path.join(data_dir, 'AAIndex_dimensionality_reduced.xlsx')
    return pd.read_excel(fpath, index_col=0).to_dict()


def esmfold_encode(df_in, scope, data_dir):
    """Load pre-computed ESMFold Cα distance-matrix features; scope 'partial' (catalytic+pocket pairs) or 'full' (lower-triangle matrix)."""
    library = df_in.iloc[0].get('Library', 'unknown')
    safe_lib = str(library).replace(' ', '_').replace('/', '_')
    fpath = os.path.join(data_dir, f'ESMFold_features_{safe_lib}_{scope}.xlsx')
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f'ESMFold features not found:\n  {fpath}\n'
            'Run 03_esmfold_preprocessing.ipynb first.'
        )
    return pd.read_excel(fpath, index_col=0).reset_index(drop=True)


def get_features(df_in, encoding, scope, data_dir=None):
    """Return feature DataFrame(s): single X for scope 'partial'/'full', or (X_partial, X_full) for 'both'."""
    if encoding == 'one_hot':
        full    = one_hot_encode(df_in)
        partial = _one_hot_partial(full)

    elif encoding == 'protparams':
        full    = protparams_encode(df_in)
        partial = _protparams_partial(full)

    elif encoding == 'aaindex':
        if data_dir is None:
            raise ValueError('data_dir required for encoding="aaindex"')
        feat_dict = load_aaindex(data_dir)
        partial   = aaindex_encode(df_in, feat_dict, partial=True)
        full      = aaindex_encode(df_in, feat_dict, partial=False)

    elif encoding == 'esmfold':
        if data_dir is None:
            raise ValueError('data_dir required for encoding="esmfold"')
        partial = esmfold_encode(df_in, 'partial', data_dir)
        full    = esmfold_encode(df_in, 'full',    data_dir)

    else:
        raise ValueError(f'Unknown encoding: {encoding!r}. '
                         'Choose one of: one_hot, protparams, aaindex, esmfold')

    if scope == 'partial':
        return partial
    if scope == 'full':
        return full
    return partial, full
