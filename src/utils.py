"""Shared utilities: variant-library loading, WT sequence, round labels, plot style.

Figures: shared across the analysis; activity/round loaders feed the round
activity distributions (Fig 1) and the design-space generator feeds library
design (Fig 2).
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from collections import Counter


WT_SEQ = "PVPFQGHINPILQLANVLYSKG"   # UGT76G1 helix; AAR06912.1

STD_AAS = 'ACDEFGHIKLMNPQRSTVWY'

ROUND_ORDER = [
    'LD-BT 0',
    'LD-BT 1',
    'LD-BT 2',
    'LD-BT 3',
]

ROUND_NS: dict[str, int] = {
    'LD-BT 0': 20,
    'LD-BT 1': 82,
    'LD-BT 2':  7,
    'LD-BT 3': 64,
}

ROUND_COLORS: dict[str, str] = {
    'LD-BT 0': '#BDBDBD',
    'LD-BT 1': '#5B9BD5',
    'LD-BT 2': '#2E75B6',
    'LD-BT 3': '#1F3864',
    'WT':            '#4EA72A',
}

RCPARAMS: dict = {
    'figure.facecolor':    'white',
    'axes.facecolor':      'white',
    'axes.edgecolor':      '#333333',
    'axes.labelcolor':     '#222222',
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'xtick.color':         '#444444',
    'ytick.color':         '#444444',
    'xtick.direction':     'out',
    'ytick.direction':     'out',
    'font.family':         'sans-serif',
    'font.sans-serif':     ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':           7,
    'axes.labelsize':      7,
    'axes.titlesize':      7,
    'axes.titleweight':    'bold',
    'legend.fontsize':     6,
    'legend.frameon':      False,
    'figure.dpi':          150,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.facecolor':   'white',
}


def load_data(path: str = 'SrUGT76G1_variant_library.xlsx') -> pd.DataFrame:
    """Read the variant-library xlsx and add alias columns Activity, WT_Hamming, Above_WT."""
    df = pd.read_excel(path)
    df['Activity'] = df['RSA_RebA_xWT']
    df['WT_Hamming'] = df['Motif_Sequence'].apply(
        lambda s: hamming(WT_SEQ, s) if isinstance(s, str) else np.nan
    )
    df['Above_WT'] = df['Activity'] > 1.0
    return df


def generate_all_combos(wt: str, positions, alphabet: str = STD_AAS) -> list[str]:
    """Every motif sequence formed by substituting `alphabet` at the 0-based
    `positions` of `wt`, all other positions fixed to wild-type."""
    wt = list(wt)
    combos = []
    for combo in itertools.product(alphabet, repeat=len(positions)):
        seq = wt[:]
        for pos, aa in zip(positions, combo):
            seq[pos] = aa
        combos.append(''.join(seq))
    return combos


def hamming(s1: str, s2: str) -> float:
    s1 = s1.strip() if isinstance(s1, str) else ''
    s2 = s2.strip() if isinstance(s2, str) else ''
    if len(s1) != len(s2) or not s1:
        return np.nan
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def round_activities(df: pd.DataFrame, cycle: str) -> pd.Series:
    """Return the RebA relative specific activity series for one DBTL cycle."""
    mask = df['DBTL Cycle'] == cycle
    return (
        df.loc[mask, 'RebA Relative Specific Activity to WT']
        .dropna()
        .reset_index(drop=True)
    )


def position_stats(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """Per-position substitution frequency and amino acid diversity for one DBTL cycle."""
    sub = df.loc[df['DBTL Cycle'] == cycle, 'Candidate_Sequence'].dropna()
    seqs = [s.strip() for s in sub if isinstance(s, str) and len(s.strip()) == len(WT_SEQ)]

    rows = []
    for i, wt_aa in enumerate(WT_SEQ):
        aas = [s[i] for s in seqs]
        cnt = Counter(aas)
        n = max(len(aas), 1)
        rows.append({
            'position':  i + 1,
            'wt_aa':     wt_aa,
            'mut_freq':  1 - cnt.get(wt_aa, 0) / n,
            'unique_aas': len(cnt),
            'wt_freq':   cnt.get(wt_aa, 0) / n,
            'aa_counts': dict(cnt),
        })

    return pd.DataFrame(rows)
