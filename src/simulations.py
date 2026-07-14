"""Simulation constants, seeds, and functions for the UGT76G1 active-learning campaign.

Figures: library-construction strategy comparison (Fig 6).
"""

from __future__ import annotations
import numpy as np
from collections import Counter

from utils import WT_SEQ, ROUND_ORDER, ROUND_NS

n_pos        = len(WT_SEQ)
x            = np.arange(n_pos)
wt_aa_labels = ['%d\n%s' % (i + 1, aa) for i, aa in enumerate(WT_SEQ)]

CONTACT_IDX      = [3, 4, 5, 6, 8]          # 0-based SSM target positions
COLONY_NS_UNIQUE = [25, 100, 7, 48]
COLONY_NS        = [3 * n for n in COLONY_NS_UNIQUE]   # 3x coverage = [75, 300, 21, 144]

NT_SEQ     = "CCAGTGCCATTCCAGGGCCACATCAACCCAATCCTGCAGCTGGCAAACGTGCTGTACTCCAAGGGC"
CODONS_SEQ = [NT_SEQ[i:i+3] for i in range(0, len(NT_SEQ), 3)]
L_BP       = len(NT_SEQ)

GENETIC_CODE: dict[str, str] = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

TRANSITIONS:   dict[str, str]       = {'A':'G', 'G':'A', 'C':'T', 'T':'C'}
TRANSVERSIONS: dict[str, list[str]] = {'A':['T','C'], 'G':['T','C'],
                                        'C':['A','G'], 'T':['A','G']}

# Taq Ts:Tv = 4:1 (Cadwell & Joyce 1994; Vanhercke et al. 2005)
P_TS       = 0.80
P_TV       = 0.10   # per transversion type (2 types -> 20% total)
# Mutation rate: 0.5%/bp (Cadwell & Joyce 1992; McCullum et al. 2010)
MU_PER_BP  = 0.005
LAMBDA_EP  = MU_PER_BP * L_BP   # Poisson mean mut/seq

_NT_P_NONSYN: list[float] = []
for _codon in CODONS_SEQ:
    _wt_aa = GENETIC_CODE[_codon]
    for _cpos in range(3):
        _nt   = _codon[_cpos]
        _muts = [(TRANSITIONS[_nt], P_TS)] + [(tv, P_TV) for tv in TRANSVERSIONS[_nt]]
        _nonsyn = sum(
            p for _new_nt, p in _muts
            if GENETIC_CODE.get(_codon[:_cpos] + _new_nt + _codon[_cpos+1:], '*')
               not in ('*', _wt_aa)
        )
        _NT_P_NONSYN.append(_nonsyn)
P_NONSYN_EFF = float(np.mean(_NT_P_NONSYN))

N_SEEDS  = 200
N_EPPCR  = 100

EPPCR_BASE_SEED     = 7777
SSM_BASE_SEED       = 42
EPPCR_AVG_BASE_SEED = 4
EPPCR_BIAS_SEED     = 12
EPPCR_THEO_SEED     = 16

STD_AAS = list('ACDEFGHIKLMNPQRSTVWY')
AA_IDX  = {aa: i for i, aa in enumerate(STD_AAS)}
N_AAS   = len(STD_AAS)

wt_aa_indices = [AA_IDX.get(aa, -1) for aa in WT_SEQ]


def ssm_coupon_expected(n: int, n_aa: int = 20) -> int:
    """Expected unique AAs from n colony picks (coupon-collector formula)."""
    if n == 0:
        return 1
    return round(n_aa * (1 - ((n_aa - 1) / n_aa) ** n))


def simulate_eppcr(N: int, seed: int):
    """Single epPCR run under Poisson + Taq Ts:Tv bias; returns (mut_freq_pct, unique_aas, N, n_unique_seqs)."""
    rng       = np.random.default_rng(seed)
    mut_count = np.zeros(n_pos, dtype=int)
    aa_sets   = [set() for _ in range(n_pos)]
    n_unique  = 0
    for _ in range(N):
        k = rng.poisson(LAMBDA_EP)
        if k == 0:
            continue
        seq_changed = False
        for nt_pos in rng.integers(0, L_BP, size=k):
            aa_pos = int(nt_pos // 3);  cpos = int(nt_pos % 3)
            codon  = CODONS_SEQ[aa_pos];  nt = codon[cpos]
            wt_aa  = GENETIC_CODE[codon]
            r      = rng.random()
            new_nt = (TRANSITIONS[nt] if r < P_TS else
                      TRANSVERSIONS[nt][0] if r < P_TS + P_TV else TRANSVERSIONS[nt][1])
            new_aa = GENETIC_CODE.get(codon[:cpos] + new_nt + codon[cpos+1:], '*')
            if new_aa not in ('*', wt_aa):
                aa_sets[aa_pos].add(new_aa)
                mut_count[aa_pos] += 1
                seq_changed = True
        if seq_changed:
            n_unique += 1
    return (mut_count / N * 100,
            np.array([max(1, len(s)) for s in aa_sets]),
            N, n_unique)


def simulate_eppcr_avg(N: int, n_seeds: int = N_SEEDS, base_seed: int = 0):
    """Average of n_seeds independent epPCR runs."""
    mf_acc = np.zeros(n_pos);  ua_acc = np.zeros(n_pos);  nu_acc = 0.0
    for s in range(n_seeds):
        mf, ua, _, nu = simulate_eppcr(N, seed=base_seed + s)
        mf_acc += mf;  ua_acc += ua;  nu_acc += nu
    return mf_acc / n_seeds, ua_acc / n_seeds, N, nu_acc / n_seeds


_STD_AAS_LIST = STD_AAS
_OTHER_AAS    = {aa: [a for a in _STD_AAS_LIST if a != aa] for aa in _STD_AAS_LIST}


def simulate_eppcr_theoretical(N: int, seed: int):
    """Theoretical-maximum epPCR: uniform substitution model (no Taq bias)."""
    rng       = np.random.default_rng(seed)
    mut_count = np.zeros(n_pos, dtype=int)
    aa_sets   = [set() for _ in range(n_pos)]
    n_unique  = 0
    for _ in range(N):
        k = rng.poisson(LAMBDA_EP)
        if k == 0:
            continue
        seq_changed = False
        for _ in range(k):
            pos    = int(rng.integers(0, n_pos))
            new_aa = _OTHER_AAS[WT_SEQ[pos]][int(rng.integers(0, 19))]
            aa_sets[pos].add(new_aa)
            mut_count[pos] += 1
            seq_changed = True
        if seq_changed:
            n_unique += 1
    return (mut_count / N * 100,
            np.array([max(1, len(s)) for s in aa_sets]),
            N, n_unique)


def simulate_eppcr_theoretical_avg(N: int, n_seeds: int = N_SEEDS, base_seed: int = 0):
    """Average of n_seeds theoretical epPCR runs."""
    mf_acc = np.zeros(n_pos);  ua_acc = np.zeros(n_pos);  nu_acc = 0.0
    for s in range(n_seeds):
        mf, ua, _, nu = simulate_eppcr_theoretical(N, seed=base_seed + s)
        mf_acc += mf;  ua_acc += ua;  nu_acc += nu
    return mf_acc / n_seeds, ua_acc / n_seeds, N, nu_acc / n_seeds


def round_position_stats_df(sub_df):
    """Per-position substitution frequency and unique AA count from a variants DataFrame."""
    seqs = [s.strip() for s in sub_df['Candidate_Sequence'].dropna()
            if isinstance(s, str) and len(s.strip()) == n_pos]
    mf = np.zeros(n_pos);  ua = np.ones(n_pos, dtype=int)
    for i, wt_aa in enumerate(WT_SEQ):
        aas = [s[i] for s in seqs]
        if not aas:
            continue
        cnt  = Counter(aas)
        mf[i] = (1 - cnt.get(wt_aa, 0) / len(aas)) * 100
        ua[i] = len(cnt)
    return mf, ua


# Grantham (1974) pairwise physicochemical distance matrix
GRANTHAM: dict[tuple[str, str], int] = {
    ('A','R'):112,('A','N'):111,('A','D'):126,('A','C'):195,('A','Q'):91,
    ('A','E'):107,('A','G'):60, ('A','H'):86, ('A','I'):94, ('A','L'):96,
    ('A','K'):106,('A','M'):84, ('A','F'):113,('A','P'):27, ('A','S'):99,
    ('A','T'):58, ('A','W'):148,('A','Y'):112,('A','V'):64,
    ('R','N'):86, ('R','D'):96, ('R','C'):180,('R','Q'):43, ('R','E'):54,
    ('R','G'):125,('R','H'):29, ('R','I'):97, ('R','L'):102,('R','K'):26,
    ('R','M'):91, ('R','F'):97, ('R','P'):103,('R','S'):110,('R','T'):71,
    ('R','W'):101,('R','Y'):77, ('R','V'):96,
    ('N','D'):23, ('N','C'):139,('N','Q'):46, ('N','E'):42, ('N','G'):80,
    ('N','H'):68, ('N','I'):149,('N','L'):153,('N','K'):94, ('N','M'):142,
    ('N','F'):158,('N','P'):91, ('N','S'):46, ('N','T'):65, ('N','W'):174,
    ('N','Y'):143,('N','V'):133,
    ('D','C'):154,('D','Q'):61, ('D','E'):45, ('D','G'):94, ('D','H'):81,
    ('D','I'):168,('D','L'):172,('D','K'):101,('D','M'):160,('D','F'):177,
    ('D','P'):108,('D','S'):65, ('D','T'):85, ('D','W'):181,('D','Y'):160,
    ('D','V'):152,
    ('C','Q'):154,('C','E'):170,('C','G'):159,('C','H'):174,('C','I'):198,
    ('C','L'):198,('C','K'):202,('C','M'):196,('C','F'):205,('C','P'):169,
    ('C','S'):112,('C','T'):149,('C','W'):215,('C','Y'):194,('C','V'):192,
    ('Q','E'):29, ('Q','G'):87, ('Q','H'):24, ('Q','I'):109,('Q','L'):113,
    ('Q','K'):53, ('Q','M'):101,('Q','F'):116,('Q','P'):76, ('Q','S'):68,
    ('Q','T'):42, ('Q','W'):130,('Q','Y'):99, ('Q','V'):96,
    ('E','G'):98, ('E','H'):40, ('E','I'):134,('E','L'):138,('E','K'):56,
    ('E','M'):126,('E','F'):140,('E','P'):93, ('E','S'):80, ('E','T'):65,
    ('E','W'):152,('E','Y'):122,('E','V'):121,
    ('G','H'):98, ('G','I'):135,('G','L'):138,('G','K'):127,('G','M'):127,
    ('G','F'):153,('G','P'):42, ('G','S'):56, ('G','T'):59, ('G','W'):184,
    ('G','Y'):147,('G','V'):109,
    ('H','I'):94, ('H','L'):99, ('H','K'):32, ('H','M'):87, ('H','F'):100,
    ('H','P'):77, ('H','S'):89, ('H','T'):47, ('H','W'):115,('H','Y'):83,
    ('H','V'):84,
    ('I','L'):5,  ('I','K'):102,('I','M'):10, ('I','F'):21, ('I','P'):95,
    ('I','S'):142,('I','T'):89, ('I','W'):61, ('I','Y'):33, ('I','V'):29,
    ('L','K'):107,('L','M'):15, ('L','F'):22, ('L','P'):98, ('L','S'):145,
    ('L','T'):92, ('L','W'):61, ('L','Y'):36, ('L','V'):32,
    ('K','M'):95, ('K','F'):102,('K','P'):103,('K','S'):121,('K','T'):78,
    ('K','W'):110,('K','Y'):85, ('K','V'):97,
    ('M','F'):28, ('M','P'):87, ('M','S'):135,('M','T'):81, ('M','W'):67,
    ('M','Y'):36, ('M','V'):21,
    ('F','P'):114,('F','S'):155,('F','T'):103,('F','W'):40, ('F','Y'):22,
    ('F','V'):50,
    ('P','S'):74, ('P','T'):38, ('P','W'):147,('P','Y'):110,('P','V'):29,
    ('S','T'):58, ('S','W'):177,('S','Y'):144,('S','V'):124,
    ('T','W'):128,('T','Y'):92, ('T','V'):69,
    ('W','Y'):37, ('W','V'):88,
    ('Y','V'):55,
}
_G: dict[tuple[str, str], int] = {}
for (_a, _b), _d in GRANTHAM.items():
    _G[(_a, _b)] = _d
    _G[(_b, _a)] = _d


def grantham_dist(a: str, b: str) -> float:
    if a == b:
        return 0
    return _G.get((a, b), float('nan'))


def total_grantham(seq, wt: str = WT_SEQ) -> float:
    seq = str(seq).strip() if isinstance(seq, str) else ''
    if len(seq) != len(wt):
        return float('nan')
    return sum(grantham_dist(w, s) for w, s in zip(wt, seq) if w != s)
