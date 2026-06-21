"""Structural techniques: ESMFold Ca-Ca distance features and structure superposition.

Two methods used by the structural supplementary panels:

  * ESMPartial distance features - parse an ESMFold PDB and compute the within-group
    Ca-Ca distances between the catalytic and pocket residues (the "ESM-Partial"
    encoding fed to the two-stage model and the structural landscapes).
  * Structure superposition - sequence-anchored SVD superposition RMSD (BLOSUM62
    correspondence) and structure-based TM-align, used to compare an ESMFold model
    against a crystal structure (e.g. SrUGT76G1 6O88) or WT-vs-variant.
  * Structure-activity correlation - all pairwise Ca-Ca distances per structure
    (104,653 for a 458-residue model), each correlated (Pearson) with measured
    activity; pairs above a correlation threshold rank the reference residues whose
    geometry tracks activity (Fig 4f / Extended Data).

Residue sets are enzyme-specific (SrUGT76G1: 7 catalytic + 9 pocket = 57 pairs;
UGTSL2 differs). Inputs are ESMFold PDB structures and the 6O88 crystal, which are
deposited on Zenodo (see the manuscript Data Availability statement); they are not
shipped in this repository.

Figures: structural superposition / RMSD (Fig S50); ESMFold pLDDT and Ca-distance
panels (Fig S18, S51); the ESM-Partial encoding behind Fig 4, Fig 5, and the
ESM-Partial AL panels (Fig S34, S37, S58, S64); the Ca-distance vs activity
correlation (Fig 4f, Extended Data).
"""

import numpy as np
import pandas as pd

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}

# SrUGT76G1 ESMPartial residues (PDB numbering): 7 catalytic + 9 pocket.
CATALYTIC = [25, 124, 146, 147, 155, 380, 381]
POCKET = [85, 88, 90, 195, 196, 199, 200, 203, 204]


def parse_ca(pdb_path, residues=None, chain='A'):
    """Ca coordinates of the first model. Returns ordered list of (resnum, aa, xyz).

    residues: optional iterable of residue numbers to keep (default: all).
    """
    keep = set(residues) if residues is not None else None
    out = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
                continue
            if line[21] not in (chain, ' '):
                continue
            try:
                rn = int(line[22:26])
            except ValueError:
                continue
            if keep is not None and rn not in keep:
                continue
            aa = THREE_TO_ONE.get(line[17:20].strip(), 'X')
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            out.append((rn, aa, xyz))
    return out


def _within_pairs(res_list):
    res = sorted(res_list)
    return [(a, b) for i, a in enumerate(res) for b in res[i + 1:]]


def esmpartial_distances(pdb_path, catalytic=CATALYTIC, pocket=POCKET):
    """Within-group Ca-Ca distances for one ESMFold structure.

    Returns {'res_{hi}_res_{lo}_contact': distance_angstrom} for every pair within
    the catalytic group and within the pocket group (cross-group pairs excluded).
    Raises if any requested residue lacks a Ca atom.
    """
    coords = {rn: xyz for rn, _, xyz in parse_ca(pdb_path, residues=catalytic + pocket)}
    missing = [r for r in catalytic + pocket if r not in coords]
    if missing:
        raise ValueError('PDB %s missing Ca for residues %s' % (pdb_path, missing))
    out = {}
    for lo, hi in _within_pairs(catalytic) + _within_pairs(pocket):
        out['res_%d_res_%d_contact' % (hi, lo)] = float(np.linalg.norm(coords[lo] - coords[hi]))
    return out


def full_ca_distances(pdb_path, chain='A'):
    """All pairwise Ca-Ca distances for one structure (lower triangle).

    Returns (labels, distances): labels are 'res_{hi}_res_{lo}_contact' (1-based
    residue numbers, hi > lo) and distances the matching Angstrom values, one per
    residue pair. For a 458-residue model this is 458*457/2 = 104,653 pairs -- the
    full structural feature set behind the Fig 4f / Extended Data correlation.
    """
    ca = parse_ca(pdb_path, chain=chain)
    nums = [rn for rn, _, _ in ca]
    coords = np.array([xyz for _, _, xyz in ca])
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    rows, cols = np.tril_indices(len(nums), k=-1)
    labels = ['res_%d_res_%d_contact' % (nums[i], nums[j]) for i, j in zip(rows, cols)]
    return labels, d[rows, cols].astype(float)


def distance_activity_correlations(dist_df, activities, method='pearson'):
    """Per-pair correlation of Ca-Ca distance with activity.

    dist_df    : variants x pairs DataFrame (columns = labels from full_ca_distances)
    activities : matching per-variant activity values
    method     : 'pearson' (default) or any pandas-supported correlation

    Returns a Series of correlation coefficients indexed by pair label, sorted
    descending. Apply any variant filter (e.g. product-forming only) before calling.
    """
    acts = pd.Series(list(activities), index=dist_df.index, dtype=float)
    r = dist_df.corrwith(acts, method=method)
    r.name = '%s_r' % method
    return r.sort_values(ascending=False)


def rank_reference_residues(corrs, pos_thresh=0.45, neg_thresh=-0.5):
    """Rank residues by how often they appear in strongly-correlated distance pairs.

    corrs      : per-pair correlation Series (from distance_activity_correlations)
    pos_thresh : keep pairs with r above this as positively correlated
    neg_thresh : keep pairs with r below this as negatively correlated

    Counts the residue numbers in each retained pair label. Returns (pos_counts,
    neg_counts), each a DataFrame with columns Residue, Count sorted by Count.
    """
    import re

    def _counts(selected):
        nums = []
        for label in selected.index:
            nums.extend(re.findall(r'\d+', label))
        out = pd.Series(nums).value_counts().reset_index()
        out.columns = ['Residue', 'Count']
        return out

    return _counts(corrs[corrs > pos_thresh]), _counts(corrs[corrs < neg_thresh])


def _parse_ca_seq(pdb_path, chain='A'):
    rec = parse_ca(pdb_path, chain=chain)
    order = [rn for rn, _, _ in rec]
    seq = ''.join(aa for _, aa, _ in rec)
    ca = {rn: xyz for rn, _, xyz in rec}
    return order, seq, ca


def superpose_rmsd(pdb_ref, pdb_mov, chain='A', core_trim=True, trim_cutoff=2.0):
    """Sequence-anchored SVD superposition RMSD between two structures.

    Establishes Ca correspondence by BLOSUM62 global alignment, then superposes with
    Bio.SVDSuperimposer. Returns dict with global RMSD (all matched Ca) and, if
    core_trim, the rigid-core RMSD after iterative outlier trimming at trim_cutoff A.
    Requires biopython.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices
    from Bio.SVDSuperimposer import SVDSuperimposer

    ord_r, seq_r, ca_r = _parse_ca_seq(pdb_ref, chain)
    ord_m, seq_m, ca_m = _parse_ca_seq(pdb_mov, chain)

    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = 'global'
    aln = aligner.align(seq_m, seq_r)[0]

    fixed, moving = [], []
    for (mA, mB), (rA, rB) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(mB - mA):
            rn_m, rn_r = ord_m[mA + k], ord_r[rA + k]
            if rn_m in ca_m and rn_r in ca_r:
                moving.append(ca_m[rn_m])
                fixed.append(ca_r[rn_r])
    fixed, moving = np.array(fixed), np.array(moving)
    n_match = len(moving)

    sup = SVDSuperimposer()
    sup.set(fixed, moving)
    sup.run()
    result = {'global_rmsd': float(sup.get_rms()), 'n_match': n_match,
              'local_rmsd': None, 'n_core': n_match}

    if core_trim:
        keep = np.ones(n_match, bool)
        for _ in range(20):
            sup.set(fixed[keep], moving[keep])
            sup.run()
            rot, tran = sup.get_rotran()
            dev = np.linalg.norm((moving @ rot + tran) - fixed, axis=1)
            nk = dev <= trim_cutoff
            if nk.sum() < 0.5 * n_match:
                nk = dev <= np.quantile(dev, 0.5)
            if nk.sum() == keep.sum() and np.array_equal(nk, keep):
                break
            keep = nk
        sup.set(fixed[keep], moving[keep])
        sup.run()
        result['local_rmsd'] = float(sup.get_rms())
        result['n_core'] = int(keep.sum())
    return result


def tmalign(pdb_ref, pdb_mov, chain='A'):
    """Structure-based TM-align (length-normalized) via tmtools.

    TM-align computes its own structural alignment (no sequence correspondence
    needed). Returns dict with TM-scores normalized by each chain and the alignment
    RMSD. Requires the optional `tmtools` package.
    """
    from tmtools import tm_align
    from tmtools.io import get_structure, get_residue_data

    def chain_data(path):
        model = next(get_structure(path).get_models())
        return get_residue_data(model[chain])

    coords_r, seq_r = chain_data(pdb_ref)
    coords_m, seq_m = chain_data(pdb_mov)
    res = tm_align(coords_r, coords_m, seq_r, seq_m)
    return {'tm_norm_ref': float(res.tm_norm_chain1),
            'tm_norm_mov': float(res.tm_norm_chain2),
            'rmsd': float(res.rmsd)}
