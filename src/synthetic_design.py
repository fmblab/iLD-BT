"""Synthetic sequence design: position-specific residue sets and constrained mutation.

The LD-BT 3 (Round 3) acquisition step. Observed activity-positive variants define,
per motif position, the set of amino acids seen at or above wild-type activity
("residue sets"). New candidate motifs are then sampled by applying a bounded number
of those allowed substitutions to template sequences drawn from the prior rounds.
The sampled motifs are scored by the two-stage model (src/ml/two_stage.py) and the
top-ranked designs are selected for the next build.

Underlies Fig 4 (SrUGT76G1 LD-BT 3 synthetic design) and the UGTSL2 LD-BT 1 analog.
"""
import random
from typing import Dict, List, Optional, Set

import pandas as pd


def build_residue_sets(
    sequences: List[str],
    activities: List[float],
    threshold: float = 1.0,
) -> Dict[int, Set[str]]:
    """Build per-position residue sets from variants above an activity threshold.

    sequences  : equal-length motif (Candidate_Sequence) strings
    activities : matching relative specific activities (xWT)
    threshold  : only variants strictly above this activity contribute residues

    Returns a dict mapping 0-based position index to the set of observed amino acids.
    """
    active_seqs = [seq for seq, act in zip(sequences, activities) if act > threshold]
    if not active_seqs:
        raise ValueError("No variants exceed activity threshold %s" % threshold)

    residue_sets: Dict[int, Set[str]] = {}
    for seq in active_seqs:
        for i, aa in enumerate(seq):
            residue_sets.setdefault(i, set()).add(aa)
    return residue_sets


def generate_by_interset_shuffling(
    residue_sets: Dict[int, Set[str]],
    n_variants: int,
    seed: Optional[int] = 42,
) -> List[str]:
    """Recombine observed residues across positions (one residue drawn per position).

    For each variant, one amino acid is drawn uniformly at random from each position's
    residue set. Positions are taken in ascending index order.

    Returns a list of generated motif strings.
    """
    rng = random.Random(seed)
    sets_as_lists = {pos: sorted(aa) for pos, aa in sorted(residue_sets.items())}

    variants = []
    for _ in range(n_variants):
        variants.append("".join(rng.choice(sets_as_lists[pos]) for pos in sorted(sets_as_lists)))
    return variants


def generate_constrained_mutations(
    wt_seq: str,
    residue_sets: Dict[int, Set[str]],
    n_variants: int,
    seed: Optional[int] = 42,
    max_mutations: int = 7,
    base_sequences: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Sample motifs by applying bounded, residue-set-constrained mutations to templates.

    Each design starts from a randomly chosen template (base_sequences, default [wt_seq]),
    draws a random mutation count in 1..max_mutations, then substitutes that many eligible
    positions with a residue drawn from the position's residue set (excluding the current one).

    wt_seq         : wild-type motif, used as the sole template when base_sequences is None
    residue_sets   : output of build_residue_sets
    n_variants     : number of designs to sample
    seed           : random seed (reproducibility)
    max_mutations  : upper bound on substitutions per design
    base_sequences : templates to sample from (default [wt_seq])

    Returns a DataFrame with columns original_sequence, Candidate_Sequence,
    num_mutations, mutated_positions.
    """
    if base_sequences is None:
        base_sequences = [wt_seq]

    rng = random.Random(seed)
    data = []

    for _ in range(n_variants):
        base = rng.choice(base_sequences)
        seq_list = list(base)
        mutation_details = []

        num_mutations = rng.randint(1, max_mutations)

        valid_sites = [
            idx for idx in residue_sets
            if idx < len(seq_list) and len(residue_sets[idx] - {seq_list[idx]}) > 0
        ]
        selected_sites = rng.sample(valid_sites, min(num_mutations, len(valid_sites)))

        for idx in selected_sites:
            original_aa = seq_list[idx]
            options = sorted(residue_sets[idx] - {original_aa})
            new_aa = rng.choice(options)
            seq_list[idx] = new_aa
            mutation_details.append("%d:%s->%s" % (idx, original_aa, new_aa))

        data.append({
            "original_sequence": base,
            "Candidate_Sequence": "".join(seq_list),
            "num_mutations": len(selected_sites),
            "mutated_positions": "; ".join(mutation_details),
        })

    return pd.DataFrame(data)
