"""Candidate-motif library design: WT-contact filter cascade, sequence logos, and cloning prep.

Figures: candidate-motif contact-filter cascades (Fig 2a; Fig S2-S8, S15) and
sequence logos (Fig S52).
"""

import pandas as pd
import re
from Bio import SeqIO
from Bio.SeqUtils import MeltingTemp as mt

A1        = r"P\w{20}[KRH]G"
A6B5      = r"D\w{22}S"
A16B11    = r"SV\w{29}FLW"   # WT-matched inner-29 residues
A18B14A19 = r"H[SC]\w{22}[DE]Q"
B12A17B13 = r"[RKH]G\w{22}H"

nterminal = "MENKTETTVRRRRRIILF"
cterminal = (
    "FSITIFHTNFNKPKTSNYPHFTFRFILDNDPQDERISNLPTHGPLAGMRIPIINEHGADELRRELELLMLASEE"
    "DEEVSCLITDALWYFAQSVADSLNLRRLVLMTSSLFNFHAHVSLPQFDELGYLDPDDKTRLEEQASGFPMLKV"
    "KDIKSAYSNWQILKEILGKMIKQTKASSGVIWNSFKELEESELETVIREIPAPSFLIPLPKHLTASSSSLLDHD"
    "RTVFQWLDQQPPSSVLYVSFGSTSEVDEKDFLEIARGLVDSKQSFLWVVRPGFVKGSTWVEPLPDGFLGERGR"
    "IVKWVPQQEVLAHGAIGAFWTHSGWNSTLESVCEGVPMIFSDFGLDQPLNARYMSDVLKVGVYLENGWERGEIA"
    "NAIRRVMVDEEGEYIRQNARVLKQKADVSLMKGGSSYESLESLVSYISSL*"
)

CODON_DICT = {
    '*': 'TAA', 'A': 'GCA', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA',
    'F': 'TTC', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'K': 'AAG',
    'L': 'CTG', 'M': 'ATG', 'N': 'AAC', 'P': 'CCA', 'Q': 'CAG',
    'R': 'CGC', 'S': 'TCC', 'T': 'ACC', 'V': 'GTG', 'W': 'TGG',
    'Y': 'TAC',
}

SaltCorrection = mt.salt_correction(Na=2, Tris=1, Mg=5, dNTPs=0.2, method=4)


def find_motif(motif, df, motif_name):
    def find_overlapping_matches(pattern, sequence):
        matches = []
        for i in range(len(sequence)):
            match = re.match(pattern, sequence[i:])
            if match:
                matches.append(match.group(0))
        return matches

    out_list = []
    for _, row in df.iterrows():
        for match in find_overlapping_matches(motif, row['AA Sequence']):
            out_list.append({
                'Phyla':             row['Phyla'],
                'Species':           row['Species'],
                'Motif':             motif_name,
                'Parent':            row['NCBI Accession'],
                'Motif_Sequence': match,
                'Length':            len(match),
                'Start':             row['AA Sequence'].index(match) + 1,
            })
    return pd.DataFrame(out_list)


def process_for_seq_logo(df):
    raw_sequences = []
    with open("Raw Sequences.txt", 'w') as f:
        for seq in df['Motif_Sequence']:
            f.write(seq[1:-1] + '\n')
            raw_sequences.append(seq)
    return raw_sequences


def make_logo(seq_list):
    import numpy as np
    import logomaker as lm
    import matplotlib.pyplot as plt

    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    ignore = set("-XBZJOU")
    L = min(len(s) for s in seq_list)
    seqs_use = [s[:L] for s in seq_list]

    counts = pd.DataFrame(0, index=range(L), columns=alphabet)
    for seq in seqs_use:
        for pos, aa in enumerate(seq):
            if aa not in ignore and aa in alphabet:
                counts.loc[pos, aa] += 1

    lm.Logo(counts, color_scheme="chemistry")
    plt.show()


def coryne_codon_optimize(sequence):
    return "".join(CODON_DICT[aa] for aa in sequence)


def gg_modify(nt_seq, motif_name):
    if motif_name == 'A1':
        return "GGTCTCTC" + nt_seq + "TTCAGAGACC"
    elif motif_name == 'A6B5':
        return "GGTCTCTACC" + nt_seq + "TAGAGACC"
    elif motif_name == 'B12A17B13':
        return "GGTCTCTCGAA" + nt_seq + "TAGAGACC"
    elif motif_name == 'A18B14A19':
        return "GGTCTCT" + nt_seq + "CAGAGACC"
    elif motif_name == 'A16B11':
        return "GGTCTCTCTCT" + nt_seq[:-4] + "CTGG" + "AGAGACC"
    else:
        return nt_seq


def Tm_calc(seq_input):
    seq = seq_input.upper()
    counts = {c: seq.count(c) for c in "ATGC"}
    return (counts["A"] + counts["T"]) * 2 + (counts["G"] + counts["C"]) * 4 - 5


def Benchling_Anneal_calc(seq_input):
    seq = seq_input.upper()
    counts = {c: seq.count(c) for c in "ATGC"}
    return 0.3 * ((counts["A"] + counts["T"]) * 2 + (counts["G"] + counts["C"]) * 4 - 5)


def Annealing(Tm_primer, Tm_product):
    return 0.3 * Tm_primer + 0.7 * Tm_product - 14.3


def enzyme_site_free(x):
    return "GGTCTCT" not in x and "AGAGACC" not in x


def gc_anchor(x):
    return x[0] in "GC" and x[-1] in "GC"


def load_genome(gb_path):
    record = SeqIO.read(gb_path, "genbank")
    return record.seq


def make_genome_free(genome):
    def Genome_free(x):
        return x not in genome
    return Genome_free


def assign_ids(df, prefix, wt_accession='AAR06912.1'):
    i = 1
    id_col = []
    for _, row in df.iterrows():
        if row['Parent'] == wt_accession:
            id_col.append('WT')
        else:
            id_col.append(prefix + str(i))
            i += 1
    df = df.copy()
    df['ID'] = id_col
    return df


def run_cloning_prep(df, motif_label, genome):
    Genome_free = make_genome_free(genome)
    df = df.copy()
    df['Full_Sequence']         = df['Motif_Sequence'].apply(lambda x: nterminal + x + cterminal)
    df['NT_Sequence']           = df['Motif_Sequence'].apply(coryne_codon_optimize)
    df['BsaI Free']             = df['NT_Sequence'].apply(enzyme_site_free)
    df['Genome Free']           = df['NT_Sequence'].apply(Genome_free)
    df['NT_Sequence for Cloning'] = df['NT_Sequence'].apply(lambda x: gg_modify(x, motif_label))
    df['Tm']                    = df['NT_Sequence for Cloning'].apply(lambda x: mt.Tm_NN(x) + SaltCorrection)
    return df
