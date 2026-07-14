"""Data loading, sequence construction, and train/val splitting.

Figures: two-stage prediction pipeline (Fig 3, Fig 4; UGTSL2 transfer Fig 7).
"""

import os
import pandas as pd

PREFIX = 'MENKTETTVRRRRRIILF'
SUFFIX = (
    'FSITIFHTNFNKPKTSNYPHFTFRFILDNDPQDERISNLPTHGPLAGMRIPIINEHGADELRRELELLMLAS'
    'EEDEEVSCLITDALWYFAQSVADSLNLRRLVLMTSSLFNFHAHVSLPQFDELGYLDPDDKTRLEEQASGF'
    'PMLKVKDIKSAYSNWQILKEILGKMIKQTKASSGVIWNSFKELEESELETVIREIPAPSFLIPLPKHLTASS'
    'SSLLDHDRTVFQWLDQQPPSSVLYVSFGSTSEVDEKDFLEIARGLVDSKQSFLWVVRPGFVKGSTWVEPLP'
    'DGFLGERGRIVKWVPQQEVLAHGAIGAFWTHSGWNSTLESVCEGVPMIFSDFGLDQPLNARYMSDVLKVGV'
    'YLENGWERGEIANAIRRVMVDEEGEYIRQNARVLKQKADVSLMKGGSSYESLESLVSYISSL*'
)

COL_CANDIDATE   = 'Candidate_Sequence'
COL_FULL_SEQ    = 'Full_Sequence'
COL_LIBRARY     = 'Library'
COL_CONVERSION  = 'Conversion'
COL_ACTIVITY    = 'RebA Relative Specific Activity to WT'

LIB_L1 = 'A1_L1'
LIB_L2 = 'A1_L2'
LIB_WT = 'WT SrUGT76G1'


def load_and_prepare(data_dir):
    """Load the master activity spreadsheet, attach full-protein sequences, and drop any sequence containing 'X'."""
    fpath = os.path.join(data_dir, 'training_activity_data.xlsx')
    df = pd.read_excel(fpath, index_col=0)
    df[COL_FULL_SEQ] = df[COL_CANDIDATE].apply(lambda x: PREFIX + str(x) + SUFFIX)
    df = df[~df[COL_FULL_SEQ].str.contains('X', na=False)].reset_index(drop=True)
    return df


def split_train_val(df, train_filter='all', val_data='l2', data_dir=None):
    """Split into (df_train, df_val); train_filter 'all'|'pos_only', val_data 'l2'|'l2_predconv'."""
    df_l1 = df[(df[COL_LIBRARY] == LIB_L1) | (df[COL_LIBRARY] == LIB_WT)
               ].reset_index(drop=True)
    df_l2 = df[df[COL_LIBRARY] == LIB_L2].reset_index(drop=True)

    df_train = (
        df_l1[df_l1[COL_CONVERSION] == 1].reset_index(drop=True)
        if train_filter == 'pos_only' else df_l1.copy()
    )

    if val_data == 'l2_predconv':
        if data_dir is None:
            raise ValueError('data_dir required when val_data="l2_predconv"')
        pred_path = os.path.join(data_dir, 'validation_predictions.xlsx')
        df_val = pd.read_excel(pred_path, index_col=0)
        df_val[COL_FULL_SEQ] = df_val[COL_CANDIDATE].apply(
            lambda x: PREFIX + str(x) + SUFFIX)
        df_val = df_val[~df_val[COL_FULL_SEQ].str.contains('X', na=False)
                        ].reset_index(drop=True)
    else:
        df_val = df_l2.copy()

    return df_train, df_val
