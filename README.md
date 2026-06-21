# Closed-loop active learning–guided engineering of UGT76G1

Code and data for:

> **Closed-loop active learning–guided engineering of UDP-glucose transferase enhances rebaudiosides conversion from stevioside**
> Dongjae Kim & Han Min Woo. Sungkyunkwan University (SKKU).

This repository provides the active-learning pipeline behind the study, which
applies an iterative Design–Build–Test–Learn (LD-BT) workflow to engineer
*Sr*UGT76G1 for improved rebaudioside (RebA → RebD) production and transfers the
strategy to UGTSL2. The three steps of a campaign — generate the design space,
simulate library-construction strategies, and train/validate the two-stage model —
run as parameterized notebooks over the curated data included here.

The numerical values plotted in the paper's figures are provided as the
manuscript's Source Data; this repository ships the method code and the curated
datasets it operates on, not per-figure plotting scripts.

---

## Repository layout

```
ugt76g1-active-learning/
├── README.md
├── LICENSE                     # MIT
├── CITATION.cff
├── environment.yml             # conda environment
├── generate_domain.ipynb       # build the combinatorial design space
├── generate_synthetic_designs.ipynb  # sample LD-BT 3 synthetic designs (constrained mutation)
├── execute_simulation.ipynb    # in-silico library-strategy simulation
├── execute_production.ipynb    # train/validate the two-stage AL model
├── analysis/
│   └── visualization.ipynb     # example plots from shipped data + pipeline outputs
├── data/
│   ├── SrUGT76G1_variant_library.xlsx           # curated variant library
│   ├── SrUGT76G1_variant_id_crosswalk.tsv        # per-variant index (ID, round, motif)
│   ├── supplementary/                            # auxiliary prediction/results tables
│   └── README.md
├── src/                        # importable Python modules the notebooks call (analysis/method code)
│   ├── utils.py                # data loading, WT sequence, round labels, design-space generator
│   ├── simulations.py          # library-strategy simulation (epPCR / SSM)
│   ├── motif_functions.py      # candidate-motif contact-filter cascade, sequence logos
│   ├── synthetic_design.py     # residue-set + constrained-mutation synthetic design (LD-BT 3)
│   ├── landscape_data.py       # t-SNE embedding + Nadaraya-Watson activity terrain
│   ├── colormap.py             # SR landscape colormap
│   ├── rankcorr.py             # descending average-tie rank-correlation
│   ├── structure.py            # ESMFold Ca-distance features + structure superposition (RMSD / TM-align)
│   ├── al.py                   # active-learning acquisition (GP-UCB, DNN Thompson-sampling)
│   └── ml/                     # two-stage ML pipeline (classifier → regressor)
│       ├── data_loading.py
│       ├── feature_engineering.py
│       ├── models.py
│       └── evaluation.py
└── results/                    # output directory (created at runtime)
```

---

## Installation

```bash
git clone https://github.com/fmblab/iLD-BT.git
cd ugt76g1-active-learning
conda env create -f environment.yml
conda activate ugt76g1
```

## Data

The curated data needed to run the pipeline is included in this repository (the
dataset is small). The variant library is `data/SrUGT76G1_variant_library.xlsx`
(174 rows, standardized columns and published `76G1-A1-*` variant IDs), with a
per-variant index crosswalk alongside it; auxiliary prediction/results tables are
in `data/supplementary/`. See [`data/README.md`](data/README.md) for the schema.

The per-residue activity workbook used to train the production model
(`data/SrUGT76G1/training_activity_data.xlsx`) and the ESMFold structures
used for structural encodings are not redistributed here; see
`execute_production.ipynb` for the expected input layout and the manuscript's Data
Availability statement for their deposit location.

## Active-learning workflow

The campaign runs through four notebooks. Each opens with a **Parameters** cell
holding the published configuration (sizes are derived from inputs, not hardcoded);
edit that cell and Run All to reproduce or vary a step:

1. **`generate_domain.ipynb`** — build the combinatorial design space for a target motif.
2. **`execute_simulation.ipynb`** — simulate library-construction strategies (epPCR / SSM) over rounds.
3. **`execute_production.ipynb`** — train + validate the two-stage model on collected DBTL data
   (place the activity workbook under `data/SrUGT76G1/` — see the notebook's data-requirement note).
4. **`generate_synthetic_designs.ipynb`** — sample the LD-BT 3 synthetic-design library by
   constrained mutation of prior-round templates, using the shipped variant library; runs
   end-to-end without external data.

Launch from the repo root (`jupyter lab` / `jupyter notebook`) so the notebooks
resolve `src/` and `data/` correctly. Outputs go to `results/`.

`analysis/visualization.ipynb` gives example plots built from the shipped data and
the pipeline outputs (round activity distributions, the strategy simulation, and a
preview of the two-stage model tables). These are illustrative; the values behind
the paper's main and supplementary figures are provided as the manuscript's Source
Data, not regenerated here.

The shared method code these notebooks call lives in `src/` — the project's
importable Python package (`src/` is added to the path by each notebook's setup
cell, so the modules import as `utils`, `simulations`, `ml.*`, etc.):

- `src/utils.py` — variant-library loading, WT sequence, round labels, and the
  combinatorial design-space generator.
- `src/simulations.py` — AL simulation constants (mutation rates, colony counts,
  seeds) and functions.
- `src/ml/` — the two-stage prediction pipeline. `evaluation.py` tunes and validates
  the classifier and regressor stages independently; `two_stage.py` composes the
  models into the explicit cascade — the classifier calls active/inactive and the
  regressor ranks **only** the predicted-active variants (`two_stage_predict`). It also
  auto-selects the top-ranked model per stage from the validation tables (`rank_models`,
  `select_model` — pass a `model_name` to pick a specific one) and fits it with its best
  stored hyperparameters (`fit_stage_model`). `execute_production.ipynb` ends with a cascade
  cell. **Models are not retrained for reporting** — published metrics come from the
  pipeline's stored validation output.

## Methods code for supplementary techniques

The supplementary figures reuse a small set of computational techniques rather than
one script per panel. The algorithm for each lives in `src/` (each module's
docstring lists the figures it underlies):

| Module | Technique | Supplementary figures |
|--------|-----------|-----------------------|
| `src/motif_functions.py` | WT-contact filter cascade, sequence logos | S2–S8, S15, S52 |
| `src/landscape_data.py` + `src/colormap.py` | t-SNE embedding + Nadaraya-Watson activity terrain | S13, S16 |
| `src/structure.py` | ESMFold Cα-distance features; SVD / TM-align superposition | S18, S50, S51 |
| `src/al.py` | GP-UCB and DNN Thompson-sampling acquisition | S32–S45, S54–S67 |
| `src/rankcorr.py` | descending average-tie rank correlation | S14, S19, + AL panels |

These modules document and reproduce the *methods*. Their large inputs — ESMFold
PDB structures and the cached per-round AL predictions — are deposited on Zenodo
(see the manuscript Data Availability statement), not shipped in this repository.
The TM-align path in `src/structure.py` additionally needs the optional `tmtools`
package (`pip install tmtools`); the SVD-superposition and Cα-distance paths use
only biopython / numpy.

## Nomenclature

Code identifiers map to the manuscript terms as follows.

**Workflow & rounds**

| Code | Manuscript |
|------|-----------|
| `LD-BT 0` … `LD-BT 3` (`utils.ROUND_ORDER`) | LD-BT Round 0–3 (iterative Learn-Design / Build-Test) |
| `generate_domain` / `execute_simulation` / `execute_production` | design-space build / library-strategy simulation / model training (Learn) |

**Activity & targets**

| Code | Manuscript |
|------|-----------|
| `RebA Relative Specific Activity to WT`, `RSA_RebA_xWT` | relative specific activity, Reb A (×wild-type) |
| `RSA_RebI_xWT` | relative specific activity, Reb I (×wild-type) |
| `Conversion` (`COL_CONVERSION`) | active/inactive label; `Conversion = 1` = product-forming |
| `WT_SEQ` | wild-type *Sr*UGT76G1 A1 α-helix motif |

**Encodings & residues** (`ml/feature_engineering.py`, `structure.py`)

| Code | Manuscript |
|------|-----------|
| `one_hot` | one-hot encoding |
| `protparams` | physicochemical descriptors (ProtParams) |
| `aaindex` | PCA-reduced AAindex descriptors |
| `esmfold` / `esmpartial_distances` | ESMFold Cα–Cα distance matrix (ESM-Partial) |
| scope `partial` / `full` | partial (catalytic + pocket residues) / full (whole chain) |
| `CATALYTIC` / `POCKET` | catalytically relevant residues / substrate-pocket residues |

**Models** (`ml/models.py`) — paper abbreviations in parentheses

| Classifier key | Regressor key |
|----------------|---------------|
| `LogisticRegression` (LogReg), `SVC`, `RandomForest` (RFC), `XGBoost` (XGBC) | `LinearRegression`, `Ridge`, `Lasso`, `SVR`, `RandomForest` (RFR), `XGBoost` (XGBR) |

**Prediction, acquisition & build**

| Code | Manuscript |
|------|-----------|
| `two_stage_predict` | two-stage prediction (classifier → regressor on predicted-actives) |
| `al.gp_ucb` | GP-UCB acquisition (μ + 2σ) |
| `al.dnn_thompson` | DNN-ensemble Thompson sampling |
| `simulate_eppcr` / `ssm_coupon_expected` | epPCR / SSM library-construction strategies |
| `superpose_rmsd` / `tmalign` | structure superposition (RMSD / TM-align) |

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)).

## License

MIT — see [`LICENSE`](LICENSE).
