# Data

## `SrUGT76G1_variant_library.xlsx`

Curated variant library for the *Sr*UGT76G1 active-learning campaign: 174 rows
(173 variants + wild type), one row per variant, with standardized column names
matching the paper.

| Column | Description |
|--------|-------------|
| `Variant_ID` | Published variant ID, `76G1-A1-{PF\|FR\|NC\|SYN}-NNNN` (WT = `WT (SrUGT76G1)`) |
| `Provenance` | Design origin: `PF` position-filtered pool, `FR` filter relaxation, `NC` non-CAZy, `SYN` synthetic, `WT` wild type |
| `Round` | Active-learning round: `LD-BT 0` … `LD-BT 3` (+ `Additions After Filter Relaxation`, `Wild type`) |
| `Motif_Sequence` | 22-residue A1 variable-region sequence (WT = `PVPFQGHINPILQLANVLYSKG`) |
| `Full_Sequence` | Full-length protein sequence |
| `N_Mutations` | Number of substitutions from WT |
| `Mutated_Positions` | Substituted positions |
| `RSA_RebA_xWT` | RebA relative specific activity, ×WT (assay floor ≈ 0.0328 ×WT) |
| `Active_RebA` | RebA active (1) / inactive (0), measured |
| `RSA_RebI_xWT` | RebI relative specific activity, ×WT |
| `Active_RebI` | RebI active (1) / inactive (0), measured |
| `Reg_Predictions` | SVR (ProtParams) predicted RSA |
| `Class_Predictions` | XGBoost (ESMPartial) predicted active/inactive |
| `Type` | `Variant` or `Wild Type` |

## `SrUGT76G1_variant_id_crosswalk.tsv`

Per-variant index for traceability: `Variant_ID`, `Provenance`, `Round`,
`Motif_Sequence`. Published IDs were assigned by joining each variant's A1 motif
to the paper's per-round supplementary crosswalks (Tables S13/S16/S19/S24/S25).
Templates and per-round detail live in those supplementary tables.

## `supplementary/`

Auxiliary design + prediction tables for the constrained-mutation (synthetic)
candidate pools, in the paper's formatted, standardized-column form.

### `A1_LDBT3_synthetic_designs.xlsx`

*Sr*UGT76G1 A1 LD-BT 3 synthetic designs with two-stage model predictions
(944 unique sequences; 1,000 designed). Source: supplementary Table S25.

| Column | Description |
|--------|-------------|
| `Synthetic_ID` | Published synthetic-design ID, `76G1-A1-SYN-NNN` |
| `N_Design_Variants` | Number of designs collapsed into this unique sequence |
| `Candidate_Motif` | 22-residue A1 variable-region sequence |
| `Template_DBTL_ID` / `Template_Round` / `Template_Motif` / `Template_Full_Sequence` | Parent variant the design was derived from |
| `Num_Mutations` / `Template_Mutations` | Substitutions from the template |
| `Stage1_Classifier_Positive` | Two-stage stage-1 classifier call (predicted active) |
| `Stage2_RebA_RSA_Prediction` | Two-stage stage-2 regressor predicted RebA RSA (×WT) |
| `Screened_LDBT3` / `Selection_Outcome` | Whether the design was built/screened, and its selection result |
| `Full_Sequence` | Full-length protein sequence |

`UGTSL2_LDBT1_designs_with_predictions.xlsx` is the UGTSL2 LD-BT 1 analog; its
formatted replacement is pending supplementary-table number assignment.

## Provenance & integrity

Built from the curated DBTL variant library by standardizing column names and
attaching published IDs. All measured values (RSA, active/inactive calls,
sequences, predictions) are copied verbatim; no value was altered, imputed, or
reordered.

## Cycle 1.4 filtering policy

Figures restrict LD-BT 3 to `Class_Predictions == 1`. Variants with
`Class_Predictions != 1` were screened under a different sampling regime and are
excluded from downstream figures.
