# AeroHealth

**Explainable Deep Learning for Aircraft Engine Failure Prediction and Remaining Useful Life (RUL) Estimation**

Predicts how many flight cycles a turbofan engine has left before failure from multivariate sensor time series, converts that into a calibrated failure risk, and explains each prediction with SHAP.

## Dataset
NASA C-MAPSS turbofan engine degradation simulation (FD001 to FD004).
Download: https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data

The raw `.txt` files are included in `data/raw/`. If you download them yourself, place them there.
Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008 (PDF in `data/raw/`). Sensor names used in the dashboard come from its Table 2.

| Subset | Operating conditions | Fault modes | Train / test engines |
|---|---|---|---|
| FD001 | 1 | HPC degradation | 100 / 100 |
| FD002 | 6 | HPC degradation | 260 / 259 |
| FD003 | 1 | HPC + fan degradation | 100 / 100 |
| FD004 | 6 | HPC + fan degradation | 249 / 248 |

## Pipeline
Sensor data → preprocessing → 30-cycle sliding windows → model → RUL → **failure risk** → **SHAP explanation** → dashboard.

Models compared: Random Forest, XGBoost, LSTM, GRU, 1D-CNN.
Metrics: RMSE and the NASA asymmetric scoring function (late predictions penalised more than early ones).

## Project structure
```
aerohealth/
├── data/
│   ├── raw/                  # C-MAPSS files
│   └── processed/            # generated windows (rebuild with src/aerohealth.py, see below)
├── notebooks/
│   ├── 01_eda.ipynb          # EDA + FD001 preprocessing -> fd001_windows.npz
│   ├── 02_baselines.ipynb    # Random Forest, XGBoost (5 seeds)
│   ├── 03_deep_models.ipynb  # LSTM, GRU, 1D-CNN (5 seeds)
│   ├── 04_shap.ipynb         # SHAP explanations for XGBoost (FD001)
│   ├── 05_failure_risk.ipynb # RUL -> calibrated failure probability -> risk level (FD001)
│   └── 06_all_datasets.ipynb # whole pipeline on FD001-FD004
├── src/
│   ├── aerohealth.py         # shared pipeline code: prepare(), features, XGBoost, calibration, risk levels
│   ├── deep.py               # 1D-CNN + training loop (same as 03)
│   ├── run_cnn_all.py        # CNN on all datasets, run as a separate process (see Notes)
│   └── export_frontend.py    # exports results to web/data/*.json for the 3D frontend
├── app/dashboard.py          # Streamlit dashboard
├── web/                      # Three.js 3D frontend (index.html, main.js, data/*.json, vendor/three)
├── results/                  # metric tables (.csv) and plots (.png)
└── requirements.txt          # pinned versions
```

## Setup
Requires Python 3.13. Package versions are pinned in `requirements.txt` so every machine gets identical results (unpinned XGBoost versions gave different test RMSE: 13.72 vs 14.09).

**With conda (macOS / Linux):**
```
git clone https://github.com/agupta30be24-create/aerohealth.git
cd aerohealth
conda create -n aerohealth python=3.13 -y
conda activate aerohealth
pip install -r requirements.txt
```

**With venv on Windows (e.g. VS Code):**
```
git clone https://github.com/agupta30be24-create/aerohealth.git
cd aerohealth
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
(macOS/Linux venv: `source venv/bin/activate`.)

## How to reproduce
Run the notebooks in order from the `notebooks/` folder: 01 → 06. 01 builds `fd001_windows.npz`. 06 builds the FD002-FD004 windows itself if they are missing (they are not tracked because `fd004_windows.npz` is about 111 MB, over GitHub's 100 MB file limit). To build them by hand:
```
cd src
python -c "import aerohealth as ah; [ah.prepare(fd) for fd in ah.DATASETS]"
```
Dashboard (from the repo root):
```
streamlit run app/dashboard.py
```

## Preprocessing
**FD001** (`01_eda`):
- Dropped 6 constant sensors (s1, s5, s10, s16, s18, s19), leaving 15 features
- RUL capped at 125
- Train/validation split by engine (80/20), never by row
- MinMax scaling fitted on training engines only
- Sliding windows of 30 cycles
- Test: one window per engine (its last 30 cycles)

**All datasets** (`src/aerohealth.py: prepare()`): same rules, generalised. For FD001 it reproduces `fd001_windows.npz` **exactly** (checked array by array).
- **Operating regimes:** k-means on the 3 operating settings. For FD002/FD004 the within-cluster error drops to exactly 0 at k=6 (matching the 6 conditions in the dataset readme), and the regime alone explains about 100% of raw sensor variance. So sensors are **MinMax-scaled within each regime**, otherwise the model would learn the flight condition instead of wear. FD001/FD003 have 1 regime.
- **Dropped sensors:** zero spread within every regime, decided per dataset (FD003 keeps s10. FD002/FD004 keep 17 sensors).
- **Short test engines:** 6 (FD002) and 11 (FD004) test engines have fewer than 30 cycles. They are padded by repeating their first cycle so none is dropped, and their error is reported separately.

## Models
- **Random Forest and XGBoost** use 90 summary features per window (last value, mean, std, min, max and slope of each of the 15 sensors), because trees cannot read sequences.
- **LSTM, GRU and 1D-CNN** read the raw 30 x 15 window. They use Adam (lr 1e-3), batch size 256, gradient clipping and early stopping on validation RMSE.
- **Deep models are trained on RUL / 125** and the output is multiplied back. Without this the LSTM got stuck predicting the average (validation RMSE about 41.7) and learned nothing. Do not remove the scaling.
- No hyperparameter tuning has been done. FD001 settings are reused on every dataset.

## Results

### FD001: all models, 5 seeds (42, 1, 2, 3, 4), mean ± std
| Model | Val RMSE | Test RMSE | Test NASA score |
|---|---|---|---|
| **XGBoost** | 12.42 ± 0.06 | **13.89 ± 0.22** | **286 ± 10** |
| Random Forest | 12.82 ± 0.04 | 14.19 ± 0.07 | 325 ± 10 |
| GRU | 11.85 ± 0.26 | 14.81 ± 0.27 | 362 ± 37 |
| 1D-CNN | 11.93 ± 0.08 | 14.85 ± 0.27 | 362 ± 15 |
| LSTM | 12.01 ± 0.17 | 14.98 ± 0.27 | 381 ± 32 |

Per-seed results: `results/tree_runs.csv`, `results/deep_runs.csv`. Test RMSE and NASA score use the true (uncapped) RUL of the 100 test engines. Validation RMSE is on capped labels. Lower is better.

- On the FD001 test set the **tree models beat the deep models for every seed**. The gap (about 1 RMSE) is about 4x the seed-to-seed spread.
- The deep models have the best validation RMSE but the worst test RMSE. Their validation RMSE is the best epoch picked on the validation set, so it is optimistically biased (the tree models are evaluated once). Validation also covers every window of 20 engines, while test is one final window per engine. Compare models on test, not validation.
- Seeds measure training randomness only. With 100 test engines, a different test sample could change the gap, so this is a result **on this test set**, not a general claim that trees are better.

### All datasets: XGBoost (5 seeds) and 1D-CNN (3 seeds), test RMSE mean ± std
| Dataset | XGBoost | 1D-CNN |
|---|---|---|
| FD001 | **13.89 ± 0.22** | 14.76 ± 0.34 |
| FD002 | **26.71 ± 0.17** | 27.63 ± 0.52 |
| FD003 | **14.20 ± 0.19** | 14.83 ± 0.35 |
| FD004 | 28.25 ± 0.15 | **27.88 ± 0.78** |

Details: `notebooks/06_all_datasets.ipynb`, `results/all_datasets_summary.csv`. The multi-condition datasets (FD002/FD004) are much harder with this untuned, same-recipe setup. Published methods do better on them, and no state-of-the-art claim is made. On FD004 the CNN's lead is within its seed std (a tie). Padded short engines have clearly higher error (`06`, Section 1).

## Failure risk (`05_failure_risk`)
Predicted RUL is converted into **P(true RUL ≤ H)** with a logistic regression fitted on the **validation** engines (the test set is never used for fitting). This corrects the model's bias near failure: about two times in three it predicts **too late** there.

| Level | Rule |
|---|---|
| HIGH | P(fail within H) ≥ 0.5 |
| MEDIUM | P(fail within 2H) ≥ 0.5 |
| LOW | otherwise |

The only chosen numbers are **H = 30** and the cut-off **0.5**. The RUL thresholds are derived (FD001: HIGH if predicted RUL ≤ 31.6).
- **Why H = 30:** validation recall/precision stay about 0.9 for every H from 15 to 60 (`05`, Section 3), so 30 is feasible but the data does not single it out. It is chosen because it leaves 25 FD001 test engines to evaluate on (only 10 at H = 15), and because H is an operational parameter (warning time maintenance needs). It is exposed at the top of the notebook and in the dashboard.
- **Why 0.5:** "more likely than not". `05`, Section 5 shows the trade-off (for example, 0.3 raises recall from 0.93 to 0.96 but nearly doubles false alarms).

| Dataset | Truly fail within 30 | Caught by HIGH | HIGH correct | Truly fail within 30 but rated LOW |
|---|---|---|---|---|
| FD001 | 25 | 23 | 23/24 | **0** |
| FD002 | 61 | 61 | 61/68 | **0** |
| FD003 | 20 | 19 | 19/21 | **0** |
| FD004 | 53 | 43 | 43/46 | **0** |

## Explainability (`04_shap`, `06`)
SHAP TreeExplainer on XGBoost (exact for tree models, so base value + SHAP values = prediction for every engine).
- **FD001:** the top sensors are **s11 (Ps30)** and **s3 (T30)**, both measured at the **high-pressure compressor (HPC) outlet**, and FD001's only fault mode is HPC degradation. Trends over the window matter most (slope = 43% of importance, latest value only 1%).
- **Fan-fault check (negative result):** fan-related sensors (Nf, NRf, BPR) do **not** gain importance on FD003/FD004, which add fan degradation. The model relies mainly on HPC sensors everywhere, so we **cannot** claim it detects fan faults specifically.
- SHAP explains what the model uses, not physical causation. Correlated sensors can share importance.

## Dashboard (`app/dashboard.py`)
Dataset selector (FD001-FD004). Fleet overview (HIGH/MEDIUM/LOW counts and an engine table sorted by risk), engine detail (risk, probability, plain-English SHAP reasons, waterfall plot, sensor history), and a method/limitations tab. H and the probability cut-off can be changed live. True RUL is hidden unless "evaluation mode" is on, because it is unknown in real operation.

## 3D frontend (`web/`, Three.js)
A 3D view of the same results:
- **Fleet:** one engine per tile, coloured by risk level and raised by P(fail within 30). Hover for details, click to select.
- **Engine cutaway:** a turbofan (fan → LPC → HPC → combustor → HPT → LPT) where each component is coloured by the **sum of the SHAP values of the sensors located there** (red shortens predicted life, blue extends it). Click a component for its per-sensor breakdown.
- An on-screen **integrity check** shows that fleet average + component contributions = the model's prediction (SHAP values add up exactly).

All numbers are computed in Python by `src/export_frontend.py` (same code as the Streamlit dashboard). The frontend only displays them. Sensor-to-component mapping follows the station names in Saxena et al. (2008), Table 2. It is approximate: core speed (Nc, NRc) and bleed enthalpy are drawn on the HPC, although the core shaft also drives the HPT. Component totals grow with the number of sensors assigned, so compare individual sensors before concluding that one component dominates.

```
python src/export_frontend.py          # writes web/data/FD00x.json (already included)
python -m http.server 8000 -d web      # then open http://localhost:8000
```
Prepared views via URL, e.g. `http://localhost:8000/?fd=FD004&eval=1&engine=10&comp=hpc`. Three.js 0.187.0 is included in `web/vendor/` (MIT licence) so it works offline.

## Notes
- **macOS: do not train XGBoost and PyTorch in the same Python process.** Each bundles its own OpenMP runtime, and the combination segfaults. That is why `06` runs the CNN through `src/run_cnn_all.py` as a separate process (02/03 are already separate).
- FD004 sensor s6 reads slightly below the training range in 2 test engines (39 of 126,480 values). This is a single-sensor quirk, not a regime error.
- Deep models train on CPU (`03` uses CUDA if available, otherwise CPU).

## Status
- [x] Environment setup (pinned versions)
- [x] EDA and preprocessing (FD001)
- [x] Baselines: Random Forest, XGBoost
- [x] Deep models: LSTM, GRU, 1D-CNN
- [x] Multiple seeds (5 seeds, all models, FD001)
- [x] SHAP explainability
- [x] Failure risk (calibrated)
- [x] Streamlit dashboard
- [x] FD002 to FD004 (XGBoost + 1D-CNN)
- [x] 3D frontend (Three.js)

Possible extensions: hyperparameter tuning, LSTM/GRU on FD002-FD004, a bootstrap over test engines for model comparisons.

## Authors
- agupta30be24-create
- chintansood
