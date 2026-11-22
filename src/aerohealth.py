"""Shared AeroHealth pipeline code (FD001): data -> features -> XGBoost RUL -> failure risk.

Mirrors notebooks 02_baselines (features, XGBoost settings) and 05_failure_risk (calibration, risk levels),
so the dashboard and the notebooks produce the same numbers.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
WINDOWS = ROOT / "data" / "processed" / "fd001_windows.npz"

RAW_COLS = ["engine", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]

# Saxena et al. (2008), Table 2, in column order s1..s21
SENSOR_NAMES = {
    "s1": "T2 fan inlet temperature", "s2": "T24 LPC outlet temperature", "s3": "T30 HPC outlet temperature",
    "s4": "T50 LPT outlet temperature", "s5": "P2 fan inlet pressure", "s6": "P15 bypass-duct pressure",
    "s7": "P30 HPC outlet pressure", "s8": "Nf physical fan speed", "s9": "Nc physical core speed",
    "s10": "epr engine pressure ratio", "s11": "Ps30 HPC outlet static pressure", "s12": "phi fuel flow / Ps30",
    "s13": "NRf corrected fan speed", "s14": "NRc corrected core speed", "s15": "BPR bypass ratio",
    "s16": "farB burner fuel-air ratio", "s17": "htBleed bleed enthalpy", "s18": "Nf_dmd demanded fan speed",
    "s19": "PCNfR_dmd demanded corrected fan speed", "s20": "W31 HPT coolant bleed", "s21": "W32 LPT coolant bleed",
}

STATS = ["last", "mean", "std", "min", "max", "slope"]
STAT_TEXT = {"last": "latest value", "mean": "average over last 30 cycles", "std": "variability over last 30 cycles",
             "min": "minimum over last 30 cycles", "max": "maximum over last 30 cycles",
             "slope": "trend over last 30 cycles"}


DATASETS = ["FD001", "FD002", "FD003", "FD004"]
# From data/raw/readme.txt; confirmed by k-means on op settings (inertia reaches 0 at k=6 for FD002/FD004)
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
FAULT_MODES = {"FD001": "HPC degradation", "FD002": "HPC degradation",
               "FD003": "HPC + fan degradation", "FD004": "HPC + fan degradation"}
W, CAP = 30, 125


def windows_path(fd="FD001"):
    return ROOT / "data" / "processed" / f"{fd.lower()}_windows.npz"


def load_windows(fd="FD001"):
    return np.load(windows_path(fd))


def load_raw(split="test", fd="FD001"):
    return pd.read_csv(DATA_RAW / f"{split}_{fd}.txt", sep=r"\s+", header=None, names=RAW_COLS)


def prepare(fd="FD001", save=True):
    """Raw C-MAPSS -> windows. Generalises 01_eda (identical output for FD001).

    1. Regimes: k-means on the 3 op settings (1 regime for FD001/FD003, 6 for FD002/FD004), fit on training rows.
    2. Drop sensors with zero spread inside every regime (same std < 1e-6 rule as 01_eda, applied per regime).
    3. RUL capped at 125. Train/val split by engine (80/20, random_state=42).
    4. MinMax scaling per regime, fitted on training engines only.
    5. 30-cycle windows. Test = last 30 cycles per engine; engines shorter than 30 are padded by repeating
       their first cycle (padded engine ids are saved so their errors can be reported separately).
    """
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    train, test = load_raw("train", fd), load_raw("test", fd)
    rul_test = pd.read_csv(DATA_RAW / f"RUL_{fd}.txt", header=None, names=["RUL"])["RUL"].values
    train["RUL"] = train.groupby("engine")["cycle"].transform("max") - train["cycle"]
    train["RUL_cap"] = train["RUL"].clip(upper=CAP)

    ops = ["op1", "op2", "op3"]
    k = N_REGIMES[fd]
    if k == 1:
        train["regime"], test["regime"] = 0, 0
    else:
        mu, sd = train[ops].mean(), train[ops].std()
        km = KMeans(k, n_init=10, random_state=0).fit((train[ops] - mu) / sd)
        train["regime"] = km.predict((train[ops] - mu) / sd)
        test["regime"] = km.predict((test[ops] - mu) / sd)

    sensors = [f"s{i}" for i in range(1, 22)]
    within_std = train.groupby("regime")[sensors].std().max()
    features = [s for s in sensors if within_std[s] >= 1e-6]

    engines = train["engine"].unique()
    tr_eng, val_eng = train_test_split(engines, test_size=0.2, random_state=42)

    train_s, test_s = train.copy(), test.copy()
    train_s[features] = train_s[features].astype(float)
    test_s[features] = test_s[features].astype(float)
    for r in range(k):
        scaler = MinMaxScaler().fit(train.loc[train.engine.isin(tr_eng) & (train.regime == r), features])
        for df_s, df in [(train_s, train), (test_s, test)]:
            m = df.regime == r
            if m.any():
                df_s.loc[m, features] = scaler.transform(df.loc[m, features])

    def make_windows(df):
        X, y = [], []
        for _, g in df.groupby("engine"):
            arr, rul = g[features].values, g["RUL_cap"].values
            for i in range(len(g) - W + 1):
                X.append(arr[i:i + W])
                y.append(rul[i + W - 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_tr, y_tr = make_windows(train_s[train_s.engine.isin(tr_eng)])
    X_val, y_val = make_windows(train_s[train_s.engine.isin(val_eng)])

    X_test, padded = [], []
    for eng, g in test_s.groupby("engine"):
        arr = g[features].values
        if len(arr) < W:
            arr = np.vstack([np.repeat(arr[:1], W - len(arr), axis=0), arr])
            padded.append(eng)
        X_test.append(arr[-W:])
    out = dict(X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
               X_test=np.array(X_test, dtype=np.float32), y_test=rul_test.astype(np.float32),
               features=np.array(features), test_engines=np.array(sorted(test.engine.unique())),
               padded_engines=np.array(padded, dtype=int))
    if save:
        np.savez(windows_path(fd), **out)
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def nasa_score(y, p):
    """NASA asymmetric score (sum over engines). Late predictions (p > y) are penalised more."""
    e = p - y
    return float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))


def summarize(X):
    """30x15 window -> 90 summary features (6 statistics x 15 sensors), same as 02_baselines."""
    t = np.arange(X.shape[1])
    slope = ((X - X.mean(1, keepdims=True)) * (t - t.mean())[None, :, None]).sum(1) / ((t - t.mean()) ** 2).sum()
    return np.concatenate([X[:, -1], X.mean(1), X.std(1), X.min(1), X.max(1), slope], axis=1)


def feature_names(sensors):
    return [f"{s}_{st}" for st in STATS for s in sensors]


def describe_feature(name):
    sensor, stat = name.split("_")
    return f"{sensor} ({SENSOR_NAMES[sensor]}), {STAT_TEXT[stat]}"


def train_xgb(F_tr, y_tr, seed=42):
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=-1, random_state=seed).fit(F_tr, y_tr)


def fit_calibration(pred_val, y_val, horizon):
    """Logistic map: predicted RUL -> P(true RUL <= horizon). Fit on validation engines only."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(pred_val.reshape(-1, 1), (y_val <= horizon).astype(int))


def prob(cal, pred):
    return cal.predict_proba(np.asarray(pred).reshape(-1, 1))[:, 1]


def derived_cutoff(cal, p_cut):
    """Predicted RUL at which the calibrated probability equals p_cut."""
    a, b = cal.intercept_[0], cal.coef_[0, 0]
    return (np.log(p_cut / (1 - p_cut)) - a) / b


def risk_levels(p_h, p_2h, p_cut):
    return np.where(p_h >= p_cut, "HIGH", np.where(p_2h >= p_cut, "MEDIUM", "LOW"))
