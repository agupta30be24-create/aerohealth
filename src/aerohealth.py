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


OPS = ["op1", "op2", "op3"]
# Common scale for operating-setting distances: std of op1..op3 over FD002 training rows (all 6 flight conditions).
# Measured: training noise distance <= 0.005, smallest gap between two flight conditions = 0.475.
OP_SCALE = np.array([14.747, 0.310, 14.238])
UNKNOWN_CONDITION_DIST = 0.1     # 20x the training noise, about 5x below the smallest gap between conditions
OUT_OF_RANGE_MARGIN = 0.1        # scaled value < -0.1 or > 1.1 = more than 10% of the training range outside it


class Preprocessor:
    """Learns every transform from the TRAINING engines of one dataset, then applies it to any data.

    1. Regimes: k-means on the 3 op settings (1 regime for FD001/FD003, 6 for FD002/FD004), fit on training rows.
    2. Drop sensors with zero spread inside every regime (same std < 1e-6 rule as 01_eda, applied per regime).
    3. Train/val split by engine (80/20, random_state=42).
    4. MinMax scaling per regime, fitted on training engines only.
    New data (test files, uploads) is only transformed, never used to fit anything.
    """

    def __init__(self, fd):
        from sklearn.cluster import KMeans
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        self.fd, self.k = fd, N_REGIMES[fd]
        train = load_raw("train", fd)
        train["RUL"] = train.groupby("engine")["cycle"].transform("max") - train["cycle"]
        train["RUL_cap"] = train["RUL"].clip(upper=CAP)

        if self.k == 1:
            self.km = None
            self.centres = (train[OPS].mean().values / OP_SCALE)[None, :]
        else:
            self.mu, self.sd = train[OPS].mean(), train[OPS].std()
            self.km = KMeans(self.k, n_init=10, random_state=0).fit((train[OPS] - self.mu) / self.sd)
            self.centres = (self.km.cluster_centers_ * self.sd.values + self.mu.values) / OP_SCALE
        train["regime"] = self.regime(train)

        sensors = [f"s{i}" for i in range(1, 22)]
        within_std = train.groupby("regime")[sensors].std().max()
        self.features = [s for s in sensors if within_std[s] >= 1e-6]

        engines = train["engine"].unique()
        self.tr_eng, self.val_eng = train_test_split(engines, test_size=0.2, random_state=42)
        self.scalers = {r: MinMaxScaler().fit(train.loc[train.engine.isin(self.tr_eng) & (train.regime == r), self.features])
                        for r in range(self.k)}
        self.train = train

    def regime(self, df):
        if self.k == 1:
            return np.zeros(len(df), dtype=int)
        return self.km.predict((df[OPS] - self.mu) / self.sd)

    def condition_distance(self, df):
        """Distance of each row's operating settings to the nearest flight condition seen in training."""
        Z = df[OPS].values / OP_SCALE
        return np.min(np.linalg.norm(Z[:, None, :] - self.centres[None], axis=2), axis=1)

    def scale(self, df):
        out = df.copy()
        out["regime"] = self.regime(df)
        out[self.features] = out[self.features].astype(float)
        for r, scaler in self.scalers.items():
            m = out["regime"] == r
            if m.any():
                out.loc[m, self.features] = scaler.transform(df.loc[m, self.features])
        return out

    def last_windows(self, scaled):
        """One window per engine: its last W cycles. Engines shorter than W are padded by repeating the first cycle."""
        X, engines, padded = [], [], []
        for eng, g in scaled.groupby("engine"):
            arr = g[self.features].values
            if len(arr) < W:
                arr = np.vstack([np.repeat(arr[:1], W - len(arr), axis=0), arr])
                padded.append(eng)
            X.append(arr[-W:])
            engines.append(eng)
        return np.array(X, dtype=np.float32), np.array(engines), np.array(padded, dtype=int)

    def apply(self, df):
        """New engine data -> model windows plus data-quality checks per engine."""
        scaled = self.scale(df)
        X, engines, padded = self.last_windows(scaled)
        dist = pd.Series(self.condition_distance(df), index=df.index)
        checks = []
        for i, (eng, g) in enumerate(df.groupby("engine")):
            win = X[i]
            outside = (win < -OUT_OF_RANGE_MARGIN) | (win > 1 + OUT_OF_RANGE_MARGIN)
            checks.append({"engine": eng, "cycles": len(g),
                           "unknown_condition_share": float((dist[g.index] > UNKNOWN_CONDITION_DIST).mean()),
                           "out_of_range_sensors": [self.features[j] for j in np.where(outside.any(0))[0]]})
        return X, engines, padded, pd.DataFrame(checks)


def prepare(fd="FD001", save=True):
    """Raw C-MAPSS -> windows via Preprocessor. Identical output to 01_eda for FD001.

    RUL capped at 125; 30-cycle sliding windows for train/val; test = last 30 cycles per engine
    (short engines padded, their ids saved so their errors can be reported separately).
    """
    pp = Preprocessor(fd)
    test = load_raw("test", fd)
    rul_test = pd.read_csv(DATA_RAW / f"RUL_{fd}.txt", header=None, names=["RUL"])["RUL"].values
    train_s = pp.scale(pp.train)

    def make_windows(df):
        X, y = [], []
        for _, g in df.groupby("engine"):
            arr, rul = g[pp.features].values, g["RUL_cap"].values
            for i in range(len(g) - W + 1):
                X.append(arr[i:i + W])
                y.append(rul[i + W - 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_tr, y_tr = make_windows(train_s[train_s.engine.isin(pp.tr_eng)])
    X_val, y_val = make_windows(train_s[train_s.engine.isin(pp.val_eng)])
    X_test, engines, padded = pp.last_windows(pp.scale(test))
    out = dict(X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, X_test=X_test, y_test=rul_test.astype(np.float32),
               features=np.array(pp.features), test_engines=engines, padded_engines=padded)
    if save:
        np.savez(windows_path(fd), **out)
    return out


# ---------------------------------------------------------------- user uploads
class UploadError(ValueError):
    """Raised with a user-readable message when an uploaded file cannot be used."""


ALIASES = {"unit": "engine", "unit_number": "engine", "id": "engine", "time": "cycle", "cycles": "cycle",
           "time_in_cycles": "cycle"}


def read_engine_file(data):
    """Parse an uploaded engine file (bytes or str) in C-MAPSS format.

    Accepts NASA's space-separated .txt (26 columns, no header) or CSV/TSV with or without a header
    (header names: engine, cycle, op1..op3, s1..s21). Returns (DataFrame, list of notes).
    """
    import io
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise UploadError("The file is empty.")
    import re
    known = set(RAW_COLS) | set(ALIASES)
    has_header = any(tok.lower() in known for tok in re.split(r"[\s,;]+", lines[0].strip()))
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"[\s,;]+", engine="python",
                     header=0 if has_header else None)
    df = df.dropna(axis=1, how="all")
    if has_header:
        df.columns = [ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            raise UploadError(f"Missing columns: {', '.join(missing)}. Expected: {', '.join(RAW_COLS)}.")
        df = df[RAW_COLS]
    else:
        if df.shape[1] != len(RAW_COLS):
            raise UploadError(f"Expected {len(RAW_COLS)} columns (engine, cycle, 3 operating settings, 21 sensors), "
                              f"found {df.shape[1]}.")
        df.columns = RAW_COLS
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise UploadError(f"Non-numeric or missing values in column(s): {', '.join(bad)}.")
    df[["engine", "cycle"]] = df[["engine", "cycle"]].astype(int)
    if df.duplicated(["engine", "cycle"]).any():
        raise UploadError("Some (engine, cycle) pairs appear more than once.")
    notes = []
    if not df.equals(df.sort_values(["engine", "cycle"])):
        notes.append("Rows were not ordered by engine and cycle; they have been sorted.")
    df = df.sort_values(["engine", "cycle"]).reset_index(drop=True)
    gaps = df.groupby("engine")["cycle"].apply(lambda c: bool((c.diff().dropna() != 1).any()))
    if gaps.any():
        notes.append(f"Missing cycles inside engine(s) {', '.join(map(str, gaps[gaps].index[:10]))}: "
                     "the last 30 recorded rows are used as-is.")
    return df, notes


def suggest_model(df):
    """Suggest a trained model from the operating settings of uploaded data.

    Flight conditions can be detected from the data; the fault type cannot. When it is unknown, suggest the model
    trained on both fault types (FD003 for one condition, FD004 for several).
    """
    single = Preprocessor.__new__(Preprocessor)       # only the sea-level centre is needed (FD001/FD003)
    single.centres = np.array([[0.0, 0.0, 100.0]]) / OP_SCALE
    share_other = float((Preprocessor.condition_distance(single, df) > UNKNOWN_CONDITION_DIST).mean())
    if share_other == 0:
        return "FD003", "All rows are at the single sea-level condition, so a one-condition model fits (FD003 covers both fault types)."
    return "FD004", (f"{share_other:.0%} of rows are at other flight conditions, so a six-condition model is needed "
                     "(FD004 covers both fault types).")


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
