"""Build the JSON payload shown by the Three.js frontend (web/).

One function, build_payload(), is used for both:
- the NASA test fleets, exported once to web/data/FD00x.json  (python src/export_frontend.py)
- uploaded engine files, computed on request by app/server.py
so the two can never disagree. All numbers come from src/aerohealth.py (same code as the Streamlit dashboard):
XGBoost (seed 42), calibrated failure risk (H, P_CUT below) and SHAP TreeExplainer. The frontend only displays them.
"""
import json

import numpy as np
import shap

import aerohealth as ah

H, P_CUT = 30, 0.5
OUT = ah.ROOT / "web" / "data"

# Sensor -> engine component, from the station names in Saxena et al. (2008), Table 2.
# Approximation for visualisation: core speeds (Nc, NRc) and bleed enthalpy (htBleed) are assigned to the HPC,
# although the core shaft also drives the HPT.
COMPONENTS = {
    "fan": {"label": "Fan / bypass", "sensors": ["s8", "s13", "s15", "s6", "s1", "s5", "s18", "s19"]},
    "lpc": {"label": "Low-pressure compressor (LPC)", "sensors": ["s2"]},
    "hpc": {"label": "High-pressure compressor (HPC)", "sensors": ["s3", "s7", "s11", "s9", "s14", "s17"]},
    "combustor": {"label": "Combustor", "sensors": ["s12", "s16"]},
    "hpt": {"label": "High-pressure turbine (HPT)", "sensors": ["s20"]},
    "lpt": {"label": "Low-pressure turbine (LPT) / exhaust", "sensors": ["s4", "s21", "s10"]},
}
SENSOR_TO_COMP = {s: c for c, v in COMPONENTS.items() for s in v["sensors"]}


class TrainedModel:
    """Everything learned from the TRAINING engines of one dataset (nothing here sees test or uploaded data)."""

    def __init__(self, fd):
        if not ah.windows_path(fd).exists():
            ah.prepare(fd)
        d = ah.load_windows(fd)
        self.fd, self.sensors = fd, list(d["features"])
        self.names = ah.feature_names(self.sensors)
        self.model = ah.train_xgb(ah.summarize(d["X_tr"]), d["y_tr"])
        self.explainer = shap.TreeExplainer(self.model)
        self.pred_val, self.y_val = self.model.predict(ah.summarize(d["X_val"])), d["y_val"]
        self.cal_H = ah.fit_calibration(self.pred_val, self.y_val, H)
        self.cal_2H = ah.fit_calibration(self.pred_val, self.y_val, 2 * H)
        self.windows = d
        self._pp = None

    @property
    def pp(self):
        if self._pp is None:
            self._pp = ah.Preprocessor(self.fd)
        return self._pp


def build_payload(tm, X, engine_ids, padded=(), y_true=None, checks=None, source="benchmark", extra=None):
    """Windows -> predictions, risk, SHAP per engine and component -> JSON-ready dict."""
    fd = tm.fd
    F = ah.summarize(X)
    pred = tm.model.predict(F)
    p_h, p_2h = ah.prob(tm.cal_H, pred), ah.prob(tm.cal_2H, pred)
    level = ah.risk_levels(p_h, p_2h, P_CUT)
    sv = tm.explainer.shap_values(F)
    sensor_of_feature = [n.split("_")[0] for n in tm.names]
    padded = set(np.asarray(padded).tolist())
    check_rows = {int(r.engine): r for r in checks.itertuples()} if checks is not None else {}

    engines = []
    for i in range(len(pred)):
        per_sensor = {s: 0.0 for s in tm.sensors}
        for j, s in enumerate(sensor_of_feature):
            per_sensor[s] += float(sv[i, j])
        per_comp = {c: 0.0 for c in COMPONENTS}
        for s, v in per_sensor.items():
            per_comp[SENSOR_TO_COMP[s]] += v
        top = np.argsort(-np.abs(sv[i]))[:3]
        eid = int(engine_ids[i])
        e = {"id": eid, "pred_rul": round(float(pred[i]), 1),
             "p_h": round(float(p_h[i]), 4), "p_2h": round(float(p_2h[i]), 4), "level": str(level[i]),
             "true_rul": int(y_true[i]) if y_true is not None else None, "padded": eid in padded,
             "drivers": [{"feature": tm.names[j], "text": ah.describe_feature(tm.names[j]),
                          "shap": round(float(sv[i, j]), 2)} for j in top],
             "component_shap": {c: round(v, 2) for c, v in per_comp.items()},
             "sensor_shap": {s: round(v, 2) for s, v in per_sensor.items()}}
        if eid in check_rows:
            c = check_rows[eid]
            e["cycles"] = int(c.cycles)
            e["unknown_condition_share"] = round(float(c.unknown_condition_share), 3)
            e["out_of_range_sensors"] = list(c.out_of_range_sensors)
        engines.append(e)

    abs_comp = {c: float(np.mean([abs(e["component_shap"][c]) for e in engines])) for c in COMPONENTS}
    total = sum(abs_comp.values()) or 1.0
    out = {
        "dataset": fd, "fault_mode": ah.FAULT_MODES[fd], "n_regimes": ah.N_REGIMES[fd],
        "H": H, "p_cut": P_CUT, "model": "XGBoost (seed 42)",
        "cutoff_high": round(float(ah.derived_cutoff(tm.cal_H, P_CUT)), 1),
        "cutoff_medium": round(float(ah.derived_cutoff(tm.cal_2H, P_CUT)), 1),
        "base_value": round(float(tm.explainer.expected_value), 2),
        "counts": {lv: int((level == lv).sum()) for lv in ["HIGH", "MEDIUM", "LOW"]},
        "components": {c: {"label": v["label"], "sensors": [s for s in v["sensors"] if s in tm.sensors],
                           "sensor_names": {s: ah.SENSOR_NAMES[s] for s in v["sensors"] if s in tm.sensors},
                           "fleet_share": round(abs_comp[c] / total, 4)} for c, v in COMPONENTS.items()},
        "engines": sorted(engines, key=lambda e: -e["p_h"]),
    }
    if y_true is not None:
        true, hi = y_true <= H, level == "HIGH"
        out["test_rmse"] = round(ah.rmse(y_true, pred), 2)
        out["evaluation"] = {"fail_within_H": int(true.sum()), "high_caught": int((hi & true).sum()),
                             "high_total": int(hi.sum()), "fail_within_H_rated_low": int((true & (level == "LOW")).sum())}
    if source != "benchmark":
        out["source"] = source
    if extra:
        out.update(extra)
    return out


def benchmark_payload(tm):
    d = tm.windows
    n = len(d["y_test"])
    ids = d["test_engines"] if "test_engines" in d.files else np.arange(1, n + 1)
    padded = d["padded_engines"] if "padded_engines" in d.files else []
    return build_payload(tm, d["X_test"], ids, padded, y_true=d["y_test"])


def upload_payload(tm, df, notes=(), suggestion=None):
    X, ids, padded, checks = tm.pp.apply(df)
    extra = {"notes": list(notes), "n_rows": int(len(df))}
    if suggestion:
        extra["suggested_model"], extra["suggestion_reason"] = suggestion
    return build_payload(tm, X, ids, padded, checks=checks, source="upload", extra=extra)


def export(fd):
    out = benchmark_payload(TrainedModel(fd))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{fd}.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{fd}: {len(out['engines'])} engines, counts {out['counts']}")


if __name__ == "__main__":
    for fd in ah.DATASETS:
        export(fd)
