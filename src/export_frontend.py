"""Export model results to JSON for the Three.js frontend (web/).

All numbers are computed here with the same code as the dashboard (src/aerohealth.py): XGBoost (seed 42),
calibrated failure risk (H, P_CUT below) and SHAP TreeExplainer. The frontend only displays them.

Usage (from repo root or src/):  python src/export_frontend.py
"""
import json
from pathlib import Path

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


def export(fd):
    if not ah.windows_path(fd).exists():
        ah.prepare(fd)
    d = ah.load_windows(fd)
    sensors = list(d["features"])
    names = ah.feature_names(sensors)
    model = ah.train_xgb(ah.summarize(d["X_tr"]), d["y_tr"])
    F_test = ah.summarize(d["X_test"])
    pred_val, pred = model.predict(ah.summarize(d["X_val"])), model.predict(F_test)

    cal_H = ah.fit_calibration(pred_val, d["y_val"], H)
    cal_2H = ah.fit_calibration(pred_val, d["y_val"], 2 * H)
    p_h, p_2h = ah.prob(cal_H, pred), ah.prob(cal_2H, pred)
    level = ah.risk_levels(p_h, p_2h, P_CUT)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(F_test)
    sensor_of_feature = [n.split("_")[0] for n in names]

    n = len(pred)
    engines_ids = d["test_engines"] if "test_engines" in d.files else np.arange(1, n + 1)
    padded = set(d["padded_engines"].tolist()) if "padded_engines" in d.files else set()

    engines = []
    for i in range(n):
        per_sensor = {s: 0.0 for s in sensors}
        for j, s in enumerate(sensor_of_feature):
            per_sensor[s] += float(sv[i, j])
        per_comp = {c: 0.0 for c in COMPONENTS}
        for s, v in per_sensor.items():
            per_comp[SENSOR_TO_COMP[s]] += v
        top = np.argsort(-np.abs(sv[i]))[:3]
        engines.append({
            "id": int(engines_ids[i]), "pred_rul": round(float(pred[i]), 1),
            "p_h": round(float(p_h[i]), 4), "p_2h": round(float(p_2h[i]), 4), "level": str(level[i]),
            "true_rul": int(d["y_test"][i]), "padded": int(engines_ids[i]) in padded,
            "drivers": [{"feature": names[j], "text": ah.describe_feature(names[j]),
                         "shap": round(float(sv[i, j]), 2)} for j in top],
            "component_shap": {c: round(v, 2) for c, v in per_comp.items()},
            "sensor_shap": {s: round(v, 2) for s, v in per_sensor.items()},
        })

    abs_comp = {c: float(np.mean([abs(e["component_shap"][c]) for e in engines])) for c in COMPONENTS}
    total = sum(abs_comp.values())
    true = d["y_test"] <= H
    hi = level == "HIGH"
    out = {
        "dataset": fd, "fault_mode": ah.FAULT_MODES[fd], "n_regimes": ah.N_REGIMES[fd],
        "H": H, "p_cut": P_CUT, "model": "XGBoost (seed 42)",
        "cutoff_high": round(float(ah.derived_cutoff(cal_H, P_CUT)), 1),
        "cutoff_medium": round(float(ah.derived_cutoff(cal_2H, P_CUT)), 1),
        "test_rmse": round(ah.rmse(d["y_test"], pred), 2),
        "base_value": round(float(explainer.expected_value), 2),
        "counts": {lv: int((level == lv).sum()) for lv in ["HIGH", "MEDIUM", "LOW"]},
        "evaluation": {"fail_within_H": int(true.sum()), "high_caught": int((hi & true).sum()),
                       "high_total": int(hi.sum()), "fail_within_H_rated_low": int((true & (level == "LOW")).sum())},
        "components": {c: {"label": v["label"], "sensors": [s for s in v["sensors"] if s in sensors],
                           "sensor_names": {s: ah.SENSOR_NAMES[s] for s in v["sensors"] if s in sensors},
                           "fleet_share": round(abs_comp[c] / total, 4)} for c, v in COMPONENTS.items()},
        "engines": sorted(engines, key=lambda e: -e["p_h"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{fd}.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{fd}: {n} engines, counts {out['counts']}, fleet component share "
          + ", ".join(f"{c} {v['fleet_share']:.0%}" for c, v in out["components"].items()))


if __name__ == "__main__":
    for fd in ah.DATASETS:
        export(fd)
