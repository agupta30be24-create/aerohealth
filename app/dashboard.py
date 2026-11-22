"""AeroHealth dashboard.  Run from the repo root:  streamlit run app/dashboard.py

Two modes:
- NASA test fleet (benchmark): FD001-FD004 test engines, true RUL available for evaluation.
- Upload your own engine data: C-MAPSS-format sensor file -> same trained model, risk and SHAP explanations.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import aerohealth as ah  # noqa: E402

st.set_page_config(page_title="AeroHealth", layout="wide")
LEVEL_COLOUR = {"HIGH": "#f8d7da", "MEDIUM": "#fff3cd", "LOW": "#d1e7dd"}
EXAMPLES = {"5 engines, 1 flight condition (from NASA FD001 test)": ROOT / "examples" / "sample_engines_1condition.csv",
            "5 engines, 6 flight conditions (from NASA FD004 test)": ROOT / "examples" / "sample_engines_6conditions.csv"}


# ---------------------------------------------------------------- trained model per dataset (cached)
@st.cache_resource(show_spinner="Training the model on this dataset's training engines (first time only)...")
def trained_bundle(fd):
    """Everything learned from the TRAINING engines of one dataset. Nothing here sees test or uploaded data."""
    if not ah.windows_path(fd).exists():
        ah.prepare(fd)
    d = ah.load_windows(fd)
    sensors = list(d["features"])
    model = ah.train_xgb(ah.summarize(d["X_tr"]), d["y_tr"])
    return {"fd": fd, "pp": ah.Preprocessor(fd), "model": model, "explainer": shap.TreeExplainer(model),
            "sensors": sensors, "names": ah.feature_names(sensors),
            "pred_val": model.predict(ah.summarize(d["X_val"])), "y_val": d["y_val"],
            "bench": {"X": d["X_test"], "y": d["y_test"],
                      "engines": d["test_engines"] if "test_engines" in d.files else np.arange(1, len(d["y_test"]) + 1),
                      "padded": d["padded_engines"] if "padded_engines" in d.files else np.array([], dtype=int)}}


def predict_and_explain(bundle, X, engines, padded, raw, y_true=None, checks=None):
    """Windows -> predicted RUL + SHAP. Shared by benchmark and upload modes so they cannot drift apart."""
    F = ah.summarize(X)
    sv = bundle["explainer"].shap_values(F)
    base = float(bundle["explainer"].expected_value)
    share = np.abs(sv).mean(0).reshape(len(ah.STATS), len(bundle["sensors"])).sum(0)
    return {"X": X, "engines": np.asarray(engines), "padded": set(np.asarray(padded).tolist()), "raw": raw,
            "y_true": y_true, "checks": checks, "pred": bundle["model"].predict(F), "shap": sv,
            "expl": shap.Explanation(values=sv, base_values=np.full(len(sv), base), data=F, feature_names=bundle["names"]),
            "sensor_share": pd.Series(share / share.sum(), index=bundle["sensors"]).sort_values(ascending=False)}


@st.cache_resource(show_spinner=False)
def benchmark_fleet(fd):
    b = trained_bundle(fd)
    return predict_and_explain(b, b["bench"]["X"], b["bench"]["engines"], b["bench"]["padded"],
                               ah.load_raw("test", fd), y_true=b["bench"]["y"])


# ---------------------------------------------------------------- sidebar: data source
st.sidebar.header("Data")
mode = st.sidebar.radio("Source", ["NASA test fleet (benchmark)", "Upload your own engine data"], index=0)
uploading = mode.startswith("Upload")

if not uploading:
    FD = st.sidebar.selectbox("C-MAPSS subset", ah.DATASETS, index=0,
                              format_func=lambda f: f"{f}: {ah.N_REGIMES[f]} condition{'s' if ah.N_REGIMES[f] > 1 else ''}, "
                                                    f"{ah.FAULT_MODES[f]}")
    bundle = trained_bundle(FD)
    V = benchmark_fleet(FD)
    source_txt = f"NASA C-MAPSS {FD} test fleet"
else:
    up = st.sidebar.file_uploader("Engine sensor file (.txt or .csv)", type=["txt", "csv", "tsv"],
                                  help="One row per engine per cycle: engine, cycle, op1-op3, s1-s21 (C-MAPSS format). "
                                       "NASA's space-separated .txt or CSV with/without header.")
    example = st.sidebar.selectbox("...or try an example", ["(none)"] + list(EXAMPLES), index=0 if up else 1,
                                   disabled=up is not None, help="Real engines from NASA's test sets, in the expected format.")
    if example != "(none)" and up is None:
        st.sidebar.download_button("Download this example", EXAMPLES[example].read_bytes(),
                                   file_name=EXAMPLES[example].name, mime="text/csv")
    st.title("AeroHealth: analyse your own engines")
    if up is None and example == "(none)":
        st.info("Upload a sensor file in the sidebar, or pick an example.\n\n"
                "**Expected format** (one row per engine per cycle, 26 numeric columns):  \n"
                "`engine, cycle, op1, op2, op3, s1, ..., s21`. NASA's `train_FD00x.txt` / `test_FD00x.txt` work as-is.")
        st.stop()
    try:
        df_up, notes = ah.read_engine_file(up.getvalue() if up is not None else EXAMPLES[example].read_bytes())
    except ah.UploadError as e:
        st.error(f"This file cannot be used: {e}")
        st.stop()
    for n in notes:
        st.sidebar.info(n)
    suggested, why = ah.suggest_model(df_up)
    FD = st.sidebar.selectbox("Trained model to use", ah.DATASETS, index=ah.DATASETS.index(suggested),
                              format_func=lambda f: f"{f} model ({ah.N_REGIMES[f]} condition{'s' if ah.N_REGIMES[f] > 1 else ''}, "
                                                    f"{ah.FAULT_MODES[f]})" + ("  [suggested]" if f == suggested else ""))
    st.sidebar.caption(f"Suggested: {suggested}. {why}")
    bundle = trained_bundle(FD)
    X_up, eng_up, pad_up, checks = bundle["pp"].apply(df_up)
    V = predict_and_explain(bundle, X_up, eng_up, pad_up, df_up, checks=checks)
    source_txt = f"Uploaded file ({len(eng_up)} engine{'s' if len(eng_up) != 1 else ''}, {len(df_up)} rows), {FD} model"

n_eng = len(V["pred"])
multi_regime = ah.N_REGIMES[FD] > 1
has_truth = V["y_true"] is not None

# ---------------------------------------------------------------- sidebar: risk settings
st.sidebar.header("Risk settings")
H = st.sidebar.selectbox("Warning horizon H (cycles)", [15, 20, 30, 45, 60], index=2,
                         help="'Will this engine fail within H cycles?' An operational choice: how much warning "
                              "maintenance needs. Notebook 05, Section 3 compares these values.")
P_CUT = st.sidebar.slider("Probability cut-off", 0.10, 0.90, 0.50, 0.05,
                          help="A level triggers when its failure probability reaches this value. 0.5 = more likely "
                               "than not. Lower = fewer missed failures but more false alarms (notebook 05, Section 5).")
eval_mode = st.sidebar.checkbox("Evaluation mode (show true RUL)", value=False, disabled=not has_truth,
                                help="True RUL is unknown in real operation. Only available for the benchmark fleet."
                                if has_truth else "Not available for uploaded data: its true RUL is unknown.")

cal_H = ah.fit_calibration(bundle["pred_val"], bundle["y_val"], H)
cal_2H = ah.fit_calibration(bundle["pred_val"], bundle["y_val"], 2 * H)
P_H, P_2H = ah.prob(cal_H, V["pred"]), ah.prob(cal_2H, V["pred"])
level = ah.risk_levels(P_H, P_2H, P_CUT)

st.sidebar.markdown(
    f"**Derived from calibration (not chosen):**  \n"
    f"HIGH if predicted RUL ≤ **{ah.derived_cutoff(cal_H, P_CUT):.1f}**  \n"
    f"MEDIUM if predicted RUL ≤ **{ah.derived_cutoff(cal_2H, P_CUT):.1f}**")
st.sidebar.caption("HIGH: P(fail within H) ≥ cut-off. MEDIUM: P(fail within 2H) ≥ cut-off. "
                   f"Calibration fitted on {FD} validation engines only.")


# ---------------------------------------------------------------- data-quality notes per engine
def engine_notes(k):
    notes = []
    e = V["engines"][k]
    if e in V["padded"]:
        notes.append("short history (padded)")
    if V["checks"] is not None:
        c = V["checks"].iloc[k]
        if c.unknown_condition_share > 0:
            notes.append(f"unknown flight condition ({c.unknown_condition_share:.0%} of rows)")
        if c.out_of_range_sensors:
            notes.append(f"outside training range: {', '.join(c.out_of_range_sensors)}")
    return notes


NOTES = [engine_notes(k) for k in range(n_eng)]

# ---------------------------------------------------------------- header
if not uploading:
    st.title("AeroHealth: engine failure risk")
cap = (f"{source_txt}. Model trained on {FD} training engines ({ah.N_REGIMES[FD]} operating condition(s), "
       f"fault mode: {ah.FAULT_MODES[FD]}): XGBoost on {len(bundle['names'])} window features.")
if has_truth:
    cap += f" Test RMSE {ah.rmse(V['y_true'], V['pred']):.2f} cycles."
st.caption(cap + " Explanations: SHAP (TreeExplainer).")

if uploading:
    flagged = sum(1 for n in NOTES if any(x.startswith("unknown") for x in n))
    if flagged:
        st.error(f"{flagged} of {n_eng} engines have operating settings that the {FD} model never saw in training. "
                 f"Their predictions are not reliable. Try the suggested model ({suggested}).")
    oor = sum(1 for n in NOTES if any(x.startswith("outside") for x in n))
    if oor:
        st.warning(f"{oor} engine(s) have sensor readings more than 10% beyond the training range. Treat them with caution.")

fleet, detail, method = st.tabs(["Fleet overview", "Engine detail", "Method and limits"])

# ---------------------------------------------------------------- fleet overview
with fleet:
    c1, c2, c3 = st.columns(3)
    c1.metric("HIGH: act now", int((level == "HIGH").sum()), help=f"P(fail within {H}) ≥ {P_CUT}")
    c2.metric("MEDIUM: plan maintenance", int((level == "MEDIUM").sum()), help=f"P(fail within {2*H}) ≥ {P_CUT}")
    c3.metric("LOW", int((level == "LOW").sum()))

    top_driver = [bundle["names"][j] for j in np.abs(V["shap"]).argmax(1)]
    table = pd.DataFrame({"engine": V["engines"], "risk": level, f"P(fail ≤ {H})": P_H, f"P(fail ≤ {2*H})": P_2H,
                          "predicted RUL": V["pred"], "main driver": top_driver})
    if any(NOTES):
        table["data check"] = ["; ".join(n) for n in NOTES]
    if eval_mode:
        table["true RUL"] = V["y_true"].astype(int)
    table = table.sort_values(f"P(fail ≤ {H})", ascending=False)
    st.dataframe(table.style.map(lambda v: f"background-color: {LEVEL_COLOUR[v]}", subset=["risk"])
                 .format({f"P(fail ≤ {H})": "{:.2f}", f"P(fail ≤ {2*H})": "{:.2f}", "predicted RUL": "{:.1f}"}),
                 hide_index=True, width="stretch", height=min(420, 38 + 35 * n_eng))
    if V["padded"]:
        st.caption(f"{len(V['padded'])} engine(s) have fewer than 30 recorded cycles. Their window is padded by "
                   "repeating the first cycle, so their predictions are less informed (notebook 06 shows clearly higher error).")
    if uploading:
        out = table.copy()
        out.insert(2, "H", H)
        st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"aerohealth_predictions_{FD}.csv", mime="text/csv")

    if eval_mode:
        true = V["y_true"] <= H
        hi = level == "HIGH"
        st.markdown(
            f"**Evaluation (test, nothing fitted on it):** HIGH caught **{(hi & true).sum()} of {true.sum()}** engines "
            f"that truly fail within {H} cycles. **{(hi & true).sum()} of {hi.sum()}** HIGH flags were correct. "
            f"Engines failing within {H} but rated LOW: **{int((true & (level == 'LOW')).sum())}**.")

# ---------------------------------------------------------------- engine detail
with detail:
    default = int(np.argmax(P_H))
    i = st.selectbox("Engine", np.arange(n_eng), index=default,
                     format_func=lambda k: f"Engine {V['engines'][k]} ({level[k]})" + (" ⚠" if NOTES[k] else ""))
    eng = V["engines"][i]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk level", level[i])
    c2.metric("Predicted RUL", f"{V['pred'][i]:.1f} cycles")
    c3.metric(f"P(fail within {H})", f"{P_H[i]:.2f}")
    if has_truth:
        c4.metric("True RUL", f"{int(V['y_true'][i])} cycles" if eval_mode else "hidden",
                  help="Turn on evaluation mode in the sidebar to compare.")
    else:
        c4.metric("Cycles recorded", int((V["raw"].engine == eng).sum()))
    for n in NOTES[i]:
        if n.startswith("short"):
            st.warning("Fewer than 30 recorded cycles: the window was padded by repeating the first cycle, "
                       "so the prediction is based on less information than usual.")
        elif n.startswith("unknown"):
            st.warning(f"Operating settings not seen by the {FD} model in training ({n.split('(', 1)[1].rstrip(')')}). "
                       "Prediction not reliable for this engine.")
        elif n.startswith("outside"):
            st.warning(f"Sensor readings beyond the training range ({n.split(': ', 1)[1]}). Treat with caution.")

    sv_i = V["shap"][i]
    top = np.argsort(-np.abs(sv_i))[:3]
    st.subheader("Why this prediction")
    st.markdown(f"Average prediction of the {FD} model is **{float(V['expl'].base_values[i]):.1f}** cycles. "
                "The biggest reasons this engine differs from that average:")
    for j in top:
        direction = "**shortens**" if sv_i[j] < 0 else "**extends**"
        st.markdown(f"- {ah.describe_feature(bundle['names'][j])}: {direction} predicted life by **{abs(sv_i[j]):.1f}** cycles")

    left, right = st.columns(2)
    with left:
        shap.plots.waterfall(V["expl"][i], max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 5.5)
        st.pyplot(fig, clear_figure=True)
        st.caption("Bars add up from the model's average (bottom) to this engine's predicted RUL (top).")
    with right:
        sensors = list(dict.fromkeys(bundle["names"][j].split("_")[0] for j in top))
        fig, axes = plt.subplots(len(sensors), 1, figsize=(7, 1.9 * len(sensors) + 0.6), sharex=True, squeeze=False)
        if multi_regime:
            window = V["X"][i]
            for ax, s in zip(axes[:, 0], sensors):
                ax.plot(np.arange(1, window.shape[0] + 1), window[:, bundle["sensors"].index(s)], lw=1.2)
                ax.set_ylabel(s)
                ax.set_title(f"{ah.SENSOR_NAMES[s]} (normalised within flight condition)", fontsize=9)
            axes[-1, 0].set_xlabel("step in the 30-cycle window the model sees")
            caption = ("Six-condition model: raw readings jump with the flight condition, so the per-condition "
                       "normalised window (exactly what the model sees) is shown.")
        else:
            hist = V["raw"][V["raw"].engine == eng]
            for ax, s in zip(axes[:, 0], sensors):
                ax.plot(hist["cycle"], hist[s], lw=1)
                ax.axvspan(hist["cycle"].iloc[-min(30, len(hist))], hist["cycle"].iloc[-1], color="orange", alpha=0.15)
                ax.set_ylabel(s)
                ax.set_title(ah.SENSOR_NAMES[s], fontsize=9)
            axes[-1, 0].set_xlabel("cycle")
            caption = ("Raw readings of the top drivers' sensors over this engine's recorded life. "
                       "Shaded = the last 30 cycles the model actually sees.")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(caption)

# ---------------------------------------------------------------- method
with method:
    top2 = V["sensor_share"].index[:2]
    top_txt = ", ".join(f"**{s}** ({ah.SENSOR_NAMES[s]}, {V['sensor_share'][s]:.0%} of total |SHAP|)" for s in top2)
    regime_txt = ("Operating settings form 6 distinct flight conditions (k-means error reaches 0 at k=6), and the condition "
                  "explains about 100% of raw sensor variance, so each sensor is MinMax-scaled **within its condition**."
                  if multi_regime else "Single operating condition, so sensors are MinMax-scaled once.")
    upload_txt = ("""
**Your uploaded data** is never used to fit anything. It is transformed with the clustering, sensor list and scalers
learned from the training engines, then passed to the trained model. Checks per engine:
- **Unknown flight condition:** operating settings farther than 0.1 (common scale) from every condition seen in training.
  Training noise is ≤ 0.005 and the closest two real conditions are 0.475 apart.
- **Outside training range:** a scaled sensor value below −0.1 or above 1.1 (more than 10% of the training range outside it).
- **Short history:** fewer than 30 cycles, padded by repeating the first cycle.
- **Model suggestion:** flight conditions are detected from the data. The fault type cannot be, so the model trained on
  both fault types (FD003 or FD004) is suggested.
""" if uploading else "")
    st.markdown(f"""
**Pipeline ({FD} model):** sensor data → preprocessing → 30-cycle windows → {len(bundle['names'])} summary features
→ XGBoost RUL → calibrated failure probability → risk level → SHAP explanation.

**Preprocessing** (fitted on training engines only). {regime_txt} Sensors with zero spread inside every condition are
dropped ({len(bundle['sensors'])} kept). RUL capped at 125. Train/validation split by engine (80/20).
{upload_txt}
**Failure risk.** Predicted RUL is converted to `P(true RUL ≤ H)` by a logistic regression fitted on the **validation**
engines. The RUL cut-offs in the sidebar are **derived**, not chosen. The only chosen numbers are H = {H} and the
probability cut-off = {P_CUT}. Notebook 05 shows the alternatives for both.

**Do the explanations match the physics?** Fault mode of the {FD} training data: **{ah.FAULT_MODES[FD]}**.
Most important sensors for these engines: {top_txt}. HPC = high-pressure compressor (T30, Ps30, P30 are measured at its
outlet). Fan faults show in fan speed (Nf, NRf) and bypass ratio (BPR).

**Limitations**
- Predictions are only as good as the match between your engines and the training fleet (simulated C-MAPSS turbofans).
- Calibration is fitted on windows from a small number of validation engines.
- One model (XGBoost) and seed in this dashboard. Notebook 06 reports several seeds and a 1D-CNN.
- SHAP explains what the **model** uses, not physical causation. Correlated sensors can share credit.
""")
