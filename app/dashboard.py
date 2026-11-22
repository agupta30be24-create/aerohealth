"""AeroHealth dashboard (C-MAPSS FD001-FD004 test fleets).  Run from the repo root:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aerohealth as ah  # noqa: E402

st.set_page_config(page_title="AeroHealth", layout="wide")
LEVEL_COLOUR = {"HIGH": "#f8d7da", "MEDIUM": "#fff3cd", "LOW": "#d1e7dd"}


@st.cache_resource(show_spinner="Training XGBoost and computing SHAP values (first load of this dataset only)...")
def build(fd):
    if not ah.windows_path(fd).exists():
        ah.prepare(fd)
    d = ah.load_windows(fd)
    sensors = list(d["features"])
    names = ah.feature_names(sensors)
    model = ah.train_xgb(ah.summarize(d["X_tr"]), d["y_tr"])
    F_test = ah.summarize(d["X_test"])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(F_test)
    expl = shap.Explanation(values=sv, base_values=np.full(len(sv), float(explainer.expected_value)),
                            data=F_test, feature_names=names)
    n = len(d["y_test"])
    engines = d["test_engines"] if "test_engines" in d.files else np.arange(1, n + 1)
    padded = set(d["padded_engines"].tolist()) if "padded_engines" in d.files else set()
    sensor_share = np.abs(sv).mean(0).reshape(len(ah.STATS), len(sensors)).sum(0)
    return {"y_val": d["y_val"], "y_test": d["y_test"], "X_test": d["X_test"], "sensors": sensors, "names": names,
            "shap": sv, "expl": expl, "engines": engines, "padded": padded,
            "sensor_share": pd.Series(sensor_share / sensor_share.sum(), index=sensors).sort_values(ascending=False),
            "pred_val": model.predict(ah.summarize(d["X_val"])), "pred_test": model.predict(F_test),
            "raw_test": ah.load_raw("test", fd)}


# ---------------- Sidebar ----------------
st.sidebar.header("Dataset")
FD = st.sidebar.selectbox("C-MAPSS subset", ah.DATASETS, index=0,
                          format_func=lambda f: f"{f}: {ah.N_REGIMES[f]} condition{'s' if ah.N_REGIMES[f] > 1 else ''}, "
                                                f"{ah.FAULT_MODES[f]}")
B = build(FD)
n_eng = len(B["y_test"])
multi_regime = ah.N_REGIMES[FD] > 1

st.sidebar.header("Risk settings")
H = st.sidebar.selectbox("Warning horizon H (cycles)", [15, 20, 30, 45, 60], index=2,
                         help="'Will this engine fail within H cycles?' An operational choice: how much warning "
                              "maintenance needs. Notebook 05, Section 3 compares these values.")
P_CUT = st.sidebar.slider("Probability cut-off", 0.10, 0.90, 0.50, 0.05,
                          help="A level triggers when its failure probability reaches this value. 0.5 = more likely "
                               "than not. Lower = fewer missed failures but more false alarms (notebook 05, Section 5).")
eval_mode = st.sidebar.checkbox("Evaluation mode (show true RUL)", value=False,
                                help="True RUL is unknown in real operation. It is only available here because "
                                     "C-MAPSS is a benchmark.")

cal_H = ah.fit_calibration(B["pred_val"], B["y_val"], H)
cal_2H = ah.fit_calibration(B["pred_val"], B["y_val"], 2 * H)
P_H, P_2H = ah.prob(cal_H, B["pred_test"]), ah.prob(cal_2H, B["pred_test"])
level = ah.risk_levels(P_H, P_2H, P_CUT)

st.sidebar.markdown(
    f"**Derived from calibration (not chosen):**  \n"
    f"HIGH if predicted RUL ≤ **{ah.derived_cutoff(cal_H, P_CUT):.1f}**  \n"
    f"MEDIUM if predicted RUL ≤ **{ah.derived_cutoff(cal_2H, P_CUT):.1f}**")
st.sidebar.caption("HIGH: P(fail within H) ≥ cut-off. MEDIUM: P(fail within 2H) ≥ cut-off. "
                   f"Calibration fitted on {FD} validation engines only.")

# ---------------- Header ----------------
st.title("AeroHealth: engine failure risk")
rmse = ah.rmse(B["y_test"], B["pred_test"])
st.caption(f"NASA C-MAPSS {FD} test fleet ({n_eng} engines, {ah.N_REGIMES[FD]} operating condition(s), "
           f"fault mode: {ah.FAULT_MODES[FD]}). Model: XGBoost on {len(B['names'])} window features, "
           f"test RMSE {rmse:.2f} cycles. Explanations: SHAP (TreeExplainer).")

fleet, detail, method = st.tabs(["Fleet overview", "Engine detail", "Method and limits"])

# ---------------- Fleet overview ----------------
with fleet:
    c1, c2, c3 = st.columns(3)
    c1.metric("HIGH: act now", int((level == "HIGH").sum()), help=f"P(fail within {H}) ≥ {P_CUT}")
    c2.metric("MEDIUM: plan maintenance", int((level == "MEDIUM").sum()), help=f"P(fail within {2*H}) ≥ {P_CUT}")
    c3.metric("LOW", int((level == "LOW").sum()))

    top_driver = [B["names"][j] for j in np.abs(B["shap"]).argmax(1)]
    table = pd.DataFrame({"engine": B["engines"], "risk": level,
                          f"P(fail ≤ {H})": P_H, f"P(fail ≤ {2*H})": P_2H,
                          "predicted RUL": B["pred_test"], "main driver": top_driver})
    if B["padded"]:
        table["note"] = ["short history (padded)" if e in B["padded"] else "" for e in B["engines"]]
    if eval_mode:
        table["true RUL"] = B["y_test"].astype(int)
    table = table.sort_values(f"P(fail ≤ {H})", ascending=False)
    st.dataframe(table.style.map(lambda v: f"background-color: {LEVEL_COLOUR[v]}", subset=["risk"])
                 .format({f"P(fail ≤ {H})": "{:.2f}", f"P(fail ≤ {2*H})": "{:.2f}", "predicted RUL": "{:.1f}"}),
                 hide_index=True, width="stretch", height=420)
    if B["padded"]:
        st.caption(f"{len(B['padded'])} test engines have fewer than 30 recorded cycles. Their window is padded by "
                   "repeating the first cycle, so their predictions are less informed (error reported in notebook 06).")

    if eval_mode:
        true = B["y_test"] <= H
        hi = level == "HIGH"
        st.markdown(
            f"**Evaluation (test, nothing fitted on it):** HIGH caught **{(hi & true).sum()} of {true.sum()}** engines "
            f"that truly fail within {H} cycles. **{(hi & true).sum()} of {hi.sum()}** HIGH flags were correct. "
            f"Engines failing within {H} but rated LOW: **{int((true & (level == 'LOW')).sum())}**.")

# ---------------- Engine detail ----------------
with detail:
    default = int(np.argmax(P_H))
    i = st.selectbox("Engine", np.arange(n_eng), index=default,
                     format_func=lambda k: f"Engine {B['engines'][k]} ({level[k]})"
                                           + (" - short history" if B['engines'][k] in B["padded"] else ""))
    eng = int(B["engines"][i])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk level", level[i])
    c2.metric("Predicted RUL", f"{B['pred_test'][i]:.1f} cycles")
    c3.metric(f"P(fail within {H})", f"{P_H[i]:.2f}")
    c4.metric("True RUL", f"{int(B['y_test'][i])} cycles" if eval_mode else "hidden",
              help="Turn on evaluation mode in the sidebar to compare.")
    if eng in B["padded"]:
        st.warning("This engine has fewer than 30 recorded cycles. Its window was padded by repeating the first "
                   "cycle, so the prediction is based on less information than usual.")

    sv_i = B["shap"][i]
    top = np.argsort(-np.abs(sv_i))[:3]
    st.subheader("Why this prediction")
    st.markdown(f"Average predicted RUL over the fleet is **{float(B['expl'].base_values[i]):.1f}** cycles. "
                "The biggest reasons this engine differs from that average:")
    for j in top:
        direction = "**shortens**" if sv_i[j] < 0 else "**extends**"
        st.markdown(f"- {ah.describe_feature(B['names'][j])}: {direction} predicted life by "
                    f"**{abs(sv_i[j]):.1f}** cycles")

    left, right = st.columns(2)
    with left:
        shap.plots.waterfall(B["expl"][i], max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 5.5)
        st.pyplot(fig, clear_figure=True)
        st.caption("Bars add up from the fleet average (bottom) to this engine's predicted RUL (top).")
    with right:
        sensors = list(dict.fromkeys(B["names"][j].split("_")[0] for j in top))
        fig, axes = plt.subplots(len(sensors), 1, figsize=(7, 1.9 * len(sensors) + 0.6), sharex=True, squeeze=False)
        if multi_regime:
            window = B["X_test"][i]
            for ax, s in zip(axes[:, 0], sensors):
                ax.plot(np.arange(1, window.shape[0] + 1), window[:, B["sensors"].index(s)], lw=1.2)
                ax.set_ylabel(s)
                ax.set_title(f"{ah.SENSOR_NAMES[s]} (normalised within flight condition)", fontsize=9)
            axes[-1, 0].set_xlabel("step in the 30-cycle window the model sees")
            caption = ("FD002/FD004 switch between 6 flight conditions, so raw readings jump with the condition. "
                       "Shown: the per-condition normalised window, exactly what the model sees.")
        else:
            hist = B["raw_test"][B["raw_test"].engine == eng]
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

# ---------------- Method ----------------
with method:
    top2 = B["sensor_share"].index[:2]
    top_txt = ", ".join(f"**{s}** ({ah.SENSOR_NAMES[s]}, {B['sensor_share'][s]:.0%} of total |SHAP|)" for s in top2)
    regime_txt = ("Operating settings form 6 distinct flight conditions (k-means error reaches 0 at k=6), and the condition "
                  "explains about 100% of raw sensor variance, so each sensor is MinMax-scaled **within its condition** "
                  "(fitted on training engines only)." if multi_regime else
                  "Single operating condition, so sensors are MinMax-scaled once (fitted on training engines only).")
    st.markdown(f"""
**Pipeline ({FD}):** sensor data → preprocessing → 30-cycle windows → {len(B['names'])} summary features
→ XGBoost RUL → calibrated failure probability → risk level → SHAP explanation.

**Preprocessing.** {regime_txt} Sensors with zero spread inside every condition are dropped
({len(B['sensors'])} kept). RUL capped at 125. Train/validation split by engine (80/20).

**Failure risk.** Predicted RUL is converted to `P(true RUL ≤ H)` by a logistic regression fitted on the **validation**
engines, which corrects for the model's bias near failure. The RUL cut-offs in the sidebar are **derived**, not chosen.
The only chosen numbers are H = {H} and the probability cut-off = {P_CUT}. Notebook 05 shows the alternatives for both.

**Do the explanations match the physics?** Fault mode of {FD}: **{ah.FAULT_MODES[FD]}**.
The model's two most important sensors here: {top_txt}. Compare them with the fault mode: HPC = high-pressure
compressor (T30, Ps30, P30 are measured at its outlet). Fan faults show in fan speed (Nf, NRf) and bypass ratio (BPR).

**Limitations**
- Only {int((B['y_test'] <= H).sum())} of {n_eng} test engines fail within H = {H}, so a few engines move recall noticeably.
- Calibration is fitted on windows from a small number of validation engines.
- One model (XGBoost) and seed in this dashboard. Notebook 06 reports several seeds and a 1D-CNN.
- SHAP explains what the **model** uses, not physical causation. Correlated sensors can share credit.
""" + (f"- {len(B['padded'])} test engines are shorter than 30 cycles and are padded.\n" if B["padded"] else ""))
