import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch

from models.hybrid_model import HybridTransformerAutoencoder
from pipeline.preprocess import load_timeseries_csv, normalize_zscore, window_sequence
from pipeline.inference import (
    quick_train_autoencoder,
    reconstruction_error_scores,
    expand_window_scores_to_timeline,
    pick_threshold,
    risk_label,
)
from firebase.firebase_client import init_firestore, save_run, maybe_create_alert, list_runs, list_alerts

try:
    from firebase.auth import sign_in_email_password
except Exception:
    sign_in_email_password = None

st.set_page_config(page_title="Crop Stress Early Warning (Hybrid T-AE)", layout="wide")

# ---------- Helpers ----------
def plot_series_and_anomalies(ts, values, scores, threshold, anomaly_mask):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ts, values, label="Signal (normalized)")
    ax.plot(ts, scores, label="Anomaly score")
    ax.axhline(threshold, linestyle="--", label="Threshold")
    # mark anomalies
    idx = np.where(anomaly_mask)[0]
    if len(idx) > 0:
        ax.scatter(np.array(ts)[idx], np.array(values)[idx], label="Anomaly points")
    ax.set_xlabel("Time")
    ax.set_title("Signal + Anomaly Score")
    ax.legend()
    st.pyplot(fig)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Session ----------
if "user" not in st.session_state:
    st.session_state.user = {"uid": "demo_user", "email": "demo@local", "mode": "demo"}

if "db_enabled" not in st.session_state:
    st.session_state.db_enabled = False

# ---------- Sidebar: Auth ----------
st.sidebar.title("Login")
api_key_set = bool(os.environ.get("FIREBASE_API_KEY", "").strip())
if api_key_set and sign_in_email_password is not None:
    email = st.sidebar.text_input("Email", value="")
    password = st.sidebar.text_input("Password", value="", type="password")
    if st.sidebar.button("Sign in"):
        try:
            auth = sign_in_email_password(email, password)
            st.session_state.user = {"uid": auth["localId"], "email": email, "mode": "firebase"}
            st.sidebar.success("Signed in")
        except Exception as e:
            st.sidebar.error(str(e))
else:
    st.sidebar.info("Auth in Demo mode (set FIREBASE_API_KEY to enable email/password login).")

if st.sidebar.button("Use Demo User"):
    st.session_state.user = {"uid": "demo_user", "email": "demo@local", "mode": "demo"}

st.sidebar.divider()

# ---------- Firebase Firestore init ----------
db = init_firestore()
st.session_state.db_enabled = db is not None
if st.session_state.db_enabled:
    st.sidebar.success("Firestore connected")
else:
    st.sidebar.warning("Firestore not configured (runs won't be saved). Set FIREBASE_SERVICE_ACCOUNT_JSON.")

st.sidebar.divider()
st.sidebar.caption(f"Device: {get_device()}")

# ---------- Main UI ----------
st.title("Hybrid Transformer–Autoencoder: Crop Stress & Yield Anomaly Prototype")
st.caption(f"User: {st.session_state.user['email']}  |  Mode: {st.session_state.user['mode']}")

tabs = st.tabs(["Run Analysis", "History (Firestore)", "Alerts (Firestore)", "About"])

# ---------- Tab 1: Run Analysis ----------
with tabs[0]:
    st.subheader("1) Choose Data")
    colA, colB = st.columns([2, 1])

    with colA:
        uploaded = st.file_uploader("Upload CSV (timestamp,value)", type=["csv"])
        demo_choice = st.selectbox("Or select demo dataset", ["None", "Healthy", "Early Stress", "Severe Stress"])
        use_uploaded = uploaded is not None

    with colB:
        window = st.number_input("Window size", min_value=24, max_value=256, value=64, step=8)
        train_split = st.slider("Train portion (normal baseline)", min_value=0.4, max_value=0.9, value=0.7, step=0.05)
        percentile = st.slider("Threshold percentile (train errors)", min_value=85, max_value=99, value=95, step=1)
        epochs = st.number_input("Quick-train epochs", min_value=3, max_value=50, value=12, step=1)

    df = None
    dataset_name = None

    if use_uploaded:
        df = load_timeseries_csv(uploaded.getvalue())
        dataset_name = uploaded.name
    elif demo_choice != "None":
        fname = {
            "Healthy": "data/demo_healthy.csv",
            "Early Stress": "data/demo_early_stress.csv",
            "Severe Stress": "data/demo_severe_stress.csv",
        }[demo_choice]
        df = pd.read_csv(fname)
        dataset_name = fname

    if df is None:
        st.info("Upload a CSV or select a demo dataset to continue.")
    else:
        st.subheader("2) Preview")
        st.dataframe(df.head(10), use_container_width=True)

        values_raw = df["value"].to_numpy()
        values_norm, mean, std = normalize_zscore(values_raw)
        ts = df["timestamp"].astype(str).to_list()

        st.write(f"Normalization: mean={mean:.4f}, std={std:.4f}")

        st.subheader("3) Run Hybrid T-AE Anomaly Detection")
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Preparing data..."):
                X = window_sequence(values_norm, int(window))  # (Nw, window, 1)
                Nw = X.shape[0]
                split_idx = int(Nw * float(train_split))
                X_train = X[:split_idx]
                X_all = X

            device = get_device()
            model = HybridTransformerAutoencoder()

            with st.spinner("Quick training on baseline (normal) segment..."):
                quick_train_autoencoder(model, X_train, epochs=int(epochs), device=device)

            with st.spinner("Scoring anomalies..."):
                train_scores_w = reconstruction_error_scores(model, X_train, device=device)
                all_scores_w = reconstruction_error_scores(model, X_all, device=device)
                th = pick_threshold(train_scores_w, percentile=float(percentile))

                # Expand window scores to per-timestep length
                T = len(values_norm)
                scores_t = expand_window_scores_to_timeline(all_scores_w, T=T, window=int(window))
                anomaly_mask = scores_t > th
                risk = risk_label(anomaly_mask)

            st.success(f"Done. Overall Risk: {risk}")
            st.metric("Overall Risk", risk)
            st.metric("Anomaly points", int(anomaly_mask.sum()))
            st.metric("Max score", float(scores_t.max()))

            st.subheader("4) Visualization")
            plot_series_and_anomalies(ts, values_norm, scores_t, th, anomaly_mask)

            # Save to Firestore (if enabled)
            if db is not None:
                with st.spinner("Saving run to Firestore..."):
                    uid = st.session_state.user["uid"]
                    # store compact series arrays (cap length for demo)
                    cap = min(len(values_norm), 800)
                    series = {
                        "timestamp": ts[:cap],
                        "value_norm": values_norm[:cap].tolist(),
                        "anomaly_score": scores_t[:cap].tolist(),
                        "is_anomaly": anomaly_mask[:cap].astype(int).tolist(),
                    }
                    run_id = save_run(
                        db=db,
                        uid=uid,
                        dataset_name=str(dataset_name),
                        threshold=float(th),
                        overall_risk=str(risk),
                        anomaly_count=int(anomaly_mask.sum()),
                        max_score=float(scores_t.max()),
                        series=series,
                    )
                    maybe_create_alert(db, uid=uid, run_id=run_id, overall_risk=str(risk))
                st.toast("Saved to Firestore ✅")
            else:
                st.info("Firestore not configured, so this run wasn't saved.")

# ---------- Tab 2: History ----------
with tabs[1]:
    st.subheader("Run History")
    if db is None:
        st.warning("Firestore not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON to enable history.")
    else:
        uid = st.session_state.user["uid"]
        runs = list_runs(db, uid=uid, limit=20)
        if not runs:
            st.info("No runs yet. Run an analysis first.")
        else:
            # show summary table
            rows = []
            for r in runs:
                rows.append({
                    "runId": r.get("_id"),
                    "createdAt": str(r.get("createdAt")),
                    "dataset": r.get("datasetName"),
                    "risk": r.get("overallRisk"),
                    "anomalies": r.get("anomalyCount"),
                    "threshold": r.get("threshold"),
                    "maxScore": r.get("maxScore"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------- Tab 3: Alerts ----------
with tabs[2]:
    st.subheader("Alerts")
    if db is None:
        st.warning("Firestore not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON to enable alerts.")
    else:
        uid = st.session_state.user["uid"]
        alerts = list_alerts(db, uid=uid, limit=20)
        if not alerts:
            st.info("No alerts yet.")
        else:
            for a in alerts:
                st.write(f"**{a.get('risk')}** — {a.get('message')}")
                st.caption(f"Run: {a.get('runId')}  |  Created: {a.get('createdAt')}  |  Status: {a.get('status')}")
                st.divider()

# ---------- Tab 4: About ----------
with tabs[3]:
    st.markdown("""
**What this prototype demonstrates (hackathon-ready):**
- A **Python-only** pipeline for early crop stress anomaly detection using a **Hybrid Transformer–Autoencoder**
- A dashboard showing **signal + anomaly score + threshold + anomaly points**
- Optional **Firebase Auth** (email/password) and **Firestore persistence**
- Automatic **High-risk alert creation** in Firestore

**Next upgrades (if you extend):**
- Multi-sensor/multi-variate inputs (NDVI + weather + soil)
- Real-time streaming ingestion
- Explainability: attention maps, feature attribution
- Action recommendations layer (LLM + curated agronomy KB)
""")
