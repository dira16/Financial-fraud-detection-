import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from logger_setup import setup_logger

logger = setup_logger()

# -------------------------------
# Load model and scaler
# -------------------------------
@st.cache_resource
def load_model():
    logger.info("Loading model and scaler for Streamlit app...")
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("fraud_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Financial Fraud Detection (CPU Optimized)")
st.markdown("Upload transaction data and detect fraud in real-time using a lightweight ML model.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # -------------------------------
    # Load and preprocess data
    # -------------------------------
    st.info("Loading and scaling data...")
    df = pd.read_csv(uploaded_file)
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df
    X_scaled = scaler.transform(X.values)

    # -------------------------------
    # Run inference
    # -------------------------------
    st.info("Running fraud detection...")
    BATCH_SIZE = 1000
    probs, preds = [], []

    if hasattr(model, "predict_proba"):
        # Supervised mode
        for i in tqdm(range(0, X_scaled.shape[0], BATCH_SIZE), desc="Predicting"):
            batch_probs = model.predict_proba(X_scaled[i:i+BATCH_SIZE])[:, 1]
            probs.extend(batch_probs)
            batch_preds = (batch_probs > 0.5).astype(int)
            preds.extend(batch_preds)
    else:
        # Unsupervised mode
        scores = []
        for i in tqdm(range(0, X_scaled.shape[0], BATCH_SIZE), desc="Scoring"):
            batch_scores = model.decision_function(X_scaled[i:i+BATCH_SIZE])
            scores.extend(batch_scores)
        scores = np.array(scores)
        probs = -scores
        raw_preds = model.predict(X_scaled)
        preds = np.where(raw_preds == -1, 1, 0)

    probs = np.array(probs)
    preds = np.array(preds)

    # -------------------------------
    # Display results
    # -------------------------------
    st.success("Prediction completed successfully.")
    result_df = df.copy()
    result_df["FraudScore"] = probs
    result_df["FraudFlag"] = preds

    st.dataframe(result_df.head(20))  # preview top rows

    # Download scored CSV
    csv_download = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Fraud Scores CSV",
        data=csv_download,
        file_name="fraud_scores_scored.csv",
        mime="text/csv"
    )

    # -------------------------------
    # Visualize fraud risk
    # -------------------------------
    st.subheader("Fraud Risk Distribution")
    fig, ax = plt.subplots()
    ax.hist(probs, bins=50, color='orange', alpha=0.7)
    ax.set_title("Fraud Score Distribution")
    ax.set_xlabel("Fraud Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Show fraud count
    fraud_count = np.sum(preds)
    st.metric(label="Detected Fraudulent Transactions", value=fraud_count)