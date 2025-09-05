import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
from logger_setup import setup_logger

logger = setup_logger()

try:
    logger.info("=== Running Fraud Detection Inference ===")

    # --- Load data ---
    logger.info("Loading dataset...")
    df = pd.read_csv("data\creditcard.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    logger.info(f"Dataset loaded: {df.shape[0]} rows")

    # --- Load artifacts ---
    logger.info("Loading model and scaler...")
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("fraud_scaler.pkl")
    logger.info("Artifacts loaded successfully.")

    # --- Preprocess ---
    logger.info("Scaling features for inference...")
    X_scaled = scaler.transform(X.values)

    # --- Predict in batches ---
    logger.info("Generating fraud scores...")
    probs = []
    preds = []
    BATCH_SIZE = 1000

    if hasattr(model, "predict_proba"):
        # Supervised (Random Forest)
        for i in tqdm(range(0, X_scaled.shape[0], BATCH_SIZE), desc="Predicting"):
            batch_probs = model.predict_proba(X_scaled[i:i+BATCH_SIZE])[:, 1]
            probs.extend(batch_probs)
            batch_preds = (batch_probs > 0.5).astype(int)
            preds.extend(batch_preds)
    else:
        # Unsupervised (Isolation Forest)
        scores = []
        for i in tqdm(range(0, X_scaled.shape[0], BATCH_SIZE), desc="Scoring"):
            batch_scores = model.decision_function(X_scaled[i:i+BATCH_SIZE])
            scores.extend(batch_scores)
        scores = np.array(scores)
        probs = -scores  # higher = more anomalous
        raw_preds = model.predict(X_scaled)
        preds = np.where(raw_preds == -1, 1, 0)

    preds = np.array(preds)
    probs = np.array(probs)

    # --- Evaluation ---
    logger.info("Evaluating model performance...")
    logger.info("\n" + classification_report(y, preds))
    try:
        auc = roc_auc_score(y, probs)
        logger.info(f"ROC-AUC Score: {auc:.4f}")
    except Exception:
        logger.warning("Could not compute ROC-AUC for unsupervised mode.")

    # --- Save predictions ---
    logger.info("Exporting fraud_scores.csv...")
    out_df = pd.DataFrame({
        "TransactionID": df.index,
        "FraudScore": probs,
        "FraudFlag": preds,
        "ActualClass": y
    })
    out_df.to_csv("fraud_scores.csv", index=False)
    logger.info("fraud_scores.csv exported successfully.")

    logger.info("=== Inference Completed Successfully ===")

except Exception as e:
    logger.exception(f"Error during prediction: {e}")