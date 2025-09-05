import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from logger_setup import setup_logger

logger = setup_logger()

USE_RANDOM_FOREST = True  # False → Isolation Forest

logger.info("=== Training Fraud Detection Model ===")

# --- Load data ---
logger.info("Loading dataset...")
df = pd.read_csv("data\creditcard.csv").head(300000)   # Use only first 100 rows
logger.info(f"Dataset size: {df.shape}")

# --- Preprocess ---
X = df.drop("Class", axis=1)
y = df["Class"]

logger.info("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
logger.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# --- Train model ---
if USE_RANDOM_FOREST:
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    for _ in tqdm(range(1), desc="Training RF"):
        model.fit(X_train, y_train)
else:
    logger.info("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.002,
        random_state=42
    )
    for _ in tqdm(range(1), desc="Training IF"):
        model.fit(X_train)

# --- Save artifacts ---
logger.info("Saving model and scaler...")
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "fraud_scaler.pkl")
logger.info("Artifacts saved: fraud_model.pkl, fraud_scaler.pkl")

logger.info("=== Training Complete ===")
from sklearn.metrics import roc_auc_score

# --- Inference on test set ---
logger.info("Generating predictions on test data...")

if USE_RANDOM_FOREST:
    # RF gives probability scores
    fraud_scores = model.predict_proba(X_test)[:, 1]  # probability of fraud (Class=1)
    preds = model.predict(X_test)
else:
    # IsolationForest: lower scores → anomalies
    fraud_scores = -model.decision_function(X_test)  # higher = more anomalous
    preds = model.predict(X_test)
    preds = np.where(preds == -1, 1, 0)  # map: -1 (outlier) → 1 (fraud), 1 → 0

# --- Build results DataFrame ---
logger.info("Building results DataFrame...")

results = pd.DataFrame({
    "TransactionID": df.iloc[y_test.index].index,         # Row index as transaction ID
    "Time": df.iloc[y_test.index]["Time"].values,         # Original Time column
    "Amount": df.iloc[y_test.index]["Amount"].values,     # Original Amount column
    "FraudScore": fraud_scores,
    "PredictedClass": preds
})

# --- Save results ---
results.to_csv("fraud_predictions.csv", index=False)
logger.info("Saved predictions to fraud_predictions.csv")
logger.info(f"Sample output:\n{results.head()}")
