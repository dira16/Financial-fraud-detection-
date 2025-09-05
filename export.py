import pandas as pd
from logger_setup import setup_logger

logger = setup_logger()

try:
    logger.info("=== Preparing Data for Power BI ===")

    df = pd.read_csv("fraud_scores.csv")
    logger.info(f"fraud_scores.csv loaded with {df.shape[0]} rows")

    fraud_count = df[df['FraudFlag'] == 1].shape[0]
    logger.info(f"Detected fraud cases: {fraud_count}")

    # Perform any extra aggregation if needed
    df.to_csv("fraud_scores_ready.csv", index=False)
    logger.info("fraud_scores_ready.csv exported successfully.")

    logger.info("=== Export Completed Successfully ===")

except Exception as e:
    logger.exception(f"Error during export: {e}")