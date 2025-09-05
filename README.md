# Fraud Detection Model

A machine learning system for detecting fraudulent credit card transactions using Random Forest and Isolation Forest algorithms.

## 📋 Overview

This project implements a fraud detection system that can identify potentially fraudulent transactions from credit card data. The system supports two different machine learning approaches:

- **Random Forest Classifier**: Supervised learning approach using labeled fraud data
- **Isolation Forest**: Unsupervised anomaly detection for identifying outliers

## 🚀 Features

- **Dual Algorithm Support**: Switch between Random Forest and Isolation Forest models
- **Data Preprocessing**: Automatic feature scaling using StandardScaler
- **Model Persistence**: Save and load trained models and scalers
- **Comprehensive Logging**: Detailed logging throughout the training process
- **Results Export**: Generate predictions with fraud scores and classifications
- **Stratified Splitting**: Maintains class distribution in train/test splits

## 📁 Project Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv          # Input dataset
├── fraud_model.pkl             # Trained model (generated)
├── fraud_scaler.pkl            # Fitted scaler (generated)
├── fraud_predictions.csv       # Model predictions (generated)
├── logger_setup.py             # Logging configuration
└── train_model.py              # Main training script
```

## 🛠️ Requirements

```python
pandas
numpy
scikit-learn
joblib
tqdm
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib tqdm
```

## 📊 Dataset

The system expects a CSV file with the following structure:
- **Time**: Transaction timestamp
- **Amount**: Transaction amount
- **Class**: Target variable (0 = legitimate, 1 = fraud)
- **V1, V2, ..., V28**: PCA-transformed features (typical for credit card datasets)

## 🔧 Configuration

### Model Selection
```python
USE_RANDOM_FOREST = True   # True = Random Forest, False = Isolation Forest
```

### Random Forest Parameters
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
```

### Isolation Forest Parameters
```python
IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.002,
    random_state=42
)
```

## 🏃‍♂️ Usage

### Training the Model

```bash
python train_model.py
```

### Output Files

The script generates:
1. **fraud_model.pkl**: Serialized trained model
2. **fraud_scaler.pkl**: Fitted StandardScaler for feature preprocessing
3. **fraud_predictions.csv**: Test set predictions with fraud scores

### Prediction Output Format

| Column | Description |
|--------|-------------|
| TransactionID | Unique identifier for each transaction |
| Time | Original timestamp from dataset |
| Amount | Transaction amount |
| FraudScore | Model confidence score (higher = more likely fraud) |
| PredictedClass | Binary prediction (0 = legitimate, 1 = fraud) |

## 📈 Model Performance

### Random Forest
- Uses probability scores for fraud likelihood
- Handles class imbalance with `class_weight='balanced'`
- Provides feature importance rankings

### Isolation Forest
- Detects anomalies without labeled fraud data
- Uses contamination parameter to set expected fraud rate
- Decision function scores converted to fraud probabilities

## 🔍 Key Features

### Data Preprocessing
- **Feature Scaling**: StandardScaler normalizes all features
- **Stratified Splitting**: Maintains fraud/legitimate ratio in train/test sets
- **Large Dataset Handling**: Configurable data size limit (currently 300,000 rows)

### Model Training
- **Progress Tracking**: tqdm progress bars for training visualization
- **Reproducibility**: Fixed random seeds for consistent results
- **Parallel Processing**: Multi-core support with `n_jobs=-1`

### Logging
- Comprehensive logging throughout the pipeline
- Training progress and model performance metrics
- File operations and data processing steps

## 🎯 Use Cases

- **Real-time Fraud Detection**: Deploy model for live transaction scoring
- **Batch Processing**: Analyze historical transactions for fraud patterns
- **Risk Assessment**: Generate fraud scores for transaction prioritization
- **Model Comparison**: Evaluate supervised vs. unsupervised approaches

## 📊 Sample Output

```csv
TransactionID,Time,Amount,FraudScore,PredictedClass
263020,160760.0,23.0,0.004851,0
11378,19847.0,11.85,0.013239,0
147283,88326.0,76.07,0.039237,0
```

## 🔄 Model Deployment

### Loading Saved Models
```python
import joblib

# Load trained model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('fraud_scaler.pkl')

# Make predictions on new data
scaled_features = scaler.transform(new_data)
fraud_scores = model.predict_proba(scaled_features)[:, 1]
predictions = model.predict(scaled_features)
```

## 📝 Logging

The system uses structured logging to track:
- Data loading and preprocessing steps
- Model training progress
- Performance metrics
- File I/O operations
- Error handling and debugging information

## 🚨 Important Notes

- **Data Leakage Prevention**: Features are scaled only on training data
- **Class Imbalance**: Random Forest uses balanced class weights
- **Memory Management**: Large datasets are handled efficiently with chunking
- **Reproducibility**: All random operations use fixed seeds

## 🤝 Contributing

To extend this project:
1. Add new algorithms in the model selection section
2. Implement additional preprocessing steps
3. Add hyperparameter tuning capabilities
4. Include additional evaluation metrics
5. Create visualization functions for results analysis

## 🆘 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce dataset size by modifying `.head(300000)`
2. **Import Error**: Ensure all required packages are installed
3. **File Not Found**: Check that `creditcard.csv` exists in the `data/` directory
4. **Performance Issues**: Adjust `n_estimators` or `max_depth` parameters

### Performance Optimization

- Use `n_jobs=-1` for parallel processing
- Reduce dataset size for faster experimentation
- Adjust model parameters based on available computational resources
