import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
from datetime import datetime
from ..config.config import (
    ML_MODEL_PATH, ML_SCALER_PATH, ML_PREDICTION_THRESHOLD,
    ML_TRAINING_DAYS, ML_RETRAINING_DAYS
)

logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self):
        """Initialize the ML model"""
        self.model = None
        self.scaler = None
        self.last_training_time = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Load existing model or prepare for training"""
        try:
            if os.path.exists(ML_MODEL_PATH) and os.path.exists(ML_SCALER_PATH):
                logger.info("Loading existing ML model and scaler")
                self.model = joblib.load(ML_MODEL_PATH)
                self.scaler = joblib.load(ML_SCALER_PATH)
                self.last_training_time = datetime.fromtimestamp(os.path.getmtime(ML_MODEL_PATH))
            else:
                logger.info("No existing ML model found. Will train a new one.")
                self.model = None
                self.scaler = None
                self.last_training_time = None
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            self.model = None
            self.scaler = None
            self.last_training_time = None
            
    def prepare_features(self, df):
        """Prepare features for ML model"""
        try:
            # Select features for ML
            feature_columns = [
                'rsi_14', 'rsi_21', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'sma_20', 'sma_50', 'sma_200',
                'ema_9', 'ema_21', 'volume_ratio', 'adx', 'dmi_plus', 'dmi_minus',
                'STOCHRSIk_14', 'STOCHRSId_14', 'atr', 'price_change', 'volatility'
            ]
            
            # Create target variable (1 for price increase, 0 for decrease)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Extract features and target
            X = df[feature_columns]
            y = df['target']
            
            return X, y, feature_columns
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None, None
            
    def train(self, historical_data):
        """Train the ML model"""
        try:
            logger.info("Starting ML model training")
            
            # Prepare features
            X, y, _ = self.prepare_features(historical_data)
            if X is None or y is None:
                return False
                
            # Split data into training and validation sets
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_accuracy = self.model.score(X_train_scaled, y_train)
            val_accuracy = self.model.score(X_val_scaled, y_val)
            
            logger.info(f"ML model trained. Train accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
            
            # Save model and scaler
            os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)
            joblib.dump(self.model, ML_MODEL_PATH)
            joblib.dump(self.scaler, ML_SCALER_PATH)
            
            self.last_training_time = datetime.now()
            logger.info("ML model and scaler saved")
            
            return True
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return False
            
    def predict(self, df):
        """Get prediction from ML model"""
        if self.model is None or self.scaler is None:
            return None, 0.0
            
        try:
            # Prepare features
            X, _, _ = self.prepare_features(df)
            if X is None:
                return None, 0.0
                
            # Get the latest data point
            X_latest = X.iloc[-1:]
            
            # Scale features
            X_latest_scaled = self.scaler.transform(X_latest)
            
            # Get prediction and probability
            prediction = self.model.predict(X_latest_scaled)[0]
            probabilities = self.model.predict_proba(X_latest_scaled)[0]
            
            # Get confidence (probability of the predicted class)
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            return prediction, confidence
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return None, 0.0
            
    def should_retrain(self):
        """Check if the model should be retrained"""
        if self.last_training_time is None:
            return True
            
        days_since_training = (datetime.now() - self.last_training_time).days
        return days_since_training >= ML_RETRAINING_DAYS 