"""
Failure Predictor Module
Predict pipeline failures before they occur using ML.
Enables proactive failure prevention and auto-remediation.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FailurePredictor:
    """
    Predict pipeline failures using ML models.
    
    Key Features:
    - Failure pattern detection
    - Risk scoring
    - Predictive alerts
    - Root cause analysis
    """
    
    def __init__(self):
        """Initialize failure predictor."""
        self.model = None
        self.scaler = StandardScaler()
        self.training_data = None
        self.failure_patterns = {}
    
    def train_model(self, historical_data: pd.DataFrame,
                   failure_indicator: str = 'failed') -> None:
        """
        Train failure prediction model on historical data.
        
        Args:
            historical_data: Historical job execution data
            failure_indicator: Column indicating failure (boolean)
        """
        self.training_data = historical_data
        
        # Select features (numeric columns except target)
        feature_cols = historical_data.select_dtypes(include=[np.number]).columns.tolist()
        if failure_indicator in feature_cols:
            feature_cols.remove(failure_indicator)
        
        X = historical_data[feature_cols]
        y = historical_data[failure_indicator]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train random forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Analyze failure patterns
        self._analyze_failure_patterns(historical_data, failure_indicator)
    
    def _analyze_failure_patterns(self, data: pd.DataFrame,
                                 failure_indicator: str) -> None:
        """
        Analyze patterns that lead to failures.
        """
        failures = data[data[failure_indicator] == True]
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if failure_indicator in numeric_cols:
            numeric_cols.remove(failure_indicator)
        
        for col in numeric_cols:
            if col in failures.columns:
                success_mean = data[data[failure_indicator] == False][col].mean()
                failure_mean = failures[col].mean()
                
                self.failure_patterns[col] = {
                    'success_mean': success_mean,
                    'failure_mean': failure_mean,
                    'risk_correlation': (failure_mean - success_mean) / (success_mean + 1e-8)
                }
    
    def predict_failure(self, job_metrics: Dict[str, float]) -> Tuple[bool, float]:
        """
        Predict if a job will fail.
        
        Args:
            job_metrics: Dictionary of job metrics
            
        Returns:
            Tuple of (will_fail, failure_probability)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert to DataFrame for consistency
        df = pd.DataFrame([job_metrics])
        
        # Use same features as training
        feature_cols = self.training_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]  # Probability of failure
        
        return bool(prediction), float(probability)
    
    def identify_risk_factors(self, job_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Identify factors contributing to failure risk.
        
        Args:
            job_metrics: Dictionary of job metrics
            
        Returns:
            List of risk factors with severity
        """
        risk_factors = []
        
        for metric_name, metric_value in job_metrics.items():
            if metric_name in self.failure_patterns:
                pattern = self.failure_patterns[metric_name]
                
                # Calculate deviation from safe values
                safe_value = pattern['success_mean']
                deviation = (metric_value - safe_value) / (safe_value + 1e-8)
                
                if abs(deviation) > 0.2:  # Significant deviation
                    risk_factors.append({
                        'metric': metric_name,
                        'value': metric_value,
                        'safe_range': {
                            'success_mean': round(safe_value, 2),
                            'failure_mean': round(pattern['failure_mean'], 2)
                        },
                        'deviation_percentage': round(deviation * 100, 2),
                        'risk_level': self._calculate_risk_level(deviation)
                    })
        
        return sorted(risk_factors, key=lambda x: abs(x['deviation_percentage']), reverse=True)
    
    def _calculate_risk_level(self, deviation: float) -> str:
        """
        Calculate risk level from deviation.
        """
        abs_dev = abs(deviation)
        if abs_dev > 1.0:
            return 'CRITICAL'
        elif abs_dev > 0.5:
            return 'HIGH'
        elif abs_dev > 0.2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        feature_cols = self.training_data.select_dtypes(include=[np.number]).columns.tolist()
        importances = dict(zip(feature_cols, self.model.feature_importances_))
        
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    def print_prediction_report(self, job_metrics: Dict[str, float]) -> None:
        """
        Print failure prediction report.
        
        Args:
            job_metrics: Job metrics to predict on
        """
        will_fail, probability = self.predict_failure(job_metrics)
        risk_factors = self.identify_risk_factors(job_metrics)
        
        print("\n" + "="*80)
        print("FAILURE PREDICTION REPORT")
        print("="*80 + "\n")
        
        print(f"Failure Prediction: {'YES' if will_fail else 'NO'}")
        print(f"Failure Probability: {probability*100:.1f}%")
        
        if risk_factors:
            print("\nIdentified Risk Factors:")
            for i, factor in enumerate(risk_factors, 1):
                print(f"\n{i}. {factor['metric']} [{factor['risk_level']}]")
                print(f"   Current Value: {factor['value']}")
                print(f"   Safe Range: {factor['safe_range']}")
                print(f"   Deviation: {factor['deviation_percentage']}%")
        else:
            print("\nNo significant risk factors identified.")
        
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    # Create sample historical data
    np.random.seed(42)
    historical = pd.DataFrame({
        'duration_seconds': np.random.randint(100, 3000, 100),
        'records_processed': np.random.randint(1000, 100000, 100),
        'records_failed': np.random.randint(0, 1000, 100),
        'memory_usage_mb': np.random.randint(100, 5000, 100),
        'failed': np.random.choice([0, 1], 100, p=[0.8, 0.2])
    })
    
    # Train predictor
    predictor = FailurePredictor()
    predictor.train_model(historical, 'failed')
    
    # Predict on new job
    new_job = {
        'duration_seconds': 2500,
        'records_processed': 50000,
        'records_failed': 500,
        'memory_usage_mb': 4500
    }
    
    predictor.print_prediction_report(new_job)
