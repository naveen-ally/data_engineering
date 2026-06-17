"""
Anomaly Detection Module
Detect anomalies and outliers in datasets using unsupervised machine learning.
Reduces manual validation work and enables self-healing pipelines.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Detects anomalies in datasets using multiple unsupervised ML techniques.
    
    Key Features:
    - Isolation Forest for anomaly detection
    - Statistical methods (IQR, Z-score)
    - Multi-method ensemble approach
    - Anomaly scoring and severity ranking
    """
    
    def __init__(self, contamination: float = 0.1, method: str = 'isolation_forest'):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0-1)
            method: Detection method ('isolation_forest', 'statistical', 'ensemble')
        """
        self.contamination = contamination
        self.method = method
        self.scaler = StandardScaler()
        self.detector = None
        self.anomaly_scores = {}
    
    def detect(self, data: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Detect anomalies in the dataset.
        
        Args:
            data: Input DataFrame
            threshold: Confidence threshold for anomalies (0-1)
            
        Returns:
            DataFrame containing anomalous records with scores
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            return pd.DataFrame()
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        if self.method == 'isolation_forest':
            return self._detect_isolation_forest(data, numeric_data, threshold)
        elif self.method == 'statistical':
            return self._detect_statistical(data, numeric_data, threshold)
        else:  # ensemble
            return self._detect_ensemble(data, numeric_data, threshold)
    
    def _detect_isolation_forest(self, original_data: pd.DataFrame, 
                                 numeric_data: pd.DataFrame, 
                                 threshold: float) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest algorithm.
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        # Train Isolation Forest
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Get predictions and scores
        predictions = self.detector.fit_predict(scaled_data)
        scores = self.detector.score_samples(scaled_data)
        
        # Normalize scores to 0-1 range (higher = more anomalous)
        normalized_scores = 1 / (1 + np.exp(scores))
        
        # Filter anomalies by threshold
        anomaly_indices = np.where((predictions == -1) & (normalized_scores >= threshold))[0]
        
        if len(anomaly_indices) == 0:
            return pd.DataFrame()
        
        result = original_data.iloc[anomaly_indices].copy()
        result['anomaly_score'] = normalized_scores[anomaly_indices]
        result['anomaly_type'] = 'isolation_forest'
        
        return result.sort_values('anomaly_score', ascending=False)
    
    def _detect_statistical(self, original_data: pd.DataFrame, 
                           numeric_data: pd.DataFrame, 
                           threshold: float) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods (IQR, Z-score).
        """
        anomalies = pd.DataFrame()
        anomaly_flags = np.zeros(len(numeric_data), dtype=bool)
        anomaly_scores_list = np.zeros(len(numeric_data))
        
        for col in numeric_data.columns:
            # Z-score method
            z_scores = np.abs((numeric_data[col] - numeric_data[col].mean()) / numeric_data[col].std())
            z_anomalies = z_scores > 3
            
            # IQR method
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_anomalies = (numeric_data[col] < Q1 - 1.5*IQR) | (numeric_data[col] > Q3 + 1.5*IQR)
            
            combined = (z_anomalies | iqr_anomalies)
            anomaly_flags |= combined
            anomaly_scores_list += combined.astype(int)
        
        # Normalize anomaly scores
        max_score = anomaly_scores_list.max()
        if max_score > 0:
            normalized_scores = anomaly_scores_list / max_score
        else:
            normalized_scores = anomaly_scores_list
        
        # Filter by threshold
        anomaly_indices = np.where(normalized_scores >= threshold)[0]
        
        if len(anomaly_indices) == 0:
            return pd.DataFrame()
        
        result = original_data.iloc[anomaly_indices].copy()
        result['anomaly_score'] = normalized_scores[anomaly_indices]
        result['anomaly_type'] = 'statistical'
        
        return result.sort_values('anomaly_score', ascending=False)
    
    def _detect_ensemble(self, original_data: pd.DataFrame, 
                        numeric_data: pd.DataFrame, 
                        threshold: float) -> pd.DataFrame:
        """
        Ensemble method combining multiple detection approaches.
        """
        # Get results from both methods
        if_result = self._detect_isolation_forest(original_data, numeric_data, 0.5)
        stat_result = self._detect_statistical(original_data, numeric_data, 0.5)
        
        # Combine results
        if len(if_result) == 0 and len(stat_result) == 0:
            return pd.DataFrame()
        
        all_indices = set()
        combined_scores = {}
        
        for idx in if_result.index:
            all_indices.add(idx)
            combined_scores[idx] = combined_scores.get(idx, 0) + if_result.loc[idx, 'anomaly_score']
        
        for idx in stat_result.index:
            all_indices.add(idx)
            combined_scores[idx] = combined_scores.get(idx, 0) + stat_result.loc[idx, 'anomaly_score']
        
        # Average scores and filter by threshold
        final_anomalies = []
        final_scores = []
        
        for idx in all_indices:
            avg_score = combined_scores[idx] / 2
            if avg_score >= threshold:
                final_anomalies.append(idx)
                final_scores.append(avg_score)
        
        if len(final_anomalies) == 0:
            return pd.DataFrame()
        
        result = original_data.loc[final_anomalies].copy()
        result['anomaly_score'] = final_scores
        result['anomaly_type'] = 'ensemble'
        
        return result.sort_values('anomaly_score', ascending=False)
    
    def get_anomaly_report(self, data: pd.DataFrame, threshold: float = 0.95) -> Dict[str, Any]:
        """
        Generate a comprehensive anomaly report.
        
        Args:
            data: Input DataFrame
            threshold: Confidence threshold
            
        Returns:
            Dictionary with anomaly statistics and recommendations
        """
        anomalies = self.detect(data, threshold)
        
        report = {
            'total_records': len(data),
            'anomalies_detected': len(anomalies),
            'anomaly_percentage': round((len(anomalies) / len(data)) * 100, 2),
            'average_anomaly_score': round(anomalies['anomaly_score'].mean(), 3) if len(anomalies) > 0 else 0,
            'max_anomaly_score': round(anomalies['anomaly_score'].max(), 3) if len(anomalies) > 0 else 0,
            'min_anomaly_score': round(anomalies['anomaly_score'].min(), 3) if len(anomalies) > 0 else 0,
            'detection_method': self.method,
            'recommendations': self._generate_recommendations(anomalies, data)
        }
        
        return report
    
    def _generate_recommendations(self, anomalies: pd.DataFrame, data: pd.DataFrame) -> List[str]:
        """
        Generate actionable recommendations based on detected anomalies.
        """
        recommendations = []
        
        if len(anomalies) == 0:
            recommendations.append("No anomalies detected. Data quality is acceptable.")
            return recommendations
        
        anomaly_rate = (len(anomalies) / len(data)) * 100
        
        if anomaly_rate > 5:
            recommendations.append(f"High anomaly rate ({anomaly_rate:.1f}%). Review data collection process.")
        
        # Check for specific columns with anomalies
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in anomalies.columns:
                outlier_rate = (anomalies[col].notna().sum() / len(anomalies)) * 100
                if outlier_rate > 50:
                    recommendations.append(f"Column '{col}' has frequent outliers. Consider data transformation.")
        
        if len(anomalies) > 0:
            max_score = anomalies['anomaly_score'].max()
            critical_anomalies = len(anomalies[anomalies['anomaly_score'] > 0.9])
            if critical_anomalies > 0:
                recommendations.append(f"Found {critical_anomalies} critical anomalies. Manual review recommended.")
        
        return recommendations
    
    def auto_quarantine(self, data: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Automatically separate anomalies for review.
        
        Returns:
            Tuple of (clean_data, quarantined_anomalies)
        """
        anomalies = self.detect(data, threshold)
        clean_data = data.drop(anomalies.index)
        
        return clean_data, anomalies


# Example usage
if __name__ == "__main__":
    # Create sample data with anomalies
    np.random.seed(42)
    normal_data = pd.DataFrame({
        'temperature': np.random.normal(20, 2, 95),
        'humidity': np.random.normal(60, 10, 95),
        'pressure': np.random.normal(1013, 5, 95)
    })
    
    # Add some anomalies
    anomalies = pd.DataFrame({
        'temperature': [45, -10, 22],
        'humidity': [120, 5, 65],
        'pressure': [950, 1100, 1012]
    })
    
    data = pd.concat([normal_data, anomalies], ignore_index=True)
    
    # Detect anomalies
    detector = AnomalyDetector(method='ensemble')
    detected = detector.detect(data, threshold=0.85)
    report = detector.get_anomaly_report(data, threshold=0.85)
    
    print("\nAnomaly Detection Report:")
    print("="*60)
    for key, value in report.items():
        if key != 'recommendations':
            print(f"{key}: {value}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    print("\nDetected Anomalies:")
    print(detected.head())
