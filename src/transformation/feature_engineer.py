"""
Feature Engineer Module
Automated feature extraction and engineering from raw data.
Generates useful features automatically using ML techniques.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Automatically generate features from raw data.
    
    Key Features:
    - Automated feature generation from timestamps
    - Statistical features (mean, std, median)
    - Interaction features
    - Encoding categorical variables
    - Feature scaling and normalization
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.generated_features = {}
        self.scalers = {}
    
    def generate_features(self, data: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """
        Generate features from raw data.
        
        Args:
            data: Input DataFrame
            target: Target column for supervised features
            
        Returns:
            DataFrame with original and generated features
        """
        features = data.copy()
        
        # Generate temporal features
        features = self._generate_temporal_features(features)
        
        # Generate statistical features
        features = self._generate_statistical_features(features)
        
        # Encode categorical variables
        features = self._encode_categorical(features)
        
        # Generate interaction features
        if target:
            features = self._generate_interaction_features(features, target)
        
        return features
    
    def _generate_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from datetime columns.
        """
        features = data.copy()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            # Extract components
            features[f'{col}_year'] = features[col].dt.year
            features[f'{col}_month'] = features[col].dt.month
            features[f'{col}_day'] = features[col].dt.day
            features[f'{col}_dayofweek'] = features[col].dt.dayofweek
            features[f'{col}_quarter'] = features[col].dt.quarter
            features[f'{col}_is_month_end'] = features[col].dt.is_month_end.astype(int)
            features[f'{col}_is_month_start'] = features[col].dt.is_month_start.astype(int)
            
            self.generated_features[col] = [
                f'{col}_year', f'{col}_month', f'{col}_day',
                f'{col}_dayofweek', f'{col}_quarter',
                f'{col}_is_month_end', f'{col}_is_month_start'
            ]
        
        return features
    
    def _generate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical features from numeric columns.
        """
        features = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Calculate rolling statistics if enough data
        if len(data) > 5:
            for col in numeric_cols:
                features[f'{col}_rolling_mean_5'] = features[col].rolling(window=5, min_periods=1).mean()
                features[f'{col}_rolling_std_5'] = features[col].rolling(window=5, min_periods=1).std()
                features[f'{col}_lag_1'] = features[col].shift(1)
                
                self.generated_features[col] = [
                    f'{col}_rolling_mean_5', f'{col}_rolling_std_5', f'{col}_lag_1'
                ]
        
        return features
    
    def _encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.
        """
        features = data.copy()
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # One-hot encoding for low-cardinality columns
            if data[col].nunique() <= 10:
                encoded = pd.get_dummies(features[col], prefix=col, drop_first=True)
                features = pd.concat([features, encoded], axis=1)
                self.generated_features[col] = list(encoded.columns)
            else:
                # Label encoding for high-cardinality columns
                features[f'{col}_encoded'] = pd.factorize(features[col])[0]
                self.generated_features[col] = [f'{col}_encoded']
        
        return features
    
    def _generate_interaction_features(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Generate interaction features for target variable.
        """
        features = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if target in numeric_cols:
            numeric_cols.remove(target)
        
        # Create interaction features (limited to top 5 correlations)
        if len(numeric_cols) > 0 and target in data.columns:
            correlations = data[numeric_cols].corrwith(data[target]).abs().sort_values(ascending=False)
            top_cols = correlations.head(min(5, len(numeric_cols))).index.tolist()
            
            for col in top_cols:
                features[f'{target}_x_{col}'] = features[target] * features[col]
                features[f'{target}_div_{col}'] = features[target] / (features[col] + 1e-8)
        
        return features
    
    def scale_features(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            data: DataFrame to scale
            method: 'standard' (z-score) or 'minmax' (0-1)
            
        Returns:
            Scaled DataFrame
        """
        features = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        features[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        self.scalers['numeric'] = scaler
        
        return features
    
    def get_feature_importance(self, data: pd.DataFrame, target: str) -> Dict[str, float]:
        """
        Estimate feature importance using correlation.
        
        Args:
            data: DataFrame with features
            target: Target column
            
        Returns:
            Dictionary of feature importances
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if target not in numeric_cols:
            return {}
        
        correlations = data[numeric_cols].corrwith(data[target]).abs().sort_values(ascending=False)
        return correlations.to_dict()
    
    def print_feature_report(self, data: pd.DataFrame, target: str = None) -> None:
        """
        Print a feature engineering report.
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING REPORT")
        print("="*80 + "\n")
        
        print(f"Original Features: {len(data.columns)}")
        print(f"New Features Generated: {sum(len(v) for v in self.generated_features.values())}")
        print(f"Total Features: {len(data.columns)}")
        
        print("\nGenerated Features by Source:")
        for source, features in self.generated_features.items():
            print(f"  From '{source}': {features}")
        
        if target and target in data.columns:
            importance = self.get_feature_importance(data, target)
            print(f"\nTop 10 Important Features (by correlation with '{target}'):")
            for i, (feat, imp) in enumerate(list(importance.items())[:10], 1):
                print(f"  {i}. {feat}: {imp:.4f}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100),
        'product': ['A', 'B', 'A', 'C', 'B'] * 20,
        'quantity': np.random.randint(1, 100, 100),
        'price': np.random.uniform(10, 100, 100),
        'sales': np.random.uniform(100, 1000, 100)
    })
    
    # Generate features
    engineer = FeatureEngineer()
    features = engineer.generate_features(data, target='sales')
    
    print(f"Original shape: {data.shape}")
    print(f"With features shape: {features.shape}")
    print(f"\nNew columns: {set(features.columns) - set(data.columns)}")
    
    engineer.print_feature_report(features, target='sales')
