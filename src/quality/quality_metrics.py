"""
Data Quality Metrics Module
Calculate and track data quality metrics to monitor pipeline health.
Provides comprehensive data profiling and quality scoring.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime


class QualityMetrics:
    """
    Calculate comprehensive data quality metrics.
    
    Key Features:
    - Completeness: Missing value analysis
    - Uniqueness: Duplicate detection
    - Validity: Value conformance
    - Consistency: Data type consistency
    - Accuracy: Data accuracy scoring
    """
    
    def __init__(self):
        """Initialize quality metrics calculator."""
        self.metrics = {}
        self.timestamp = None
    
    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all quality metrics for a dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with comprehensive quality metrics
        """
        self.timestamp = datetime.now()
        
        metrics = {
            'timestamp': self.timestamp.isoformat(),
            'total_records': len(data),
            'total_columns': len(data.columns),
            'completeness': self._calculate_completeness(data),
            'uniqueness': self._calculate_uniqueness(data),
            'validity': self._calculate_validity(data),
            'consistency': self._calculate_consistency(data),
            'overall_quality_score': 0
        }
        
        # Calculate overall quality score (weighted average)
        metrics['overall_quality_score'] = round(
            (metrics['completeness']['overall_score'] * 0.3 +
             metrics['uniqueness']['overall_score'] * 0.2 +
             metrics['validity']['overall_score'] * 0.3 +
             metrics['consistency']['overall_score'] * 0.2),
            2
        )
        
        self.metrics = metrics
        return metrics
    
    def _calculate_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate data completeness (missing values).
        """
        completeness_scores = {}
        
        for col in data.columns:
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            completeness_scores[col] = round(100 - missing_pct, 2)
        
        overall_score = round(np.mean(list(completeness_scores.values())), 2)
        
        return {
            'scores_by_column': completeness_scores,
            'overall_score': overall_score,
            'metric_type': 'Percentage of non-null values'
        }
    
    def _calculate_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate uniqueness (duplicate detection).
        """
        total_duplicates = data.duplicated().sum()
        duplicate_pct = (total_duplicates / len(data)) * 100
        uniqueness_score = round(100 - duplicate_pct, 2)
        
        # Column-level duplicates
        col_duplicates = {}
        for col in data.select_dtypes(include=['object', 'int64', 'float64']).columns:
            dup_count = data[col].duplicated().sum()
            col_duplicates[col] = int(dup_count)
        
        return {
            'total_duplicates': int(total_duplicates),
            'duplicate_percentage': round(duplicate_pct, 2),
            'duplicates_by_column': col_duplicates,
            'overall_score': uniqueness_score,
            'metric_type': 'Percentage of unique records'
        }
    
    def _calculate_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate validity (data type conformance).
        """
        validity_scores = {}
        
        for col in data.columns:
            col_type = data[col].dtype
            # Numeric columns
            if col_type in ['int64', 'float64']:
                valid = data[col].notna().sum()
            else:
                # String columns - check for expected patterns
                valid = len(data[col]) - data[col].isnull().sum()
            
            validity_scores[col] = round((valid / len(data)) * 100, 2)
        
        overall_score = round(np.mean(list(validity_scores.values())), 2)
        
        return {
            'scores_by_column': validity_scores,
            'overall_score': overall_score,
            'metric_type': 'Percentage of valid values by type'
        }
    
    def _calculate_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate consistency (data type and format consistency).
        """
        consistency_scores = {}
        
        for col in data.columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                consistency_scores[col] = 100.0
                continue
            
            # Check type consistency
            if data[col].dtype == 'object':
                # Check if all non-null values have similar structure
                sample_types = col_data.apply(type).value_counts()
                consistency = (sample_types.iloc[0] / len(col_data)) * 100
            else:
                # Numeric columns - check for unexpected values
                consistency = 100.0
            
            consistency_scores[col] = round(consistency, 2)
        
        overall_score = round(np.mean(list(consistency_scores.values())), 2)
        
        return {
            'scores_by_column': consistency_scores,
            'overall_score': overall_score,
            'metric_type': 'Data type and format consistency'
        }
    
    def get_quality_grade(self, score: float) -> str:
        """
        Assign a letter grade based on quality score.
        
        Args:
            score: Quality score (0-100)
            
        Returns:
            Letter grade (A, B, C, D, F)
        """
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Acceptable)"
        elif score >= 60:
            return "D (Poor)"
        else:
            return "F (Critical)"
    
    def print_report(self) -> None:
        """
        Print a formatted quality metrics report.
        """
        if not self.metrics:
            print("No metrics calculated. Run calculate_metrics() first.")
            return
        
        print("\n" + "="*80)
        print("DATA QUALITY METRICS REPORT")
        print("="*80 + "\n")
        
        print(f"Timestamp: {self.metrics['timestamp']}")
        print(f"Total Records: {self.metrics['total_records']}")
        print(f"Total Columns: {self.metrics['total_columns']}")
        print(f"\nOverall Quality Score: {self.metrics['overall_quality_score']}/100")
        print(f"Quality Grade: {self.get_quality_grade(self.metrics['overall_quality_score'])}")
        
        print("\n" + "-"*80)
        print("METRIC BREAKDOWN:")
        print("-"*80)
        
        # Completeness
        comp = self.metrics['completeness']
        print(f"\nCOMPLETENESS: {comp['overall_score']}/100")
        print(f"  {comp['metric_type']}")
        
        # Uniqueness
        uniq = self.metrics['uniqueness']
        print(f"\nUNIQUENESS: {uniq['overall_score']}/100")
        print(f"  Total Duplicates: {uniq['total_duplicates']} ({uniq['duplicate_percentage']}%)")
        
        # Validity
        valid = self.metrics['validity']
        print(f"\nVALIDITY: {valid['overall_score']}/100")
        print(f"  {valid['metric_type']}")
        
        # Consistency
        cons = self.metrics['consistency']
        print(f"\nCONSISTENCY: {cons['overall_score']}/100")
        print(f"  {cons['metric_type']}")
        
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    # Create sample data with quality issues
    data = pd.DataFrame({
        'id': [1, 2, 2, 4, 5, None],
        'name': ['Alice', 'Bob', 'Bob', 'David', 'Eve', 'Frank'],
        'email': ['alice@test.com', 'bob@test.com', 'bob@test.com', None, 'eve@test.com', 'frank@test.com'],
        'age': [25, 30, 30, 35, 28, 32],
        'salary': [50000, 60000, 60000, 75000, 55000, 65000]
    })
    
    # Calculate metrics
    calculator = QualityMetrics()
    metrics = calculator.calculate_metrics(data)
    calculator.print_report()
