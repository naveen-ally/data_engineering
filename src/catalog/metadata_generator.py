"""
Metadata Generator Module
Automatically generate metadata for datasets.
Creates comprehensive data documentation and lineage tracking.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import json


class MetadataGenerator:
    """
    Automatically generate comprehensive metadata for datasets.
    
    Key Features:
    - Schema metadata generation
    - Data profiling
    - Lineage tracking
    - Quality metrics
    - Auto-generated documentation
    """
    
    def __init__(self):
        """Initialize metadata generator."""
        self.metadata = {}
        self.lineage = []
    
    def generate_metadata(self, data: pd.DataFrame, dataset_name: str,
                         description: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for a dataset.
        
        Args:
            data: Input DataFrame
            dataset_name: Name of dataset
            description: Dataset description
            
        Returns:
            Dictionary with complete metadata
        """
        metadata = {
            'dataset_name': dataset_name,
            'description': description or 'No description provided',
            'generated_at': datetime.now().isoformat(),
            'row_count': len(data),
            'column_count': len(data.columns),
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2),
            'columns': self._generate_column_metadata(data),
            'statistics': self._generate_statistics(data),
            'quality_score': self._calculate_quality_score(data)
        }
        
        self.metadata = metadata
        return metadata
    
    def _generate_column_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate metadata for each column.
        """
        columns_meta = {}
        
        for col in data.columns:
            col_data = data[col]
            columns_meta[col] = {
                'name': col,
                'data_type': str(col_data.dtype),
                'nullable': bool(col_data.isnull().any()),
                'missing_count': int(col_data.isnull().sum()),
                'missing_percentage': round((col_data.isnull().sum() / len(data)) * 100, 2),
                'unique_count': int(col_data.nunique()),
                'unique_percentage': round((col_data.nunique() / len(data)) * 100, 2),
                'min_value': self._safe_min(col_data),
                'max_value': self._safe_max(col_data),
                'mean_value': self._safe_mean(col_data),
                'median_value': self._safe_median(col_data),
                'std_dev': self._safe_std(col_data),
                'sample_values': col_data.dropna().head(3).tolist()
            }
        
        return columns_meta
    
    def _generate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate overall dataset statistics.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        return {
            'numeric_columns': list(numeric_cols),
            'categorical_columns': list(categorical_cols),
            'datetime_columns': list(datetime_cols),
            'total_missing_values': int(data.isnull().sum().sum()),
            'total_duplicates': int(data.duplicated().sum()),
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2)
        }
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall data quality score.
        """
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        
        duplicate_ratio = (data.duplicated().sum() / len(data)) * 100 if len(data) > 0 else 0
        uniqueness = max(0, 100 - duplicate_ratio)
        
        overall_score = (completeness * 0.5 + uniqueness * 0.5)
        
        return {
            'completeness_score': round(completeness, 2),
            'uniqueness_score': round(uniqueness, 2),
            'overall_quality_score': round(overall_score, 2)
        }
    
    def track_lineage(self, source_dataset: str, target_dataset: str,
                     transformation: str, details: Dict[str, Any] = None) -> None:
        """
        Track data lineage and transformations.
        
        Args:
            source_dataset: Source dataset name
            target_dataset: Target dataset name
            transformation: Type of transformation
            details: Additional details about transformation
        """
        lineage_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source_dataset,
            'target': target_dataset,
            'transformation': transformation,
            'details': details or {}
        }
        
        self.lineage.append(lineage_entry)
    
    def export_metadata_json(self, filepath: str) -> None:
        """
        Export metadata to JSON file.
        
        Args:
            filepath: Path to save metadata JSON
        """
        with open(filepath, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def generate_data_dictionary(self, data: pd.DataFrame) -> str:
        """
        Generate a data dictionary document.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Formatted data dictionary as string
        """
        doc = "# Data Dictionary\n\n"
        doc += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        doc += f"**Total Records:** {len(data)}\n"
        doc += f"**Total Columns:** {len(data.columns)}\n\n"
        
        doc += "## Column Definitions\n\n"
        
        for col, meta in self.metadata.get('columns', {}).items():
            doc += f"### {col}\n"
            doc += f"- **Type:** {meta['data_type']}\n"
            doc += f"- **Nullable:** {meta['nullable']}\n"
            doc += f"- **Missing:** {meta['missing_count']} ({meta['missing_percentage']}%)\n"
            doc += f"- **Unique Values:** {meta['unique_count']}\n"
            
            if meta['min_value'] is not None:
                doc += f"- **Range:** {meta['min_value']} to {meta['max_value']}\n"
            
            if meta['mean_value'] is not None:
                doc += f"- **Mean:** {meta['mean_value']:.2f}\n"
            
            doc += f"- **Sample Values:** {meta['sample_values']}\n\n"
        
        return doc
    
    def print_metadata_report(self) -> None:
        """
        Print metadata report.
        """
        if not self.metadata:
            print("No metadata available. Run generate_metadata() first.")
            return
        
        print("\n" + "="*80)
        print("METADATA REPORT")
        print("="*80 + "\n")
        
        print(f"Dataset: {self.metadata['dataset_name']}")
        print(f"Description: {self.metadata['description']}")
        print(f"Generated: {self.metadata['generated_at']}")
        print(f"\nDimensions: {self.metadata['row_count']} rows × {self.metadata['column_count']} columns")
        print(f"Memory Usage: {self.metadata['memory_usage_mb']} MB")
        
        print("\nData Quality:")
        quality = self.metadata['quality_score']
        print(f"  Completeness: {quality['completeness_score']}/100")
        print(f"  Uniqueness: {quality['uniqueness_score']}/100")
        print(f"  Overall Score: {quality['overall_quality_score']}/100")
    
    @staticmethod
    def _safe_min(series):
        try:
            val = series.min()
            return float(val) if pd.notna(val) else None
        except:
            return None
    
    @staticmethod
    def _safe_max(series):
        try:
            val = series.max()
            return float(val) if pd.notna(val) else None
        except:
            return None
    
    @staticmethod
    def _safe_mean(series):
        try:
            val = series.mean()
            return float(val) if pd.notna(val) else None
        except:
            return None
    
    @staticmethod
    def _safe_median(series):
        try:
            val = series.median()
            return float(val) if pd.notna(val) else None
        except:
            return None
    
    @staticmethod
    def _safe_std(series):
        try:
            val = series.std()
            return float(val) if pd.notna(val) else None
        except:
            return None


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = pd.DataFrame({
        'customer_id': range(1, 101),
        'name': [f'Customer_{i}' for i in range(1, 101)],
        'email': [f'customer{i}@test.com' for i in range(1, 101)],
        'registration_date': pd.date_range('2024-01-01', periods=100),
        'purchase_amount': np.random.uniform(100, 5000, 100),
        'purchase_count': np.random.randint(1, 50, 100)
    })
    
    # Generate metadata
    generator = MetadataGenerator()
    metadata = generator.generate_metadata(data, 'customer_database', 'Customer transaction data')
    
    generator.print_metadata_report()
    
    # Generate data dictionary
    dictionary = generator.generate_data_dictionary(data)
    print("\n" + dictionary)
