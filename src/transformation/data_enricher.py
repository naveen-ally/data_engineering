"""
Data Enricher Module
Enrich data with external sources and inferred relationships.
Adds context and derived information to raw data.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from datetime import datetime


class DataEnricher:
    """
    Enrich data with external sources and derived features.
    
    Key Features:
    - Join with external datasets
    - Calculate derived fields
    - Add reference data
    - Infer relationships
    """
    
    def __init__(self):
        """Initialize data enricher."""
        self.enrichment_log = []
        self.external_data = {}
    
    def register_external_data(self, name: str, data: pd.DataFrame,
                              key_column: str) -> None:
        """
        Register external dataset for enrichment.
        
        Args:
            name: Name of external dataset
            data: External DataFrame
            key_column: Column to join on
        """
        self.external_data[name] = {
            'data': data,
            'key_column': key_column
        }
    
    def enrich_with_external_data(self, data: pd.DataFrame, external_name: str,
                                 join_column: str, how: str = 'left') -> pd.DataFrame:
        """
        Enrich data by joining with external dataset.
        
        Args:
            data: Input DataFrame to enrich
            external_name: Name of registered external data
            join_column: Column in input data to join on
            how: Type of join ('left', 'right', 'inner', 'outer')
            
        Returns:
            Enriched DataFrame
        """
        if external_name not in self.external_data:
            raise ValueError(f"External data '{external_name}' not registered")
        
        external = self.external_data[external_name]
        key_column = external['key_column']
        
        # Perform join
        enriched = data.merge(
            external['data'],
            left_on=join_column,
            right_on=key_column,
            how=how,
            suffixes=('', f'_{external_name}')
        )
        
        # Log enrichment
        self.enrichment_log.append({
            'timestamp': datetime.now(),
            'type': 'external_join',
            'external_name': external_name,
            'original_rows': len(data),
            'enriched_rows': len(enriched),
            'new_columns': set(enriched.columns) - set(data.columns)
        })
        
        return enriched
    
    def add_derived_field(self, data: pd.DataFrame, field_name: str,
                         calculation: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
        """
        Add a derived field calculated from existing columns.
        
        Args:
            data: Input DataFrame
            field_name: Name of new field
            calculation: Function that takes DataFrame and returns Series
            
        Returns:
            DataFrame with new derived field
        """
        enriched = data.copy()
        enriched[field_name] = calculation(data)
        
        self.enrichment_log.append({
            'timestamp': datetime.now(),
            'type': 'derived_field',
            'field_name': field_name,
            'rows_affected': len(enriched)
        })
        
        return enriched
    
    def add_reference_data(self, data: pd.DataFrame, reference_name: str,
                          reference_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Add reference data (lookup values).
        
        Args:
            data: Input DataFrame
            reference_name: Name of reference field
            reference_dict: Dictionary mapping original to reference values
            
        Returns:
            DataFrame with reference data
        """
        enriched = data.copy()
        enriched[reference_name] = data.iloc[:, 0].map(reference_dict)
        
        self.enrichment_log.append({
            'timestamp': datetime.now(),
            'type': 'reference_data',
            'field_name': reference_name,
            'unique_values': len(set(reference_dict.values()))
        })
        
        return enriched
    
    def infer_customer_segment(self, data: pd.DataFrame, 
                              value_column: str) -> pd.DataFrame:
        """
        Infer customer segments based on value.
        
        Args:
            data: Input DataFrame
            value_column: Column with values to segment
            
        Returns:
            DataFrame with segment column
        """
        enriched = data.copy()
        
        # Define segments based on quantiles
        q1 = data[value_column].quantile(0.33)
        q2 = data[value_column].quantile(0.67)
        
        def segment(value):
            if value <= q1:
                return 'Low-Value'
            elif value <= q2:
                return 'Mid-Value'
            else:
                return 'High-Value'
        
        enriched['customer_segment'] = data[value_column].apply(segment)
        
        self.enrichment_log.append({
            'timestamp': datetime.now(),
            'type': 'inferred_segment',
            'field_name': 'customer_segment',
            'segments': list(enriched['customer_segment'].unique())
        })
        
        return enriched
    
    def infer_churn_risk(self, data: pd.DataFrame,
                        activity_column: str,
                        days_inactive: int = 30) -> pd.DataFrame:
        """
        Infer churn risk based on inactivity.
        
        Args:
            data: Input DataFrame
            activity_column: Column with last activity date
            days_inactive: Days to consider as churn risk
            
        Returns:
            DataFrame with churn_risk column
        """
        enriched = data.copy()
        
        # Convert to datetime if needed
        activity_dates = pd.to_datetime(enriched[activity_column], errors='coerce')
        days_since_activity = (datetime.now() - activity_dates).dt.days
        
        enriched['churn_risk'] = days_since_activity > days_inactive
        enriched['days_inactive'] = days_since_activity
        
        self.enrichment_log.append({
            'timestamp': datetime.now(),
            'type': 'inferred_churn',
            'field_name': 'churn_risk',
            'at_risk_count': enriched['churn_risk'].sum()
        })
        
        return enriched
    
    def calculate_customer_ltv(self, data: pd.DataFrame,
                              amount_column: str,
                              frequency_column: str = None) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (LTV).
        
        Args:
            data: Input DataFrame
            amount_column: Column with transaction amounts
            frequency_column: Column with purchase frequency (optional)
            
        Returns:
            DataFrame with ltv column
        """
        enriched = data.copy()
        
        # Simple LTV = total amount * frequency (or just total amount)
        if frequency_column and frequency_column in data.columns:
            enriched['ltv'] = enriched[amount_column] * enriched[frequency_column]
        else:
            enriched['ltv'] = enriched[amount_column]
        
        self.enrichment_log.append({
            'timestamp': datetime.now(),
            'type': 'calculated_ltv',
            'field_name': 'ltv',
            'average_ltv': round(enriched['ltv'].mean(), 2)
        })
        
        return enriched
    
    def print_enrichment_report(self) -> None:
        """
        Print enrichment operations report.
        """
        print("\n" + "="*80)
        print("DATA ENRICHMENT OPERATIONS REPORT")
        print("="*80 + "\n")
        
        print(f"Total Enrichment Operations: {len(self.enrichment_log)}")
        print("\nOperations:")
        
        for i, log in enumerate(self.enrichment_log, 1):
            print(f"\n{i}. {log['type'].upper()}")
            print(f"   Timestamp: {log['timestamp']}")
            for key, value in log.items():
                if key not in ['timestamp', 'type']:
                    print(f"   {key}: {value}")
        
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    # Create customer data
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'total_purchase': [1000, 2500, 500, 3000, 1500],
        'purchase_frequency': [5, 8, 3, 10, 6],
        'last_activity': ['2024-01-10', '2024-06-01', '2023-12-01', '2024-06-15', '2024-04-01']
    })
    
    # Create region data for enrichment
    regions = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'region': ['North', 'South', 'East', 'West', 'North'],
        'region_manager': ['Alice M', 'Bob M', 'Charlie M', 'David M', 'Eve M']
    })
    
    # Enrich data
    enricher = DataEnricher()
    enricher.register_external_data('regions', regions, 'customer_id')
    
    enriched = enricher.enrich_with_external_data(customers, 'regions', 'customer_id')
    enriched = enricher.calculate_customer_ltv(enriched, 'total_purchase', 'purchase_frequency')
    enriched = enricher.infer_customer_segment(enriched, 'ltv')
    enriched = enricher.infer_churn_risk(enriched, 'last_activity', days_inactive=180)
    
    print("\nEnriched Data:")
    print(enriched)
    
    enricher.print_enrichment_report()
