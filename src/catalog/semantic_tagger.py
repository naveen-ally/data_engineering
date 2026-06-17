"""
Semantic Tagger Module
Automatically tag datasets with meaningful labels using NLP.
Enables intelligent data discovery and cataloging.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import Counter
import re


class SemanticTagger:
    """
    Automatically tag datasets with meaningful semantic labels.
    
    Key Features:
    - Column name analysis
    - Content pattern recognition
    - Domain-based tagging
    - Quality assessment tags
    """
    
    def __init__(self):
        """Initialize semantic tagger."""
        self.tags = {}
        self.domain_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Initialize domain patterns for different data types.
        """
        return {
            'financial': {
                'keywords': ['amount', 'price', 'cost', 'revenue', 'profit', 'balance', 'payment'],
                'patterns': [r'\$', r'USD', r'currency']
            },
            'customer': {
                'keywords': ['customer', 'user', 'client', 'account', 'person', 'contact'],
                'patterns': [r'email', r'phone', r'address']
            },
            'temporal': {
                'keywords': ['date', 'time', 'timestamp', 'hour', 'day', 'month', 'year', 'period'],
                'patterns': [r'\d{4}-\d{2}-\d{2}', r'\d{2}:\d{2}:\d{2}']
            },
            'location': {
                'keywords': ['address', 'city', 'country', 'region', 'location', 'state', 'zip', 'postal'],
                'patterns': [r'\d{5}', r'country']
            },
            'identifier': {
                'keywords': ['id', 'code', 'key', 'identifier', 'unique'],
                'patterns': [r'_id$', r'^id_']
            },
            'measurement': {
                'keywords': ['count', 'quantity', 'amount', 'value', 'metric', 'score', 'rating'],
                'patterns': [r'\d+', r'%']
            }
        }
    
    def tag_dataset(self, data: pd.DataFrame, dataset_name: str = None) -> Dict[str, Any]:
        """
        Tag a dataset with semantic labels.
        
        Args:
            data: Input DataFrame
            dataset_name: Name of dataset
            
        Returns:
            Dictionary with dataset tags and metadata
        """
        dataset_tags = {
            'dataset_name': dataset_name or 'unknown',
            'columns': len(data.columns),
            'rows': len(data),
            'dataset_tags': self._generate_dataset_tags(data),
            'column_tags': self._tag_columns(data),
            'quality_tags': self._generate_quality_tags(data)
        }
        
        self.tags = dataset_tags
        return dataset_tags
    
    def _generate_dataset_tags(self, data: pd.DataFrame) -> List[str]:
        """
        Generate tags for the entire dataset.
        """
        tags = []
        
        # Size tags
        if len(data) < 1000:
            tags.append('small-dataset')
        elif len(data) < 1000000:
            tags.append('medium-dataset')
        else:
            tags.append('large-dataset')
        
        # Type tags
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        if len(numeric_cols) > 0:
            tags.append('numeric-data')
        if len(categorical_cols) > 0:
            tags.append('categorical-data')
        if len(datetime_cols) > 0:
            tags.append('time-series')
        
        # Domain tags
        column_names = ' '.join([col.lower() for col in data.columns])
        for domain, patterns in self.domain_patterns.items():
            if any(keyword in column_names for keyword in patterns['keywords']):
                tags.append(f'domain:{domain}')
        
        return tags
    
    def _tag_columns(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generate tags for individual columns.
        """
        column_tags = {}
        
        for col in data.columns:
            tags = []
            col_lower = col.lower()
            col_data = data[col]
            
            # Data type tags
            if col_data.dtype in ['int64', 'float64']:
                tags.append('numeric')
            elif col_data.dtype == 'object':
                tags.append('string')
            elif col_data.dtype == 'datetime64[ns]':
                tags.append('datetime')
            
            # Domain tags
            for domain, patterns in self.domain_patterns.items():
                if any(keyword in col_lower for keyword in patterns['keywords']):
                    tags.append(f'domain:{domain}')
                
                # Check patterns in data
                if col_data.dtype == 'object':
                    for pattern in patterns['patterns']:
                        if any(re.search(pattern, str(val), re.IGNORECASE) 
                               for val in col_data.dropna().head(10)):
                            tags.append(f'contains:{domain}')
                            break
            
            # Quality tags
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100
            if missing_pct > 50:
                tags.append('mostly-missing')
            elif missing_pct > 10:
                tags.append('has-missing-values')
            
            if col_data.nunique() == 1:
                tags.append('single-value')
            elif col_data.nunique() / len(col_data) > 0.95:
                tags.append('unique-values')
            
            column_tags[col] = list(set(tags))  # Remove duplicates
        
        return column_tags
    
    def _generate_quality_tags(self, data: pd.DataFrame) -> List[str]:
        """
        Generate quality assessment tags.
        """
        tags = []
        
        # Check completeness
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if completeness > 95:
            tags.append('quality:high')
        elif completeness > 80:
            tags.append('quality:medium')
        else:
            tags.append('quality:low')
        
        # Check duplicates
        if data.duplicated().any():
            dup_count = data.duplicated().sum()
            dup_pct = (dup_count / len(data)) * 100
            if dup_pct > 10:
                tags.append('caution:high-duplicates')
            else:
                tags.append('caution:some-duplicates')
        else:
            tags.append('quality:no-duplicates')
        
        return tags
    
    def search_datasets(self, tag: str) -> List[str]:
        """
        Search datasets by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching dataset names
        """
        if not self.tags:
            return []
        
        matches = []
        all_tags = (
            self.tags.get('dataset_tags', []) +
            self.tags.get('quality_tags', [])
        )
        
        if tag.lower() in [t.lower() for t in all_tags]:
            matches.append(self.tags['dataset_name'])
        
        return matches
    
    def print_tags_report(self, tags: Dict[str, Any] = None) -> None:
        """
        Print tags report.
        """
        if tags is None:
            tags = self.tags
        
        if not tags:
            print("No tags available. Run tag_dataset() first.")
            return
        
        print("\n" + "="*80)
        print("SEMANTIC TAGGING REPORT")
        print("="*80 + "\n")
        
        print(f"Dataset: {tags['dataset_name']}")
        print(f"Rows: {tags['rows']}, Columns: {tags['columns']}")
        
        print("\nDataset Tags:")
        for tag in tags['dataset_tags']:
            print(f"  - {tag}")
        
        print("\nQuality Tags:")
        for tag in tags['quality_tags']:
            print(f"  - {tag}")
        
        print("\nColumn Tags:")
        for col, col_tags in tags['column_tags'].items():
            print(f"  {col}:")
            for tag in col_tags:
                print(f"    - {tag}")
        
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    # Create sample customer dataset
    data = pd.DataFrame({
        'customer_id': range(1, 101),
        'email': [f'customer{i}@test.com' for i in range(1, 101)],
        'registration_date': pd.date_range('2024-01-01', periods=100),
        'total_purchases': np.random.randint(1, 50, 100),
        'lifetime_value': np.random.uniform(100, 5000, 100),
        'country': np.random.choice(['USA', 'Canada', 'Mexico'], 100)
    })
    
    # Tag dataset
    tagger = SemanticTagger()
    tags = tagger.tag_dataset(data, 'customer_transactions')
    tagger.print_tags_report()
