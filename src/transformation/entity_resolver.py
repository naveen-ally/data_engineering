"""
Entity Resolver Module
Match and merge entities across datasets using fuzzy matching.
Automatically resolves duplicate and related records.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')


class EntityResolver:
    """
    Resolve entities across datasets using fuzzy matching.
    
    Key Features:
    - Fuzzy string matching
    - Record linkage and deduplication
    - Entity merging and consolidation
    - Similarity scoring
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize entity resolver.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.entity_groups = []
        self.matches = []
    
    def fuzzy_match(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        if not isinstance(str1, str) or not isinstance(str2, str):
            return 0.0
        
        str1 = str(str1).lower().strip()
        str2 = str(str2).lower().strip()
        
        if str1 == str2:
            return 1.0
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, str1, str2)
        return matcher.ratio()
    
    def find_duplicates(self, data: pd.DataFrame, key_column: str) -> List[List[int]]:
        """
        Find duplicate records based on a key column.
        
        Args:
            data: DataFrame to check
            key_column: Column to check for duplicates
            
        Returns:
            List of groups of similar indices
        """
        duplicates = []
        processed = set()
        
        for i in range(len(data)):
            if i in processed:
                continue
            
            group = [i]
            processed.add(i)
            
            for j in range(i + 1, len(data)):
                if j in processed:
                    continue
                
                similarity = self.fuzzy_match(
                    str(data.iloc[i][key_column]),
                    str(data.iloc[j][key_column])
                )
                
                if similarity >= self.similarity_threshold:
                    group.append(j)
                    processed.add(j)
            
            if len(group) > 1:
                duplicates.append(group)
        
        self.entity_groups = duplicates
        return duplicates
    
    def merge_records(self, data: pd.DataFrame, record_indices: List[int],
                      merge_strategy: str = 'first') -> pd.Series:
        """
        Merge multiple records into one.
        
        Args:
            data: DataFrame containing records
            record_indices: Indices of records to merge
            merge_strategy: 'first', 'last', 'mode', 'mean'
            
        Returns:
            Merged record as Series
        """
        records = data.iloc[record_indices]
        merged = pd.Series(dtype=object)
        
        for col in data.columns:
            if merge_strategy == 'first':
                merged[col] = records[col].iloc[0]
            elif merge_strategy == 'last':
                merged[col] = records[col].iloc[-1]
            elif merge_strategy == 'mode':
                # Use mode for categorical, mean for numeric
                if records[col].dtype == 'object':
                    merged[col] = records[col].mode()[0] if len(records[col].mode()) > 0 else records[col].iloc[0]
                else:
                    merged[col] = records[col].mean()
            elif merge_strategy == 'mean':
                if records[col].dtype in ['int64', 'float64']:
                    merged[col] = records[col].mean()
                else:
                    merged[col] = records[col].iloc[0]
        
        return merged
    
    def deduplicate(self, data: pd.DataFrame, key_column: str,
                   merge_strategy: str = 'first') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicates and merge similar records.
        
        Args:
            data: Input DataFrame
            key_column: Column to use for deduplication
            merge_strategy: How to merge duplicate records
            
        Returns:
            Tuple of (deduplicated_data, report)
        """
        duplicates = self.find_duplicates(data, key_column)
        
        if len(duplicates) == 0:
            return data, {'duplicates_found': 0, 'records_removed': 0}
        
        # Start with original data
        result = data.copy()
        merged_records = []
        indices_to_remove = set()
        
        # Merge duplicate groups
        for group in duplicates:
            merged = self.merge_records(result, group, merge_strategy)
            merged_records.append(merged)
            
            # Mark all but first for removal
            for idx in group[1:]:
                indices_to_remove.add(idx)
        
        # Remove duplicates and add merged records
        result = result.drop(list(indices_to_remove))
        merged_df = pd.DataFrame(merged_records)
        result = pd.concat([result, merged_df], ignore_index=True)
        
        report = {
            'original_records': len(data),
            'duplicates_found': len(duplicates),
            'records_removed': len(indices_to_remove),
            'final_records': len(result),
            'deduplication_rate': round((len(indices_to_remove) / len(data)) * 100, 2)
        }
        
        return result, report
    
    def link_records(self, data1: pd.DataFrame, data2: pd.DataFrame,
                     key_col1: str, key_col2: str) -> pd.DataFrame:
        """
        Link records between two datasets.
        
        Args:
            data1: First DataFrame
            data2: Second DataFrame
            key_col1: Key column in first DataFrame
            key_col2: Key column in second DataFrame
            
        Returns:
            Linked records with similarity scores
        """
        matches = []
        
        for i, row1 in data1.iterrows():
            for j, row2 in data2.iterrows():
                similarity = self.fuzzy_match(
                    str(row1[key_col1]),
                    str(row2[key_col2])
                )
                
                if similarity >= self.similarity_threshold:
                    match = {
                        f'{key_col1}_idx': i,
                        f'{key_col1}_value': row1[key_col1],
                        f'{key_col2}_idx': j,
                        f'{key_col2}_value': row2[key_col2],
                        'similarity_score': round(similarity, 3)
                    }
                    matches.append(match)
        
        self.matches = matches
        return pd.DataFrame(matches)
    
    def print_deduplication_report(self, report: Dict[str, Any]) -> None:
        """
        Print deduplication report.
        """
        print("\n" + "="*80)
        print("ENTITY RESOLUTION & DEDUPLICATION REPORT")
        print("="*80 + "\n")
        
        print(f"Original Records: {report['original_records']}")
        print(f"Duplicates Found: {report['duplicates_found']}")
        print(f"Records Removed: {report['records_removed']}")
        print(f"Final Records: {report['final_records']}")
        print(f"Deduplication Rate: {report['deduplication_rate']}%")
        print(f"Data Reduction: {100 - ((report['final_records'] / report['original_records']) * 100):.2f}%")


# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates
    data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John Smith', 'jane doe', 'John Smyth', 'Jane Doe', 'Alice Johnson'],
        'email': ['john@test.com', 'jane@test.com', 'john.s@test.com', 'jane@test.com', 'alice@test.com']
    })
    
    resolver = EntityResolver(similarity_threshold=0.85)
    deduplicated, report = resolver.deduplicate(data, 'name', merge_strategy='first')
    
    print("Original Data:")
    print(data)
    print("\nDeduplicated Data:")
    print(deduplicated)
    
    resolver.print_deduplication_report(report)
