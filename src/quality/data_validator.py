"""
Data Validator Module
Validate data quality and integrity using rule-based and ML-based approaches.
Automates data validation at scale with intelligent rules.

Based on AI-driven data engineering patterns from:
https://medium.com/area-21/how-ai-is-reshaping-data-engineering-a-practical-guide-with-examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum


class ValidationStatus(Enum):
    """Validation result statuses."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationRule:
    """Represents a single validation rule."""
    name: str
    column: str
    rule_type: str  # 'required', 'range', 'pattern', 'custom'
    rule_value: Any = None
    error_message: str = None


class DataValidator:
    """
    Validates data quality and integrity.
    
    Key Features:
    - Pre-defined validation rules (required, range, pattern)
    - Custom validation functions
    - Batch validation with detailed reporting
    - Data profiling and quality metrics
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.rules: List[ValidationRule] = []
        self.validation_results = {}
    
    def add_required_rule(self, column: str, error_message: str = None) -> None:
        """
        Add a rule to check if column has no null values.
        
        Args:
            column: Column name
            error_message: Custom error message
        """
        if error_message is None:
            error_message = f"Column '{column}' contains null values"
        
        rule = ValidationRule(
            name=f"required_{column}",
            column=column,
            rule_type="required",
            error_message=error_message
        )
        self.rules.append(rule)
    
    def add_range_rule(self, column: str, min_val: float = None, max_val: float = None,
                      error_message: str = None) -> None:
        """
        Add a rule to check if column values are within range.
        
        Args:
            column: Column name
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            error_message: Custom error message
        """
        if error_message is None:
            error_message = f"Column '{column}' values out of range [{min_val}, {max_val}]"
        
        rule = ValidationRule(
            name=f"range_{column}",
            column=column,
            rule_type="range",
            rule_value={"min": min_val, "max": max_val},
            error_message=error_message
        )
        self.rules.append(rule)
    
    def add_pattern_rule(self, column: str, pattern: str, error_message: str = None) -> None:
        """
        Add a rule to check if column values match a regex pattern.
        
        Args:
            column: Column name
            pattern: Regex pattern
            error_message: Custom error message
        """
        if error_message is None:
            error_message = f"Column '{column}' values don't match pattern '{pattern}'"
        
        rule = ValidationRule(
            name=f"pattern_{column}",
            column=column,
            rule_type="pattern",
            rule_value=pattern,
            error_message=error_message
        )
        self.rules.append(rule)
    
    def add_custom_rule(self, column: str, validator_func: Callable, 
                       error_message: str = None) -> None:
        """
        Add a custom validation rule.
        
        Args:
            column: Column name
            validator_func: Function that takes Series and returns boolean Series
            error_message: Custom error message
        """
        if error_message is None:
            error_message = f"Column '{column}' failed custom validation"
        
        rule = ValidationRule(
            name=f"custom_{column}",
            column=column,
            rule_type="custom",
            rule_value=validator_func,
            error_message=error_message
        )
        self.rules.append(rule)
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation rules against the data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_records': len(data),
            'passed': True,
            'rules_passed': 0,
            'rules_failed': 0,
            'rule_results': []
        }
        
        for rule in self.rules:
            rule_result = self._apply_rule(data, rule)
            results['rule_results'].append(rule_result)
            
            if rule_result['status'] == ValidationStatus.PASSED.value:
                results['rules_passed'] += 1
            else:
                results['rules_failed'] += 1
                results['passed'] = False
        
        self.validation_results = results
        return results
    
    def _apply_rule(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """
        Apply a single validation rule.
        """
        if rule.column not in data.columns:
            return {
                'rule_name': rule.name,
                'status': ValidationStatus.FAILED.value,
                'message': f"Column '{rule.column}' not found",
                'failed_records': len(data)
            }
        
        column_data = data[rule.column]
        
        if rule.rule_type == "required":
            failed = column_data.isnull().sum()
            status = ValidationStatus.PASSED.value if failed == 0 else ValidationStatus.FAILED.value
            return {
                'rule_name': rule.name,
                'status': status,
                'message': rule.error_message,
                'failed_records': int(failed)
            }
        
        elif rule.rule_type == "range":
            min_val = rule.rule_value.get('min')
            max_val = rule.rule_value.get('max')
            
            if min_val is not None and max_val is not None:
                failed = ((column_data < min_val) | (column_data > max_val)).sum()
            elif min_val is not None:
                failed = (column_data < min_val).sum()
            else:
                failed = (column_data > max_val).sum()
            
            status = ValidationStatus.PASSED.value if failed == 0 else ValidationStatus.FAILED.value
            return {
                'rule_name': rule.name,
                'status': status,
                'message': rule.error_message,
                'failed_records': int(failed)
            }
        
        elif rule.rule_type == "pattern":
            pattern = rule.rule_value
            valid = column_data.astype(str).str.match(pattern)
            failed = (~valid).sum()
            status = ValidationStatus.PASSED.value if failed == 0 else ValidationStatus.FAILED.value
            return {
                'rule_name': rule.name,
                'status': status,
                'message': rule.error_message,
                'failed_records': int(failed)
            }
        
        elif rule.rule_type == "custom":
            validator = rule.rule_value
            try:
                valid = validator(column_data)
                failed = (~valid).sum()
                status = ValidationStatus.PASSED.value if failed == 0 else ValidationStatus.FAILED.value
            except Exception as e:
                return {
                    'rule_name': rule.name,
                    'status': ValidationStatus.FAILED.value,
                    'message': f"Custom validation error: {str(e)}",
                    'failed_records': -1
                }
            
            return {
                'rule_name': rule.name,
                'status': status,
                'message': rule.error_message,
                'failed_records': int(failed)
            }
    
    def print_report(self) -> None:
        """
        Print a formatted validation report.
        """
        if not self.validation_results:
            print("No validation results available. Run validate() first.")
            return
        
        results = self.validation_results
        print("\n" + "="*80)
        print("DATA VALIDATION REPORT")
        print("="*80 + "\n")
        
        print(f"Total Records: {results['total_records']}")
        print(f"Overall Status: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")
        print(f"Rules Passed: {results['rules_passed']}")
        print(f"Rules Failed: {results['rules_failed']}")
        print("\nDetailed Results:")
        print("-" * 80)
        
        for rule_result in results['rule_results']:
            status_symbol = "✓" if rule_result['status'] == "passed" else "✗"
            print(f"{status_symbol} {rule_result['rule_name']}")
            print(f"  Status: {rule_result['status']}")
            print(f"  Message: {rule_result['message']}")
            print(f"  Failed Records: {rule_result['failed_records']}")
            print()


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, None],
        'email': ['user1@test.com', 'user2@test.com', 'invalid-email', 'user4@test.com', 'user5@test.com'],
        'age': [25, 30, 35, 150, 28],
        'score': [85.5, 90.0, 78.5, 95.0, 88.0]
    })
    
    # Create validator and add rules
    validator = DataValidator()
    validator.add_required_rule('user_id')
    validator.add_pattern_rule('email', r'^[\w\.-]+@[\w\.-]+\.\w+$')
    validator.add_range_rule('age', min_val=0, max_val=120)
    validator.add_range_rule('score', min_val=0, max_val=100)
    
    # Validate
    results = validator.validate(data)
    validator.print_report()
