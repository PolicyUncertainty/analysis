#!/usr/bin/env python3
"""
Comprehensive testing and validation script for model specifications.

This script tests hardcoded, derived, and data-derived specs to ensure:
- All expected keys are present
- Data types are correct for JAX/numpy compatibility  
- No NaN/inf values where they shouldn't be
- Array shapes are consistent with model expectations
- Cross-validation between related specs passes

Usage: python test_specs.py
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import yaml

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs, read_and_derive_specs

print("=== SPEC VALIDATION STARTED ===")

# Color codes for output formatting
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colored_print(text: str, color: str = Colors.ENDC) -> None:
    """Print text with color formatting."""
    import os
    # Check if we're in a terminal that supports colors
    if os.name == 'nt':  # Windows
        # Just print without colors on Windows
        print(text)
    else:
        print(f"{color}{text}{Colors.ENDC}")

def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def load_all_specs(load_precomputed: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load hardcoded, derived, and data-derived specs.
    
    Returns:
        Tuple of (hardcoded_specs, derived_specs, full_specs)
    """
    try:
        # Create path dictionary
        path_dict = create_path_dict()
        colored_print("SUCCESS: Successfully created path dictionary", Colors.OKGREEN)
        
        # Load hardcoded specs from YAML
        spec_path = path_dict["specs"]
        with open(spec_path, 'r') as f:
            hardcoded_specs = yaml.safe_load(f)
        colored_print(f"SUCCESS: Successfully loaded hardcoded specs from {spec_path}", Colors.OKGREEN)
        
        # Load hardcoded + derived specs
        derived_specs = read_and_derive_specs(spec_path)
        colored_print("SUCCESS: Successfully generated derived specs", Colors.OKGREEN)
        
        # Load full specs (hardcoded + derived + data-derived)
        full_specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=load_precomputed)
        colored_print("SUCCESS: Successfully generated full specs with data-derived components", Colors.OKGREEN)
        
        return hardcoded_specs, derived_specs, full_specs
        
    except Exception as e:
        colored_print(f"ERROR: Error loading specs: {str(e)}", Colors.FAIL)
        raise

def get_spec_categories(hardcoded_specs: Dict[str, Any], derived_specs: Dict[str, Any], full_specs: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Categorize spec keys by their origin.
    
    Returns:
        Dictionary with categories: 'hardcoded', 'derived', 'data_derived'
    """
    hardcoded_keys = set(hardcoded_specs.keys())
    derived_keys = set(derived_specs.keys()) - hardcoded_keys
    data_derived_keys = set(full_specs.keys()) - set(derived_specs.keys())
    
    return {
        'hardcoded': sorted(list(hardcoded_keys)),
        'derived': sorted(list(derived_keys)),  
        'data_derived': sorted(list(data_derived_keys))
    }

def get_array_info(value: Any) -> Dict[str, Any]:
    """
    Get detailed information about arrays/tensors.
    
    Returns:
        Dictionary with shape, dtype, min, max, mean info
    """
    info = {}
    
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        info['shape'] = value.shape
        info['dtype'] = str(value.dtype)
        info['size'] = value.size
        
        # Only compute stats for numeric arrays
        if np.issubdtype(value.dtype, np.number):
            info['min'] = float(np.min(value))
            info['max'] = float(np.max(value))
            info['mean'] = float(np.mean(value))
            info['has_nan'] = bool(np.any(np.isnan(value)))
            info['has_inf'] = bool(np.any(np.isinf(value)))
        
    return info

def print_spec_inventory(specs: Dict[str, Any], categories: Dict[str, List[str]]) -> None:
    """
    Print comprehensive inventory of all specs with types, shapes, and basic statistics.
    """
    print_section_header("SPECIFICATION INVENTORY")
    
    total_specs = len(specs)
    colored_print(f"Total specifications: {total_specs}", Colors.OKBLUE)
    
    for category, keys in categories.items():
        if not keys:
            continue
            
        colored_print(f"\n{category.upper().replace('_', ' ')} SPECS ({len(keys)} items):", Colors.OKCYAN + Colors.BOLD)
        colored_print("-" * 50, Colors.OKCYAN)
        
        for key in keys:
            value = specs[key]
            type_name = type(value).__name__
            
            # Get basic info
            info_parts = [f"Type: {type_name}"]
            
            if isinstance(value, (list, tuple)):
                info_parts.append(f"Length: {len(value)}")
                if len(value) > 0:
                    element_types = set(type(x).__name__ for x in value)
                    info_parts.append(f"Elements: {', '.join(element_types)}")
            
            elif isinstance(value, dict):
                info_parts.append(f"Keys: {len(value)}")
            
            elif isinstance(value, (np.ndarray, jnp.ndarray)):
                array_info = get_array_info(value)
                info_parts.append(f"Shape: {array_info['shape']}")
                info_parts.append(f"DType: {array_info['dtype']}")
                
                if 'min' in array_info:
                    info_parts.append(f"Range: [{array_info['min']:.4g}, {array_info['max']:.4g}]")
                    if array_info['has_nan']:
                        info_parts.append("WARNING: HAS NaN")
                    if array_info['has_inf']:
                        info_parts.append("WARNING: HAS Inf")
            
            elif isinstance(value, (int, float)):
                info_parts.append(f"Value: {value}")
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    info_parts.append("WARNING: NaN/Inf")
            
            elif isinstance(value, str):
                info_parts.append(f"Value: '{value[:50]}{'...' if len(value) > 50 else ''}'")
            
            # Print the key and its info
            colored_print(f"  {key}:", Colors.OKGREEN)
            colored_print(f"    {' | '.join(info_parts)}", Colors.ENDC)
    
    colored_print(f"\nOK: Inventory complete: {total_specs} specifications processed", Colors.OKGREEN)

def validate_data_quality(specs: Dict[str, Any], categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Validate data quality across all specs.
    
    Returns:
        Dictionary with lists of issues found
    """
    print_section_header("DATA QUALITY VALIDATION")
    
    issues = {
        'nan_values': [],
        'inf_values': [],
        'negative_counts': [],
        'empty_arrays': [],
        'suspicious_values': []
    }
    
    for key, value in specs.items():
        try:
            # Check arrays/tensors
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                if value.size == 0:
                    issues['empty_arrays'].append(key)
                
                if np.issubdtype(value.dtype, np.number):
                    if np.any(np.isnan(value)):
                        issues['nan_values'].append(key)
                    
                    if np.any(np.isinf(value)):
                        issues['inf_values'].append(key)
                    
                    # Check for negative values in count-like specs
                    if any(count_word in key.lower() for count_word in ['n_', 'num_', 'count', 'size']):
                        if np.any(value < 0):
                            issues['negative_counts'].append(key)
                    
                    # Check for suspicious probability values
                    if any(prob_word in key.lower() for prob_word in ['prob', 'rate', 'share']):
                        if np.any((value < 0) | (value > 1)):
                            issues['suspicious_values'].append(f"{key} (probabilities outside [0,1])")
            
            # Check scalars
            elif isinstance(value, float):
                if np.isnan(value):
                    issues['nan_values'].append(key)
                if np.isinf(value):
                    issues['inf_values'].append(key)
                
                # Check count specs
                if any(count_word in key.lower() for count_word in ['n_', 'num_', 'count', 'size']) and value < 0:
                    issues['negative_counts'].append(key)
                
                # Check probability specs  
                if any(prob_word in key.lower() for prob_word in ['prob', 'rate', 'share']) and (value < 0 or value > 1):
                    issues['suspicious_values'].append(f"{key} (probability outside [0,1])")
        
        except Exception as e:
            issues['suspicious_values'].append(f"{key} (validation error: {str(e)})")
    
    # Report findings
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        colored_print("OK: No data quality issues found!", Colors.OKGREEN)
    else:
        colored_print(f"WARNING: Found {total_issues} potential data quality issues:", Colors.WARNING)
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                colored_print(f"\n  {issue_type.replace('_', ' ').title()}:", Colors.WARNING)
                for issue in issue_list:
                    colored_print(f"    • {issue}", Colors.FAIL)
    
    return issues


def validate_spec_consistency(specs: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate consistency between related specs.
    
    Returns:
        Dictionary with consistency issues
    """
    print_section_header("CROSS-VALIDATION OF RELATED SPECS")
    
    issues = {
        'dimension_mismatches': [],
        'label_count_mismatches': [],
        'range_inconsistencies': [],
        'missing_dependencies': []
    }
    
    try:
        # Check label counts match array dimensions
        label_checks = [
            ('choice_labels', 'n_choices'),
            ('education_labels', 'n_education_types'),
            ('health_labels', 'n_all_health_states'),
            ('partner_labels', 'n_partner_states'),
            ('sex_labels', 'n_sexes')
        ]
        
        for label_key, count_key in label_checks:
            if label_key in specs and count_key in specs:
                if len(specs[label_key]) != specs[count_key]:
                    issues['label_count_mismatches'].append(
                        f"{label_key} length ({len(specs[label_key])}) != {count_key} ({specs[count_key]})"
                    )
        
        # Check period calculations
        if all(key in specs for key in ['start_age', 'end_age', 'n_periods']):
            expected_periods = specs['end_age'] - specs['start_age'] + 1
            if specs['n_periods'] != expected_periods:
                issues['dimension_mismatches'].append(
                    f"n_periods ({specs['n_periods']}) != end_age - start_age + 1 ({expected_periods})"
                )
        
        # Check SRA grid consistency
        if all(key in specs for key in ['min_SRA', 'max_SRA', 'SRA_grid_size', 'n_policy_states']):
            expected_states = int((specs['max_SRA'] - specs['min_SRA']) / specs['SRA_grid_size']) + 1 + 1
            if specs['n_policy_states'] != expected_states:
                issues['dimension_mismatches'].append(
                    f"n_policy_states ({specs['n_policy_states']}) != calculated states ({expected_states})"
                )
        
        # Check transition matrix dimensions
        matrix_checks = [
            ('health_trans_mat', ['n_all_health_states', 'n_periods']),
            ('partner_trans_mat', ['n_partner_states'])
        ]
        
        for matrix_key, dim_keys in matrix_checks:
            if matrix_key in specs and isinstance(specs[matrix_key], (np.ndarray, jnp.ndarray)):
                matrix = specs[matrix_key]
                for i, dim_key in enumerate(dim_keys):
                    if dim_key in specs:
                        expected_dim = specs[dim_key]
                        if i < len(matrix.shape) and matrix.shape[i] != expected_dim:
                            issues['dimension_mismatches'].append(
                                f"{matrix_key} shape[{i}] ({matrix.shape[i]}) != {dim_key} ({expected_dim})"
                            )
        
        # Check age ranges
        age_ranges = [
            ('start_age', 'end_age'),
            ('min_SRA', 'max_SRA'),
            ('start_age', 'max_est_age_labor'),
            ('min_long_insured_age', 'max_ret_age')
        ]
        
        for min_key, max_key in age_ranges:
            if min_key in specs and max_key in specs:
                if specs[min_key] >= specs[max_key]:
                    issues['range_inconsistencies'].append(
                        f"{min_key} ({specs[min_key]}) >= {max_key} ({specs[max_key]})"
                    )
        
        # Check for missing critical specs
        critical_specs = [
            'n_periods', 'n_choices', 'n_education_types', 'start_age', 'end_age',
            'discount_factor', 'interest_rate'
        ]
        
        for spec in critical_specs:
            if spec not in specs:
                issues['missing_dependencies'].append(spec)
    
    except Exception as e:
        issues['missing_dependencies'].append(f"Validation error: {str(e)}")
    
    # Report findings
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        colored_print("OK: All cross-validation checks passed!", Colors.OKGREEN)
    else:
        colored_print(f"WARNING: Found {total_issues} consistency issues:", Colors.WARNING)
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                colored_print(f"\n  {issue_type.replace('_', ' ').title()}:", Colors.WARNING)
                for issue in issue_list:
                    colored_print(f"    • {issue}", Colors.FAIL)
    
    return issues

def validate_jax_compatibility(specs: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate that specs are compatible with JAX operations.
    
    Returns:
        Dictionary with compatibility issues
    """
    print_section_header("JAX/NUMPY TYPE VALIDATION")
    
    issues = {
        'non_jax_arrays': [],
        'incompatible_dtypes': [],
        'object_arrays': [],
        'mixed_precision': []
    }
    
    expected_array_specs = [
        'health_trans_mat', 'partner_trans_mat', 'income_trans_mat',
        'informed_hazard_rate', 'uninformed_ERP', 'job_sep_probs',
        'log_job_sep_probs', 'wage_params', 'policy_states_trans_mat'
    ]
    
    float_dtypes = [np.float32, np.float64, jnp.float32, jnp.float64]
    int_dtypes = [np.int32, np.int64, np.uint8, np.uint32, jnp.int32, jnp.int64, jnp.uint8, jnp.uint32]
    
    for key, value in specs.items():
        try:
            if isinstance(value, (list, tuple)):
                # Check if this should be an array
                if key in expected_array_specs or any(word in key.lower() for word in ['trans', 'mat', 'params', 'rate', 'prob']):
                    issues['non_jax_arrays'].append(f"{key} (should be array, is {type(value).__name__})")
            
            elif isinstance(value, np.ndarray):
                # Check if numpy array should be JAX array for performance
                if key in expected_array_specs:
                    issues['non_jax_arrays'].append(f"{key} (numpy array, consider JAX array for performance)")
                
                # Check for object arrays
                if value.dtype == np.object_:
                    issues['object_arrays'].append(key)
                
                # Check dtype compatibility
                if value.dtype not in float_dtypes + int_dtypes and np.issubdtype(value.dtype, np.number):
                    issues['incompatible_dtypes'].append(f"{key} (dtype: {value.dtype})")
            
            elif isinstance(value, jnp.ndarray):
                # Check dtype compatibility  
                if value.dtype not in float_dtypes + int_dtypes and np.issubdtype(value.dtype, np.number):
                    issues['incompatible_dtypes'].append(f"{key} (dtype: {value.dtype})")
        
        except Exception as e:
            issues['incompatible_dtypes'].append(f"{key} (validation error: {str(e)})")
    
    # Report findings
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        colored_print("OK: All specs are JAX/numpy compatible!", Colors.OKGREEN)
    else:
        colored_print(f"WARNING: Found {total_issues} type compatibility issues:", Colors.WARNING)
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                colored_print(f"\n  {issue_type.replace('_', ' ').title()}:", Colors.WARNING)
                for issue in issue_list:
                    colored_print(f"    • {issue}", Colors.FAIL)
    
    return issues


def validate_spec_consistency(specs: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate consistency between related specs.
    
    Returns:
        Dictionary with consistency issues
    """
    print_section_header("CROSS-VALIDATION OF RELATED SPECS")
    
    issues = {
        'dimension_mismatches': [],
        'label_count_mismatches': [],
        'range_inconsistencies': [],
        'missing_dependencies': []
    }
    
    try:
        # Check label counts match array dimensions
        label_checks = [
            ('choice_labels', 'n_choices'),
            ('education_labels', 'n_education_types'),
            ('health_labels', 'n_all_health_states'),
            ('partner_labels', 'n_partner_states'),
            ('sex_labels', 'n_sexes')
        ]
        
        for label_key, count_key in label_checks:
            if label_key in specs and count_key in specs:
                if len(specs[label_key]) != specs[count_key]:
                    issues['label_count_mismatches'].append(
                        f"{label_key} length ({len(specs[label_key])}) != {count_key} ({specs[count_key]})"
                    )
        
        # Check period calculations
        if all(key in specs for key in ['start_age', 'end_age', 'n_periods']):
            expected_periods = specs['end_age'] - specs['start_age'] + 1
            if specs['n_periods'] != expected_periods:
                issues['dimension_mismatches'].append(
                    f"n_periods ({specs['n_periods']}) != end_age - start_age + 1 ({expected_periods})"
                )
        
        # Check SRA grid consistency
        if all(key in specs for key in ['min_SRA', 'max_SRA', 'SRA_grid_size', 'n_policy_states']):
            expected_states = int((specs['max_SRA'] - specs['min_SRA']) / specs['SRA_grid_size']) + 1 + 1
            if specs['n_policy_states'] != expected_states:
                issues['dimension_mismatches'].append(
                    f"n_policy_states ({specs['n_policy_states']}) != calculated states ({expected_states})"
                )
        
        # Check transition matrix dimensions
        matrix_checks = [
            ('health_trans_mat', ['n_all_health_states', 'n_periods']),
            ('partner_trans_mat', ['n_partner_states'])
        ]
        
        for matrix_key, dim_keys in matrix_checks:
            if matrix_key in specs and isinstance(specs[matrix_key], (np.ndarray, jnp.ndarray)):
                matrix = specs[matrix_key]
                for i, dim_key in enumerate(dim_keys):
                    if dim_key in specs:
                        expected_dim = specs[dim_key]
                        if i < len(matrix.shape) and matrix.shape[i] != expected_dim:
                            issues['dimension_mismatches'].append(
                                f"{matrix_key} shape[{i}] ({matrix.shape[i]}) != {dim_key} ({expected_dim})"
                            )
        
        # Check age ranges
        age_ranges = [
            ('start_age', 'end_age'),
            ('min_SRA', 'max_SRA'),
            ('start_age', 'max_est_age_labor'),
            ('min_long_insured_age', 'max_ret_age')
        ]
        
        for min_key, max_key in age_ranges:
            if min_key in specs and max_key in specs:
                if specs[min_key] >= specs[max_key]:
                    issues['range_inconsistencies'].append(
                        f"{min_key} ({specs[min_key]}) >= {max_key} ({specs[max_key]})"
                    )
        
        # Check for missing critical specs
        critical_specs = [
            'n_periods', 'n_choices', 'n_education_types', 'start_age', 'end_age',
            'discount_factor', 'interest_rate'
        ]
        
        for spec in critical_specs:
            if spec not in specs:
                issues['missing_dependencies'].append(spec)
    
    except Exception as e:
        issues['missing_dependencies'].append(f"Validation error: {str(e)}")
    
    # Report findings
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        colored_print("OK: All cross-validation checks passed!", Colors.OKGREEN)
    else:
        colored_print(f"WARNING: Found {total_issues} consistency issues:", Colors.WARNING)
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                colored_print(f"\n  {issue_type.replace('_', ' ').title()}:", Colors.WARNING)
                for issue in issue_list:
                    colored_print(f"    • {issue}", Colors.FAIL)
    
    return issues

def validate_model_compatibility(specs):
    """
    Validate that specs are structured for model function compatibility.
    
    Returns:
        Dictionary with model compatibility issues
    """
    print_section_header("MODEL COMPATIBILITY CHECKS")
    
    issues = {
        "missing_model_specs": [],
        "wrong_lookup_dimensions": [],
        "incompatible_indexing": [],
        "transition_matrix_issues": []
    }
    
    # Check for critical model specs based on usage patterns we found
    critical_model_specs = {
        "informed_hazard_rate": "array indexed by education",
        "uninformed_ERP": "array indexed by education", 
        "n_policy_states": "integer for state space",
        "policy_step_periods": "array for policy transitions",
        "ERP": "float for penalty calculation",
        "disabled_health_var": "integer for health state lookup",
        "death_health_var": "integer for health state lookup",
        "good_health_var": "integer for health state lookup",
        "bad_health_var": "integer for health state lookup",
        "experience_threshold_very_long_insured": "array indexed by sex",
        "years_before_SRA_long_insured": "integer for age calculations",
        "min_long_insured_age": "integer for age calculations"
    }
    
    for spec_key, expected_type in critical_model_specs.items():
        if spec_key not in specs:
            issues["missing_model_specs"].append(f"{spec_key} ({expected_type})")
    
    # Report findings
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        colored_print("OK: All model compatibility checks passed!", Colors.OKGREEN)
    else:
        colored_print(f"WARNING: Found {total_issues} model compatibility issues:", Colors.WARNING)
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                colored_print(f"\\n  {issue_type.replace('_', ' ').title()}:", Colors.WARNING)
                for issue in issue_list:
                    colored_print(f"    • {issue}", Colors.FAIL)
    
    return issues

def main():
    """
    Main function to run all specification tests.
    """
    print_section_header("SPECIFICATION TESTING AND VALIDATION")
    colored_print("Starting comprehensive spec validation...", Colors.OKBLUE)
    
    try:
        # Load all specs
        colored_print("\n1. Loading specifications...", Colors.OKCYAN)
        hardcoded_specs, derived_specs, full_specs = load_all_specs(load_precomputed=True)
        
        # Categorize specs
        categories = get_spec_categories(hardcoded_specs, derived_specs, full_specs)
        
        # Run all validations
        colored_print("\n2. Running validation tests...", Colors.OKCYAN)
        
        # Inventory
        print_spec_inventory(full_specs, categories)
        
        # Data quality validation
        data_quality_issues = validate_data_quality(full_specs, categories)
        
        # JAX compatibility
        jax_issues = validate_jax_compatibility(full_specs)
        
        # Cross-validation
        consistency_issues = validate_spec_consistency(full_specs)
        
        # Model compatibility
        model_issues = validate_model_compatibility(full_specs)
        
        # Summary report
        print_section_header("VALIDATION SUMMARY")
        
        all_issues = {
            "Data Quality": data_quality_issues,
            "JAX Compatibility": jax_issues,
            "Consistency": consistency_issues,
            "Model Compatibility": model_issues
        }
        
        total_issues = 0
        for test_name, issue_dict in all_issues.items():
            test_issue_count = sum(len(issue_list) for issue_list in issue_dict.values())
            total_issues += test_issue_count
            
            if test_issue_count == 0:
                colored_print(f"OK: {test_name}: PASSED", Colors.OKGREEN)
            else:
                colored_print(f"WARNING: {test_name}: {test_issue_count} issues", Colors.WARNING)
        
        colored_print(f"\n{'='*60}", Colors.HEADER)
        if total_issues == 0:
            colored_print("SUCCESS: ALL TESTS PASSED! Specifications are ready for use.", Colors.OKGREEN + Colors.BOLD)
        else:
            colored_print(f"WARNING: FOUND {total_issues} TOTAL ISSUES that need attention.", Colors.WARNING + Colors.BOLD)
            colored_print("Review the detailed output above to fix these issues.", Colors.WARNING)
        
        colored_print(f"{'='*60}", Colors.HEADER)
        
        return total_issues == 0
        
    except Exception as e:
        colored_print(f"ERROR: Critical error during validation: {str(e)}", Colors.FAIL)
        import traceback
        colored_print(traceback.format_exc(), Colors.FAIL)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

