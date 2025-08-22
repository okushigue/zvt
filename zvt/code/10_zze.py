#!/usr/bin/env python3
"""
zvt_expanded_experiments.py - Expansion of ZVT experiments
Tests additional combinations and more complex patterns
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Additional constants from your reports
ADDITIONAL_CONSTANTS = {
    'speed_of_light': 299792458.0,
    'elementary_charge': 1.602176634e-19,
    'bohr_radius': 5.29177210903e-11,
    'mu_0': 1.25663706212e-06,
    'vacuum_permittivity': 8.8541878128e-12,
    'rydberg': 10973731.56816,
    'avogadro': 6.02214076e23,
    'boltzmann': 1.380649e-23,
    'stefan_boltzmann': 5.670374419e-08,
}

# Additional zeros (based on your reports)
ADDITIONAL_ZEROS = {
    'speed_of_light_zero': 299792.458,  # Scaled value
    'elementary_charge_zero': 160217.6634,  # Scaled value
    'bohr_radius_zero': 52917.72109,  # Scaled value
    'mu_0_zero': 125663.706212,  # Scaled value
    'vacuum_permittivity_zero': 88541.878128,  # Scaled value
}

print("Expanded ZVT Experiments")
print("="*50)
print(f"Testing {len(ADDITIONAL_CONSTANTS)} additional constants")
print(f"With {len(ADDITIONAL_ZEROS)} corresponding zeros")

# Theoretical scale analysis
print("\nScale Analysis:")
print("-" * 50)

for const_name, const_value in ADDITIONAL_CONSTANTS.items():
    if const_name in ADDITIONAL_ZEROS:
        zero_value = ADDITIONAL_ZEROS[const_name]
        log_ratio = np.log10(zero_value / abs(const_value))
        print(f"{const_name:20} | log₁₀(zero/const) = {log_ratio:6.2f}")

# Identify patterns
print("\nIdentified Patterns:")
print("-" * 50)

# Group by order of magnitude
groups = {}
for const_name, const_value in ADDITIONAL_CONSTANTS.items():
    order = int(np.log10(abs(const_value)))
    if order not in groups:
        groups[order] = []
    groups[order].append(const_name)

for order in sorted(groups.keys()):
    print(f"Order 10^{order}: {', '.join(groups[order])}")

# Save analysis
analysis = {
    'timestamp': datetime.now().isoformat(),
    'additional_constants': ADDITIONAL_CONSTANTS,
    'additional_zeros': ADDITIONAL_ZEROS,
    'magnitude_groups': groups,
    'suggested_experiments': []
}

# Suggest promising experiments
print("\nSuggested Experiments:")
print("-" * 50)

for const_name in ADDITIONAL_CONSTANTS.keys():
    if const_name in ADDITIONAL_ZEROS:
        analysis['suggested_experiments'].append({
            'constant': const_name,
            'zero': f"{const_name}_zero",
            'priority': 'high' if 'speed' in const_name or 'elementary' in const_name else 'medium'
        })
        print(f"- {const_name} + {const_name}_zero (priority: {'high' if 'speed' in const_name or 'elementary' in const_name else 'medium'})")

with open('zvt_expanded_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print("\nAnalysis saved in 'zvt_expanded_analysis.json'")
