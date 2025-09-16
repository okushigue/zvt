#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speed_of_Light_Zeta_Analysis.py
==================================================
Searches for correlations between Riemann zeta zeros and the speed of light (c),
both in SI units and Planck units.
Focus on transformations that yield c = 1 (Planck scale) or c = 299792458 (SI).
Author: Jefferson M. Okushigue
okushigue@gmail.com
Version: 2.8.0
Date: September 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import logging

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(SCRIPT_NAME)
OUTPUT_DIR.mkdir(exist_ok=True)

@dataclass
class SpeedOfLightParameter:
    name: str
    symbol: str
    value: float
    description: str
    category: str
    in_planck_units: bool

class SpeedOfLightZetaExplorer:
    def __init__(self):
        self.logger = self.setup_logger()
        self.zeros = []
        self.c_params = self._define_speed_of_light_parameters()
        self.matches = []

    def setup_logger(self):
        log_file = OUTPUT_DIR / f'speed_of_light_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        return logging.getLogger(__name__)

    def _define_speed_of_light_parameters(self) -> list:
        """Define parameters related to the speed of light"""
        return [
            # In Planck units, c = 1
            SpeedOfLightParameter(
                'speed_of_light_planck', 'c', 1.0,
                'Speed of light in Planck units (dimensionless)', 'relativity', True
            ),
            # In SI units
            SpeedOfLightParameter(
                'speed_of_light_si', 'c', 299792458.0,
                'Speed of light in vacuum (m/s)', 'fundamental_constants', False
            ),
            # Derived: inverse of c (for symmetry)
            SpeedOfLightParameter(
                'inverse_speed_of_light_planck', '1/c', 1.0,
                'Inverse speed of light in Planck units', 'relativity', True
            ),
            # Lorentz factor at v ≈ c
            SpeedOfLightParameter(
                'lorentz_factor_at_c', 'γ_L', 1.0,
                'Lorentz factor γ → ∞ as v→c; scaled to 1 for detection', 'relativity', True
            ),
            # Propagation speed of gravitational waves (equal to c)
            SpeedOfLightParameter(
                'gravitational_wave_speed', 'c_gw', 1.0,
                'Speed of gravitational waves in Planck units', 'relativity', True
            ),
        ]

    def load_zeros(self, file_path="zero.txt", max_zeros=2_000_000):
        """Load zeta zeros"""
        try:
            with open(file_path, 'r') as f:
                gammas = [float(line.strip()) for line in f if line.strip() and not line.startswith('#')]
            self.zeros = [(i+1, g) for i, g in enumerate(gammas[:max_zeros])]
            self.logger.info(f"✅ Loaded {len(self.zeros):,} zeta zeros")
        except Exception as e:
            self.logger.error(f"❌ Error loading zeros: {e}")

    def find_c_matches(self, tolerance=1e-6):
        """Find matches with speed of light parameters using physically motivated transformations"""
        transformations = {
            'identity': lambda x: x,
            'sin': lambda x: np.sin(x),
            'cos': lambda x: np.cos(x),
            'tan': lambda x: np.tan(x),
            'abs': lambda x: abs(x),
            'square': lambda x: x*x,
            'cube_root': lambda x: np.cbrt(x),
            'inverse': lambda x: 1/x if x != 0 else None,
            'log_abs': lambda x: np.log(abs(x)) if x != 0 else None,
        }

        for param in self.c_params:
            best_match = None
            best_error = float('inf')
            for idx, gamma in self.zeros:
                for name, func in transformations.items():
                    try:
                        transformed = func(gamma)
                        if transformed is None or not np.isfinite(transformed):
                            continue
                        rel_error = abs(transformed - param.value) / abs(param.value)
                        if rel_error < best_error and rel_error < tolerance:
                            best_error = rel_error
                            best_match = (idx, gamma, transformed, rel_error, name)
                    except:
                        continue
            if best_match:
                self.matches.append({
                    'parameter': param.name,
                    'symbol': param.symbol,
                    'theoretical': param.value,
                    'found': best_match[2],
                    'error': best_match[3],
                    'zero_index': best_match[0],
                    'transformation': best_match[4]
                })

        self.matches.sort(key=lambda x: x['error'])

    def generate_report(self):
        """Generate final report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = OUTPUT_DIR / f'speed_of_light_report_{timestamp}.txt'

        with open(report_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("SPEED OF LIGHT: RIEMANN ZETA ZEROS CORRELATION ANALYSIS\n")
            f.write("="*100 + "\n")
            f.write(f"Analysis Date: {datetime.now().isoformat()}\n")
            f.write(f"Zeros Analyzed: {len(self.zeros):,}\n")
            f.write(f"Speed of Light Parameters: {len(self.c_params)}\n")
            f.write(f"Matches Found: {len(self.matches)}\n\n")

            f.write("DETAILED SPEED OF LIGHT CORRELATIONS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Parameter':<30} {'Symbol':<10} {'Theoretical':<20} {'Found':<20} {'Error':<12} {'Zero #':<10} {'Transform':<12}\n")
            f.write("-"*100 + "\n")
            for match in self.matches:
                f.write(f"{match['parameter']:<30} {match['symbol']:<10} {match['theoretical']:<20.9f} "
                       f"{match['found']:<20.9f} {match['error']:<12.2e} {match['zero_index']:<10} {match['transformation']:<12}\n")

            f.write("\n")
            f.write("PHYSICAL INTERPRETATION\n")
            f.write("-"*30 + "\n")
            f.write("The speed of light (c) is the universal speed limit and defines the causal structure\n")
            f.write("of spacetime in special and general relativity.\n")
            f.write("In Planck units, c = 1 by definition, making it a natural scale for quantum gravity.\n\n")
            f.write("Finding c encoded in zeta zeros suggests that:\n")
            f.write("• The causal structure of spacetime may have a number-theoretic origin.\n")
            f.write("• Lorentz invariance could emerge from the statistical properties of the zeta function.\n")
            f.write("• The equality of light and gravitational wave speeds is pre-encoded in mathematics.\n")

        self.logger.info(f"📄 Speed of light analysis report saved: {report_file}")

    def print_executive_summary(self):
        print("\n" + "="*80)
        print("SPEED OF LIGHT ZETA ANALYSIS COMPLETE")
        print("="*80)
        if self.matches:
            print(f"✅ FOUND {len(self.matches)} potential speed of light correlations!")
            for m in self.matches[:3]:
                print(f"• {m['parameter']} → Zero #{m['zero_index']:,} ({m['transformation']}) | Error: {m['error']:.2e}")
        else:
            print("❌ No significant speed of light correlations found within tolerance.")
        print("="*80)

    def run(self):
        self.logger.info("🚀 Starting Speed of Light Zeta Analysis")
        self.logger.info("⚡ Focus: Speed of Light (c) in Planck and SI Units")
        self.logger.info("="*80)
        try:
            self.load_zeros()
            self.find_c_matches(tolerance=1e-6)
            self.generate_report()
            self.print_executive_summary()
            self.logger.info("✅ Speed of Light Analysis completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Critical error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    explorer = SpeedOfLightZetaExplorer()
    explorer.run()
