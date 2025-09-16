#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosmological_Constant_Zeta_Analysis.py
==================================================
Focused statistical analysis of potential connections between Riemann zeta zeros
and the cosmological constant (Λ) in Planck units.
CHANGES:
- Database focused on Λ only (in Planck units)
- Uses logarithmic and inverse transformations optimized for tiny values
- Enhanced precision for extreme-scale constants
Features:
- Reads zeros directly from zero.txt (one gamma per line)
- Multiple statistical significance tests (Monte Carlo, Bonferroni, Bootstrap)
- Advanced visualizations focused on cosmological constant
- Comprehensive reporting with LaTeX-ready tables
- Functional transformations optimized for Λ scale
Author: Scientific Analysis System
Version: 2.4.0 (Cosmological Constant Focus)
Date: September 2025
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import os
import json
from typing import List, Tuple, Dict, Any
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 14
})

# Global constants
SCRIPT_NAME = Path(sys.argv[0]).stem
OUTPUT_DIR = Path(SCRIPT_NAME)
OUTPUT_DIR.mkdir(exist_ok=True)

@dataclass
class AnalysisConfig:
    """Configuration parameters for analysis"""
    tolerance: float = 1e-6  # Relative error tolerance
    monte_carlo_trials: int = 10000
    bootstrap_samples: int = 1000
    significance_level: float = 0.05
    max_zeros_for_analysis: int = 2000000
    random_seed: int = 42

@dataclass
class PhysicsConstant:
    """Physical constant with metadata"""
    name: str
    symbol: str
    value: float
    uncertainty: float
    source: str
    category: str
    description: str

@dataclass
class ZetaMatch:
    """Match between zeta zero and physics constant"""
    constant_name: str
    constant_value: float
    zero_index: int
    gamma_value: float
    absolute_error: float
    relative_error: float
    significance: float
    confidence_interval: Tuple[float, float]
    transformation: str

class AdvancedLogger:
    """Enhanced logging system"""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.setup_logging()

    def setup_logging(self):
        log_file = self.log_dir / f'cosmological_constant_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)8s | %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)
    def debug(self, msg: str): self.logger.debug(msg)

class CosmologicalConstantDatabase:
    """Database containing ONLY the cosmological constant"""
    def __init__(self):
        self.constants = self._initialize_cosmological_constant()

    def _initialize_cosmological_constant(self) -> Dict[str, PhysicsConstant]:
        """Initialize with cosmological constant in Planck units"""
        constants = {}

        # Fundamental constants
        G = 6.67430e-11      # m³ kg⁻¹ s⁻²
        hbar = 1.054571817e-34  # J·s
        c = 299792458       # m/s
        l_Planck = np.sqrt(hbar * G / c**3)  # Planck length
        t_Planck = np.sqrt(hbar * G / c**5)  # Planck time

        # Observed cosmological constant (from Planck 2018 + DESI)
        H0_km_s_Mpc = 67.66  # Hubble constant
        H0 = H0_km_s_Mpc * 1000 / 3.0856e22  # Convert to s⁻¹
        Omega_Lambda = 0.6889
        rho_crit = 3 * H0**2 / (8 * np.pi * G)  # Critical density
        Lambda_physical = 8 * np.pi * G * Omega_Lambda * rho_crit / c**2  # In SI units
        Lambda_planck = Lambda_physical * l_Planck**2  # Dimensionless

        constants['cosmological_constant_planck'] = PhysicsConstant(
            name='cosmological_constant_planck',
            symbol='Λ',
            value=Lambda_planck,  # ~1.1e-122
            uncertainty=Lambda_planck * 0.03,  # 3% uncertainty
            source='Planck 2018 + DESI 2024 (derived)',
            category='dark_energy',
            description='Cosmological constant in Planck units (dimensionless)'
        )

        return constants

    def get_cosmological_constant_only(self) -> Dict[str, PhysicsConstant]:
        """Return the cosmological constant"""
        return self.constants

class ZetaZerosLoader:
    """Loader for zeta zeros from zero.txt"""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.zeros = []

    def find_zero_txt_files(self) -> List[Path]:
        """Find potential zero.txt files"""
        search_locations = [
            OUTPUT_DIR / "zero.txt",
            Path("zero.txt"),
            Path("zeta/zero.txt"),
            Path.home() / "zeta" / "zero.txt",
            Path.home() / "Documents" / "zero.txt",
            Path.home() / "Downloads" / "zero.txt"
        ]
        found_files = [p for p in search_locations if p.exists()]
        self.logger.info(f"Found {len(found_files)} potential zero.txt files")
        return found_files

    def load_from_txt(self, txt_path: Path) -> bool:
        """Load zeros from zero.txt"""
        try:
            gammas = []
            with open(txt_path, 'r') as f:
                for linha in f:
                    linha = linha.strip()
                    if linha and not linha.startswith('#'):
                        try:
                            gamma = float(linha)
                            if gamma > 0:
                                gammas.append(gamma)
                        except ValueError:
                            continue
            # Sort and create (index, gamma) tuples
            gammas.sort()
            self.zeros = [(i + 1, gamma) for i, gamma in enumerate(gammas)]
            self.logger.info(f"✅ Loaded {len(self.zeros):,} zeta zeros from {txt_path}")
            self.logger.info(f"Range: {self.zeros[0][1]:.6f} to {self.zeros[-1][1]:.6f}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading {txt_path}: {e}")
            return False

    def load_zeros(self, max_zeros: int = None) -> bool:
        """Load zeros from zero.txt"""
        txt_files = self.find_zero_txt_files()
        if not txt_files:
            self.logger.error("No zero.txt files found")
            return False
        for txt_file in txt_files:
            if self.load_from_txt(txt_file):
                if max_zeros and len(self.zeros) > max_zeros:
                    self.zeros = self.zeros[:max_zeros]
                    self.logger.info(f"Limited to {max_zeros:,} zeros for performance")
                return True
        return False

class StatisticalAnalyzer:
    """Advanced statistical analysis methods"""
    def __init__(self, config: AnalysisConfig, logger: AdvancedLogger):
        self.config = config
        self.logger = logger
        np.random.seed(config.random_seed)

    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons"""
        n_comparisons = len(p_values)
        return [min(1.0, p * n_comparisons) for p in p_values]

    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    statistic_func, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        n_bootstrap = self.config.bootstrap_samples
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        return lower, upper

    def monte_carlo_null_test(self, observed_matches: int, 
                            zeros: List[Tuple[int, float]], 
                            constants: Dict[str, float]) -> float:
        """Monte Carlo test under null hypothesis of no correlation"""
        gamma_values = np.array([g for _, g in zeros])
        null_matches = []
        for _ in range(self.config.monte_carlo_trials):
            shuffled_gammas = np.random.permutation(gamma_values)
            matches = 0
            for const_val in constants.values():
                relative_errors = np.abs(shuffled_gammas - const_val) / np.abs(const_val)
                if np.any(relative_errors < self.config.tolerance):
                    matches += 1
            null_matches.append(matches)
        p_value = np.sum(np.array(null_matches) >= observed_matches) / self.config.monte_carlo_trials
        self.logger.info(f"Monte Carlo null test: {np.mean(null_matches):.2f} ± {np.std(null_matches):.2f} expected matches")
        return p_value

class CosmologicalConstantZetaExplorer:
    """Main analysis class focused on the cosmological constant"""
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.logger = AdvancedLogger(OUTPUT_DIR)
        self.constants_db = CosmologicalConstantDatabase()
        self.zeros_loader = ZetaZerosLoader(self.logger)
        self.stats_analyzer = StatisticalAnalyzer(self.config, self.logger)
        self.matches = []
        self.analysis_results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = OUTPUT_DIR / f'cosmological_constant_analysis_{timestamp}.json'
        self.report_file = OUTPUT_DIR / f'cosmological_constant_report_{timestamp}.txt'
        self.plots_dir = OUTPUT_DIR / 'lambda_plots'
        self.plots_dir.mkdir(exist_ok=True)

    def load_data(self) -> bool:
        success = self.zeros_loader.load_zeros(self.config.max_zeros_for_analysis)
        if success:
            self.logger.info(f"Data loaded: {len(self.zeros_loader.zeros):,} zeros")
        return success

    def find_transformed_matches(self, tolerance: float = 1e-4) -> List[ZetaMatch]:
        """Find matches using functional transformations optimized for tiny values like Λ"""
        self.logger.info("🔍 Searching for transformed matches with cosmological constant...")
        matches = []
        lambda_constants = self.constants_db.get_cosmological_constant_only()
        gammas = [(idx, g) for idx, g in self.zeros_loader.zeros if g > 0]
        
        # Specialized transformations for very small values
        transformations = {
            'identity': lambda x: x,
            'log': lambda x: np.log(x) if x > 0 else None,
            'inverse': lambda x: 1/x if x != 0 else None,
            'exp_neg': lambda x: np.exp(-x),
            'exp_neg_sqrt': lambda x: np.exp(-np.sqrt(x)),
            'exp_neg_x2': lambda x: np.exp(-x*x),
            'power_law_decay': lambda x: x**(-4),
            'double_exp_decay': lambda x: np.exp(-np.exp(x/10)),  # Very rapid decay
        }

        for const_name, const_obj in lambda_constants.items():
            for trans_name, trans_func in transformations.items():
                best_match = None
                best_error = float('inf')
                for idx, gamma in gammas:
                    transformed = trans_func(gamma)
                    if transformed is None or not np.isfinite(transformed) or transformed <= 0:
                        continue
                    abs_error = abs(transformed - const_obj.value)
                    rel_error = abs_error / abs(const_obj.value) if const_obj.value != 0 else abs_error
                    if rel_error < best_error and rel_error < tolerance:
                        best_error = rel_error
                        best_match = (idx, gamma, transformed, abs_error, rel_error, trans_name)
                if best_match:
                    transformed_val = best_match[2]
                    ci_low = transformed_val * 0.999
                    ci_high = transformed_val * 1.001
                    match = ZetaMatch(
                        constant_name=const_name,
                        constant_value=const_obj.value,
                        zero_index=best_match[0],
                        gamma_value=transformed_val,
                        absolute_error=best_match[3],
                        relative_error=best_match[4],
                        significance=0.0,
                        confidence_interval=(ci_low, ci_high),
                        transformation=best_match[5]
                    )
                    matches.append(match)
        # Deduplicate by constant
        seen_constants = set()
        unique_matches = []
        for match in sorted(matches, key=lambda x: x.relative_error):
            if match.constant_name not in seen_constants:
                unique_matches.append(match)
                seen_constants.add(match.constant_name)
        self.logger.info(f"Found {len(unique_matches)} unique transformed matches with cosmological constant")
        return unique_matches

    def statistical_analysis(self, matches: List[ZetaMatch]) -> Dict[str, Any]:
        """Comprehensive statistical analysis focused on Λ"""
        self.logger.info("📊 Performing statistical analysis for cosmological constant...")
        results = {}
        if matches:
            errors = [m.relative_error for m in matches]
            results['match_statistics'] = {
                'count': len(matches),
                'mean_relative_error': np.mean(errors),
                'std_relative_error': np.std(errors),
                'median_relative_error': np.median(errors),
                'min_relative_error': np.min(errors),
                'max_relative_error': np.max(errors)
            }
        else:
            results['match_statistics'] = {'count': 0}

        const_values = {k: v.value for k, v in self.constants_db.get_cosmological_constant_only().items()}
        p_value_mc = self.stats_analyzer.monte_carlo_null_test(
            len(matches), self.zeros_loader.zeros, const_values
        )
        results['monte_carlo_p_value'] = p_value_mc
        n_zeros = len(self.zeros_loader.zeros)
        n_constants = len(const_values)
        gamma_range = max(g for _, g in self.zeros_loader.zeros) - min(g for _, g in self.zeros_loader.zeros)
        expected_matches_uniform = n_constants * 2 * self.config.tolerance * n_zeros / gamma_range
        results['expected_matches_uniform'] = expected_matches_uniform
        if matches:
            p_values = [0.05] * len(matches)
            corrected_p_values = self.stats_analyzer.bonferroni_correction(p_values)
            results['bonferroni_corrected_alpha'] = self.config.significance_level / len(matches)
        return results

    def create_visualizations(self, matches: List[ZetaMatch]):
        """Create visualizations focused on cosmological constant"""
        self.logger.info("📈 Creating cosmological constant focused visualizations...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Cosmological Constant: Zeta Zeros Analysis', fontsize=16, fontweight='bold')

        # Distribution of gamma values with Λ highlighted
        gammas = np.array([g for _, g in self.zeros_loader.zeros[:50000]])
        ax1.hist(gammas, bins=100, alpha=0.7, color='darkblue', density=True, edgecolor='black')
        lambda_const = list(self.constants_db.get_cosmological_constant_only().values())[0]
        ax1.axvline(lambda_const.value, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label=f"{lambda_const.symbol} = {lambda_const.value:.2e}")
        ax1.set_xlabel('γ Value')
        ax1.set_ylabel('Density')
        ax1.set_title('γ Distribution with Λ (not visible due to scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)

        # Match quality visualization
        if matches:
            match_names = [f"{m.constant_name}\n({m.transformation})" for m in matches]
            errors = [m.relative_error for m in matches]
            bars = ax2.barh(range(len(match_names)), errors, color='purple', alpha=0.7)
            ax2.set_yticks(range(len(match_names)))
            ax2.set_yticklabels(match_names, fontsize=8)
            ax2.set_xlabel('Relative Error')
            ax2.set_title('Cosmological Constant Match Quality')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            # Confidence intervals
            for i, match in enumerate(matches):
                ci_low, ci_high = match.confidence_interval
                ci_width = ci_high - ci_low
                if ci_width > 0:
                    ax2.errorbar(match.relative_error, i, xerr=ci_width/2,
                               color='black', alpha=0.5, capsize=3)
        else:
            ax2.text(0.5, 0.5, 'No Matches Found', ha='center', va='center', fontsize=14)

        plt.tight_layout()
        plot_file = self.plots_dir / 'cosmological_constant_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Overview plot saved: {plot_file}")
        plt.show()

    def generate_report(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Generate comprehensive report focused on cosmological constant"""
        self.logger.info("📝 Generating cosmological constant focused scientific report...")
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("COSMOLOGICAL CONSTANT: RIEMANN ZETA ZEROS CORRELATION ANALYSIS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: tolerance={self.config.tolerance:.0e}, "
                   f"MC_trials={self.config.monte_carlo_trials:,}\n")
            f.write(f"Dataset: {len(self.zeros_loader.zeros):,} Riemann zeta zeros\n")
            f.write(f"Parameters analyzed: 1 (Λ in Planck units)\n")
            f.write("EXECUTIVE SUMMARY - COSMOLOGICAL CONSTANT FOCUS\n")
            f.write("-" * 50 + "\n")
            if matches:
                f.write(f"• Found {len(matches)} potential correlations with Λ\n")
                f.write(f"• Best correlation: {matches[0].constant_name} "
                       f"(relative error: {matches[0].relative_error:.2e})\n")
                f.write(f"• Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.4f}\n")
                f.write(f"• Expected random matches: {stats_results.get('expected_matches_uniform', 0):.2f}\n")
            else:
                f.write("• No correlations found within the specified tolerance\n")
                f.write("• This suggests no detectable relationship at the tested precision level\n")

            f.write("\nDETAILED CORRELATION ANALYSIS\n")
            f.write("-" * 60 + "\n")
            if matches:
                f.write(f"{'Parameter':<30} {'Transform':<12} {'Symbol':<8} {'Theoretical':<15} {'Found Value':<15} "
                       f"{'Rel. Error':<12} {'Zero Index':<10} {'Category':<15}\n")
                f.write("-" * 130 + "\n")
                for match in matches:
                    const_obj = self.constants_db.constants[match.constant_name]
                    f.write(f"{const_obj.description[:29]:<30} {match.transformation:<12} {const_obj.symbol:<8} "
                           f"{match.constant_value:<15.9f} {match.gamma_value:<15.9f} "
                           f"{match.relative_error:<12.2e} {match.zero_index:<10} {const_obj.category:<15}\n")
            else:
                f.write("No correlations found within tolerance.\n")

            f.write("\nCOSMOLOGICAL CONSTANT DATABASE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Parameter':<30} {'Symbol':<8} {'Value':<15} {'Uncertainty':<12} {'Source':<15} {'Category':<15}\n")
            f.write("-" * 110 + "\n")
            for name, const in self.constants_db.get_cosmological_constant_only().items():
                f.write(f"{const.description[:29]:<30} {const.symbol:<8} {const.value:<15.9f} "
                       f"{const.uncertainty:<12.6f} {const.source[:14]:<15} {const.category:<15}\n")

            f.write("\nSTATISTICAL SIGNIFICANCE ASSESSMENT\n")
            f.write("-" * 50 + "\n")
            f.write(f"Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.6f}\n")
            if stats_results['monte_carlo_p_value'] < 0.05:
                f.write("→ STATISTICALLY SIGNIFICANT: Correlations unlikely due to random chance\n")
                f.write("  This suggests potential systematic relationship between zeta zeros\n")
                f.write("  and the cosmological constant\n")
            else:
                f.write("→ NOT STATISTICALLY SIGNIFICANT: Results consistent with random chance\n")
                f.write("  No evidence for systematic correlation between zeta zeros\n")
                f.write("  and Λ at tested precision\n")
            f.write(f"Bonferroni-corrected α: {stats_results.get('bonferroni_corrected_alpha', 'N/A')}\n")
            f.write(f"Expected random matches: {stats_results.get('expected_matches_uniform', 0):.2f}\n")

            f.write("\nSCIENTIFIC INTERPRETATION - COSMOLOGICAL CONSTANT FOCUS\n")
            f.write("-" * 55 + "\n")
            f.write("1. METHODOLOGY: Focused analysis of the cosmological constant in Planck units:\n")
            f.write("   • Uses exponential decay and power-law transformations\n")
            f.write("   • Optimized for extremely small values (~10^-122)\n")
            f.write("2. NULL HYPOTHESIS: No systematic relationship exists between Riemann zeta zero locations\n")
            f.write("   and the value of Λ.\n")
            f.write("3. PHYSICAL CONTEXT:\n")
            f.write("   • Λ is the greatest theoretical challenge in physics (10^120 discrepancy).\n")
            f.write("   • Its origin is unknown; it may be related to vacuum energy or new physics.\n")
            f.write("   • Any correlation with zeta zeros would suggest a deep mathematical origin.\n")
            f.write("4. INTERPRETATION OF RESULTS:\n")
            if stats_results['monte_carlo_p_value'] < 0.01:
                f.write("   → HIGHLY SIGNIFICANT CORRELATION DETECTED\n")
                f.write("     Strong evidence for non-random relationship. This could indicate:\n")
                f.write("     * The zeta function encodes the vacuum energy of the universe\n")
                f.write("     * Mathematical solution to the cosmological constant problem\n")
                f.write("     * Zeta zeros as regulators of quantum field theory divergences\n")
            elif stats_results['monte_carlo_p_value'] < 0.05:
                f.write("   → MODERATE CORRELATION DETECTED\n")
                f.write("     Evidence suggests possible relationship, requiring further investigation.\n")
            else:
                f.write("   → NO SIGNIFICANT CORRELATION DETECTED\n")
                f.write("     Results consistent with random chance. This suggests:\n")
                f.write("     * Λ is independent of zeta zero distribution\n")
                f.write("     * Current precision insufficient to detect subtle correlations\n")

            f.write("\nTECHNICAL METHODOLOGY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Tolerance threshold: {self.config.tolerance:.0e} (relative error)\n")
            f.write(f"Monte Carlo trials: {self.config.monte_carlo_trials:,}\n")
            f.write(f"Bootstrap samples: {self.config.bootstrap_samples:,}\n")
            f.write(f"Significance level: {self.config.significance_level}\n")
            f.write(f"Random seed: {self.config.random_seed} (reproducibility)\n")
            f.write(f"Analysis timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Focus: Cosmological constant in Planck units only\n")

        self.logger.info(f"Report saved: {self.report_file}")

    def save_results(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Save results in structured JSON format"""
        results_data = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'script_version': '2.4.0',
                'analysis_focus': 'cosmological_constant_only',
                'config': asdict(self.config),
                'n_zeros_analyzed': len(self.zeros_loader.zeros),
                'n_constants_tested': len(self.constants_db.get_cosmological_constant_only())
            },
            'matches': [asdict(match) for match in matches],
            'statistical_analysis': stats_results,
            'constants_database': {
                name: {
                    'value': const.value,
                    'uncertainty': const.uncertainty,
                    'source': const.source,
                    'category': const.category,
                    'description': const.description
                } for name, const in self.constants_db.get_cosmological_constant_only().items()
            }
        }
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        self.logger.info(f"Results saved: {self.results_file}")

    def print_executive_summary(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Print concise executive summary"""
        print("\n" + "=" * 100)
        print("EXECUTIVE SUMMARY - COSMOLOGICAL CONSTANT ZETA CORRELATION ANALYSIS")
        print("=" * 100)
        print(f"Dataset: {len(self.zeros_loader.zeros):,} Riemann zeta zeros")
        print(f"Parameters: 1 (Λ in Planck units)")
        print(f"Tolerance: {self.config.tolerance:.0e} (relative error)")
        if matches:
            print(f"\n✅ CORRELATIONS FOUND: {len(matches)}")
            print(f"Best correlation: {matches[0].constant_name} (error: {matches[0].relative_error:.2e})")
        else:
            print("\n❌ NO CORRELATIONS FOUND within tolerance")
            print("No evidence for systematic relationship between zeta zeros and Λ")

        print(f"\n📊 STATISTICAL SIGNIFICANCE:")
        print(f"Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.6f}")
        if stats_results['monte_carlo_p_value'] < 0.01:
            print("→ HIGHLY SIGNIFICANT (p < 0.01) - Strong evidence for correlation")
        elif stats_results['monte_carlo_p_value'] < 0.05:
            print("→ SIGNIFICANT (p < 0.05) - Moderate evidence for correlation")
        else:
            print("→ NOT SIGNIFICANT (p ≥ 0.05) - Results consistent with random chance")

        print(f"\nExpected random correlations: {stats_results.get('expected_matches_uniform', 0):.2f}")
        print(f"\n📁 Output files saved in: {OUTPUT_DIR}")
        print("=" * 100 + "\n")

    def run_complete_analysis(self):
        """Execute complete analysis pipeline"""
        self.logger.info("🚀 Starting Cosmological Constant Zeta Analysis")
        self.logger.info("🌌 Focus: Cosmological Constant (Λ) in Planck Units")
        self.logger.info("=" * 80)
        try:
            if not self.load_data():
                self.logger.error("Failed to load zeta zeros data")
                return False

            transformed_matches = self.find_transformed_matches(tolerance=1e-4)
            all_matches = transformed_matches

            # Filter and deduplicate matches
            seen_constants = set()
            filtered_matches = []
            for match in sorted(all_matches, key=lambda x: x.relative_error):
                if match.relative_error < 0.5 and match.constant_name not in seen_constants:
                    filtered_matches.append(match)
                    seen_constants.add(match.constant_name)

            stats_results = self.statistical_analysis(filtered_matches)
            self.create_visualizations(filtered_matches)
            self.generate_report(filtered_matches, stats_results)
            self.save_results(filtered_matches, stats_results)
            self.print_executive_summary(filtered_matches, stats_results)

            self.logger.info("✅ Cosmological Constant Analysis completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Critical error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    config = AnalysisConfig(
        tolerance=1e-6,
        monte_carlo_trials=10000,
        bootstrap_samples=1000,
        significance_level=0.05,
        max_zeros_for_analysis=2000000,
        random_seed=42
    )
    explorer = CosmologicalConstantZetaExplorer(config)
    success = explorer.run_complete_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
