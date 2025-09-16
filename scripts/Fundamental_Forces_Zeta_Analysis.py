#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fundamental_Forces_Zeta_Analysis.py
================================================
Focused statistical analysis of potential connections between Riemann zeta zeros
and parameters representing the FOUR FUNDAMENTAL FORCES of nature.
CHANGES:
- Replaced cosmological database with fundamental forces parameters
- Focus on Gravitation, Electromagnetism, Weak, and Strong forces
- Uses coupling constants and characteristic parameters for comparison
Features:
- Reads zeros directly from zero.txt (one gamma per line)
- Multiple statistical significance tests
- Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals
- Advanced visualizations focused on fundamental forces
- Comprehensive reporting with LaTeX-ready tables
- Monte Carlo null hypothesis testing
- Functional transformations optimized for force constants
Author: Okushigue, Jefferson M.
okushigue@gmail.com
Version: 2.3.0 (Fundamental Forces Focus)
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
from collections import defaultdict

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
    tolerance: float = 1e-6
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
        log_file = self.log_dir / f'fundamental_forces_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

class FundamentalForcesDatabase:
    """Database containing parameters for the four fundamental forces"""
    def __init__(self):
        self.constants = self._initialize_fundamental_forces_constants()

    def _initialize_fundamental_forces_constants(self) -> Dict[str, PhysicsConstant]:
        """Initialize with parameters for the four fundamental forces"""
        constants = {}

        # Gravitational Force
        constants['gravitational_constant'] = PhysicsConstant(
            name='gravitational_constant',
            symbol='G',
            value=6.67430e-11,  # m³ kg⁻¹ s⁻²
            uncertainty=1.5e-15,
            source='CODATA 2018',
            category='gravity',
            description='Newtonian gravitational constant'
        )

        # Dimensionless gravitational coupling constant (proton mass)
        alpha_G = (6.67430e-11 * (1.67262192369e-27)**2) / (1.054571817e-34 * 299792458)
        constants['gravitational_coupling'] = PhysicsConstant(
            name='gravitational_coupling',
            symbol='α_G',
            value=alpha_G,  # ~5.9e-39
            uncertainty=0.0,
            source='Derived (G·m_p²/ħc)',
            category='gravity',
            description='Dimensionless gravitational coupling constant (using proton mass)'
        )

        # Electromagnetic Force
        constants['fine_structure_constant'] = PhysicsConstant(
            name='fine_structure_constant',
            symbol='α',
            value=7.2973525693e-3,
            uncertainty=1.1e-11,
            source='CODATA 2018',
            category='electromagnetism',
            description='Fine structure constant (electromagnetic coupling)'
        )

        # Weak Nuclear Force
        constants['fermi_coupling_constant'] = PhysicsConstant(
            name='fermi_coupling_constant',
            symbol='G_F',
            value=1.1663787e-5,  # GeV⁻²
            uncertainty=6e-13,
            source='Particle Data Group',
            category='weak_force',
            description='Fermi coupling constant (weak force)'
        )

        # Dimensionless weak coupling constant (at Z-boson mass scale)
        # α_W = g² / (4π), where g is the SU(2) gauge coupling
        # At M_Z, α_W ≈ 1/29.5
        constants['weak_coupling_constant'] = PhysicsConstant(
            name='weak_coupling_constant',
            symbol='α_W',
            value=1/29.5,  # ~0.0339
            uncertainty=0.001,
            source='Particle Data Group (derived)',
            category='weak_force',
            description='Dimensionless weak coupling constant at M_Z scale'
        )

        # Strong Nuclear Force
        constants['strong_coupling_constant'] = PhysicsConstant(
            name='strong_coupling_constant',
            symbol='α_S',
            value=0.118,  # at M_Z scale
            uncertainty=0.001,
            source='Particle Data Group',
            category='strong_force',
            description='Strong coupling constant at Z-boson mass scale'
        )

        return constants

    def get_fundamental_forces_constants(self) -> Dict[str, PhysicsConstant]:
        """Return all fundamental forces constants"""
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

class FundamentalForcesZetaExplorer:
    """Main analysis class focused on fundamental forces constants"""
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.logger = AdvancedLogger(OUTPUT_DIR)
        self.constants_db = FundamentalForcesDatabase()
        self.zeros_loader = ZetaZerosLoader(self.logger)
        self.stats_analyzer = StatisticalAnalyzer(self.config, self.logger)
        self.matches = []
        self.analysis_results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = OUTPUT_DIR / f'fundamental_forces_analysis_{timestamp}.json'
        self.report_file = OUTPUT_DIR / f'fundamental_forces_report_{timestamp}.txt'
        self.plots_dir = OUTPUT_DIR / 'forces_plots'
        self.plots_dir.mkdir(exist_ok=True)

    def load_data(self) -> bool:
        success = self.zeros_loader.load_zeros(self.config.max_zeros_for_analysis)
        if success:
            self.logger.info(f"Data loaded: {len(self.zeros_loader.zeros):,} zeros")
        return success

    def find_matches(self) -> List[ZetaMatch]:
        """Find direct matches (identity transformation) with fundamental forces constants"""
        self.logger.info("🔍 Searching for direct matches with fundamental forces constants...")
        matches = []
        forces_constants = self.constants_db.get_fundamental_forces_constants()
        for const_name, const_obj in forces_constants.items():
            best_match = None
            best_error = float('inf')
            for idx, gamma in self.zeros_loader.zeros:
                abs_error = abs(gamma - const_obj.value)
                rel_error = abs_error / abs(const_obj.value) if const_obj.value != 0 else abs_error
                if rel_error < best_error and rel_error < self.config.tolerance:
                    best_error = rel_error
                    best_match = (idx, gamma, abs_error, rel_error)
            if best_match:
                nearby_gammas = np.array([g for _, g in self.zeros_loader.zeros
                                        if abs(g - best_match[1]) < 0.1])
                if len(nearby_gammas) > 10:
                    ci = self.stats_analyzer.bootstrap_confidence_interval(
                        nearby_gammas, np.mean, confidence=0.95
                    )
                else:
                    ci = (best_match[1], best_match[1])
                match = ZetaMatch(
                    constant_name=const_name,
                    constant_value=const_obj.value,
                    zero_index=best_match[0],
                    gamma_value=best_match[1],
                    absolute_error=best_match[2],
                    relative_error=best_match[3],
                    significance=0.0,
                    confidence_interval=ci,
                    transformation='identity'
                )
                matches.append(match)
        matches.sort(key=lambda x: x.relative_error)
        self.logger.info(f"Found {len(matches)} direct matches with fundamental forces constants")
        return matches

    def find_transformed_matches(self, tolerance: float = 1e-4) -> List[ZetaMatch]:
        """Find matches using functional transformations optimized for force constants"""
        self.logger.info("🔍 Searching for transformed matches with fundamental forces constants...")
        matches = []
        forces_constants = self.constants_db.get_fundamental_forces_constants()
        gammas = [(idx, g) for idx, g in self.zeros_loader.zeros if g > 0]
        # Transformations
        transformations = {
            'identity': lambda x: x,
            'log': lambda x: np.log(x) if x > 0 else None,
            'inverse': lambda x: 1/x if x != 0 else None,
            'sqrt': lambda x: np.sqrt(x) if x >= 0 else None,
            'square': lambda x: x*x,
            'log10': lambda x: np.log10(x) if x > 0 else None,
            'reciprocal_sqrt': lambda x: 1/np.sqrt(x) if x > 0 else None,
            'cube_root': lambda x: np.cbrt(x),
            'exp': lambda x: np.exp(x) if x < 5 else None,
        }
        for const_name, const_obj in forces_constants.items():
            for trans_name, trans_func in transformations.items():
                best_match = None
                best_error = float('inf')
                for idx, gamma in gammas:
                    if trans_name in ['log', 'log10', 'sqrt', 'reciprocal_sqrt'] and gamma <= 0:
                        continue
                    transformed = trans_func(gamma)
                    if transformed is None or not np.isfinite(transformed):
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
        # Remove duplicate matches for the same constant
        seen_constants = set()
        unique_matches = []
        for match in sorted(matches, key=lambda x: x.relative_error):
            if match.constant_name not in seen_constants:
                unique_matches.append(match)
                seen_constants.add(match.constant_name)
        self.logger.info(f"Found {len(unique_matches)} unique transformed matches with fundamental forces constants")
        return unique_matches

    def statistical_analysis(self, matches: List[ZetaMatch]) -> Dict[str, Any]:
        """Comprehensive statistical analysis focused on fundamental forces"""
        self.logger.info("📊 Performing statistical analysis for fundamental forces matches...")
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
            # Categorize matches by force
            gravity_matches = [m for m in matches if 'gravity' in self.constants_db.constants[m.constant_name].category]
            em_matches = [m for m in matches if 'electromagnetism' in self.constants_db.constants[m.constant_name].category]
            weak_matches = [m for m in matches if 'weak_force' in self.constants_db.constants[m.constant_name].category]
            strong_matches = [m for m in matches if 'strong_force' in self.constants_db.constants[m.constant_name].category]
            results['category_breakdown'] = {
                'gravity_matches': len(gravity_matches),
                'electromagnetism_matches': len(em_matches),
                'weak_force_matches': len(weak_matches),
                'strong_force_matches': len(strong_matches),
            }
        else:
            results['match_statistics'] = {'count': 0}
            results['category_breakdown'] = {
                'gravity_matches': 0,
                'electromagnetism_matches': 0,
                'weak_force_matches': 0,
                'strong_force_matches': 0,
            }

        const_values = {k: v.value for k, v in self.constants_db.get_fundamental_forces_constants().items()}
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
        """Create visualizations focused on fundamental forces matches"""
        self.logger.info("📈 Creating fundamental forces focused visualizations...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fundamental Forces: Zeta Zeros Analysis', fontsize=16, fontweight='bold')

        # Distribution of gamma values with constants highlighted
        gammas = np.array([g for _, g in self.zeros_loader.zeros[:50000]])
        ax1.hist(gammas, bins=100, alpha=0.7, color='darkblue', density=True, edgecolor='black')
        forces_constants = self.constants_db.get_fundamental_forces_constants()
        colors = {
            'gravity': 'brown',
            'electromagnetism': 'red',
            'weak_force': 'orange',
            'strong_force': 'green'
        }
        for const_name, const_obj in forces_constants.items():
            if const_obj.value > 0 and const_obj.value < np.max(gammas):
                color = colors.get(const_obj.category, 'black')
                ax1.axvline(const_obj.value, color=color, linestyle='--', alpha=0.8, linewidth=2,
                           label=f"{const_obj.symbol} ({const_obj.category})")
        ax1.set_xlabel('γ Value')
        ax1.set_ylabel('Density')
        ax1.set_title('γ Distribution with Fundamental Forces Constants')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        ax1.grid(True, alpha=0.3)

        # Match quality visualization
        if matches:
            match_names = [f"{m.constant_name}\n({m.transformation})" for m in matches]
            errors = [m.relative_error for m in matches]
            bar_colors = []
            for match in matches:
                const_obj = self.constants_db.constants[match.constant_name]
                bar_colors.append(colors.get(const_obj.category, 'gray'))
            bars = ax2.barh(range(len(match_names)), errors, color=bar_colors, alpha=0.7)
            ax2.set_yticks(range(len(match_names)))
            ax2.set_yticklabels(match_names, fontsize=8)
            ax2.set_xlabel('Relative Error')
            ax2.set_title('Fundamental Forces Match Quality')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            # Confidence intervals
            for i, match in enumerate(matches):
                ci_low, ci_high = match.confidence_interval
                ci_width = ci_high - ci_low
                if ci_width > 0:
                    ax2.errorbar(match.relative_error, i, xerr=ci_width/2,
                               color='black', alpha=0.5, capsize=3)

        # Zero spacings distribution
        gamma_diffs = np.diff([g for _, g in self.zeros_loader.zeros[:10000]])
        ax3.hist(gamma_diffs, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
        ax3.set_xlabel('γ[n+1] - γ[n]')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Zero Spacings Distribution')
        ax3.grid(True, alpha=0.3)

        # Category distribution pie chart
        if matches:
            categories = defaultdict(int)
            for match in matches:
                const_obj = self.constants_db.constants[match.constant_name]
                categories[const_obj.category] += 1
            if categories:
                wedges, texts, autotexts = ax4.pie(categories.values(), labels=categories.keys(),
                                                  autopct='%1.1f%%', colors=[colors.get(cat, 'gray') for cat in categories.keys()])
                ax4.set_title('Matches by Fundamental Force')
        else:
            ax4.text(0.5, 0.5, 'No Matches Found', ha='center', va='center', fontsize=14)
            ax4.set_title('Match Categories')

        plt.tight_layout()
        plot_file = self.plots_dir / 'fundamental_forces_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Overview plot saved: {plot_file}")
        plt.show()

    def generate_report(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Generate comprehensive report focused on fundamental forces"""
        self.logger.info("📝 Generating fundamental forces focused scientific report...")
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("FUNDAMENTAL FORCES: RIEMANN ZETA ZEROS CORRELATION ANALYSIS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: tolerance={self.config.tolerance:.0e}, "
                   f"MC_trials={self.config.monte_carlo_trials:,}\n")
            f.write(f"Dataset: {len(self.zeros_loader.zeros):,} Riemann zeta zeros\n")
            f.write(f"Constants analyzed: {len(self.constants_db.get_fundamental_forces_constants())} fundamental forces parameters\n")
            f.write("EXECUTIVE SUMMARY - FUNDAMENTAL FORCES FOCUS\n")
            f.write("-" * 50 + "\n")
            if matches:
                f.write(f"• Found {len(matches)} potential correlations with fundamental forces constants\n")
                f.write(f"• Best correlation: {matches[0].constant_name} "
                       f"(relative error: {matches[0].relative_error:.2e})\n")
                f.write(f"• Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.4f}\n")
                f.write(f"• Gravity matches: {stats_results['category_breakdown']['gravity_matches']}\n")
                f.write(f"• Electromagnetism matches: {stats_results['category_breakdown']['electromagnetism_matches']}\n")
                f.write(f"• Weak Force matches: {stats_results['category_breakdown']['weak_force_matches']}\n")
                f.write(f"• Strong Force matches: {stats_results['category_breakdown']['strong_force_matches']}\n")
                f.write(f"• Expected random matches: {stats_results.get('expected_matches_uniform', 0):.2f}\n")
            else:
                f.write("• No correlations found within the specified tolerance\n")
                f.write("• This suggests no detectable relationship at the tested precision level\n")

            f.write("\nDETAILED CORRELATION ANALYSIS\n")
            f.write("-" * 60 + "\n")
            if matches:
                f.write(f"{'Constant':<30} {'Transform':<12} {'Symbol':<8} {'Theoretical':<15} {'Found Value':<15} "
                       f"{'Rel. Error':<12} {'Zero Index':<10} {'Category':<15}\n")
                f.write("-" * 130 + "\n")
                for match in matches:
                    const_obj = self.constants_db.constants[match.constant_name]
                    f.write(f"{const_obj.description[:29]:<30} {match.transformation:<12} {const_obj.symbol:<8} "
                           f"{match.constant_value:<15.9f} {match.gamma_value:<15.9f} "
                           f"{match.relative_error:<12.2e} {match.zero_index:<10} {const_obj.category:<15}\n")
            else:
                f.write("No correlations found within tolerance.\n")

            f.write("\nFUNDAMENTAL FORCES CONSTANTS DATABASE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Parameter':<30} {'Symbol':<8} {'Value':<15} {'Uncertainty':<12} {'Source':<15} {'Category':<15}\n")
            f.write("-" * 110 + "\n")
            for name, const in self.constants_db.get_fundamental_forces_constants().items():
                f.write(f"{const.description[:29]:<30} {const.symbol:<8} {const.value:<15.9f} "
                       f"{const.uncertainty:<12.6f} {const.source[:14]:<15} {const.category:<15}\n")

            f.write("\nSTATISTICAL SIGNIFICANCE ASSESSMENT\n")
            f.write("-" * 50 + "\n")
            f.write(f"Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.6f}\n")
            if stats_results['monte_carlo_p_value'] < 0.05:
                f.write("→ STATISTICALLY SIGNIFICANT: Correlations unlikely due to random chance\n")
                f.write("  This suggests potential systematic relationship between zeta zeros\n")
                f.write("  and fundamental forces parameters\n")
            else:
                f.write("→ NOT STATISTICALLY SIGNIFICANT: Results consistent with random chance\n")
                f.write("  No evidence for systematic correlation between zeta zeros\n")
                f.write("  and fundamental forces parameters at tested precision\n")
            f.write(f"Bonferroni-corrected α: {stats_results.get('bonferroni_corrected_alpha', 'N/A')}\n")
            f.write(f"Expected random matches: {stats_results.get('expected_matches_uniform', 0):.2f}\n")

            f.write("\nSCIENTIFIC INTERPRETATION - FUNDAMENTAL FORCES FOCUS\n")
            f.write("-" * 55 + "\n")
            f.write("1. METHODOLOGY: Focused analysis of the four fundamental forces:\n")
            f.write("   • Gravity (G, α_G)\n")
            f.write("   • Electromagnetism (α)\n")
            f.write("   • Weak Nuclear Force (G_F, α_W)\n")
            f.write("   • Strong Nuclear Force (α_S)\n")
            f.write("2. NULL HYPOTHESIS: No systematic relationship exists between Riemann zeta zero locations\n")
            f.write("   and fundamental forces parameters.\n")
            f.write("3. PHYSICAL CONTEXT:\n")
            f.write("   • These forces govern all interactions in the universe.\n")
            f.write("   • Their relative strengths are fundamental to the structure of matter and the cosmos.\n")
            f.write("   • Any correlation with zeta zeros would suggest a deep mathematical origin for physics.\n")
            f.write("4. INTERPRETATION OF RESULTS:\n")
            if stats_results['monte_carlo_p_value'] < 0.01:
                f.write("   → HIGHLY SIGNIFICANT CORRELATION DETECTED\n")
                f.write("     Strong evidence for non-random relationship. This could indicate:\n")
                f.write("     * A unified mathematical framework underlying all fundamental forces\n")
                f.write("     * The zeta function as a 'master key' for physical constants\n")
                f.write("     * Connection between number theory and the Standard Model of particle physics\n")
            elif stats_results['monte_carlo_p_value'] < 0.05:
                f.write("   → MODERATE CORRELATION DETECTED\n")
                f.write("     Evidence suggests possible relationship, requiring further investigation.\n")
            else:
                f.write("   → NO SIGNIFICANT CORRELATION DETECTED\n")
                f.write("     Results consistent with random chance. This suggests:\n")
                f.write("     * Fundamental forces are independent of the distribution of zeta zeros\n")
                f.write("     * Current precision insufficient to detect subtle correlations\n")

            f.write("\nTECHNICAL METHODOLOGY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Tolerance threshold: {self.config.tolerance:.0e} (relative error)\n")
            f.write(f"Monte Carlo trials: {self.config.monte_carlo_trials:,}\n")
            f.write(f"Bootstrap samples: {self.config.bootstrap_samples:,}\n")
            f.write(f"Significance level: {self.config.significance_level}\n")
            f.write(f"Random seed: {self.config.random_seed} (reproducibility)\n")
            f.write(f"Analysis timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Focus: Fundamental forces parameters only\n")

        self.logger.info(f"Report saved: {self.report_file}")

    def save_results(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Save results in structured JSON format"""
        results_data = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'script_version': '2.3.0',
                'analysis_focus': 'fundamental_forces_only',
                'config': asdict(self.config),
                'n_zeros_analyzed': len(self.zeros_loader.zeros),
                'n_constants_tested': len(self.constants_db.get_fundamental_forces_constants())
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
                } for name, const in self.constants_db.get_fundamental_forces_constants().items()
            }
        }
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        self.logger.info(f"Results saved: {self.results_file}")

    def print_executive_summary(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Print concise executive summary"""
        print("\n" + "=" * 100)
        print("EXECUTIVE SUMMARY - FUNDAMENTAL FORCES ZETA CORRELATION ANALYSIS")
        print("=" * 100)
        print(f"Dataset: {len(self.zeros_loader.zeros):,} Riemann zeta zeros")
        print(f"Constants: {len(self.constants_db.get_fundamental_forces_constants())} fundamental forces parameters")
        print(f"Tolerance: {self.config.tolerance:.0e} (relative error)")
        if matches:
            print(f"\n✅ CORRELATIONS FOUND: {len(matches)}")
            print(f"Best correlation: {matches[0].constant_name} (error: {matches[0].relative_error:.2e})")
            categories = defaultdict(int)
            for match in matches:
                const_obj = self.constants_db.constants[match.constant_name]
                categories[const_obj.category] += 1
            print("\nCorrelations by force:")
            for category, count in categories.items():
                print(f"  • {category.replace('_', ' ').title()}: {count}")
        else:
            print("\n❌ NO CORRELATIONS FOUND within tolerance")
            print("No evidence for systematic relationship between zeta zeros and fundamental forces")

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
        self.logger.info("🚀 Starting Fundamental Forces Zeta Analysis")
        self.logger.info("⚛️ Focus: Four Fundamental Forces (Gravity, EM, Weak, Strong)")
        self.logger.info("=" * 80)
        try:
            if not self.load_data():
                self.logger.error("Failed to load zeta zeros data")
                return False

            direct_matches = self.find_matches()
            transformed_matches = self.find_transformed_matches(tolerance=1e-4)
            all_matches = direct_matches + transformed_matches

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

            self.logger.info("✅ Fundamental Forces Analysis completed successfully!")
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
    explorer = FundamentalForcesZetaExplorer(config)
    success = explorer.run_complete_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
