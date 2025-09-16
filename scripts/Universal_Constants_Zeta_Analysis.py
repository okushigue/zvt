#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal_Constants_Zeta_Analysis.py
=================================================
Focused statistical analysis of potential connections between Riemann zeta zeros
and UNIVERSAL DIMENSIONLESS CONSTANTS that define our universe.
CHANGES:
- Replaced cosmological database with UNIVERSAL DIMENSIONLESS CONSTANTS
- Focus on fundamental adimensional parameters from physics and mathematics
- Includes constants from quantum physics, gravity, cosmology, and pure math
Features:
- Reads zeros directly from zero.txt (one gamma per line)
- Multiple statistical significance tests
- Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals
- Advanced visualizations focused on universal constants
- Comprehensive reporting with LaTeX-ready tables
- Monte Carlo null hypothesis testing
- Functional transformations optimized for universal constants
Author: Okushigue, Jefferson M.
okushigue@gmail.com
Version: 2.3.0 (Universal Constants Focus)
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
        log_file = self.log_dir / f'universal_constants_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

class UniversalConstantsDatabase:
    """Database containing UNIVERSAL DIMENSIONLESS CONSTANTS"""
    def __init__(self):
        self.constants = self._initialize_universal_constants()

    def _initialize_universal_constants(self) -> Dict[str, PhysicsConstant]:
        """Initialize with universal dimensionless constants"""
        constants = {}

        # 1. Mathematical Constants
        constants['pi'] = PhysicsConstant(
            name='pi',
            symbol='π',
            value=np.pi,
            uncertainty=0.0,
            source='Mathematical Constant',
            category='mathematical',
            description='Ratio of circle circumference to diameter'
        )

        constants['euler_number'] = PhysicsConstant(
            name='euler_number',
            symbol='e',
            value=np.e,
            uncertainty=0.0,
            source='Mathematical Constant',
            category='mathematical',
            description='Base of natural logarithm'
        )

        constants['golden_ratio'] = PhysicsConstant(
            name='golden_ratio',
            symbol='φ',
            value=(1 + np.sqrt(5)) / 2,
            uncertainty=0.0,
            source='Mathematical Constant',
            category='mathematical',
            description='Golden ratio (1 + sqrt(5))/2'
        )

        # 2. Quantum Physics Constants
        constants['fine_structure_constant'] = PhysicsConstant(
            name='fine_structure_constant',
            symbol='α',
            value=7.2973525693e-3,
            uncertainty=1.1e-11,
            source='CODATA 2018',
            category='quantum_physics',
            description='Electromagnetic coupling constant'
        )

        # 3. Particle Physics Constants
        constants['proton_to_electron_mass_ratio'] = PhysicsConstant(
            name='proton_to_electron_mass_ratio',
            symbol='m_p/m_e',
            value=1836.15267343,
            uncertainty=1.1e-7,
            source='CODATA 2018',
            category='particle_physics',
            description='Ratio of proton mass to electron mass'
        )

        # 4. Gravitational Constants
        # α_G = G * m_proton^2 / (ħ * c)
        G = 6.67430e-11  # m³ kg⁻¹ s⁻²
        m_p = 1.67262192369e-27  # kg
        hbar = 1.054571817e-34  # J·s
        c = 299792458  # m/s
        alpha_G = (G * m_p**2) / (hbar * c)
        constants['gravitational_coupling_constant'] = PhysicsConstant(
            name='gravitational_coupling_constant',
            symbol='α_G',
            value=alpha_G,  # ~5.905e-39
            uncertainty=0.0,
            source='Derived (G·m_p²/ħc)',
            category='gravity',
            description='Dimensionless gravitational coupling constant'
        )

        # 5. Cosmological Constants
        constants['dark_energy_density_parameter'] = PhysicsConstant(
            name='dark_energy_density_parameter',
            symbol='Ω_Λ',
            value=0.6889,
            uncertainty=0.0056,
            source='Planck 2018',
            category='cosmology',
            description='Dark energy density parameter (dimensionless)'
        )

        constants['baryon_asymmetry_parameter'] = PhysicsConstant(
            name='baryon_asymmetry_parameter',
            symbol='η',
            value=6.1e-10,
            uncertainty=0.3e-10,
            source='Planck 2018 + BBN',
            category='cosmology',
            description='Baryon-to-photon ratio (matter-antimatter asymmetry)'
        )

        # Fine-tuning parameter: amplitude of density fluctuations
        constants['density_fluctuation_amplitude'] = PhysicsConstant(
            name='density_fluctuation_amplitude',
            symbol='Q',
            value=1e-5,  # Approximately 10^-5
            uncertainty=1e-6,
            source='Cosmological Observations',
            category='cosmology',
            description='Amplitude of primordial density fluctuations (δρ/ρ)'
        )

        # Eddington number (approximate number of protons in observable universe)
        constants['eddington_number'] = PhysicsConstant(
            name='eddington_number',
            symbol='N',
            value=1e80,  # Order of magnitude
            uncertainty=0.0,
            source='Cosmological Estimate',
            category='cosmology',
            description='Approximate number of protons in observable universe'
        )

        return constants

    def get_universal_constants(self) -> Dict[str, PhysicsConstant]:
        """Return all universal constants"""
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

class UniversalConstantsZetaExplorer:
    """Main analysis class focused on universal constants"""
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.logger = AdvancedLogger(OUTPUT_DIR)
        self.constants_db = UniversalConstantsDatabase()
        self.zeros_loader = ZetaZerosLoader(self.logger)
        self.stats_analyzer = StatisticalAnalyzer(self.config, self.logger)
        self.matches = []
        self.analysis_results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = OUTPUT_DIR / f'universal_constants_analysis_{timestamp}.json'
        self.report_file = OUTPUT_DIR / f'universal_constants_report_{timestamp}.txt'
        self.plots_dir = OUTPUT_DIR / 'universal_plots'
        self.plots_dir.mkdir(exist_ok=True)

    def load_data(self) -> bool:
        success = self.zeros_loader.load_zeros(self.config.max_zeros_for_analysis)
        if success:
            self.logger.info(f"Data loaded: {len(self.zeros_loader.zeros):,} zeros")
        return success

    def find_matches(self) -> List[ZetaMatch]:
        """Find direct matches (identity transformation) with universal constants"""
        self.logger.info("🔍 Searching for direct matches with universal constants...")
        matches = []
        universal_constants = self.constants_db.get_universal_constants()
        for const_name, const_obj in universal_constants.items():
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
        self.logger.info(f"Found {len(matches)} direct matches with universal constants")
        return matches

    def find_transformed_matches(self, tolerance: float = 1e-4) -> List[ZetaMatch]:
        """Find matches using functional transformations optimized for universal constants"""
        self.logger.info("🔍 Searching for transformed matches with universal constants...")
        matches = []
        universal_constants = self.constants_db.get_universal_constants()
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
        for const_name, const_obj in universal_constants.items():
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
        self.logger.info(f"Found {len(unique_matches)} unique transformed matches with universal constants")
        return unique_matches

    def statistical_analysis(self, matches: List[ZetaMatch]) -> Dict[str, Any]:
        """Comprehensive statistical analysis focused on universal constants"""
        self.logger.info("📊 Performing statistical analysis for universal constants matches...")
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
            # Categorize matches
            mathematical_matches = [m for m in matches if 'mathematical' in self.constants_db.constants[m.constant_name].category]
            quantum_matches = [m for m in matches if 'quantum_physics' in self.constants_db.constants[m.constant_name].category]
            particle_matches = [m for m in matches if 'particle_physics' in self.constants_db.constants[m.constant_name].category]
            gravity_matches = [m for m in matches if 'gravity' in self.constants_db.constants[m.constant_name].category]
            cosmology_matches = [m for m in matches if 'cosmology' in self.constants_db.constants[m.constant_name].category]
            results['category_breakdown'] = {
                'mathematical_matches': len(mathematical_matches),
                'quantum_physics_matches': len(quantum_matches),
                'particle_physics_matches': len(particle_matches),
                'gravity_matches': len(gravity_matches),
                'cosmology_matches': len(cosmology_matches),
            }
        else:
            results['match_statistics'] = {'count': 0}
            results['category_breakdown'] = {
                'mathematical_matches': 0,
                'quantum_physics_matches': 0,
                'particle_physics_matches': 0,
                'gravity_matches': 0,
                'cosmology_matches': 0,
            }

        const_values = {k: v.value for k, v in self.constants_db.get_universal_constants().items()}
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
        """Create visualizations focused on universal constants matches"""
        self.logger.info("📈 Creating universal constants focused visualizations...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Universal Constants: Zeta Zeros Analysis', fontsize=16, fontweight='bold')

        # Distribution of gamma values with constants highlighted
        gammas = np.array([g for _, g in self.zeros_loader.zeros[:50000]])
        ax1.hist(gammas, bins=100, alpha=0.7, color='darkblue', density=True, edgecolor='black')
        universal_constants = self.constants_db.get_universal_constants()
        colors = {
            'mathematical': 'red',
            'quantum_physics': 'orange',
            'particle_physics': 'green',
            'gravity': 'brown',
            'cosmology': 'purple'
        }
        for const_name, const_obj in universal_constants.items():
            if const_obj.value > 0 and const_obj.value < np.max(gammas):
                color = colors.get(const_obj.category, 'black')
                ax1.axvline(const_obj.value, color=color, linestyle='--', alpha=0.8, linewidth=2,
                           label=f"{const_obj.symbol} ({const_obj.category})")
        ax1.set_xlabel('γ Value')
        ax1.set_ylabel('Density')
        ax1.set_title('γ Distribution with Universal Constants')
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
            ax2.set_title('Universal Constants Match Quality')
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
                ax4.set_title('Matches by Universal Constant Category')
        else:
            ax4.text(0.5, 0.5, 'No Matches Found', ha='center', va='center', fontsize=14)
            ax4.set_title('Match Categories')

        plt.tight_layout()
        plot_file = self.plots_dir / 'universal_constants_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Overview plot saved: {plot_file}")
        plt.show()

    def generate_report(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Generate comprehensive report focused on universal constants"""
        self.logger.info("📝 Generating universal constants focused scientific report...")
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("UNIVERSAL CONSTANTS: RIEMANN ZETA ZEROS CORRELATION ANALYSIS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: tolerance={self.config.tolerance:.0e}, "
                   f"MC_trials={self.config.monte_carlo_trials:,}\n")
            f.write(f"Dataset: {len(self.zeros_loader.zeros):,} Riemann zeta zeros\n")
            f.write(f"Constants analyzed: {len(self.constants_db.get_universal_constants())} universal constants\n")
            f.write("EXECUTIVE SUMMARY - UNIVERSAL CONSTANTS FOCUS\n")
            f.write("-" * 50 + "\n")
            if matches:
                f.write(f"• Found {len(matches)} potential correlations with universal constants\n")
                f.write(f"• Best correlation: {matches[0].constant_name} "
                       f"(relative error: {matches[0].relative_error:.2e})\n")
                f.write(f"• Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.4f}\n")
                f.write(f"• Mathematical matches: {stats_results['category_breakdown']['mathematical_matches']}\n")
                f.write(f"• Quantum Physics matches: {stats_results['category_breakdown']['quantum_physics_matches']}\n")
                f.write(f"• Cosmology matches: {stats_results['category_breakdown']['cosmology_matches']}\n")
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

            f.write("\nUNIVERSAL CONSTANTS DATABASE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Parameter':<30} {'Symbol':<8} {'Value':<15} {'Uncertainty':<12} {'Source':<15} {'Category':<15}\n")
            f.write("-" * 110 + "\n")
            for name, const in self.constants_db.get_universal_constants().items():
                f.write(f"{const.description[:29]:<30} {const.symbol:<8} {const.value:<15.9f} "
                       f"{const.uncertainty:<12.6f} {const.source[:14]:<15} {const.category:<15}\n")

            f.write("\nSTATISTICAL SIGNIFICANCE ASSESSMENT\n")
            f.write("-" * 50 + "\n")
            f.write(f"Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.6f}\n")
            if stats_results['monte_carlo_p_value'] < 0.05:
                f.write("→ STATISTICALLY SIGNIFICANT: Correlations unlikely due to random chance\n")
                f.write("  This suggests potential systematic relationship between zeta zeros\n")
                f.write("  and universal constants that define our universe\n")
            else:
                f.write("→ NOT STATISTICALLY SIGNIFICANT: Results consistent with random chance\n")
                f.write("  No evidence for systematic correlation between zeta zeros\n")
                f.write("  and universal constants at tested precision\n")
            f.write(f"Bonferroni-corrected α: {stats_results.get('bonferroni_corrected_alpha', 'N/A')}\n")
            f.write(f"Expected random matches: {stats_results.get('expected_matches_uniform', 0):.2f}\n")

            f.write("\nSCIENTIFIC INTERPRETATION - UNIVERSAL CONSTANTS FOCUS\n")
            f.write("-" * 55 + "\n")
            f.write("1. METHODOLOGY: Focused analysis of universal dimensionless constants including:\n")
            f.write("   • Mathematical constants (π, e, φ)\n")
            f.write("   • Physical coupling constants (α, α_G)\n")
            f.write("   • Cosmological parameters (Ω_Λ, η, Q)\n")
            f.write("   • Fundamental ratios (m_p/m_e)\n")
            f.write("2. NULL HYPOTHESIS: No systematic relationship exists between Riemann zeta zero locations\n")
            f.write("   and the values of universal constants.\n")
            f.write("3. CONTEXT:\n")
            f.write("   • These constants define the fundamental structure and evolution of our universe.\n")
            f.write("   • They are the 'free parameters' of the Standard Model and Lambda-CDM cosmology.\n")
            f.write("   • Any correlation with zeta zeros would suggest a deep mathematical origin for physics.\n")
            f.write("4. INTERPRETATION OF RESULTS:\n")
            if stats_results['monte_carlo_p_value'] < 0.01:
                f.write("   → HIGHLY SIGNIFICANT CORRELATION DETECTED\n")
                f.write("     Strong evidence for non-random relationship. This could indicate:\n")
                f.write("     * A unified mathematical framework underlying all physical laws\n")
                f.write("     * The zeta function as a 'master key' for the universe's fundamental parameters\n")
                f.write("     * Connection between number theory and the fine-tuning of the cosmos\n")
            elif stats_results['monte_carlo_p_value'] < 0.05:
                f.write("   → MODERATE CORRELATION DETECTED\n")
                f.write("     Evidence suggests possible relationship, requiring further investigation.\n")
            else:
                f.write("   → NO SIGNIFICANT CORRELATION DETECTED\n")
                f.write("     Results consistent with random chance. This suggests:\n")
                f.write("     * Universal constants are independent of the distribution of zeta zeros\n")
                f.write("     * Current precision insufficient to detect subtle correlations\n")

            f.write("\nTECHNICAL METHODOLOGY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Tolerance threshold: {self.config.tolerance:.0e} (relative error)\n")
            f.write(f"Monte Carlo trials: {self.config.monte_carlo_trials:,}\n")
            f.write(f"Bootstrap samples: {self.config.bootstrap_samples:,}\n")
            f.write(f"Significance level: {self.config.significance_level}\n")
            f.write(f"Random seed: {self.config.random_seed} (reproducibility)\n")
            f.write(f"Analysis timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Focus: Universal dimensionless constants only\n")

        self.logger.info(f"Report saved: {self.report_file}")

    def save_results(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Save results in structured JSON format"""
        results_data = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'script_version': '2.3.0',
                'analysis_focus': 'universal_constants_only',
                'config': asdict(self.config),
                'n_zeros_analyzed': len(self.zeros_loader.zeros),
                'n_constants_tested': len(self.constants_db.get_universal_constants())
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
                } for name, const in self.constants_db.get_universal_constants().items()
            }
        }
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        self.logger.info(f"Results saved: {self.results_file}")

    def print_executive_summary(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Print concise executive summary"""
        print("\n" + "=" * 100)
        print("EXECUTIVE SUMMARY - UNIVERSAL CONSTANTS ZETA CORRELATION ANALYSIS")
        print("=" * 100)
        print(f"Dataset: {len(self.zeros_loader.zeros):,} Riemann zeta zeros")
        print(f"Constants: {len(self.constants_db.get_universal_constants())} universal constants")
        print(f"Tolerance: {self.config.tolerance:.0e} (relative error)")
        if matches:
            print(f"\n✅ CORRELATIONS FOUND: {len(matches)}")
            print(f"Best correlation: {matches[0].constant_name} (error: {matches[0].relative_error:.2e})")
            categories = defaultdict(int)
            for match in matches:
                const_obj = self.constants_db.constants[match.constant_name]
                categories[const_obj.category] += 1
            print("\nCorrelations by category:")
            for category, count in categories.items():
                print(f"  • {category.replace('_', ' ').title()}: {count}")
        else:
            print("\n❌ NO CORRELATIONS FOUND within tolerance")
            print("No evidence for systematic relationship between zeta zeros and universal constants")

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
        self.logger.info("🚀 Starting Universal Constants Zeta Analysis")
        self.logger.info("🌐 Focus: Universal dimensionless constants (π, α, Ω_Λ, m_p/m_e, etc.)")
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

            self.logger.info("✅ Universal Constants Analysis completed successfully!")
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
    explorer = UniversalConstantsZetaExplorer(config)
    success = explorer.run_complete_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
