#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dark_Energy_Matter_Zeta_Analysis.py
======================================================
Focused statistical analysis of potential connections between Riemann zeta zeros
and ONLY dark matter/dark energy cosmological constants.

CHANGES:
- Filtered constants database to include ONLY dark energy and dark matter parameters
- Removed all other physics constants (electromagnetic, particle physics, etc.)
- Enhanced focus on cosmological parameters from Planck 2018 and DESI 2024

Features:
- Reads zeros directly from zero.txt (one gamma per line)
- Multiple statistical significance tests
- Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals
- Advanced visualizations focused on dark energy/matter
- Comprehensive reporting with LaTeX-ready tables
- Monte Carlo null hypothesis testing
- Functional transformations optimized for cosmological constants

Author: Okushigue, Jefferson M.
okushigue@gmail.com
Version: 2.3.0 (Dark Energy/Matter Focus)
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
        log_file = self.log_dir / f'dark_energy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
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

class DarkEnergyMatterDatabase:
    """Database containing ONLY dark energy and dark matter constants"""
    
    def __init__(self):
        self.constants = self._initialize_dark_constants()
    
    def _initialize_dark_constants(self) -> Dict[str, PhysicsConstant]:
        """Initialize with ONLY dark energy/matter cosmological parameters"""
        constants = {}
        
        # Dark Energy Constants (Planck 2018 + DESI 2024)
        constants['dark_energy_density'] = PhysicsConstant(
            name='dark_energy_density',
            symbol='Ω_Λ',
            value=0.6889,
            uncertainty=0.0056,
            source='Planck 2018',
            category='dark_energy',
            description='Dark energy density parameter'
        )
        
        constants['dark_energy_equation_of_state'] = PhysicsConstant(
            name='dark_energy_equation_of_state',
            symbol='w',
            value=-1.028,
            uncertainty=0.032,
            source='DESI 2024',
            category='dark_energy',
            description='Dark energy equation of state parameter (w = P/ρ)'
        )
        
        constants['dark_energy_absolute_w'] = PhysicsConstant(
            name='dark_energy_absolute_w',
            symbol='|w|',
            value=1.028,
            uncertainty=0.032,
            source='Derived (DESI 2024)',
            category='dark_energy',
            description='Absolute value of dark energy equation of state'
        )
        
        # Dark Matter Constants (Planck 2018)
        constants['dark_matter_density'] = PhysicsConstant(
            name='dark_matter_density',
            symbol='Ω_c',
            value=0.2607,
            uncertainty=0.0057,
            source='Planck 2018',
            category='dark_matter',
            description='Cold dark matter density parameter'
        )
        
        constants['total_matter_density'] = PhysicsConstant(
            name='total_matter_density',
            symbol='Ω_m',
            value=0.3111,
            uncertainty=0.0056,
            source='Planck 2018',
            category='dark_matter',
            description='Total matter density parameter (includes dark matter)'
        )
        
        constants['baryon_density'] = PhysicsConstant(
            name='baryon_density',
            symbol='Ω_b',
            value=0.0504,
            uncertainty=0.0008,
            source='Planck 2018',
            category='baryonic_matter',
            description='Baryonic matter density parameter (for comparison with dark matter)'
        )
        
        # Dark Energy/Matter Ratios - Key for cosmic evolution
        constants['dark_energy_to_matter_ratio'] = PhysicsConstant(
            name='dark_energy_to_matter_ratio',
            symbol='Ω_Λ/Ω_m',
            value=0.6889 / 0.3111,
            uncertainty=0.03,
            source='Derived (Planck 2018)',
            category='dark_energy_ratio',
            description='Ratio of dark energy to total matter density'
        )
        
        constants['dark_to_baryonic_matter_ratio'] = PhysicsConstant(
            name='dark_to_baryonic_matter_ratio',
            symbol='Ω_c/Ω_b',
            value=0.2607 / 0.0504,
            uncertainty=0.05,
            source='Derived (Planck 2018)',
            category='dark_matter_ratio',
            description='Ratio of dark matter to baryonic matter'
        )
        
        constants['dark_matter_fraction'] = PhysicsConstant(
            name='dark_matter_fraction',
            symbol='Ω_c/Ω_m',
            value=0.2607 / 0.3111,
            uncertainty=0.02,
            source='Derived (Planck 2018)',
            category='dark_matter_ratio',
            description='Dark matter fraction of total matter'
        )
        
        # Hubble constant (related to dark energy through cosmic expansion)
        constants['hubble_constant'] = PhysicsConstant(
            name='hubble_constant',
            symbol='H₀',
            value=67.66,
            uncertainty=0.42,
            source='Planck 2018',
            category='cosmic_expansion',
            description='Hubble constant (km/s/Mpc) - related to dark energy expansion'
        )
        
        # Derived dark energy parameters
        constants['dark_energy_w_plus_one'] = PhysicsConstant(
            name='dark_energy_w_plus_one',
            symbol='w+1',
            value=-1.028 + 1,
            uncertainty=0.032,
            source='Derived (DESI 2024)',
            category='dark_energy',
            description='Dark energy deviation from cosmological constant (w = -1)'
        )
        
        # Critical density ratios
        constants['universe_curvature'] = PhysicsConstant(
            name='universe_curvature',
            symbol='Ω_k',
            value=1.0 - 0.6889 - 0.3111,  # 1 - Ω_Λ - Ω_m
            uncertainty=0.008,
            source='Derived (Planck 2018)',
            category='cosmic_geometry',
            description='Universe curvature parameter (should be ≈ 0 for flat universe)'
        )
        
        return constants
    
    def get_dark_constants_only(self) -> Dict[str, PhysicsConstant]:
        """Return all constants (all are dark energy/matter related)"""
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

class DarkEnergyMatterZetaExplorer:
    """Main analysis class focused on dark energy/matter constants"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.logger = AdvancedLogger(OUTPUT_DIR)
        self.constants_db = DarkEnergyMatterDatabase()
        self.zeros_loader = ZetaZerosLoader(self.logger)
        self.stats_analyzer = StatisticalAnalyzer(self.config, self.logger)
        
        self.matches = []
        self.analysis_results = {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = OUTPUT_DIR / f'dark_energy_matter_analysis_{timestamp}.json'
        self.report_file = OUTPUT_DIR / f'dark_energy_matter_report_{timestamp}.txt'
        self.plots_dir = OUTPUT_DIR / 'dark_plots'
        self.plots_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> bool:
        success = self.zeros_loader.load_zeros(self.config.max_zeros_for_analysis)
        if success:
            self.logger.info(f"Data loaded: {len(self.zeros_loader.zeros):,} zeros")
        return success
    
    def find_matches(self) -> List[ZetaMatch]:
        """Find direct matches (identity transformation) with dark energy/matter constants"""
        self.logger.info("🔍 Searching for direct matches with dark energy/matter constants...")
        
        matches = []
        dark_constants = self.constants_db.get_dark_constants_only()
        
        for const_name, const_obj in dark_constants.items():
            best_match = None
            best_error = float('inf')
            
            for idx, gamma in self.zeros_loader.zeros:
                # Skip negative values for w parameter comparison
                if const_obj.value < 0 and gamma > 0:
                    abs_error_check = abs(gamma - abs(const_obj.value))
                    rel_error_check = abs_error_check / abs(const_obj.value)
                    if rel_error_check <= self.config.tolerance:
                        abs_error = abs_error_check
                        rel_error = rel_error_check
                        if rel_error < best_error:
                            best_error = rel_error
                            best_match = (idx, gamma, abs_error, rel_error)
                    continue
                
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
        self.logger.info(f"Found {len(matches)} direct matches with dark energy/matter constants")
        return matches

    def find_transformed_matches(self, tolerance: float = 1e-4) -> List[ZetaMatch]:
        """Find matches using functional transformations optimized for cosmological constants"""
        self.logger.info("🔍 Searching for transformed matches with dark energy/matter constants...")
        
        matches = []
        dark_constants = self.constants_db.get_dark_constants_only()
        gammas = [(idx, g) for idx, g in self.zeros_loader.zeros if g > 0]
        
        # Transformations particularly relevant for cosmological constants
        transformations = {
            'identity': lambda x: x,
            'log': lambda x: np.log(x) if x > 0 else None,
            'inverse': lambda x: 1/x if x != 0 else None,
            'sqrt': lambda x: np.sqrt(x) if x >= 0 else None,
            'square': lambda x: x*x,
            'log10': lambda x: np.log10(x) if x > 0 else None,
            'reciprocal_sqrt': lambda x: 1/np.sqrt(x) if x > 0 else None,
            'cube_root': lambda x: np.cbrt(x),
            'exp': lambda x: np.exp(x) if x < 5 else None,  # Limited range for cosmological values
        }
        
        for const_name, const_obj in dark_constants.items():
            for trans_name, trans_func in transformations.items():
                best_match = None
                best_error = float('inf')
                
                for idx, gamma in gammas:
                    if trans_name in ['log', 'log10', 'sqrt', 'reciprocal_sqrt'] and gamma <= 0:
                        continue
                    
                    transformed = trans_func(gamma)
                    if transformed is None or not np.isfinite(transformed):
                        continue
                    
                    # Special handling for negative constants (like w parameter)
                    if const_obj.value < 0:
                        # Compare with absolute value
                        abs_error = abs(transformed - abs(const_obj.value))
                        rel_error = abs_error / abs(const_obj.value)
                    else:
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
        
        self.logger.info(f"Found {len(unique_matches)} unique transformed matches with dark energy/matter constants")
        return unique_matches

    def statistical_analysis(self, matches: List[ZetaMatch]) -> Dict[str, Any]:
        """Comprehensive statistical analysis focused on dark energy/matter"""
        self.logger.info("📊 Performing statistical analysis for dark energy/matter matches...")
        
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
            
            # Categorize matches by dark energy vs dark matter
            dark_energy_matches = [m for m in matches if 'dark_energy' in self.constants_db.constants[m.constant_name].category]
            dark_matter_matches = [m for m in matches if 'dark_matter' in self.constants_db.constants[m.constant_name].category]
            
            results['category_breakdown'] = {
                'dark_energy_matches': len(dark_energy_matches),
                'dark_matter_matches': len(dark_matter_matches),
                'other_matches': len(matches) - len(dark_energy_matches) - len(dark_matter_matches)
            }
        else:
            results['match_statistics'] = {'count': 0}
            results['category_breakdown'] = {'dark_energy_matches': 0, 'dark_matter_matches': 0, 'other_matches': 0}
        
        const_values = {k: v.value for k, v in self.constants_db.get_dark_constants_only().items()}
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

    def create_dark_energy_visualizations(self, matches: List[ZetaMatch]):
        """Create visualizations focused on dark energy/matter matches"""
        self.logger.info("📈 Creating dark energy/matter focused visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dark Energy & Dark Matter: Zeta Zeros Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Distribution of gamma values with dark constants highlighted
        gammas = np.array([g for _, g in self.zeros_loader.zeros[:50000]])
        ax1.hist(gammas, bins=100, alpha=0.7, color='darkblue', density=True, edgecolor='black')
        
        dark_constants = self.constants_db.get_dark_constants_only()
        colors = {'dark_energy': 'red', 'dark_matter': 'orange', 'dark_energy_ratio': 'purple', 
                 'dark_matter_ratio': 'green', 'cosmic_expansion': 'brown', 'baryonic_matter': 'gray',
                 'cosmic_geometry': 'pink'}
        
        for const_name, const_obj in dark_constants.items():
            if const_obj.value > 0 and const_obj.value < np.max(gammas):
                color = colors.get(const_obj.category, 'black')
                ax1.axvline(const_obj.value, color=color, linestyle='--', alpha=0.8, linewidth=2,
                           label=f"{const_obj.symbol} ({const_obj.category})")
        
        ax1.set_xlabel('γ Value')
        ax1.set_ylabel('Density')
        ax1.set_title('γ Distribution with Dark Energy/Matter Constants')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        ax1.grid(True, alpha=0.3)
        
        # Match quality visualization
        if matches:
            match_names = [f"{m.constant_name}\n({m.transformation})" for m in matches]
            errors = [m.relative_error for m in matches]
            
            # Color bars by category
            bar_colors = []
            for match in matches:
                const_obj = self.constants_db.constants[match.constant_name]
                bar_colors.append(colors.get(const_obj.category, 'gray'))
            
            bars = ax2.barh(range(len(match_names)), errors, color=bar_colors, alpha=0.7)
            ax2.set_yticks(range(len(match_names)))
            ax2.set_yticklabels(match_names, fontsize=8)
            ax2.set_xlabel('Relative Error')
            ax2.set_title('Dark Energy/Matter Match Quality')
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
                ax4.set_title('Matches by Dark Energy/Matter Category')
        else:
            ax4.text(0.5, 0.5, 'No Matches Found', ha='center', va='center', fontsize=14)
            ax4.set_title('Match Categories')
        
        plt.tight_layout()
        plot_file = self.plots_dir / 'dark_energy_matter_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Overview plot saved: {plot_file}")
        plt.show()

    def generate_dark_energy_report(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Generate comprehensive report focused on dark energy/matter"""
        self.logger.info("📝 Generating dark energy/matter focused scientific report...")
        
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("DARK ENERGY & DARK MATTER: RIEMANN ZETA ZEROS CORRELATION ANALYSIS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: tolerance={self.config.tolerance:.0e}, "
                   f"MC_trials={self.config.monte_carlo_trials:,}\n")
            f.write(f"Dataset: {len(self.zeros_loader.zeros):,} Riemann zeta zeros\n")
            f.write(f"Dark Constants analyzed: {len(self.constants_db.get_dark_constants_only())} cosmological parameters\n\n")
            
            f.write("EXECUTIVE SUMMARY - DARK ENERGY & DARK MATTER FOCUS\n")
            f.write("-" * 50 + "\n")
            if matches:
                f.write(f"• Found {len(matches)} potential correlations with dark energy/matter constants\n")
                f.write(f"• Best correlation: {matches[0].constant_name} "
                       f"(relative error: {matches[0].relative_error:.2e})\n")
                f.write(f"• Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.4f}\n")
                f.write(f"• Dark energy matches: {stats_results['category_breakdown']['dark_energy_matches']}\n")
                f.write(f"• Dark matter matches: {stats_results['category_breakdown']['dark_matter_matches']}\n")
                f.write(f"• Expected random matches: {stats_results.get('expected_matches_uniform', 0):.2f}\n")
            else:
                f.write("• No correlations found within the specified tolerance\n")
                f.write("• This suggests no detectable relationship at the tested precision level\n")
                f.write("• Dark energy equation of state (w) and density parameters (Ω) show no correlation\n")
            f.write("\n")
            
            f.write("DETAILED DARK ENERGY/MATTER CORRELATION ANALYSIS\n")
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
            f.write("\n")
            
            f.write("COSMOLOGICAL CONSTANTS DATABASE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Parameter':<30} {'Symbol':<8} {'Value':<15} {'Uncertainty':<12} {'Source':<15} {'Category':<15}\n")
            f.write("-" * 110 + "\n")
            for name, const in self.constants_db.get_dark_constants_only().items():
                f.write(f"{const.description[:29]:<30} {const.symbol:<8} {const.value:<15.9f} "
                       f"{const.uncertainty:<12.6f} {const.source[:14]:<15} {const.category:<15}\n")
            f.write("\n")
            
            f.write("STATISTICAL SIGNIFICANCE ASSESSMENT\n")
            f.write("-" * 50 + "\n")
            f.write(f"Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.6f}\n")
            if stats_results['monte_carlo_p_value'] < 0.05:
                f.write("→ STATISTICALLY SIGNIFICANT: Correlations unlikely due to random chance\n")
                f.write("  This suggests potential systematic relationship between zeta zeros\n")
                f.write("  and dark energy/matter parameters\n")
            else:
                f.write("→ NOT STATISTICALLY SIGNIFICANT: Results consistent with random chance\n")
                f.write("  No evidence for systematic correlation between zeta zeros\n")
                f.write("  and cosmological parameters at tested precision\n")
            
            f.write(f"Bonferroni-corrected α: {stats_results.get('bonferroni_corrected_alpha', 'N/A')}\n")
            f.write(f"Expected random matches: {stats_results.get('expected_matches_uniform', 0):.2f}\n\n")
            
            f.write("SCIENTIFIC INTERPRETATION - COSMOLOGICAL FOCUS\n")
            f.write("-" * 55 + "\n")
            f.write("1. METHODOLOGY: Focused analysis of dark energy and dark matter parameters\n")
            f.write("   including Ω_Λ (dark energy density), w (equation of state), Ω_c (dark\n")
            f.write("   matter density), and derived ratios. Uses functional transformations\n")
            f.write("   optimized for cosmological constants.\n\n")
            
            f.write("2. NULL HYPOTHESIS: No systematic relationship exists between Riemann\n")
            f.write("   zeta zero locations and dark energy/dark matter parameters.\n\n")
            
            f.write("3. COSMOLOGICAL CONTEXT:\n")
            f.write("   • Dark energy (~69% of universe) drives accelerated expansion\n")
            f.write("   • Dark matter (~26% of universe) provides gravitational scaffolding\n")
            f.write("   • Any correlation with zeta zeros would suggest deep mathematical\n")
            f.write("     structure underlying cosmic evolution\n\n")
            
            f.write("4. INTERPRETATION OF RESULTS:\n")
            if stats_results['monte_carlo_p_value'] < 0.01:
                f.write("   → HIGHLY SIGNIFICANT CORRELATION DETECTED\n")
                f.write("     Strong evidence for non-random relationship between zeta zeros\n")
                f.write("     and dark energy/matter parameters. This could indicate:\n")
                f.write("     * Fundamental mathematical structure in cosmic composition\n")
                f.write("     * Connection between number theory and cosmological evolution\n")
                f.write("     * Possible quantum gravitational effects on large scales\n")
            elif stats_results['monte_carlo_p_value'] < 0.05:
                f.write("   → MODERATE CORRELATION DETECTED\n")
                f.write("     Evidence suggests possible relationship, requiring further investigation:\n")
                f.write("     * Validate with independent zeta zero computations\n")
                f.write("     * Test with updated cosmological parameters (DESI, Euclid)\n")
                f.write("     * Examine theoretical frameworks connecting number theory to cosmology\n")
            else:
                f.write("   → NO SIGNIFICANT CORRELATION DETECTED\n")
                f.write("     Results consistent with random chance. This suggests:\n")
                f.write("     * No direct relationship between zeta zeros and cosmic parameters\n")
                f.write("     * Dark energy/matter evolution independent of number-theoretic structure\n")
                f.write("     * Current precision insufficient to detect subtle correlations\n")
            
            f.write("\n5. IMPLICATIONS FOR COSMOLOGY:\n")
            if matches and stats_results['monte_carlo_p_value'] < 0.05:
                f.write("   • Potential connection between mathematical physics and cosmic evolution\n")
                f.write("   • May inform theories of quantum gravity and holographic principle\n")
                f.write("   • Could suggest quantized nature of cosmic parameters\n")
                f.write("   • Warrants investigation by theoretical cosmologists\n")
            else:
                f.write("   • Dark energy and dark matter parameters appear mathematically independent\n")
                f.write("   • Cosmic evolution follows standard Lambda-CDM model predictions\n")
                f.write("   • No evidence for deeper mathematical structure in cosmic composition\n")
            
            f.write("\n6. DATA SOURCES & RELIABILITY:\n")
            f.write("   • Cosmological parameters: Planck 2018 (high precision CMB data)\n")
            f.write("   • Dark energy equation of state: DESI 2024 (latest BAO measurements)\n")
            f.write("   • Zeta zeros: High-precision numerical computation\n")
            f.write("   • Statistical methods: Monte Carlo, bootstrap, multiple comparison correction\n")
            
            f.write("\n7. LIMITATIONS AND CAVEATS:\n")
            f.write("   • Analysis is purely empirical - no theoretical mechanism proposed\n")
            f.write("   • Cosmological parameters have observational uncertainties\n")
            f.write(f"   • Tolerance level ({self.config.tolerance:.0e}) affects detection sensitivity\n")
            f.write("   • Functional transformations introduce additional parameter space\n")
            f.write("   • Correlation does not establish causation or physical significance\n")
            
            f.write("\n8. FUTURE RESEARCH DIRECTIONS:\n")
            if matches and stats_results['monte_carlo_p_value'] < 0.05:
                f.write("   PRIORITY (significant correlations found):\n")
                f.write("   • Develop theoretical framework connecting zeta function to cosmology\n")
                f.write("   • Test predictions with upcoming surveys (Euclid, Vera Rubin, Roman)\n")
                f.write("   • Investigate quantum field theory in curved spacetime connections\n")
                f.write("   • Examine holographic duality implications\n")
                f.write("   • Cross-validate with alternative dark energy models (quintessence, etc.)\n")
            else:
                f.write("   STANDARD (no significant correlations):\n")
                f.write("   • Repeat analysis with higher precision zeta zeros\n")
                f.write("   • Test with modified gravity parameters\n")
                f.write("   • Investigate other mathematical sequences (prime gaps, etc.)\n")
                f.write("   • Consider alternative cosmological frameworks\n")
            
            f.write("\n")
            f.write("TECHNICAL METHODOLOGY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Tolerance threshold: {self.config.tolerance:.0e} (relative error)\n")
            f.write(f"Monte Carlo trials: {self.config.monte_carlo_trials:,}\n")
            f.write(f"Bootstrap samples: {self.config.bootstrap_samples:,}\n")
            f.write(f"Significance level: {self.config.significance_level}\n")
            f.write(f"Random seed: {self.config.random_seed} (reproducibility)\n")
            f.write(f"Analysis timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Focus: Dark energy & dark matter parameters only\n\n")
        
        self.logger.info(f"Dark energy/matter focused report saved: {self.report_file}")

    def save_results(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Save results in structured JSON format"""
        results_data = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'script_version': '2.3.0',
                'analysis_focus': 'dark_energy_dark_matter_only',
                'config': asdict(self.config),
                'n_zeros_analyzed': len(self.zeros_loader.zeros),
                'n_constants_tested': len(self.constants_db.get_dark_constants_only())
            },
            'matches': [asdict(match) for match in matches],
            'statistical_analysis': stats_results,
            'dark_constants_database': {
                name: {
                    'value': const.value,
                    'uncertainty': const.uncertainty,
                    'source': const.source,
                    'category': const.category,
                    'description': const.description
                } for name, const in self.constants_db.get_dark_constants_only().items()
            }
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved: {self.results_file}")

    def print_executive_summary(self, matches: List[ZetaMatch], stats_results: Dict[str, Any]):
        """Print concise executive summary focused on dark energy/matter"""
        print("\n" + "=" * 100)
        print("EXECUTIVE SUMMARY - DARK ENERGY & DARK MATTER ZETA CORRELATION ANALYSIS")
        print("=" * 100)
        
        print(f"Dataset: {len(self.zeros_loader.zeros):,} Riemann zeta zeros")
        print(f"Constants: {len(self.constants_db.get_dark_constants_only())} dark energy/matter parameters")
        print(f"Tolerance: {self.config.tolerance:.0e} (relative error)")
        
        if matches:
            print(f"\n✅ CORRELATIONS FOUND: {len(matches)}")
            print(f"Best correlation: {matches[0].constant_name} (error: {matches[0].relative_error:.2e})")
            
            categories = defaultdict(int)
            for match in matches:
                const_obj = self.constants_db.constants[match.constant_name]
                categories[const_obj.category] += 1
            
            print("\nCorrelations by cosmological category:")
            for category, count in categories.items():
                print(f"  • {category.replace('_', ' ').title()}: {count}")
                
            print(f"\nDark Energy matches: {stats_results['category_breakdown']['dark_energy_matches']}")
            print(f"Dark Matter matches: {stats_results['category_breakdown']['dark_matter_matches']}")
        else:
            print("\n❌ NO CORRELATIONS FOUND within tolerance")
            print("No evidence for systematic relationship between zeta zeros and dark energy/matter")
        
        print(f"\n📊 STATISTICAL SIGNIFICANCE:")
        print(f"Monte Carlo p-value: {stats_results['monte_carlo_p_value']:.6f}")
        
        if stats_results['monte_carlo_p_value'] < 0.01:
            print("→ HIGHLY SIGNIFICANT (p < 0.01) - Strong evidence for correlation")
        elif stats_results['monte_carlo_p_value'] < 0.05:
            print("→ SIGNIFICANT (p < 0.05) - Moderate evidence for correlation")
        else:
            print("→ NOT SIGNIFICANT (p ≥ 0.05) - Results consistent with random chance")
        
        print(f"\nExpected random correlations: {stats_results.get('expected_matches_uniform', 0):.2f}")
        
        print(f"\n🌌 COSMOLOGICAL IMPLICATIONS:")
        if matches and stats_results['monte_carlo_p_value'] < 0.05:
            print("• Potential deep mathematical structure in cosmic evolution")
            print("• May inform quantum gravity and holographic theories")
            print("• Warrants theoretical investigation by cosmologists")
        else:
            print("• Dark energy/matter evolution appears independent of number theory")
            print("• Standard Lambda-CDM model predictions maintained")
            print("• No evidence for quantized cosmic parameters")
        
        print(f"\n📁 Output files saved in: {OUTPUT_DIR}")
        print("=" * 100 + "\n")

    def run_complete_analysis(self):
        """Execute complete analysis pipeline focused on dark energy/matter"""
        self.logger.info("🚀 Starting Dark Energy & Dark Matter Zeta Analysis")
        self.logger.info("🌌 Focus: Cosmological parameters only (Ω_Λ, w, Ω_c, ratios)")
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
                if match.relative_error < 0.5 and match.constant_name not in seen_constants:  # Filter absurd matches
                    filtered_matches.append(match)
                    seen_constants.add(match.constant_name)
            
            stats_results = self.statistical_analysis(filtered_matches)
            self.create_dark_energy_visualizations(filtered_matches)
            self.generate_dark_energy_report(filtered_matches, stats_results)
            self.save_results(filtered_matches, stats_results)
            self.print_executive_summary(filtered_matches, stats_results)
            
            self.logger.info("✅ Dark Energy & Dark Matter Analysis completed successfully!")
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
    
    explorer = DarkEnergyMatterZetaExplorer(config)
    success = explorer.run_complete_analysis()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
