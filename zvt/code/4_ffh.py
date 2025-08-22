#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_FOUR_FORCES_HUNTER.py - Zeta/4 Fundamental Forces resonance hunter (Optimized Version)
Author: Jefferson M. Okushigue
Date: 2025-08-13
Modified to search for resonances with the four fundamental forces of physics
"""

import numpy as np
from mpmath import mp
import time
import concurrent.futures
import pickle
import os
import signal
import sys
from datetime import datetime
from scipy import stats
import warnings
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import psutil
from tqdm import tqdm

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zvt_four_forces.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
mp.dps = 50  # High precision

# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

@dataclass
class Resonance:
    """Class to store information about a resonance"""
    zero_index: int
    gamma: float
    quality: float
    tolerance: float
    force_name: str
    constant_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'zero_index': self.zero_index,
            'gamma': self.gamma,
            'quality': self.quality,
            'tolerance': self.tolerance,
            'force_name': self.force_name,
            'constant_value': self.constant_value,
            'energy_gev': self.gamma / 10  # Energy estimate
        }

@dataclass
class StatisticalResult:
    """Class to store results of statistical analyses"""
    basic_stats: Dict[str, float]
    chi2_test: Optional[Dict[str, float]] = None
    binomial_test: Optional[Dict[str, float]] = None
    poisson_test: Optional[Dict[str, float]] = None
    
    def is_significant(self, criteria: Dict[str, float]) -> bool:
        """Check if the result is statistically significant"""
        # Check significance factor
        if self.basic_stats.get('significance_factor', 0) < criteria.get('min_significance_factor', 2.0):
            return False
        
        # Check statistical tests
        significant_tests = 0
        
        if self.chi2_test and self.chi2_test.get('significant', False):
            significant_tests += 1
            
        if self.binomial_test and self.binomial_test.get('significant', False):
            significant_tests += 1
            
        if self.poisson_test and self.poisson_test.get('significant', False):
            significant_tests += 1
            
        # Consider significant if at least one test is significant
        return significant_tests > 0

@dataclass
class BatchResult:
    """Class to store results of a batch analysis"""
    batch_number: int
    timestamp: str
    zeros_analyzed: int
    batch_time: float
    forces_analysis: Dict[str, Dict[float, List[Resonance]]]
    comparative_analysis: Dict[str, Any]
    best_resonances: Dict[str, Resonance]
    progress_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'batch_number': self.batch_number,
            'timestamp': self.timestamp,
            'zeros_analyzed': self.zeros_analyzed,
            'batch_time': self.batch_time,
            'progress_percent': self.progress_percent,
            'best_resonances': {k: v.to_dict() for k, v in self.best_resonances.items()}
        }

class ZetaFourForcesHunter:
    """Main class for analyzing resonances between zeta zeros and fundamental forces"""
    
    # 4 FUNDAMENTAL FORCES OF PHYSICS
    FUNDAMENTAL_FORCES = {
        'electromagnetic': 1 / 137.035999084,      # α (fine structure constant)
        'strong': 0.1185,                           # αs (strong coupling constant at MZ)
        'weak': 0.0338,                             # αW (weak coupling constant)
        'gravitational': 5.906e-39                 # αG (dimensionless gravitational constant)
    }
    
    # Specific tolerances for each force (adjusted to their magnitudes)
    FORCE_TOLERANCES = {
        'electromagnetic': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        'strong': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        'weak': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        'gravitational': [1e-38, 1e-39, 1e-40, 1e-41, 1e-42, 1e-43]  # Appropriate tolerances for gravitational scale
    }
    
    # Control constants for statistical validation
    CONTROL_CONSTANTS = {
        'electromagnetic': FUNDAMENTAL_FORCES['electromagnetic'],
        'strong': FUNDAMENTAL_FORCES['strong'],
        'weak': FUNDAMENTAL_FORCES['weak'], 
        'random_1': 1 / 142.7,
        'random_2': 1 / 129.3,
        'golden_ratio': (np.sqrt(5) - 1) / 2,
        'pi_scale': np.pi / 100,
        'e_scale': np.e / 100
    }
    
    # CRITERIA FOR SIGNIFICANT RESONANCE
    SIGNIFICANCE_CRITERIA = {
        'min_resonances': 10,           # Minimum number of resonances
        'min_significance_factor': 2.0, # Minimum significance factor (2x expected)
        'max_p_value': 0.01,           # Maximum p-value for significance
        'min_chi2_stat': 6.635         # χ² critical for p < 0.01
    }
    
    def __init__(self, zeros_file: str, results_dir: str = "zvt_4forces_results", 
                 cache_file: str = "zeta_zeros_cache.pkl", increment: int = 10000):
        """
        Initialize the resonance hunter
        
        Args:
            zeros_file: Path to the file with zeta function zeros
            results_dir: Directory to save results
            cache_file: Cache file for loaded zeros
            increment: Batch size for processing
        """
        self.zeros_file = zeros_file
        self.results_dir = results_dir
        self.cache_file = cache_file
        self.increment = increment
        self.all_zeros = []
        self.session_results = []
        self.best_overall = {}
        self.shutdown_requested = False
        
        # Set up directories
        os.makedirs(results_dir, exist_ok=True)
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        logger.info(f"ZVT 4 Fundamental Forces Hunter initialized")
        logger.info(f"Zeros file: {zeros_file}")
        logger.info(f"Results directory: {results_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handler for interrupt signals"""
        self.shutdown_requested = True
        logger.info(f"⏸️ Shutdown requested. Completing current batch and saving...")
    
    def load_zeros_from_file(self) -> List[Tuple[int, float]]:
        """Load zeros from file with progress indicator"""
        zeros = []
        try:
            logger.info(f"📂 Loading zeros from file: {self.zeros_file}")
            
            # First, count total lines to show progress
            logger.info("📊 Counting file lines...")
            with open(self.zeros_file, 'r') as f:
                total_lines = sum(1 for line in f if line.strip())
            logger.info(f"📊 Total lines found: {total_lines:,}")
            
            with open(self.zeros_file, 'r') as f:
                progress_counter = 0
                for line_num, line in enumerate(f, start=1):
                    if self.shutdown_requested:
                        break
                    line = line.strip()
                    if line:
                        try:
                            zero = float(line)
                            zeros.append((line_num, zero))  # Format: (index, value)
                            progress_counter += 1
                            
                            # Show progress every 100,000 zeros
                            if progress_counter % 100000 == 0:
                                percent = (progress_counter / total_lines) * 100
                                logger.info(f"📈 Loaded {progress_counter:,} zeros ({percent:.1f}%)")
                                
                        except ValueError:
                            logger.warning(f"⚠️ Invalid line {line_num}: '{line}'")
                            continue
            
            logger.info(f"✅ {len(zeros):,} zeros successfully loaded from {total_lines:,} lines")
            return zeros
        except Exception as e:
            logger.error(f"❌ Error reading file: {e}")
            return []
    
    def save_enhanced_cache(self, backup=True):
        """Save zeros to cache with backup option"""
        try:
            if backup and os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.cache_file, backup_file)
                logger.info(f"📦 Cache backup created: {backup_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.all_zeros, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"💾 Cache saved: {len(self.all_zeros)} zeros")
        except Exception as e:
            logger.error(f"❌ Error saving cache: {e}")
    
    def load_enhanced_cache(self, force_reload=False):
        """Load zeros from cache or file"""
        if not force_reload and os.path.exists(self.cache_file):
            try:
                logger.info(f"🔍 Checking existing cache...")
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        logger.info(f"✅ Valid cache: {len(data):,} zeros loaded")
                        
                        # Check if cache seems complete
                        file_size = os.path.getsize(self.zeros_file)
                        expected_zeros = file_size // 20  # Rough estimate (20 bytes per zero)
                        
                        if len(data) < expected_zeros * 0.5:  # If cache has less than 50% of expected
                            logger.warning(f"⚠️ Cache seems incomplete ({len(data):,} vs ~{expected_zeros:,} expected)")
                            logger.info(f"🔄 Forcing reload from file...")
                            force_reload = True
                        else:
                            return data
            except Exception as e:
                logger.warning(f"⚠️ Invalid cache ({e}), loading from file...")
                force_reload = True
        
        if force_reload or not os.path.exists(self.cache_file):
            logger.info("📂 Loading all zeros from original file...")
            zeros = self.load_zeros_from_file()
            if zeros:
                logger.info(f"💾 Saving {len(zeros):,} zeros to cache...")
                self.save_enhanced_cache(zeros)
                return zeros
        return []
    
    def find_resonances_for_force(self, zeros_batch: List[Tuple[int, float]], force_name: str, 
                                 constant_value: float, tolerance: float) -> List[Resonance]:
        """
        Find resonances for a specific force with a given tolerance
        
        Args:
            zeros_batch: Batch of zeros to analyze
            force_name: Name of the force
            constant_value: Value of the force constant
            tolerance: Tolerance for resonance
            
        Returns:
            List of found resonances
        """
        resonances = []
        for n, gamma in zeros_batch:
            mod_val = gamma % constant_value
            min_distance = min(mod_val, constant_value - mod_val)
            if min_distance < tolerance:
                resonances.append(Resonance(
                    zero_index=n,
                    gamma=gamma,
                    quality=min_distance,
                    tolerance=tolerance,
                    force_name=force_name,
                    constant_value=constant_value
                ))
        return resonances
    
    def find_multi_tolerance_resonances(self, zeros_batch: List[Tuple[int, float]], 
                                      constants_dict: Dict[str, float] = None) -> Dict[str, Dict[float, List[Resonance]]]:
        """
        Find resonances at multiple tolerance levels
        
        Args:
            zeros_batch: Batch of zeros to analyze
            constants_dict: Dictionary of constants (uses FUNDAMENTAL_FORCES if None)
            
        Returns:
            Nested dictionary with resonances by force and tolerance
        """
        if constants_dict is None:
            constants_dict = self.FUNDAMENTAL_FORCES
            
        all_results = {}
        
        # Use ThreadPoolExecutor to parallelize by force
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(constants_dict)) as executor:
            futures = {}
            
            for const_name, const_value in constants_dict.items():
                # Use specific tolerances for each force or generic tolerances
                if const_name in self.FORCE_TOLERANCES:
                    tolerances_to_use = self.FORCE_TOLERANCES[const_name]
                else:
                    tolerances_to_use = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
                
                # Submit task for each tolerance
                for tolerance in tolerances_to_use:
                    future = executor.submit(
                        self.find_resonances_for_force, 
                        zeros_batch, const_name, const_value, tolerance
                    )
                    futures[(const_name, tolerance)] = future
            
            # Collect results
            for (const_name, tolerance), future in futures.items():
                try:
                    resonances = future.result()
                    
                    # Initialize nested dictionary if needed
                    if const_name not in all_results:
                        all_results[const_name] = {}
                    
                    all_results[const_name][tolerance] = resonances
                except Exception as e:
                    logger.error(f"Error processing {const_name} with tolerance {tolerance}: {e}")
        
        return all_results
    
    def enhanced_statistical_analysis(self, zeros_batch: List[Tuple[int, float]], 
                                    resonances: List[Resonance]) -> StatisticalResult:
        """
        Perform enhanced statistical analysis of resonances
        
        Args:
            zeros_batch: Batch of zeros analyzed
            resonances: List of found resonances
            
        Returns:
            Statistical analysis result
        """
        if len(zeros_batch) == 0 or len(resonances) == 0:
            return StatisticalResult(basic_stats={})
        
        total_zeros = len(zeros_batch)
        resonant_count = len(resonances)
        
        # Use first resonance to get constant and tolerance
        constant_value = resonances[0].constant_value
        tolerance = resonances[0].tolerance
        
        expected_random = total_zeros * (2 * tolerance / constant_value)
        
        # Basic validation to avoid invalid values
        if constant_value <= 0 or tolerance <= 0:
            return StatisticalResult(basic_stats={})
            
        basic_stats = {
            'total_zeros': total_zeros,
            'resonant_count': resonant_count,
            'expected_random': expected_random,
            'resonance_rate': resonant_count / total_zeros,
            'significance_factor': resonant_count / expected_random if expected_random > 0 else float('inf')
        }
        
        result = StatisticalResult(basic_stats=basic_stats)
        
        # Chi-square test (only if expected >= 5)
        if expected_random >= 5:
            chi2_stat = (resonant_count - expected_random)**2 / expected_random
            chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=1)
            result.chi2_test = {
                'statistic': chi2_stat,
                'p_value': chi2_pvalue,
                'critical_value_05': 3.841,
                'significant': chi2_stat > 3.841
            }
        
        # Binomial test with probability validation
        p_expected = 2 * tolerance / constant_value
        
        # Check if p_expected is in valid range [0,1]
        if 0 <= p_expected <= 1:
            try:
                binom_result = stats.binomtest(resonant_count, total_zeros, p_expected, alternative='two-sided')
                binom_pvalue = binom_result.pvalue
            except AttributeError:
                try:
                    binom_pvalue = stats.binom_test(resonant_count, total_zeros, p_expected, alternative='two-sided')
                except AttributeError:
                    from scipy.stats import binom
                    binom_pvalue = 2 * min(binom.cdf(resonant_count, total_zeros, p_expected),
                                          1 - binom.cdf(resonant_count - 1, total_zeros, p_expected))
            except Exception as e:
                logger.warning(f"Warning: Binomial test failed: {e}")
                binom_pvalue = 1.0  # Neutral p-value in case of error
                
            result.binomial_test = {
                'p_value': binom_pvalue,
                'significant': binom_pvalue < 0.05
            }
        else:
            # If p_expected is invalid, skip binomial test
            logger.warning(f"Warning: p_expected={p_expected:.2e} out of range [0,1] - skipping binomial test")
            result.binomial_test = {
                'p_value': 1.0,  # Neutral p-value
                'significant': False,
                'invalid_probability': True
            }
        
        # Poisson test (more robust for extreme cases)
        if expected_random > 0:
            try:
                poisson_pvalue = 1 - stats.poisson.cdf(resonant_count - 1, expected_random)
                result.poisson_test = {
                    'p_value': poisson_pvalue,
                    'significant': poisson_pvalue < 0.05
                }
            except Exception as e:
                logger.warning(f"Warning: Poisson test failed: {e}")
                result.poisson_test = {
                    'p_value': 1.0,
                    'significant': False
                }
        
        return result
    
    def comparative_constant_analysis(self, zeros_batch: List[Tuple[int, float]], 
                                    tolerance: float = 1e-5) -> Dict[str, Any]:
        """
        Perform comparative analysis between different constants
        
        Args:
            zeros_batch: Batch of zeros to analyze
            tolerance: Tolerance for analysis
            
        Returns:
            Comparative analysis results
        """
        multi_results = self.find_multi_tolerance_resonances(zeros_batch, self.CONTROL_CONSTANTS)
        comparative_stats = {}
        
        for const_name, const_results in multi_results.items():
            if tolerance in const_results:
                resonances = const_results[tolerance]
                const_value = self.CONTROL_CONSTANTS[const_name]
                stats_result = self.enhanced_statistical_analysis(zeros_batch, resonances)
                comparative_stats[const_name] = {
                    'constant_value': const_value,
                    'resonance_count': len(resonances),
                    'resonance_rate': len(resonances) / len(zeros_batch) * 100,
                    'stats': stats_result
                }
        
        return comparative_stats
    
    def log_significant_resonance(self, force_name: str, constant_value: float, 
                                 tolerance: float, resonances: List[Resonance], 
                                 stats_result: StatisticalResult, zeros_count: int):
        """
        Log significant resonance
        
        Args:
            force_name: Name of the force
            constant_value: Value of the constant
            tolerance: Tolerance used
            resonances: List of resonances
            stats_result: Statistical analysis result
            zeros_count: Number of zeros analyzed
        """
        logger.info("\n" + "🚨" * 40)
        logger.info("🚨 SIGNIFICANT RESONANCE DETECTED! 🚨")
        logger.info("🚨" * 40)
        logger.info(f"🔬 FORCE: {force_name.upper()}")
        logger.info(f"🎯 CONSTANT: {constant_value:.15e}")
        logger.info(f"📏 TOLERANCE: {tolerance:.0e}")
        logger.info(f"📊 ZEROS ANALYZED: {zeros_count:,}")
        logger.info(f"🔍 RESONANCES: {len(resonances):,}")
        
        if stats_result:
            basic = stats_result.basic_stats
            logger.info(f"📈 Rate: {basic['resonance_rate']*100:.6f}% | Significance: {basic['significance_factor']:.3f}x")
            
            if stats_result.chi2_test:
                chi2 = stats_result.chi2_test
                logger.info(f"🧪 χ²={chi2['statistic']:.3f}, p={chi2['p_value']:.2e}")
        
        # Show top 5 best resonances
        if resonances:
            best_resonances = sorted(resonances, key=lambda x: x.quality)[:5]
            logger.info(f"\n💎 TOP 5 BEST RESONANCES:")
            logger.info("| Rank | Zero #    | Gamma            | Quality        | Energy (GeV) |")
            logger.info("|------|-----------|------------------|------------------|---------------|")
            for i, res in enumerate(best_resonances, 1):
                logger.info(f"| {i:4d} | {res.zero_index:9,} | {res.gamma:16.12f} | {res.quality:.6e} | {res.gamma/10:13.3f} |")
        
        logger.info("🚨" * 40)
        logger.info("▶️ AUTOMATICALLY CONTINUING ANALYSIS...\n")
    
    def analyze_batch_with_significance_detection(self, zeros_batch: List[Tuple[int, float]], 
                                                batch_num: int) -> BatchResult:
        """
        Analyze a batch of zeros with significance detection
        
        Args:
            zeros_batch: Batch of zeros to analyze
            batch_num: Batch number
            
        Returns:
            Batch analysis result
        """
        logger.info(f"\n🔬 BATCH #{batch_num}: {len(zeros_batch):,} zeros")
        
        # Analysis of the 4 fundamental forces
        forces_results = self.find_multi_tolerance_resonances(zeros_batch, self.FUNDAMENTAL_FORCES)
        
        best_resonances = {}  # Store best resonances by force
        
        logger.info(f"\n🌌 ANALYSIS OF THE 4 FUNDAMENTAL FORCES:")
        logger.info("| Force         | Constant    | Tolerance | Resonances | Rate (%) | Significance |")
        logger.info("|---------------|--------------|------------|--------------|----------|---------------|")
        
        for force_name, force_value in self.FUNDAMENTAL_FORCES.items():
            # Use specific tolerances for each force
            tolerances_to_check = self.FORCE_TOLERANCES.get(force_name, [1e-4, 1e-5, 1e-6])
            
            for tolerance in tolerances_to_check[:3]:  # Check only top 3 tolerances of each force
                try:
                    if tolerance in forces_results[force_name]:
                        resonances = forces_results[force_name][tolerance]
                        count = len(resonances)
                        rate = count / len(zeros_batch) * 100 if len(zeros_batch) > 0 else 0
                        
                        stats_result = self.enhanced_statistical_analysis(zeros_batch, resonances)
                        sig_factor = stats_result.basic_stats['significance_factor'] if stats_result else 0
                        
                        # Show significance status in table
                        sig_marker = "🚨" if sig_factor > self.SIGNIFICANCE_CRITERIA['min_significance_factor'] else "  "
                        logger.info(f"|{sig_marker}{force_name:11s} | {force_value:.12e} | {tolerance:8.0e} | {count:10d} | {rate:8.3f} | {sig_factor:8.2f}x |")
                        
                        # Save best resonance for each force
                        if resonances:
                            current_best = min(resonances, key=lambda x: x.quality)
                            if force_name not in best_resonances or current_best.quality < best_resonances[force_name].quality:
                                best_resonances[force_name] = current_best
                        
                        # Check if significant and log it
                        if stats_result and stats_result.is_significant(self.SIGNIFICANCE_CRITERIA):
                            self.log_significant_resonance(
                                force_name, force_value, tolerance, resonances, 
                                stats_result, len(zeros_batch)
                            )
                    else:
                        logger.info(f"|  {force_name:11s} | {force_value:.12e} | {tolerance:8.0e} |        N/A |      N/A |      N/A |")
                except Exception as e:
                    logger.error(f"|❌{force_name:11s} | {force_value:.12e} | {tolerance:8.0e} |      ERROR |      N/A |      N/A |")
                    logger.error(f"    Error in analysis of {force_name}: {e}")
                    continue
        
        # Show best resonances found for each force
        if best_resonances:
            logger.info(f"\n💎 BEST RESONANCES FOUND:")
            logger.info("| Force         | Zero #    | Gamma            | Quality      | Energy (GeV) |")
            logger.info("|---------------|-----------|------------------|----------------|---------------|")
            for force_name, res in best_resonances.items():
                logger.info(f"| {force_name:13s} | {res.zero_index:9,} | {res.gamma:16.12f} | {res.quality:.6e} | {res.gamma/10:13.3f} |")
        
        # Comparative analysis always continues (with error handling)
        try:
            comparative_analysis = self.comparative_constant_analysis(zeros_batch)
        except Exception as e:
            logger.warning(f"⚠️ Error in comparative analysis: {e}")
            comparative_analysis = {}
        
        return BatchResult(
            batch_number=batch_num,
            timestamp=datetime.now().isoformat(),
            zeros_analyzed=len(zeros_batch),
            batch_time=0,  # Will be filled later
            forces_analysis=forces_results,
            comparative_analysis=comparative_analysis,
            best_resonances=best_resonances,
            progress_percent=0  # Will be filled later
        )
    
    def generate_comprehensive_report(self, final_batch: int) -> str:
        """
        Generate a comprehensive report of results
        
        Args:
            final_batch: Number of the last batch processed
            
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"Report_4Forces_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ZVT 4 FUNDAMENTAL FORCES HUNTER - COMPREHENSIVE REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Version: Four Forces Hunter v2.0\n")
            f.write(f"Final Batch: #{final_batch}\n")
            f.write(f"Zeros Analyzed: {len(self.all_zeros):,}\n")
            f.write(f"Source File: {self.zeros_file}\n\n")
            
            f.write("CONFIGURATION OF THE 4 FUNDAMENTAL FORCES:\n")
            for force_name, force_value in self.FUNDAMENTAL_FORCES.items():
                tolerances_str = ", ".join([f"{t:.0e}" for t in self.FORCE_TOLERANCES.get(force_name, [1e-4, 1e-5, 1e-6])])
                f.write(f"  {force_name.capitalize()}: {force_value:.15e} (tolerances: {tolerances_str})\n")
            
            f.write(f"\nPrecision: {mp.dps} decimal places\n\n")
            
            f.write("SIGNIFICANCE CRITERIA:\n")
            for key, value in self.SIGNIFICANCE_CRITERIA.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "="*80 + "\n")
            
            # Add best global resonances
            if self.best_overall:
                f.write("\nBEST GLOBAL RESONANCES FOUND:\n")
                f.write("| Force         | Zero #        | Gamma                | Quality        | Energy (GeV) |\n")
                f.write("|---------------|---------------|----------------------|------------------|---------------|\n")
                for force_name, res in self.best_overall.items():
                    f.write(f"| {force_name:13s} | {res.zero_index:13,} | {res.gamma:20.15f} | {res.quality:.6e} | {res.gamma/10:13.6f} |\n")
                
                f.write("\nDETAILS OF BEST RESONANCES:\n")
                for force_name, res in self.best_overall.items():
                    error_percent = (res.quality / self.FUNDAMENTAL_FORCES[force_name]) * 100
                    f.write(f"\n{force_name.upper()}:\n")
                    f.write(f"   Zero #{res.zero_index:,} (γ = {res.gamma:.15f})\n")
                    f.write(f"   Quality: {res.quality:.15e}\n")
                    f.write(f"   Relative error: {error_percent:.12f}%\n")
                    f.write(f"   Tolerance: {res.tolerance:.0e}\n")
                    f.write(f"   Estimated energy: {res.gamma/10:.6f} GeV\n")
        
        logger.info(f"📊 Report saved: {report_file}")
        return report_file
    
    def create_visualizations(self):
        """Create visualizations of results"""
        if not self.session_results:
            logger.warning("No results to visualize")
            return
        
        # Prepare data for visualization
        forces_data = {force: [] for force in self.FUNDAMENTAL_FORCES.keys()}
        
        for result in self.session_results:
            for force_name, res in result.best_resonances.items():
                forces_data[force_name].append({
                    'batch': result.batch_number,
                    'quality': res.quality,
                    'zero_index': res.zero_index,
                    'gamma': res.gamma
                })
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Resonance Analysis - 4 Fundamental Forces', fontsize=16)
        
        # 1. Quality of resonances by batch
        ax = axes[0, 0]
        for force_name, data in forces_data.items():
            if data:
                batches = [d['batch'] for d in data]
                qualities = [d['quality'] for d in data]
                ax.plot(batches, qualities, 'o-', label=force_name.capitalize())
        
        ax.set_yscale('log')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Quality (log)')
        ax.set_title('Quality of Resonances by Batch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Number of zeros vs quality
        ax = axes[0, 1]
        for force_name, data in forces_data.items():
            if data:
                zero_indices = [d['zero_index'] for d in data]
                qualities = [d['quality'] for d in data]
                ax.scatter(zero_indices, qualities, label=force_name.capitalize(), alpha=0.7)
        
        ax.set_yscale('log')
        ax.set_xlabel('Zero Index')
        ax.set_ylabel('Quality (log)')
        ax.set_title('Quality vs Zero Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Quality distribution
        ax = axes[1, 0]
        for force_name, data in forces_data.items():
            if data:
                qualities = [d['quality'] for d in data]
                ax.hist(qualities, bins=20, alpha=0.5, label=force_name.capitalize())
        
        ax.set_xscale('log')
        ax.set_xlabel('Quality (log)')
        ax.set_ylabel('Frequency')
        ax.set_title('Quality Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Best global resonances
        ax = axes[1, 1]
        if self.best_overall:
            forces = list(self.best_overall.keys())
            qualities = [res.quality for res in self.best_overall.values()]
            
            bars = ax.bar(forces, qualities)
            ax.set_yscale('log')
            ax.set_ylabel('Quality (log)')
            ax.set_title('Best Global Resonances')
            
            # Add values on bars
            for bar, quality in zip(bars, qualities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{quality:.2e}',
                        ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = os.path.join(self.results_dir, f"visualizations_{timestamp}.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualizations saved: {viz_file}")
        
        # Close figure to free memory
        plt.close(fig)
    
    def run_analysis(self, force_reload=False):
        """
        Run complete analysis
        
        Args:
            force_reload: Whether to force cache reload
            
        Returns:
            Tuple with zeros, results, and best resonances
        """
        logger.info(f"🚀 ZVT 4 FUNDAMENTAL FORCES HUNTER")
        logger.info(f"🌌 Searching for resonances with the 4 fundamental forces of physics")
        logger.info("=" * 80)
        
        for force_name, force_value in self.FUNDAMENTAL_FORCES.items():
            tolerances_str = ", ".join([f"{t:.0e}" for t in self.FORCE_TOLERANCES.get(force_name, [1e-4, 1e-5, 1e-6])[:3]])
            logger.info(f"🔬 {force_name.upper()}: {force_value:.15e} (tolerances: {tolerances_str})")
        
        logger.info(f"\n📁 File: {self.zeros_file}")
        logger.info("🛑 Ctrl+C to stop")
        logger.info("🚨 Significant resonances will be automatically highlighted")
        logger.info("=" * 80)
        
        # Load zeros
        self.all_zeros = self.load_enhanced_cache(force_reload=force_reload)
        current_count = len(self.all_zeros)
        
        if current_count == 0:
            logger.error("❌ No zeros loaded. Check the file.")
            return [], [], None
        
        # Diagnose original file
        try:
            file_size = os.path.getsize(self.zeros_file)
            logger.info(f"📊 Original file: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            with open(self.zeros_file, 'r') as f:
                total_lines = sum(1 for line in f if line.strip())
            logger.info(f"📊 Lines in file: {total_lines:,}")
            logger.info(f"📊 Zeros loaded: {current_count:,}")
            
            if current_count < total_lines * 0.9:  # If loaded less than 90% of lines
                logger.warning(f"⚠️ WARNING: Loaded only {(current_count/total_lines)*100:.1f}% of the file!")
                logger.info(f"💡 Run with: python3 {sys.argv[0]} --force-reload")
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking file: {e}")
        
        logger.info(f"🎯 PROCESSING ALL {current_count:,} ZEROS!")
        logger.info(f"📦 Batches of {self.increment:,} zeros each")
        logger.info(f"⏱️ Estimate: ~{(current_count//self.increment)} batches")
        
        batch_num = 1
        
        # Create progress bar
        pbar = tqdm(total=current_count, desc="Processing zeros", unit="zeros")
        
        for i in range(0, current_count, self.increment):
            if self.shutdown_requested:
                break
            
            batch_start = i
            batch_end = min(i + self.increment, current_count)
            batch = self.all_zeros[batch_start:batch_end]
            
            # Progress indicator
            progress_percent = (batch_end / current_count) * 100
            logger.info(f"\n🔬 BATCH #{batch_num}: Zeros {batch_start:,} to {batch_end:,} ({progress_percent:.1f}% complete)")
            start_time = time.time()
            
            batch_result = self.analyze_batch_with_significance_detection(batch, batch_num)
            
            # Update time and progress information
            elapsed = time.time() - start_time
            batch_result.batch_time = elapsed
            batch_result.progress_percent = progress_percent
            
            # Update best global resonances
            if batch_result.best_resonances:
                for force_name, res in batch_result.best_resonances.items():
                    if force_name not in self.best_overall or res.quality < self.best_overall[force_name].quality:
                        self.best_overall[force_name] = res
                        logger.info(f"    🎯 NEW GLOBAL BEST for {force_name.upper()}!")
                        logger.info(f"    Zero #{res.zero_index:,} → γ={res.gamma:.12f}, quality={res.quality:.6e}")
            
            # Calculate performance statistics
            zeros_per_sec = len(batch) / elapsed if elapsed > 0 else 0
            remaining_zeros = current_count - batch_end
            eta_seconds = remaining_zeros / zeros_per_sec if zeros_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
            
            logger.info(f"⏱️ Batch processed in {elapsed:.1f}s ({zeros_per_sec:,.0f} zeros/s)")
            if eta_hours > 0:
                logger.info(f"📈 ETA for completion: {eta_hours:.1f} hours")
            
            self.session_results.append(batch_result)
            
            # Update progress bar
            pbar.update(len(batch))
            
            batch_num += 1
        
        # Close progress bar
        pbar.close()
        
        logger.info(f"\n📊 Generating final report...")
        report_file = self.generate_comprehensive_report(batch_num-1)
        
        # Create visualizations
        logger.info(f"\n📊 Creating visualizations...")
        self.create_visualizations()
        
        # Show best global resonances
        if self.best_overall:
            logger.info(f"\n" + "🏆" * 60)
            logger.info(f"🏆 BEST GLOBAL RESONANCES FOUND 🏆")
            logger.info(f"🏆" * 60)
            logger.info("| Force         | Zero #        | Gamma                | Quality        | Energy (GeV) |")
            logger.info("|---------------|---------------|----------------------|------------------|---------------|")
            for force_name, res in self.best_overall.items():
                logger.info(f"| {force_name:13s} | {res.zero_index:13,} | {res.gamma:20.15f} | {res.quality:.6e} | {res.gamma/10:13.6f} |")
            
            logger.info(f"\n💎 DETAILS OF BEST RESONANCES:")
            for force_name, res in self.best_overall.items():
                error_percent = (res.quality / self.FUNDAMENTAL_FORCES[force_name]) * 100
                logger.info(f"\n🔬 {force_name.upper()}:")
                logger.info(f"   Zero #{res.zero_index:,} (γ = {res.gamma:.15f})")
                logger.info(f"   Quality: {res.quality:.15e}")
                logger.info(f"   Relative error: {error_percent:.12f}%")
                logger.info(f"   Tolerance: {res.tolerance:.0e}")
                logger.info(f"   Estimated energy: {res.gamma/10:.6f} GeV")
            logger.info(f"🏆" * 60)
        
        # Final summary
        total_processed = self.session_results[-1].zeros_analyzed if self.session_results else 0
        logger.info(f"\n" + "="*60)
        logger.info(f"📊 FINAL ANALYSIS SUMMARY")
        logger.info(f"="*60)
        logger.info(f"📈 Total zeros processed: {total_processed:,} of {current_count:,}")
        logger.info(f"📈 Percentage completed: {(total_processed/current_count)*100:.1f}%")
        logger.info(f"📈 Batches processed: {len(self.session_results)}")
        if self.session_results:
            total_time = sum(r.batch_time for r in self.session_results)
            avg_time = total_time / len(self.session_results)
            logger.info(f"📈 Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
            logger.info(f"📈 Average time per batch: {avg_time:.1f}s")
            logger.info(f"📈 Average speed: {total_processed/total_time:,.0f} zeros/s")
        logger.info(f"="*60)
        
        return self.all_zeros, self.session_results, self.best_overall

# Main execution
if __name__ == "__main__":
    zeros_file = os.path.expanduser("~/zeta/zero.txt")  # Path to the zeros file
    
    # Check if should force reload
    force_reload = len(sys.argv) > 1 and sys.argv[1] == '--force-reload'
    
    try:
        logger.info("🌟 Starting ZVT 4 Fundamental Forces Hunter")
        hunter = ZetaFourForcesHunter(zeros_file)
        zeros, results, best = hunter.run_analysis(force_reload=force_reload)
        
        if zeros and len(zeros) > 0:
            logger.info(f"\n🎯 Analysis Complete!")
            logger.info(f"📁 Results in: {hunter.results_dir}/")
            logger.info(f"💾 Cache: {hunter.cache_file}")
    except KeyboardInterrupt:
        logger.info(f"\n⏹️ Analysis interrupted. Progress saved.")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info(f"\n🔬 Session completed!")
