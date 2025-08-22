#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fch.py - Zeta/Fundamental Constants resonance hunter (Optimized Version)
Author: Jefferson M. Okushigue
Date: 2025-08-14
Searches for resonances with 4 fundamental physics constants
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
import json

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zvt_fundamental_constants.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
mp.dps = 50  # High precision

# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("magma")

@dataclass
class Resonance:
    """Class to store information about a resonance"""
    zero_index: int
    gamma: float
    quality: float
    tolerance: float
    constant_name: str
    constant_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'zero_index': self.zero_index,
            'gamma': self.gamma,
            'quality': self.quality,
            'tolerance': self.tolerance,
            'constant_name': self.constant_name,
            'constant_value': self.constant_value,
            'energy_gev': self.gamma / 10
        }

@dataclass
class SessionState:
    """Class to store session state for resumption"""
    last_processed_index: int = 0
    best_resonances: Dict[str, Resonance] = field(default_factory=dict)
    session_results: List[Any] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    total_zeros: int = 0

class ZetaFundamentalConstantsHunter:
    """Main class for analyzing resonances between zeta zeros and fundamental constants"""
    
    # 4 FUNDAMENTAL PHYSICS CONSTANTS
    FUNDAMENTAL_CONSTANTS = {
        'fine_structure': 1/137.035999084,        # α (fine structure constant)
        'electron_mass': 9.1093837015e-31,      # m_e (electron mass)
        'rydberg': 1.0973731568160e7,          # R_∞ (Rydberg constant)
        'avogadro': 6.02214076e23              # N_A (Avogadro's number)
    }
    
    # Specific tolerances for each constant (adjusted to their magnitudes)
    CONSTANT_TOLERANCES = {
        'fine_structure': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        'electron_mass': [1e-30, 1e-31, 1e-32, 1e-33, 1e-34, 1e-35],
        'rydberg': [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
        'avogadro': [1e-21, 1e-22, 1e-23, 1e-24, 1e-25, 1e-26]
    }
    
    # Control constants for statistical validation
    CONTROL_CONSTANTS = {
        'fine_structure': FUNDAMENTAL_CONSTANTS['fine_structure'],
        'electron_mass': FUNDAMENTAL_CONSTANTS['electron_mass'],
        'rydberg': FUNDAMENTAL_CONSTANTS['rydberg'],
        'avogadro': FUNDAMENTAL_CONSTANTS['avogadro'],
        'random_1': 1/136.0,
        'random_2': 9.1e-31,
        'golden_ratio': (np.sqrt(5) - 1) / 2,
        'pi_scale': np.pi / 10000,
        'e_scale': np.e / 10000
    }
    
    # CRITERIA FOR SIGNIFICANT RESONANCE
    SIGNIFICANCE_CRITERIA = {
        'min_resonances': 10,           # Minimum number of resonances
        'min_significance_factor': 2.0, # Minimum significance factor (2x expected)
        'max_p_value': 0.01,           # Maximum p-value for significance
        'min_chi2_stat': 6.635         # χ² critical for p < 0.01
    }
    
    def __init__(self, zeros_file: str, results_dir: str = "zvt_fundamental_results", 
                 cache_file: str = "zeta_zeros_cache_fundamental.pkl", 
                 state_file: str = "session_state_fundamental.json",
                 increment: int = 50000):
        """
        Initialize the resonance hunter
        
        Args:
            zeros_file: Path to the file with zeta function zeros
            results_dir: Directory to save results
            cache_file: Cache file for loaded zeros
            state_file: State file for resumption
            increment: Batch size for processing
        """
        self.zeros_file = zeros_file
        self.results_dir = results_dir
        self.cache_file = cache_file
        self.state_file = state_file
        self.increment = increment
        self.all_zeros = []
        self.session_state = SessionState()
        self.shutdown_requested = False
        
        # Set up directories
        os.makedirs(results_dir, exist_ok=True)
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        logger.info(f"ZVT Fundamental Constants Hunter initialized")
        logger.info(f"Zeros file: {zeros_file}")
        logger.info(f"Results directory: {results_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handler for interrupt signals"""
        self.shutdown_requested = True
        logger.info(f"⏸️ Shutdown requested. Saving state and finalizing current batch...")
    
    def save_session_state(self):
        """Save session state for resumption"""
        try:
            state_data = {
                'last_processed_index': self.session_state.last_processed_index,
                'best_resonances': {k: v.to_dict() for k, v in self.session_state.best_resonances.items()},
                'start_time': self.session_state.start_time,
                'total_zeros': self.session_state.total_zeros
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"💾 Session state saved: {self.state_file}")
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")
    
    def load_session_state(self):
        """Load session state for resumption"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.session_state.last_processed_index = state_data.get('last_processed_index', 0)
                self.session_state.start_time = state_data.get('start_time', time.time())
                self.session_state.total_zeros = state_data.get('total_zeros', 0)
                
                # Rebuild Resonance objects
                for name, res_data in state_data.get('best_resonances', {}).items():
                    self.session_state.best_resonances[name] = Resonance(**res_data)
                
                logger.info(f"🔍 Session state loaded: last index {self.session_state.last_processed_index:,}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Error loading state: {e}")
                return False
        return False
    
    def load_zeros_from_file(self, start_index: int = 0) -> List[Tuple[int, float]]:
        """Load zeros from file with progress indicator and resumption"""
        zeros = []
        try:
            logger.info(f"📂 Loading zeros from file: {self.zeros_file}")
            logger.info(f"🔄 Resuming from index {start_index:,}")
            
            # First, count total lines
            logger.info("📊 Counting file lines...")
            with open(self.zeros_file, 'r') as f:
                total_lines = sum(1 for line in f if line.strip())
            logger.info(f"📊 Total lines found: {total_lines:,}")
            
            with open(self.zeros_file, 'r') as f:
                # Skip already processed lines
                for _ in range(start_index):
                    next(f, None)
                
                progress_counter = 0
                for line_num, line in enumerate(f, start=start_index + 1):
                    if self.shutdown_requested:
                        break
                    line = line.strip()
                    if line:
                        try:
                            zero = float(line)
                            zeros.append((line_num, zero))
                            progress_counter += 1
                            
                            # Show progress every 100,000 zeros
                            if progress_counter % 100000 == 0:
                                percent = ((start_index + progress_counter) / total_lines) * 100
                                logger.info(f"📈 Loaded {start_index + progress_counter:,} zeros ({percent:.1f}%)")
                                
                        except ValueError:
                            logger.warning(f"⚠️ Invalid line {line_num}: '{line}'")
                            continue
            
            logger.info(f"✅ {len(zeros):,} zeros loaded (total: {start_index + len(zeros):,} of {total_lines:,})")
            return zeros
        except Exception as e:
            logger.error(f"❌ Error reading file: {e}")
            return []
    
    def save_enhanced_cache(self, zeros_list, backup=True):
        """Save zeros to cache with backup option"""
        try:
            if backup and os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.cache_file, backup_file)
                logger.info(f"📦 Cache backup created: {backup_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(zeros_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"💾 Cache saved: {len(zeros_list)} zeros")
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
                        return data
            except Exception as e:
                logger.warning(f"⚠️ Invalid cache ({e}), loading from file...")
        
        # Load from file
        start_index = self.session_state.last_processed_index
        zeros = self.load_zeros_from_file(start_index)
        if zeros:
            self.save_enhanced_cache(zeros)
        return zeros
    
    def find_resonances_for_constant(self, zeros_batch: List[Tuple[int, float]], constant_name: str, 
                                 constant_value: float, tolerance: float) -> List[Resonance]:
        """Find resonances for a specific constant (optimized)"""
        resonances = []
        constant_val = float(constant_value)  # Convert once for better performance
        
        for n, gamma in zeros_batch:
            mod_val = gamma % constant_val
            min_distance = min(mod_val, constant_val - mod_val)
            if min_distance < tolerance:
                resonances.append(Resonance(
                    zero_index=n,
                    gamma=gamma,
                    quality=min_distance,
                    tolerance=tolerance,
                    constant_name=constant_name,
                    constant_value=constant_val
                ))
        return resonances
    
    def find_multi_tolerance_resonances(self, zeros_batch: List[Tuple[int, float]], 
                                      constants_dict: Dict[str, float] = None) -> Dict[str, Dict[float, List[Resonance]]]:
        """Find resonances at multiple tolerance levels (optimized)"""
        if constants_dict is None:
            constants_dict = self.FUNDAMENTAL_CONSTANTS
            
        all_results = {}
        
        # Use ThreadPoolExecutor to parallelize by constant
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(constants_dict)) as executor:
            futures = {}
            
            for const_name, const_value in constants_dict.items():
                tolerances_to_use = self.CONSTANT_TOLERANCES.get(const_name, [1e-4, 1e-5, 1e-6])
                
                # Submit task for each tolerance
                for tolerance in tolerances_to_use:
                    future = executor.submit(
                        self.find_resonances_for_constant, 
                        zeros_batch, const_name, const_value, tolerance
                    )
                    futures[(const_name, tolerance)] = future
            
            # Collect results
            for (const_name, tolerance), future in futures.items():
                try:
                    resonances = future.result()
                    
                    if const_name not in all_results:
                        all_results[const_name] = {}
                    
                    all_results[const_name][tolerance] = resonances
                except Exception as e:
                    logger.error(f"Error processing {const_name} with tolerance {tolerance}: {e}")
        
        return all_results
    
    def enhanced_statistical_analysis(self, zeros_batch: List[Tuple[int, float]], 
                                    resonances: List[Resonance]) -> Dict[str, Any]:
        """Perform enhanced statistical analysis (simplified)"""
        if len(zeros_batch) == 0 or len(resonances) == 0:
            return {}
        
        total_zeros = len(zeros_batch)
        resonant_count = len(resonances)
        constant_value = resonances[0].constant_value
        tolerance = resonances[0].tolerance
        
        expected_random = total_zeros * (2 * tolerance / constant_value)
        
        basic_stats = {
            'total_zeros': total_zeros,
            'resonant_count': resonant_count,
            'expected_random': expected_random,
            'resonance_rate': resonant_count / total_zeros,
            'significance_factor': resonant_count / expected_random if expected_random > 0 else float('inf')
        }
        
        return basic_stats
    
    def analyze_batch_optimized(self, zeros_batch: List[Tuple[int, float]], batch_num: int) -> Dict[str, Any]:
        """Analyze a batch of zeros in an optimized way"""
        logger.info(f"\n🔬 BATCH #{batch_num}: {len(zeros_batch):,} zeros")
        
        # Analysis of the 4 fundamental constants
        constants_results = self.find_multi_tolerance_resonances(zeros_batch, self.FUNDAMENTAL_CONSTANTS)
        
        best_resonances = {}
        
        logger.info(f"\n🌌 ANALYSIS OF THE 4 FUNDAMENTAL CONSTANTS:")
        logger.info("| Constant         | Value         | Tolerance | Resonances | Rate (%) | Significance |")
        logger.info("|-------------------|---------------|------------|--------------|----------|---------------|")
        
        for constant_name, constant_value in self.FUNDAMENTAL_CONSTANTS.items():
            tolerances_to_check = self.CONSTANT_TOLERANCES.get(constant_name, [1e-4, 1e-5, 1e-6])
            
            for tolerance in tolerances_to_check[:3]:  # Check only the top 3 tolerances
                try:
                    if tolerance in constants_results[constant_name]:
                        resonances = constants_results[constant_name][tolerance]
                        count = len(resonances)
                        rate = count / len(zeros_batch) * 100 if len(zeros_batch) > 0 else 0
                        
                        stats_result = self.enhanced_statistical_analysis(zeros_batch, resonances)
                        sig_factor = stats_result.get('significance_factor', 0)
                        
                        # Show significance status
                        sig_marker = "🚨" if sig_factor > 2.0 else "  "
                        logger.info(f"|{sig_marker}{constant_name:17s} | {constant_value:.12e} | {tolerance:8.0e} | {count:10d} | {rate:8.3f} | {sig_factor:8.2f}x |")
                        
                        # Save best resonance
                        if resonances:
                            current_best = min(resonances, key=lambda x: x.quality)
                            if constant_name not in best_resonances or current_best.quality < best_resonances[constant_name].quality:
                                best_resonances[constant_name] = current_best
                        
                        # Register significant resonance
                        if sig_factor > 2.0:
                            logger.info(f"\n🚨 SIGNIFICANT RESONANCE: {constant_name.upper()}!")
                            logger.info(f"   Zero #{current_best.zero_index:,} → γ={current_best.gamma:.12f}")
                            logger.info(f"   Quality: {current_best.quality:.6e} (tolerance: {tolerance:.0e})")
                    
                except Exception as e:
                    logger.error(f"❌ Error in analysis of {constant_name}: {e}")
                    continue
        
        return {
            'batch_number': batch_num,
            'zeros_analyzed': len(zeros_batch),
            'best_resonances': best_resonances,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_final_report(self):
        """Generate a complete final report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"Final_Report_Fundamentals_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ZVT FUNDAMENTAL CONSTANTS HUNTER - FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Zeros file: {self.zeros_file}\n")
            f.write(f"Total zeros processed: {self.session_state.last_processed_index:,}\n\n")
            
            f.write("FUNDAMENTAL CONSTANTS ANALYZED:\n")
            f.write("-" * 80 + "\n")
            for name, value in self.FUNDAMENTAL_CONSTANTS.items():
                f.write(f"{name.replace('_', ' ').title()}: {value:.15e}\n")
            
            f.write("\nDESCRIPTION OF CONSTANTS:\n")
            f.write("-" * 80 + "\n")
            f.write("Fine Structure (α): Dimensionless constant that measures the strength of electromagnetic interaction\n")
            f.write("Electron Mass (m_e): Rest mass of the electron, fundamental for atomic physics\n")
            f.write("Rydberg (R_∞): Constant for atomic spectroscopy and electronic transitions\n")
            f.write("Avogadro (N_A): Number of particles in a mole, fundamental for chemistry\n")
            
            f.write("\nRESULTS OF BEST RESONANCES:\n")
            f.write("-" * 80 + "\n")
            
            for constant_name, res in self.session_state.best_resonances.items():
                error_percent = (res.quality / self.FUNDAMENTAL_CONSTANTS[constant_name]) * 100
                f.write(f"\n{constant_name.upper().replace('_', ' ')}:\n")
                f.write(f"  Zero #{res.zero_index:,}\n")
                f.write(f"  Gamma: {res.gamma:.15f}\n")
                f.write(f"  Quality: {res.quality:.15e}\n")
                f.write(f"  Relative error: {error_percent:.12f}%\n")
                f.write(f"  Tolerance: {res.tolerance:.0e}\n")
                f.write(f"  Estimated energy: {res.gamma/10:.6f} GeV\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("STATISTICAL ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            
            # Calculate general statistics
            total_time = time.time() - self.session_state.start_time
            avg_speed = self.session_state.last_processed_index / total_time
            
            f.write(f"Total processing time: {total_time:.1f} seconds\n")
            f.write(f"Average speed: {avg_speed:,.0f} zeros/s\n")
            f.write(f"Batches processed: {len(self.session_state.session_results)}\n")
            f.write(f"Batch size: {self.increment:,} zeros\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("OBSERVATIONS:\n")
            f.write("-" * 80 + "\n")
            f.write("1. Analysis of fundamental constants of modern physics\n")
            f.write("2. Includes dimensionless and dimensional constants\n")
            f.write("3. Search for connections with atomic and molecular structure\n")
            f.write("4. Results may indicate fundamental relationships of nature\n")
            
        logger.info(f"📊 Final report saved: {report_file}")
        return report_file
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations of the results"""
        if not self.session_state.session_results:
            logger.warning("No results to visualize")
            return
        
        # Prepare data for visualization
        constants_data = {constant: [] for constant in self.FUNDAMENTAL_CONSTANTS.keys()}
        
        for result in self.session_state.session_results:
            for constant_name, res in result.get('best_resonances', {}).items():
                constants_data[constant_name].append({
                    'batch': result['batch_number'],
                    'quality': res.quality,
                    'zero_index': res.zero_index,
                    'gamma': res.gamma
                })
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Resonance Analysis - 4 Fundamental Constants', fontsize=16)
        
        # 1. Quality of resonances by batch
        ax = axes[0, 0]
        for constant_name, data in constants_data.items():
            if data:
                batches = [d['batch'] for d in data]
                qualities = [d['quality'] for d in data]
                ax.plot(batches, qualities, 'o-', label=constant_name.replace('_', ' ').title())
        
        ax.set_yscale('log')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Quality (log)')
        ax.set_title('Quality of Resonances by Batch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Number of zeros vs quality
        ax = axes[0, 1]
        for constant_name, data in constants_data.items():
            if data:
                zero_indices = [d['zero_index'] for d in data]
                qualities = [d['quality'] for d in data]
                ax.scatter(zero_indices, qualities, label=constant_name.replace('_', ' ').title(), alpha=0.7, s=50)
        
        ax.set_yscale('log')
        ax.set_xlabel('Zero Index')
        ax.set_ylabel('Quality (log)')
        ax.set_title('Quality vs Zero Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Quality distribution
        ax = axes[0, 2]
        for constant_name, data in constants_data.items():
            if data:
                qualities = [d['quality'] for d in data]
                ax.hist(qualities, bins=20, alpha=0.5, label=constant_name.replace('_', ' ').title())
        
        ax.set_xscale('log')
        ax.set_xlabel('Quality (log)')
        ax.set_ylabel('Frequency')
        ax.set_title('Quality Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Comparison of constant magnitudes
        ax = axes[1, 0]
        constants = list(self.FUNDAMENTAL_CONSTANTS.keys())
        values = [self.FUNDAMENTAL_CONSTANTS[c] for c in constants]
        
        bars = ax.bar(range(len(constants)), values)
        ax.set_yscale('log')
        ax.set_ylabel('Constant Value (log)')
        ax.set_title('Magnitudes of Fundamental Constants')
        ax.set_xticks(range(len(constants)))
        ax.set_xticklabels([c.replace('_', ' ').title() for c in constants], rotation=45)
        
        # Add values on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}',
                    ha='center', va='bottom', rotation=90)
        
        ax.grid(True, alpha=0.3)
        
        # 5. Best global resonances
        ax = axes[1, 1]
        if self.session_state.best_resonances:
            constants = list(self.session_state.best_resonances.keys())
            qualities = [res.quality for res in self.session_state.best_resonances.values()]
            
            bars = ax.bar(constants, qualities)
            ax.set_yscale('log')
            ax.set_ylabel('Quality (log)')
            ax.set_title('Best Global Resonances')
            ax.set_xticklabels([c.replace('_', ' ').title() for c in constants], rotation=45)
            
            # Add values on bars
            for bar, quality in zip(bars, qualities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{quality:.2e}',
                        ha='center', va='bottom', rotation=90)
        
        ax.grid(True, alpha=0.3)
        
        # 6. Estimated energies
        ax = axes[1, 2]
        if self.session_state.best_resonances:
            constants = list(self.session_state.best_resonances.keys())
            energies = [res.gamma/10 for res in self.session_state.best_resonances.values()]
            
            bars = ax.bar(constants, energies)
            ax.set_ylabel('Estimated Energy (GeV)')
            ax.set_title('Estimated Energies of Best Resonances')
            ax.set_xticklabels([c.replace('_', ' ').title() for c in constants], rotation=45)
            
            # Add values on bars
            for bar, energy in zip(bars, energies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{energy:.1f}',
                        ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = os.path.join(self.results_dir, f"comprehensive_visualizations_{timestamp}.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualizations saved: {viz_file}")
        
        # Close figure to free memory
        plt.close(fig)
    
    def run_analysis_optimized(self, force_reload=False):
        """Run complete analysis with resumption and optimized performance"""
        logger.info(f"🚀 ZVT FUNDAMENTAL CONSTANTS HUNTER")
        logger.info(f"🌌 Searching for resonances with 4 fundamental physics constants")
        logger.info("=" * 80)
        
        # Try to load session state
        session_loaded = self.load_session_state()
        
        if session_loaded:
            logger.info(f"🔄 Resuming previous session...")
        else:
            logger.info(f"🆕 Starting new session...")
        
        # Show constants
        for constant_name, constant_value in self.FUNDAMENTAL_CONSTANTS.items():
            tolerances_str = ", ".join([f"{t:.0e}" for t in self.CONSTANT_TOLERANCES.get(constant_name, [1e-4, 1e-5, 1e-6])[:3]])
            logger.info(f"🔬 {constant_name.upper().replace('_', ' ')}: {constant_value:.15e} (tolerances: {tolerances_str})")
        
        logger.info(f"\n📁 File: {self.zeros_file}")
        logger.info("🛑 Ctrl+C to stop (state will be saved)")
        logger.info("🚨 Significant resonances will be automatically highlighted")
        logger.info("=" * 80)
        
        # Load zeros
        self.all_zeros = self.load_enhanced_cache(force_reload=force_reload)
        current_count = len(self.all_zeros)
        
        if current_count == 0:
            logger.error("❌ No zeros loaded. Check the file.")
            return [], [], None
        
        # Update total zeros in state
        self.session_state.total_zeros = current_count
        
        logger.info(f"🎯 PROCESSING {current_count:,} ZEROS!")
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
            
            batch_result = self.analyze_batch_optimized(batch, batch_num)
            
            # Update information
            elapsed = time.time() - start_time
            batch_result['batch_time'] = elapsed
            batch_result['progress_percent'] = progress_percent
            
            # Update best global resonances
            if batch_result['best_resonances']:
                for constant_name, res in batch_result['best_resonances'].items():
                    if constant_name not in self.session_state.best_resonances or res.quality < self.session_state.best_resonances[constant_name].quality:
                        self.session_state.best_resonances[constant_name] = res
                        logger.info(f"    🎯 NEW GLOBAL BEST for {constant_name.upper()}!")
                        logger.info(f"    Zero #{res.zero_index:,} → γ={res.gamma:.12f}, quality={res.quality:.6e}")
            
            # Calculate performance statistics
            zeros_per_sec = len(batch) / elapsed if elapsed > 0 else 0
            remaining_zeros = current_count - batch_end
            eta_seconds = remaining_zeros / zeros_per_sec if zeros_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
            
            logger.info(f"⏱️ Batch processed in {elapsed:.1f}s ({zeros_per_sec:,.0f} zeros/s)")
            if eta_hours > 0:
                logger.info(f"📈 ETA for completion: {eta_hours:.1f} hours")
            
            self.session_state.session_results.append(batch_result)
            self.session_state.last_processed_index = batch_end
            
            # Save state periodically
            if batch_num % 10 == 0:  # Every 10 batches
                self.save_session_state()
            
            # Update progress bar
            pbar.update(len(batch))
            
            batch_num += 1
        
        # Close progress bar
        pbar.close()
        
        # Save final state
        self.save_session_state()
        
        # Generate final report
        self.generate_final_report()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Show best global resonances
        if self.session_state.best_resonances:
            logger.info(f"\n" + "🏆" * 60)
            logger.info(f"🏆 BEST GLOBAL RESONANCES FOUND 🏆")
            logger.info(f"🏆" * 60)
            logger.info("| Constant         | Zero #        | Gamma                | Quality        | Energy (GeV) |")
            logger.info("|-------------------|---------------|----------------------|------------------|---------------|")
            for constant_name, res in self.session_state.best_resonances.items():
                logger.info(f"| {constant_name:17s} | {res.zero_index:13,} | {res.gamma:20.15f} | {res.quality:.6e} | {res.gamma/10:13.6f} |")
            
            logger.info(f"\n💎 DETAILS OF BEST RESONANCES:")
            for constant_name, res in self.session_state.best_resonances.items():
                error_percent = (res.quality / self.FUNDAMENTAL_CONSTANTS[constant_name]) * 100
                logger.info(f"\n🔬 {constant_name.upper().replace('_', ' ')}:")
                logger.info(f"   Zero #{res.zero_index:,} (γ = {res.gamma:.15f})")
                logger.info(f"   Quality: {res.quality:.15e}")
                logger.info(f"   Relative error: {error_percent:.12f}%")
                logger.info(f"   Tolerance: {res.tolerance:.0e}")
                logger.info(f"   Estimated energy: {res.gamma/10:.6f} GeV")
            logger.info(f"🏆" * 60)
        
        # Final summary
        total_processed = self.session_state.last_processed_index
        elapsed_time = time.time() - self.session_state.start_time
        
        logger.info(f"\n" + "="*60)
        logger.info(f"📊 FINAL ANALYSIS SUMMARY")
        logger.info(f"="*60)
        logger.info(f"📈 Total zeros processed: {total_processed:,} of {current_count:,}")
        logger.info(f"📈 Percentage completed: {(total_processed/current_count)*100:.1f}%")
        logger.info(f"📈 Batches processed: {len(self.session_state.session_results)}")
        logger.info(f"📈 Total time: {elapsed_time:.1f}s ({elapsed_time/3600:.1f}h)")
        logger.info(f"📈 Average speed: {total_processed/elapsed_time:,.0f} zeros/s")
        logger.info(f"="*60)
        
        return self.all_zeros, self.session_state.session_results, self.session_state.best_resonances

# Main execution
if __name__ == "__main__":
    zeros_file = os.path.expanduser("~/zeta/zero.txt")  # Path to the zeros file
    
    # Check if should force reload
    force_reload = len(sys.argv) > 1 and sys.argv[1] == '--force-reload'
    
    try:
        logger.info("🌟 Starting ZVT Fundamental Constants Hunter")
        hunter = ZetaFundamentalConstantsHunter(zeros_file)
        zeros, results, best = hunter.run_analysis_optimized(force_reload=force_reload)
        
        if zeros and len(zeros) > 0:
            logger.info(f"\n🎯 Analysis Complete!")
            logger.info(f"📁 Results in: {hunter.results_dir}/")
            logger.info(f"💾 Cache: {hunter.cache_file}")
            logger.info(f"🔄 State: {hunter.state_file}")
    except KeyboardInterrupt:
        logger.info(f"\n⏹️ Analysis interrupted. State saved for resumption.")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info(f"\n🔬 Session completed!")
