#!/usr/bin/env python3
"""
revalidacao.py - Final version compatible with Qiskit 2.1.1+
Without deprecated dependencies
"""
import argparse
import json
import os
import time
import math
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

# Corrected imports for Qiskit 2.1.1
try:
    from qiskit import QuantumCircuit, transpile, qpy
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import QiskitRuntimeService
    # Removed: from qiskit.tools.monitor import job_monitor (deprecated)
except ImportError as e:
    print("Error importing Qiskit:", e)
    print("Execute: pip install qiskit-aer qiskit-ibm-runtime")
    raise

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    shots: int
    seed: int
    p0: float
    target_states: List[str]
    backends: List[str]
    qpy_path: Optional[str] = None
    max_retries: int = 3
    timeout: int = 3600

@dataclass
class ExecutionResult:
    backend: str
    circuit_index: int
    counts: Dict[str, int]
    summary: Dict[str, Union[float, int]]
    stats: Dict[str, float]
    execution_time: float

# Statistical functions
def se_binomial(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0

def z_two_sided(p_obs: float, p0: float, n: int) -> float:
    se = se_binomial(p0, n)
    return (p_obs - p0) / se if se != 0 else float('inf')

def p_from_z(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

def ic95(p_obs: float, n: int) -> Tuple[float, float]:
    se = se_binomial(p_obs, n)
    z = 1.96
    return max(0.0, p_obs - z * se), min(1.0, p_obs + z * se)

def effect_size(p_obs: float, p0: float) -> float:
    return 2 * math.asin(math.sqrt(p_obs)) - 2 * math.asin(math.sqrt(p0))

def bh_fdr(pvals: List[float]) -> List[float]:
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    for i, (idx, p) in enumerate(indexed, start=1):
        q = p * n / i
        adjusted[idx] = min(q, 1.0)
    return adjusted

def bonferroni_correction(pvals: List[float], alpha: float = 0.05) -> float:
    return alpha / max(1, len(pvals))

# Simplified Qiskit functions
def load_qpy(path: str) -> List[QuantumCircuit]:
    try:
        with open(path, 'rb') as fd:
            circuits = qpy.load(fd)
        logger.info(f"Loaded {len(circuits)} circuits from {path}")
        return circuits
    except Exception as e:
        logger.error(f"Failed to load QPY: {e}")
        raise

def generate_grover_circuit(n_qubits: int, marked_states: List[str]) -> QuantumCircuit:
    """Generates Grover circuit for marked states"""
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initial superposition
    for q in range(n_qubits):
        qc.h(q)
    
    # Grover iterations (simplified)
    for _ in range(1):
        # Oracle for marked states
        qc.barrier()
        for state in marked_states:
            # Prepare the state
            for i, bit in enumerate(reversed(state)):
                if bit == '0':
                    qc.x(i)
            
            # Marking with multi-controlled Z
            if n_qubits > 1:
                qc.h(n_qubits-1)
                qc.mcx(list(range(n_qubits-1)), n_qubits-1)
                qc.h(n_qubits-1)
            else:
                qc.z(0)
            
            # Undo preparation
            for i, bit in enumerate(reversed(state)):
                if bit == '0':
                    qc.x(i)
        
        # Diffuser
        qc.barrier()
        for q in range(n_qubits):
            qc.h(q)
            qc.x(q)
        
        qc.h(n_qubits-1)
        qc.mcx(list(range(n_qubits-1)), n_qubits-1)
        qc.h(n_qubits-1)
        
        for q in range(n_qubits):
            qc.x(q)
            qc.h(q)
    
    qc.barrier()
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def get_backend(backend_name: str):
    """Gets backend for execution"""
    try:
        if backend_name.startswith('ibm_'):
            service = QiskitRuntimeService()
            return service.backend(backend_name)
        elif backend_name == 'aer_simulator':
            return AerSimulator()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    except Exception as e:
        logger.error(f"Failed to get backend {backend_name}: {e}")
        raise

def monitor_job_simple(job, timeout: int = 300):
    """Simple job monitoring (replacement for job_monitor)"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            status = job.status()
            if status.name == 'DONE':
                logger.info("Job completed successfully")
                return True
            elif status.name in ['CANCELLED', 'ERROR']:
                logger.error(f"Job failed with status: {status.name}")
                return False
            else:
                logger.info(f"Job status: {status.name} ({int(time.time() - start_time)}s)")
                time.sleep(5)
        except Exception as e:
            logger.warning(f"Error checking status: {e}")
            time.sleep(5)
    
    logger.error("Timeout exceeded")
    return False

def run_on_backend(qc: QuantumCircuit, backend, config: ExperimentConfig) -> Tuple[Dict[str, int], Dict, float]:
    """Executes circuit with retry and timeout"""
    start_time = time.time()
    
    for attempt in range(config.max_retries):
        try:
            logger.info(f"Attempt {attempt+1}/{config.max_retries}")
            
            # Transpilation
            t_qc = transpile(qc, backend=backend, seed_transpiler=config.seed, optimization_level=1)
            
            # Execution
            job = backend.run(t_qc, shots=config.shots)
            
            # Simple monitoring
            if not monitor_job_simple(job, config.timeout):
                raise Exception("Job failed or timeout")
            
            result = job.result()
            counts = result.get_counts()
            
            execution_time = time.time() - start_time
            return counts, result, execution_time
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == config.max_retries - 1:
                logger.error("All attempts failed")
                raise
            time.sleep(5)

# Analysis pipeline
def summarize_counts(counts: Dict[str, int], target_states: List[str]) -> Dict[str, Union[float, int]]:
    total = sum(counts.values())
    succ = sum(counts.get(s, 0) for s in target_states)
    p_obs = succ / total if total > 0 else 0.0
    
    return {
        'total_shots': total,
        'success_counts': succ,
        'p_obs': p_obs,
        'effect_size': effect_size(p_obs, 0.25)
    }

def stat_summary(p_obs: float, p0: float, n: int) -> Dict[str, float]:
    z = z_two_sided(p_obs, p0, n)
    pval = p_from_z(z)
    ci_low, ci_high = ic95(p_obs, n)
    
    return {
        'z': z,
        'p_value': pval,
        'ci95_low': ci_low,
        'ci95_high': ci_high,
        'standard_error': se_binomial(p_obs, n)
    }

def execute_single_backend(backend_name: str, circuits: List[QuantumCircuit], config: ExperimentConfig) -> List[ExecutionResult]:
    """Executes circuits on a specific backend"""
    logger.info(f"Executing on backend: {backend_name}")
    
    try:
        backend_obj = get_backend(backend_name)
    except Exception as e:
        logger.error(f"Failed to configure backend {backend_name}: {e}")
        return []
    
    results = []
    for i, qc in enumerate(circuits):
        try:
            counts, qres, exec_time = run_on_backend(qc, backend_obj, config)
            summ = summarize_counts(counts, config.target_states)
            stats = stat_summary(summ['p_obs'], config.p0, summ['total_shots'])
            
            result = ExecutionResult(
                backend=backend_name,
                circuit_index=i,
                counts=counts,
                summary=summ,
                stats=stats,
                execution_time=exec_time
            )
            results.append(result)
            logger.info(f"Circuit {i} on {backend_name}: p_obs={summ['p_obs']:.4f}, p_value={stats['p_value']:.4g}")
            
        except Exception as e:
            logger.error(f"Failed to execute circuit {i} on {backend_name}: {e}")
    
    return results

def print_results_summary(results: List[ExecutionResult]):
    """Prints results summary"""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\nBackend: {result.backend}")
        print(f"Circuit: {result.circuit_index}")
        print(f"Success rate: {result.summary['p_obs']:.4f} ({result.summary['success_counts']}/{result.summary['total_shots']})")
        print(f"P-value: {result.stats['p_value']:.4g}")
        print(f"CI95: [{result.stats['ci95_low']:.4f}, {result.stats['ci95_high']:.4f}]")
        print(f"Effect size: {result.summary['effect_size']:.4f}")
        print(f"Execution time: {result.execution_time:.2f}s")
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Experimental revalidation script for ZVT")
    parser.add_argument('--qpy', help='Path to QPY file with circuits')
    parser.add_argument('--shots', type=int, default=20000, help='Number of shots')
    parser.add_argument('--backends', type=str, default='aer_simulator', help='Backends')
    parser.add_argument('--out', type=str, default='revalidacao_results.json', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--target_states', type=str, default='0000,0001,0010,0011', help='Target states')
    parser.add_argument('--p0', type=float, default=0.25, help='Baseline')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum attempts')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout (seconds)')
    
    args = parser.parse_args()
    
    # Configuration
    config = ExperimentConfig(
        shots=args.shots,
        seed=args.seed,
        p0=args.p0,
        target_states=[s.strip() for s in args.target_states.split(',') if s.strip()],
        backends=[b.strip() for b in args.backends.split(',') if b.strip()],
        qpy_path=args.qpy,
        max_retries=args.max_retries,
        timeout=args.timeout
    )
    
    logger.info(f"Configuration: {config}")
    
    # Load or generate circuits
    circuits = []
    if args.qpy:
        circuits = load_qpy(args.qpy)
    else:
        n_qubits = len(config.target_states[0])
        circuits = [generate_grover_circuit(n_qubits, config.target_states)]
    
    logger.info(f"Processing {len(circuits)} circuits on {len(config.backends)} backends")
    
    # Execution
    all_results = []
    for backend in config.backends:
        results = execute_single_backend(backend, circuits, config)
        all_results.extend(results)
    
    # Statistical corrections
    if all_results:
        pvals = [r.stats['p_value'] for r in all_results]
        bonf_alpha = bonferroni_correction(pvals)
        fdr_adj = bh_fdr(pvals)
        
        for i, result in enumerate(all_results):
            result.stats['p_value_bonferroni_threshold'] = bonf_alpha
            result.stats['p_value_fdr_adj'] = fdr_adj[i]
    
    # Print summary
    print_results_summary(all_results)
    
    # Save results
    final_results = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'shots': config.shots,
            'seed': config.seed,
            'backends': config.backends,
            'p0': config.p0,
            'target_states': config.target_states,
            'total_executions': len(all_results)
        },
        'results': [r.__dict__ for r in all_results]
    }
    
    with open(args.out, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved to {args.out}")

if __name__ == '__main__':
    main()
