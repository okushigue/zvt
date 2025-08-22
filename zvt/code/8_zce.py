#!/usr/bin/env python3
"""
zvt_constants_experiment.py - Experimento ZVT com Constantes Físicas
Incorpora as constantes dos relatórios do Jefferson para testar ressonâncias com zeros de Riemann
"""
import argparse
import json
import os
import time
import math
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
except ImportError as e:
    print("Erro ao importar Qiskit:", e)
    raise

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constantes Físicas dos Relatórios do Jefferson
PHYSICAL_CONSTANTS = {
    'fine_structure': 7.297352569283802e-03,  # α
    'speed_of_light': 2.997924580000000e+08,  # c
    'gravitational': 6.674299999999999e-11,  # G
    'hbar': 1.054571817000000e-34,  # ħ
    'electron_mass': 9.109383701500001e-31,  # m_e
    'proton_mass': 1.672621923690000e-27,  # m_p
    'elementary_charge': 1.602176634000000e-19,  # e
    'bohr_radius': 5.291772109030000e-11,  # a₀
}

# Zeros de Riemann relevantes dos relatórios
RIEMANN_ZEROS = {
    'alpha_zero': 87144.853030040001613,  # Zero #118412 (conectado com α)
    'gravitational_zero': 508397.511089391016867,  # Zero #833507
    'hbar_zero': 1051303.966566361021250,  # Zero #1845153
    'proton_mass_zero': 123502.256272351994994,  # Zero #174667
    'electron_mass_zero': 953397.367270938004367,  # Zero #1658483
}

@dataclass
class ZVTExperimentConfig:
    shots: int
    seed: int
    constant_name: str
    zero_name: str
    backends: List[str]
    max_retries: int = 3
    timeout: int = 300

@dataclass
class ZVTResult:
    constant_name: str
    zero_name: str
    backend: str
    counts: Dict[str, int]
    success_rate: float
    p_value: float
    effect_size: float
    resonance_strength: float
    execution_time: float

def constant_to_angle(constant_value: float, scale_factor: float = 1e20) -> float:
    """Converte constante física para ângulo de rotação quântica"""
    # Normaliza o valor para o intervalo [0, 2π]
    normalized = (constant_value * scale_factor) % (2 * math.pi)
    return normalized

def zero_to_phase(zero_value: float, scale_factor: float = 1e-5) -> float:
    """Converte zero de Riemann para fase quântica"""
    # Normaliza para fase [0, 2π]
    phase = (zero_value * scale_factor) % (2 * math.pi)
    return phase

def create_zvt_circuit(constant_name: str, zero_name: str, n_qubits: int = 4) -> QuantumCircuit:
    """Cria circuito quântico que incorpora constante e zero de Riemann"""
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Obtém valores
    constant_value = PHYSICAL_CONSTANTS[constant_name]
    zero_value = RIEMANN_ZEROS[zero_name]
    
    # Converte para parâmetros quânticos
    theta = constant_to_angle(constant_value)
    phase = zero_to_phase(zero_value)
    
    logger.info(f"Constante {constant_name}: {constant_value:.2e} → θ = {theta:.4f}")
    logger.info(f"Zero {zero_name}: {zero_value:.2f} → φ = {phase:.4f}")
    
    # Estado inicial com superposição
    for q in range(n_qubits):
        qc.h(q)
    
    # Aplica rotações baseadas na constante física
    for i, q in enumerate(range(n_qubits)):
        angle = theta * (i + 1) / n_qubits
        qc.rz(angle, q)
    
    # Aplica fase baseada no zero de Riemann
    qc.p(phase, 0)
    
    # Porta de entrelaçamento controlada pela constante
    if n_qubits > 1:
        control_angle = theta / math.pi
        qc.crx(control_angle, 0, 1)
    
    # Medição
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc

def calculate_resonance_strength(success_rate: float, expected_rate: float = 0.25) -> float:
    """Calcula a força da ressonância"""
    # Ressonância perfeita = 1.0, sem ressonância = 0.0
    if expected_rate == 0:
        return 0.0
    
    ratio = success_rate / expected_rate
    # Normaliza para [0, 1]
    resonance = min(ratio, 1.0) if ratio >= 1.0 else ratio
    return resonance

def se_binomial(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0

def z_two_sided(p_obs: float, p0: float, n: int) -> float:
    se = se_binomial(p0, n)
    return (p_obs - p0) / se if se != 0 else float('inf')

def p_from_z(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

def effect_size(p_obs: float, p0: float) -> float:
    return 2 * math.asin(math.sqrt(p_obs)) - 2 * math.asin(math.sqrt(p0))

def run_zvt_experiment(config: ZVTExperimentConfig) -> List[ZVTResult]:
    """Executa experimento ZVT completo"""
    logger.info(f"Executando experimento: {config.constant_name} + {config.zero_name}")
    
    # Cria circuito
    qc = create_zvt_circuit(config.constant_name, config.zero_name)
    
    results = []
    
    for backend_name in config.backends:
        try:
            # Configura backend
            if backend_name == 'aer_simulator':
                backend = AerSimulator()
            else:
                logger.warning(f"Backend {backend_name} não suportado, usando aer_simulator")
                backend = AerSimulator()
            
            # Transpila e executa
            t_qc = transpile(qc, backend=backend, seed_transpiler=config.seed, optimization_level=1)
            job = backend.run(t_qc, shots=config.shots)
            
            # Monitoramento simples
            start_time = time.time()
            while time.time() - start_time < config.timeout:
                try:
                    status = job.status()
                    if status.name == 'DONE':
                        break
                    elif status.name in ['CANCELLED', 'ERROR']:
                        raise Exception(f"Job failed: {status.name}")
                    time.sleep(1)
                except:
                    time.sleep(1)
            
            result = job.result()
            counts = result.get_counts()
            
            # Calcula estatísticas
            total = sum(counts.values())
            # Considera "sucesso" os estados com maior probabilidade (top 25%)
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            success_states = sorted_counts[:max(1, len(sorted_counts) // 4)]
            success_count = sum(count for state, count in success_states)
            success_rate = success_count / total if total > 0 else 0.0
            
            # Estatísticas
            p_value = p_from_z(z_two_sided(success_rate, 0.25, total))
            e_size = effect_size(success_rate, 0.25)
            resonance = calculate_resonance_strength(success_rate)
            
            execution_time = time.time() - start_time
            
            zvt_result = ZVTResult(
                constant_name=config.constant_name,
                zero_name=config.zero_name,
                backend=backend_name,
                counts=counts,
                success_rate=success_rate,
                p_value=p_value,
                effect_size=e_size,
                resonance_strength=resonance,
                execution_time=execution_time
            )
            
            results.append(zvt_result)
            
            logger.info(f"Resultado: taxa={success_rate:.4f}, ressonância={resonance:.4f}, p-value={p_value:.2e}")
            
        except Exception as e:
            logger.error(f"Falha no backend {backend_name}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Experimento ZVT com Constantes Físicas")
    parser.add_argument('--shots', type=int, default=10000, help='Número de shots')
    parser.add_argument('--backends', type=str, default='aer_simulator', help='Backends')
    parser.add_argument('--out', type=str, default='zvt_results.json', help='Arquivo de saída')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout')
    parser.add_argument('--all_combinations', action='store_true', help='Testar todas combinações')
    parser.add_argument('--constant', help='Constante específica para testar')
    parser.add_argument('--zero', help='Zero específico para testar')
    
    args = parser.parse_args()
    
    # Prepara combinações para testar
    combinations = []
    
    if args.all_combinations:
        # Testa todas combinações de constantes e zeros
        for constant in PHYSICAL_CONSTANTS.keys():
            for zero in RIEMANN_ZEROS.keys():
                combinations.append((constant, zero))
    elif args.constant and args.zero:
        # Testa combinação específica
        combinations.append((args.constant, args.zero))
    else:
        # Testa combinações selecionadas (mais relevantes)
        combinations = [
            ('fine_structure', 'alpha_zero'),
            ('gravitational', 'gravitational_zero'),
            ('hbar', 'hbar_zero'),
            ('electron_mass', 'electron_mass_zero'),
            ('proton_mass', 'proton_mass_zero'),
        ]
    
    logger.info(f"Testando {len(combinations)} combinações")
    
    # Executa experimentos
    all_results = []
    backends = [b.strip() for b in args.backends.split(',') if b.strip()]
    
    for constant_name, zero_name in combinations:
        config = ZVTExperimentConfig(
            shots=args.shots,
            seed=args.seed,
            constant_name=constant_name,
            zero_name=zero_name,
            backends=backends,
            timeout=args.timeout
        )
        
        results = run_zvt_experiment(config)
        all_results.extend(results)
    
    # Salva resultados
    final_results = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'shots': args.shots,
            'seed': args.seed,
            'total_experiments': len(all_results),
            'physical_constants': PHYSICAL_CONSTANTS,
            'riemann_zeros': RIEMANN_ZEROS
        },
        'results': [r.__dict__ for r in all_results]
    }
    
    with open(args.out, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Resultados salvos em {args.out}")
    
    # Imprime resumo
    print("\n" + "="*80)
    print("RESUMO DOS EXPERIMENTOS ZVT")
    print("="*80)
    print(f"{'Constante':<15} {'Zero':<15} {'Taxa':<8} {'Ressonância':<10} {'p-value':<12} {'Efeito':<8}")
    print("-" * 80)
    
    for r in all_results:
        print(f"{r.constant_name:<15} {r.zero_name:<15} {r.success_rate:<8.4f} {r.resonance_strength:<10.4f} {r.p_value:<12.2e} {r.effect_size:<8.4f}")
    
    # Encontra as melhores ressonâncias
    best_resonances = sorted(all_results, key=lambda x: x.resonance_strength, reverse=True)[:5]
    print("\nTOP 5 RESSONÂNCIAS:")
    print("-" * 50)
    for i, r in enumerate(best_resonances, 1):
        print(f"{i}. {r.constant_name} + {r.zero_name}: {r.resonance_strength:.4f}")

if __name__ == '__main__':
    main()
