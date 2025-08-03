#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_ALPHA_HUNTER_10K.py - Zeta/α resonance hunter for 10k zeros
Author: Jefferson M. Okushigue
Date: 2025-07-26
"""
import numpy as np
from mpmath import mp
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import pickle
import os

# ----------------- CORE CONFIGURATION -----------------
mp.dps = 50  # Absolute precision
N_ZEROS = 10000
ALPHA = 1 / 137.035999084
TOLERANCE = 1e-4
MAX_WORKERS = os.cpu_count()  # Use all cores
CACHE_FILE = "zeta_zeros_10k.pkl"

# ----------------- PARALLELIZED FUNCTION -----------------
def compute_zeros_range(start, end):
    return [(k, float(mp.im(mp.zetazero(k)))) for k in range(start, end+1)]

# ----------------- HUNT RESONANCES -----------------
def find_resonances(zeros):
    return [
        (n, gamma, gamma % ALPHA)
        for n, gamma in zeros
        if (gamma % ALPHA) < TOLERANCE or (ALPHA - (gamma % ALPHA)) < TOLERANCE
    ]

# ----------------- LOAD/CALCULATE ZEROS -----------------
def get_zeros():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    
    chunk_size = N_ZEROS // MAX_WORKERS
    ranges = [(i*chunk_size+1, (i+1)*chunk_size) for i in range(MAX_WORKERS)]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(compute_zeros_range, *zip(*ranges)))
    
    zeros = []
    for result in results:
        zeros.extend(result)
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(zeros, f)
    
    return zeros

# ----------------- ADVANCED ANALYSIS -----------------
def enhanced_analysis(resonances):
    # Cross patterns with π and φ
    patterns = []
    for n, gamma, mod in resonances:
        phi_pattern = gamma % (1.61803398875)
        pi_pattern = gamma % (np.pi)
        patterns.append((n, gamma, mod, phi_pattern, pi_pattern))
    return patterns

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    print(f"🔍 STARTING SEARCH IN {N_ZEROS} ZEROS (α = {ALPHA:.10f})...")
    start_time = time.time()
    
    # Step 1: Get zeros (cache or calculation)
    zeros = get_zeros()
    print(f"⚡ ZEROS LOADED/CALCULATED IN {time.time()-start_time:.2f}s")
    
    # Step 2: Hunt resonances
    resonances = find_resonances(zeros)
    resonances.sort(key=lambda x: x[2])  # Sort by resonance
    
    # Step 3: Deep analysis
    enhanced_data = enhanced_analysis(resonances[:100])  # Top 100
    
    # ----------------- RESULTS -----------------
    print(f"\n🎯 {len(resonances)} RESONANT ZEROS FOUND (γ mod α < {TOLERANCE})")
    print("| Zero # | γ (Imag.) | γ mod α | φ-mod | π-mod |")
    print("|--------|-----------|---------|-------|-------|")
    for data in enhanced_data[:20]:
        print(f"| {data[0]} | {data[1]:.5f} | {data[2]:.7f} | {data[3]:.5f} | {data[4]:.5f} |")
    
    # ----------------- 3D PLOT (OPTIONAL) -----------------
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    x = [d[0] for d in enhanced_data]
    y = [d[3] for d in enhanced_data]  # φ-mod
    z = [d[4] for d in enhanced_data]  # π-mod
    
    ax.scatter(x, y, z, c='cyan', s=50, alpha=0.7)
    ax.set_xlabel('Zero Index (n)')
    ax.set_ylabel('γ mod φ (Golden Ratio)')
    ax.set_zlabel('γ mod π')
    ax.set_title(f'CROSS PATTERNS IN {len(resonances)} RESONANT ZEROS', pad=20)
    
    plt.savefig('ZVT_3D_Patterns.png', dpi=300)
    print("\n📊 3D PLOT SAVED: 'ZVT_3D_Patterns.png'")

    # ----------------- ZVT PREDICTIONS -----------------
    if len(resonances) > 0:
        best_zero = min(resonances, key=lambda x: x[2])
        print(f"\n💎 BEST ZERO FOUND:")
        print(f"   • Zero #{best_zero[0]} → γ = {best_zero[1]:.10f}")
        print(f"   • γ mod α = {best_zero[2]:.10f} (Error: {best_zero[2]/ALPHA*100:.5f}%)")
        print(f"   • ZVT Prediction: New particle at ~{best_zero[1]/10:.2f} GeV")

    print(f"\n⏱ TOTAL TIME: {time.time()-start_time:.2f} seconds")
