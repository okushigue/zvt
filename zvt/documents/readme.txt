ZETA VIBRATION THEORY (ZVT) - COMPUTATIONAL FRAMEWORK
====================================================

REQUIREMENTS:
------------
Python 3.8 or higher

DEPENDENCIES:
--------------
Install the necessary packages with pip:

pip install numpy scipy matplotlib pandas tqdm
pip install numba h5py seaborn jupyter

For quantum simulations (optional):
pip install qiskit

The cache is already included with over 2 million zeros.

HOW TO RUN THE SCRIPTS:
----------------------

1. Resonance Analysis:
python scripts/analysis/find_resonances.py --input data/zeta_zeros/zeros1.txt --output data/results/resonances.json

2. Analysis of Fundamental Forces: 
python scripts/analysis/find_resonances.py --constants forces --input data/zeta_zeros/zeros1.txt --output data/results/forces_resonances.json

3. Analysis of Particle Constants: 
python scripts/analysis/find_resonances.py --constants particles --input data/zeta_zeros/zeros1.txt --output data/results/particles_resonances.json

4. Search for the Z' Boson: 
python scripts/experiments/z_prime_search.py ​​--coupling 0.1 --mass-range 1.0 2.0 --output data/results/z_prime_prediction.json

5. Cosmological Tests: 
python scripts/experiments/cosmological_tests.py --input data/zeta_zeros/zeros1.txt --output data/results/cosmological_params.json

6. Generate Charts: 
python scripts/analysis/plot_results.py --input data/results/resonances.json --output plots/

EXAMPLE OF USE IN PYTHON:
-------------------------
import numpy as np
from scripts.core.zvt_theory import ZetaVibrationTheory
from scripts.utils.data_loader import load_zeta_zeros

# Load zeros from zeta
zeros = load_zeta_zeros('data/zeta_zeros/zeros1.txt', max_zeros=100000)

# Initialize ZVT
zvt = ZetaVibrationTheory(zeros)

# Find resonances
resonances = zvt.find_all_resonances()
print(f"Spectral constant C: {zvt.spectral_constant(): .6f}")

# Analyze specific resonance
em_resonance = zvt.find_resonance('electromagnetic')
print(f"EM Resonance: Zero #{em_resonance['zero_index']}, Quality: {em_resonance['quality']: .2e}")

FOR MORE INFORMATION:
------------------------
- Full documentation: docs/
- Interactive notebooks: notebooks/
- Report issues: https://github.com/okushigue/zvt/issues
