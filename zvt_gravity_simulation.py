import numpy as np
import matplotlib.pyplot as plt

# Configurações
x = np.linspace(-10, 10, 1000)  # Coordenada espacial adimensional
tau = np.linspace(-10, 10, 1000)  # Escala de tempo adimensional
A_0 = 1.0  # Amplitude base
tau_scale = 1.0  # Escala de tempo

# Zeros não-triviais da função zeta
zeta_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
n_zeros = len(zeta_zeros)

# Frequências e pesos
omega_n = np.array(zeta_zeros) / tau_scale
w_n = np.array([np.log(n + 2) / (1 + 0.01 * t_n**2) for n, t_n in enumerate(zeta_zeros)])

# Campo Phi(x, tau) em 2D (simplificado para uma dimensão espacial)
X, T = np.meshgrid(x, tau)
Phi = np.zeros_like(X)
for i in range(n_zeros):
    Phi += w_n[i] * A_0 * np.exp(-0.5 * (T * omega_n[i])**2) * np.cos(omega_n[i] * T)

# Gradiente de Phi (para gravidade)
dPhi_dx = np.gradient(Phi, x, axis=1)
dPhi_dt = np.gradient(Phi, tau, axis=0)

# Métrica simplificada g_00 ~ (dPhi_dt)^2 (componente temporal)
g_00 = dPhi_dt**2

# Visualização
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
c = ax.contourf(X, T, g_00, cmap='viridis')
cbar = plt.colorbar(c, ax=ax)
cbar.set_label('Métrica g_00 (Curvatura)', color='#e0e0e0')
ax.set_title('Curvatura do Espaço-Tempo (ZVT)', color='#e0e0e0')
ax.set_xlabel('x (Espaço)', color='#e0e0e0')
ax.set_ylabel(r'$\tau$ (Tempo)', color='#e0e0e0')
plt.savefig('zvt_gravity_simulation.png', dpi=300, bbox_inches='tight')
plt.show()
