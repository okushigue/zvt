import numpy as np
import matplotlib.pyplot as plt

# Configurações
tau = np.linspace(-10, 10, 1000)  # Escala de tempo adimensional
A_0 = 1.0  # Amplitude base
tau_scale = 1.0  # Escala de tempo para normalizar frequências

# Zeros não-triviais da função zeta (valores aproximados)
zeta_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
n_zeros = len(zeta_zeros)

# Frequências e pesos
omega_n = np.array(zeta_zeros) / tau_scale
w_n = np.array([np.log(n + 2) / (1 + 0.01 * t_n**2) for n, t_n in enumerate(zeta_zeros)])

# Campo Phi(tau)
Phi = np.zeros_like(tau)
for i in range(n_zeros):
    Phi += w_n[i] * A_0 * np.exp(-0.5 * (tau * omega_n[i])**2) * np.cos(omega_n[i] * tau)

# Densidade rho = |Phi|^2
rho = np.abs(Phi)**2

# Visualização
plt.style.use('dark_background')  # Tema escuro para alinhar com a página HTML
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot do campo Phi
ax1.plot(tau, Phi, color='#00CED1', label=r'$\Phi(\tau)$')
ax1.set_title('Campo Vibracional da ZVT', color='#e0e0e0')
ax1.set_xlabel(r'$\tau$', color='#e0e0e0')
ax1.set_ylabel(r'$\Phi(\tau)$', color='#e0e0e0')
ax1.legend()
ax1.grid(True, color='#333')

# Plot da densidade rho
ax2.plot(tau, rho, color='#00CED1', label=r'$\rho = |\Phi|^2$')
ax2.set_title('Densidade de Energia (Matéria Escura e Galáxias)', color='#e0e0e0')
ax2.set_xlabel(r'$\tau$', color='#e0e0e0')
ax2.set_ylabel(r'$\rho$', color='#e0e0e0')
ax2.legend()
ax2.grid(True, color='#333')

plt.tight_layout()
plt.savefig('zvt_field_simulation.png', dpi=300, bbox_inches='tight')
plt.show()
