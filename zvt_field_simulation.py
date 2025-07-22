import numpy as np
import matplotlib.pyplot as plt

tau = np.linspace(-10, 10, 1000)
A_0 = 1.0
tau_scale = 1.0

t_n_valid = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
t_n_invalid = [14.134725 * 1.2, 21.022040 * 1.2, 25.010858 * 1.2, 30.424876 * 1.2, 32.935062 * 1.2]
n_zeros = len(t_n_valid)

omega_n_valid = np.array(t_n_valid) / tau_scale
omega_n_invalid = np.array(t_n_invalid) / tau_scale
w_n_valid = np.array([np.log(n + 2) / (1 + 0.01 * t_n**2) for n, t_n in enumerate(t_n_valid)])
w_n_invalid = np.array([np.log(n + 2) / (1 + 0.01 * t_n**2) for n, t_n in enumerate(t_n_invalid)])

Phi_valid = np.zeros_like(tau)
Phi_invalid = np.zeros_like(tau)
for i in range(n_zeros):
    Phi_valid += w_n_valid[i] * A_0 * np.exp(-0.5 * (tau * omega_n_valid[i])**2) * np.cos(omega_n_valid[i] * tau)
    Phi_invalid += w_n_invalid[i] * A_0 * np.exp(-0.5 * (tau * omega_n_invalid[i])**2) * np.cos(omega_n_invalid[i] * tau)

rho_valid = np.abs(Phi_valid)**2
rho_invalid = np.abs(Phi_invalid)**2

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(tau, Phi_valid, color='#00CED1', label=r'$\Phi(\tau), \sigma = 1/2$')
ax1.plot(tau, Phi_invalid, color='#ff4500', label=r'$\Phi(\tau), \sigma \neq 1/2$')
ax1.set_title('Campo Vibracional sob Hipótese de Riemann', color='#e0e0e0')
ax1.set_xlabel(r'$\tau$', color='#e0e0e0')
ax1.set_ylabel(r'$\Phi(\tau)$', color='#e0e0e0')
ax1.legend()
ax1.grid(True, color='#333')

ax2.plot(tau, rho_valid, color='#00CED1', label=r'$\rho, \sigma = 1/2$')
ax2.plot(tau, rho_invalid, color='#ff4500', label=r'$\rho, \sigma \neq 1/2$')
ax2.set_title('Densidade de Energia', color='#e0e0e0')
ax2.set_xlabel(r'$\tau$', color='#e0e0e0')
ax2.set_ylabel(r'$\rho$', color='#e0e0e0')
ax2.legend()
ax2.grid(True, color='#333')

plt.tight_layout()
plt.savefig('riemann_hypothesis_test.png', dpi=300, bbox_inches='tight')
plt.show()
