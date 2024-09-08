import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
beta = 0.5
j = 5

Jx = jmat(j, 'x')
Jy = jmat(j, 'y')
Jz = jmat(j, 'z')

alpha = np.linspace(0, 15, 50)
phases_vs_alpha = np.zeros((len(alpha), (2*j+1)))

# Loop over alpha and compute the Floquet operator and its eigenvalues
for row, i in enumerate(alpha):
    F0 = (-1j * beta * Jx).expm()
    F1 = (-1j * i * (Jz * Jz) / (2 * j)).expm()  # Use .expm() for the correct matrix exponentiation
    F = F0 * F1
    F = F.full()
    eigs = np.linalg.eigvals(F)
    phases = np.angle(eigs)
    # Convert phases from [-π, π] to [0, 2π]
    phases = (phases + 2 * np.pi) % (2 * np.pi)
    phases_vs_alpha[row, :] = -phases


# Plotting for visualization
plt.figure(figsize=(10, 6))
for i in range(phases_vs_alpha.shape[1]):
    plt.plot(alpha, phases_vs_alpha[:, i], '.',markersize = 1)
plt.xlabel(r'$\alpha$')
plt.ylabel('Phase')
plt.title(r'Phases of Eigenvalues vs $\alpha$')
# plt.legend()
plt.show()

