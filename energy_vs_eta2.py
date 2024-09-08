import numpy as np
import matplotlib.pyplot as plt
from qutip import * 

# Parameters
j = 7
alpha = [1.2,1.5]
beta = [1,2,3]

# Jx, Jy,Jz matrix
Jx = jmat(j, 'x')
Jy = jmat(j, 'y')
Jz = jmat(j, 'z')

# Range of eta
eta_val = np.linspace(0.5,3,1000)

# Phases vs eta matrix(Each row is different for different eta value)
phases_vs_eta = np.zeros((len(eta_val), (2*j +1)))

# Loop over eta and compute Foquet operator and eigenvalues
for row,i in enumerate(eta_val):
    F2 = (-1j * beta[2] * (Jz * Jz) / (2 * j + 1)).expm() * (-1j * i * Jz).expm()
    F1 = (-1j * beta[1] * (Jy * Jy) / (2 * j + 1)).expm() * (-1j * alpha[1] * Jy).expm()
    F0 = (-1j * beta[0] * (Jx * Jx) / (2 * j + 1)).expm() * (-1j * alpha[0] * Jx).expm()
    F = F2*F1*F0
    F = F.full()
    eigs = np.linalg.eigvals(F)
    phases = np.angle(eigs)
    phases_vs_eta[row,:] = -phases

# plotting for energy 
plt.figure(figsize=(10,6))
for i in range(phases_vs_eta.shape[1]):
    plt.plot(eta_val, phases_vs_eta[:, i], '.', markersize = 0.5, color = 'k')
# plt.ylim(-2.5,2.5)
plt.xlabel(r'$\eta$')
plt.ylabel('phases')
plt.title(r'Phases of Eigenvalues vs $\eta$')
plt.show()