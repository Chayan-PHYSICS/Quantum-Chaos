import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameter
N = 2**10  # Dimension of Hilbert space
q = 6
K = 2
alpha = 2*np.pi/q
eta = [0.459,0.469,0.464]
a = destroy(N)
psi0 = coherent(N,0)
# no of kicks 
n = 600

'''classical mean energy calculation'''
n_sample = 1000 # Number of trajectories for averaging
c_energy = np.zeros(shape=n) # Array to store the energy after each kick
init_pts = 1.56*np.random.rand(n_sample,2) # initial conditions 

# Vectorized loop over initial conditions
v, u = init_pts[:, 0], init_pts[:, 1]

# Precompute trig functions
cos_alpha = np.cos(alpha)
sin_alpha = np.sin(alpha)

for i in range(1, n):
    #The classical mapping
    v_new = v * cos_alpha + sin_alpha * (u + K * np.sin(v))
    u_new = (u + K * np.sin(v)) * cos_alpha - v * sin_alpha
    
    # Update v and u
    v, u = v_new, u_new
    # Calculate energy and accumulate
    E_n = 0.5 * (v**2 + u**2)
    c_energy[i] = np.mean(E_n)  # Vectorized mean energy calculation

#-----------------------------------@@@@@@@@@----------------------------------

'''Quantum energy calculation'''
# Cosine function 
def cosine(Op):
    ''' Or you can use the `cosm()`function'''
    return 0.5*((1j*Op).expm() + (-1j*Op).expm())

def mean_energy(psi0,eta,n):
    # Floquet operator
    F0 = (-1j*alpha*a.dag()*a).expm()
    F1_cosine = cosine(eta * (a.dag() + a))
    F1 = (-1j * (K / (2 * eta**2)) * F1_cosine).expm()
    F = F0*F1

    # Energy of HO
    H = a.dag()*a
    # mean energy array 
    energy = np.zeros(shape=n,dtype='complex')
    # Zeroth element (mean energy without kicks)
    energy[0] = (psi0.dag() * H * psi0).full()[0, 0]

    #evolve psi according to the kick and calculate the mean energy
    for i in range(1,n):
        psi0 = F*psi0
        E = (psi0.dag() * H * psi0).full()[0, 0]
        energy[i] = E

    return energy

'''plot'''
# Plot classical mean energy
plt.plot(c_energy, label='Classical')

# plot Quantum mean energy 
for eta in eta :
    energy = mean_energy(psi0,eta,n)
    plt.plot(np.real(energy),label = f"$\eta$ = {eta}")

plt.xlabel("$n$(No. of kicks)")
plt.ylabel("Mean energy")
plt.legend()
plt.show()