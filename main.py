from qutip import *
import numpy as np
import matplotlib.pyplot as plt

#Parameters
N = 800 #no of states in hilbart space
eta = 0.25
k1 = 0.2
k2= 0.225
del_k = k2-k1
alpha = np.pi/(2*eta) # Initial eigenstate of coherent state
#alpha = 1j*np.pi/(np.sqrt(3)*eta)
#Oparetors
a = destroy(N) #Anhilator
psi_0 = coherent(N,alpha) #Initial coherent state
psi_0_dag = coherent(N,alpha).dag() # complex conjugate of Initial coherent state
wt = 2*np.pi/6

# Cosine function 
''' Or you can use the `cosm()`function '''
def cosine(Op):
    return 0.5*((1j*Op).expm() + (-1j*Op).expm())

# Floquet time evolution Operators
F0 = (-1j*a.dag()*a*wt).expm()
F1 = (-1j*k1*cosine(2*eta*(a.dag()+a))/(np.sqrt(2)*eta**2)).expm()
F_1 = F0*F1
F2 = (-1j*k2*cosine(2*eta*(a.dag()+a))/(np.sqrt(2)*eta**2)).expm()
F_2 = (F0*F2).dag()

# Initial inner product
inner_product = psi_0_dag*psi_0

ip = [] #Array to store inner product for each kick
ip.append(inner_product[0][0][0]) 

for i in range(1,1000):
    psi_0 = F_1*psi_0
    psi_0_dag = psi_0_dag*F_2
    inner_product = psi_0_dag*psi_0
    ip.append(inner_product[0][0][0]) 

# No. of pulse 
n = np.arange(0, 1000, 1)
# probability to find the system in g1 ground state
P_g = 0.5*(1 - np.cos(del_k*n/(np.sqrt(2)*eta**2))*np.real(ip)
            + np.sin(del_k*n/(np.sqrt(2)*eta**2))*np.imag(ip))

# probability to find the system in g2 ground state
P_g_prime = 0.5*(1 - np.sin(del_k*n/(np.sqrt(2)*eta**2))*np.real(ip) 
                 - np.cos(del_k*n/(np.sqrt(2)*eta**2))*np.imag(ip))

# Plotting
plt.rc('font', **{'family': 'serif', 'size': '13'})
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_ylim(0,1)
ax1.set_xlim(0,1000)
ax1.plot(n,P_g, 'b',label='$P_g$')
ax1.plot(n,P_g_prime, '-r', label = '$P_g^\prime$')
ax1.legend()

ax2.set_xlim(0,1000)
ax2.set_ylim(-1,1)
ax2.plot(n,(np.abs(ip))**2, 'b',linewidth=1.3, label='$o$')
ax2.plot(n,np.real(ip),'-r',linewidth=1.3,
          label = '$Re(\langle\\alpha|\hat{U}_2^\dag \hat{U}_1|\\alpha\\rangle)$')
ax2.plot(n,np.imag(ip),'-g', linewidth=1.3, 
         label = '$Im(\langle\\alpha|\hat{U}_2^\dag \hat{U}_1|\\alpha\\rangle)$')
ax2.legend(fontsize ='10', loc = 'lower right')

plt.xlabel("$n$")



plt.show()     

