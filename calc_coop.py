import numpy as np
import matplotlib.pyplot as plt
from utils import dbm_to_mw

''' Calculates reflectivity and cooperativity for cavity '''

# def resonance_shape(f, fa, fm, ka, km, gam, kext, mismatch):
#    return 1 - np.exp(1j*mismatch)*kext/(1j*(f - fa) + ka/2 + gam**2/(1j*(f - fm) + km/2))
from hybridmode import resonance_shape

dbm = 17
hbar = 1.05457182e-34
sphere_choice = 0

if sphere_choice == 0: # YIG on a stick
    g_am = 9.54e6
    gamma_m = 1.47e6
    Omega_b = 12.457e6
    gmb0 = 4.5e-3
    Gamma_b = 3600
    omega_a = 7.111e9
    #k = 6.3e6
    kappa_a = 5.4e6
    #kappa_a = 6.3e6
    k1 = 2.62e6
    #k2 = 1.068e6
    k2 = 0
    kint = kappa_a - k1 - k2
    #k2 = 1e6
    #k1 = 3e6
    #k2 = 3e6
    k1 = 2.72e6
    k = kint + k1 + k2
    mismatch = 0
    omega_m = np.linspace(7.05e9, 7.2e9, 100)

elif sphere_choice == 1: # PRX YIG
    g_am = 5.43e6
    gamma_m = 1.01e6
    Omega_b = 12.627e6
    gmb0 = 4.38e-3
    Gamma_b = 286
    omega_a = 7.077e9
    kint = 1.56e6
    k1 = 1.11e6
    k2 = 1.20e6
    k = kint + k1 + k2
    mismatch = 0
    omega_m = np.linspace(7.05e9, 7.2e9, 100)

elif sphere_choice == 2: # 200um YIG
    g_am = 5129660.642938008
    gamma_m = 2517511.12518074
    Omega_b = 15606699.501112761
    gmb0 = 3e-3
    Gamma_b = 260
    omega_a = 7337048981.135661
    k1 = 486187.65041748405
    k2 = 490000
    k = 2815384.133676843
    kint = k - k1 - k2
    k1 = 1.2e6
    k2 = 0
    k = kint + k1 + k2
    mismatch = 0
    omega_m = np.linspace(7.3e9, 7.4e9, 100)

Sam = (omega_a + omega_m)/2
Dam = (omega_a - omega_m)/2
delta = (g_am**2 + Dam**2)**.5

omega_minus = Sam - delta
omega_plus = Sam + delta

omega_d = omega_minus - Omega_b

theta = np.arctan2(2*g_am, omega_m - omega_a)/2
print(theta/np.pi)

cos, sin = np.cos(theta), np.sin(theta)

k_e_minus = (1 + np.cos(2*theta))*k1/2
k_minus = cos**2*k + sin**2*gamma_m

'''
for kp, ke, km in zip(omega_minus, k_e_minus, k_minus):
    print(f'{kp:10g} {ke:10g} {km:10g}')
'''

P = dbm_to_mw(dbm)/1000
epsilon_d = (P/(hbar*2*np.pi*omega_d))**.5

# resonance_shape(f, fa, fm, ka, km, gam, kext, mismatch):
bottom = resonance_shape(omega_minus, omega_a, omega_m, kint + k1 + k2,
                gamma_m, g_am, k1, mismatch)

twopi = 2*np.pi

# -------------------------------------
Delta_a = omega_d - omega_a
Delta_m = omega_d - omega_m

a_expect = (1j*Delta_m - gamma_m/2)*epsilon_d*k1**.5/ \
        ((1j*Delta_a - k/2)*(1j*Delta_m - gamma_m/2) + g_am**2)

m_expect = 1j*g_am*a_expect/(1j*Delta_m - gamma_m/2)
print(np.abs(m_expect)**2)
C1 = np.abs(m_expect)**2 * gmb0**2/(k_minus*Gamma_b) # magnon cooperativity
refl1 = (1 - 2*(k_e_minus/k_minus) + C1)/(1 + C1)
# --------------------------------------

#C = 4*P*gmb0**2/(hbar*omega_d*twopi*Omega_b**2*Gamma_b)*k1*sin**4*cos**2/(k*cos**2 + gamma_m*sin**2)
C = 4*epsilon_d**2 *gmb0**2/(Gamma_b*k_minus)*k_e_minus*sin**4/(Omega_b**2 + k_minus**2/4)
refl = (1 - 2*(k_e_minus/k_minus) + C)/(1 + C)

f = omega_minus
fig, ax = plt.subplots(1, 2)
ax[0].set_title('reflectivity')
ax[0].plot(f, refl, label='r')
ax[0].plot(f, np.abs(bottom), '--', label='bottom')
ax[0].plot(f, refl - np.abs(bottom), label='diff')
ax[0].legend()

ax[1].set_title('Cooperativity')
ax[1].plot(f, C)

plt.show()

