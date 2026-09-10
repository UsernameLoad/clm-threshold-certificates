# sigma=1 exact excursion laws from Sakajo's closed form q_m=(t/2)^(m-1)e^(-m nu t)
# Sigma(t) = sum q_m = e^{-nu t}/(1-r), r=(t/2)e^{-nu t}; nu* = 1/(2e)
import numpy as np
from scipy.optimize import brentq, minimize_scalar
NUS = 1.0/(2.0*np.e)
t0 = 2.0*np.e

def Sig(t, nu):
    r = (t/2.0)*np.exp(-nu*t)
    return np.exp(-nu*t)/(1.0-r) if r < 1 else np.inf

print("nu* exact = 1/(2e) =", NUS, "  t0 = 2e =", t0)
print("\n# decay side: Sigma_peak scaling (predict gamma=1: Sigma_peak*eps -> const = nu* e^{-1}? check)")
print(f"{'eps':>10} {'Sig_pk':>14} {'Sig_pk*eps':>12} {'t_pk':>10} {'(t0-t_pk)/eps':>14} {'d_min/eps':>10}")
for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
    nu = NUS + eps
    m = minimize_scalar(lambda t: -Sig(t, nu), bounds=(1.0, 3.0/nu), method='bounded',
                        options={'xatol':1e-13})
    tpk, spk = m.x, -m.fun
    rmax = 1.0/(2.0*nu*np.e)
    dmin = -np.log(rmax)
    print(f"{eps:>10.1e} {spk:>14.6e} {spk*eps:>12.6f} {tpk:>10.6f} {(t0-tpk)/eps:>14.4f} {dmin/eps:>10.4f}")
print("predicted coefficients: (t0-t_pk)/eps -> 2/nu*^2 =", 2.0/NUS**2, "; d_min/eps -> 1/nu* =", 1.0/NUS)

print("\n# blowup side: T*(nu) = first root of r(t)=1 (predict t0-T* ~ sqrt(2/nu*^3) sqrt(eps'))")
for epsp in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    nu = NUS - epsp
    f = lambda t: (t/2.0)*np.exp(-nu*t) - 1.0
    Ts = brentq(f, 1.0, 1.0/nu, xtol=1e-14)
    print(f"eps'={epsp:>8.1e}  T*={Ts:>12.8f}  (t0-T*)/sqrt(eps') = {(t0-Ts)/np.sqrt(epsp):>10.5f}")
print("predicted: sqrt(2/nu*^3) =", np.sqrt(2.0/NUS**3))
