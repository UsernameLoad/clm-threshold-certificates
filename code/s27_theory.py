#!/usr/bin/env python
# S27 — the m0-row residual theory pass (; s27_reg.txt).
# NUMERICS-ANCHORED derivation: every analytic label is verified against the
# exact convolution sums before it is used (the N19/N21 lesson: no derived law
# without a numeric anchor). Modes:
#   verify — exact-sum verification of the convolution expansions at the six
#            banked sigma (the m0-row correction terms with verified coeffs)
#   traj   — the along-trajectory corrected-identity test: e_true(t) from the
#            corrected row vs the fitted e-hat(t), per banked column (F2)
#   proj   — fit-projection arithmetic: what unit-coefficient unmodeled content
#            at m^{sigma-2}, m^{-1}, m^{1-sigma} shifts (A-hat, D-hat, e-hat) by,
#            per sigma/window; plus the required-coefficient lines (labeled)
import os, sys
import numpy as np
from math import gamma, pi, sin
from scipy.special import zeta as _zeta
from s25_threeterm import fit_slot
import s17_kappa as sk

sys.stdout.reconfigure(encoding="utf-8")
DIR = os.path.dirname(os.path.abspath(__file__))

def zeta(s):
    # scipy zeta handles s>1; for s<1 use the functional equation via mpmath-free
    # route: scipy.special.zeta supports s<1? (scipy >= 1.9 supports complex/neg
    # via zetac? safest: use the reflection formula with gamma)
    if s > 1:
        return float(_zeta(s))
    if s == 0.5:
        return -1.4603545088095868
    # reflection: zeta(s) = 2^s pi^{s-1} sin(pi s/2) Gamma(1-s) zeta(1-s)
    return float(2**s * pi**(s-1) * sin(pi*s/2) * gamma(1-s) * _zeta(1-s))

def B(a, b): return gamma(a)*gamma(b)/gamma(a+b)
def B1(s):   return pi/sin(pi*s)

SIGMAS = [0.75, 1.25, 1.375, 1.5, 1.75, 1.9]

def mode_verify():
    print("===")
    print("S_a(m) = sum_{j=1}^{m-1} j^a  ==  zeta(-a) + m^{a+1}/(a+1) - m^a/2 + a m^{a-1}/12 + ...")
    print("C(m)   = sum j^{s-1}(m-j)^{-s} == zeta(s) m^{s-1} + B(s,1-s) + zeta(1-s) m^{-s} + [next]")
    print("G(m)   = sum j^{s-1}(m-j)^{s-1} == B(s,s) m^{2s-1} + 2 zeta(1-s) m^{s-1} + [next]")
    ms = np.array([256, 512, 1024, 2048, 4096])
    for s in SIGMAS:
        zs, z1s = zeta(s), zeta(1.0 - s)
        b1, bss = B1(s), B(s, s)
        print("--- sigma = %.3f : zeta(s)=%+.6f zeta(1-s)=%+.6f B(s,1-s)=%+.6f B(s,s)=%+.6f" % (s, zs, z1s, b1, bss))
        rowsC, rowsG = [], []
        for m in ms:
            j = np.arange(1, m)
            C = float(np.sum(j**(s-1.0)*(m-j)**(-s)))
            G = float(np.sum(j**(s-1.0)*(m-j)**(s-1.0)))
            remC = C - (zs*m**(s-1.0) + b1 + z1s*m**(-s))
            remG = G - (bss*m**(2*s-1.0) + 2.0*z1s*m**(s-1.0))
            rowsC.append((m, C, remC)); rowsG.append((m, G, remG))
        # the residual's local power (log-log slope between successive m)
        pc = [np.log(abs(rowsC[i+1][2]/rowsC[i][2]))/np.log(ms[i+1]/ms[i]) for i in range(len(ms)-1) if rowsC[i][2] != 0]
        pg = [np.log(abs(rowsG[i+1][2]/rowsG[i][2]))/np.log(ms[i+1]/ms[i]) for i in range(len(ms)-1) if rowsG[i][2] != 0]
        print("    C(m): residual after 3 terms at m=4096: %+.3e ; local power = %s  [candidate s-2 = %.3f]"
              % (rowsC[-1][2], ", ".join("%.3f" % p for p in pc), s-2.0))
        print("    G(m): residual after 2 terms at m=4096: %+.3e ; local power = %s  [candidate s-2 = %.3f]"
              % (rowsG[-1][2], ", ".join("%.3f" % p for p in pg), s-2.0))
        # implied next-coefficient (residual / m^{s-2}) — printed for the record
        cC = rowsC[-1][2]/ms[-1]**(s-2.0); cG = rowsG[-1][2]/ms[-1]**(s-2.0)
        print("    implied next coeffs: C: %+.5f * m^{s-2} ; G: %+.5f * m^{s-2}" % (cC, cG))
        # Euler-Maclaurin S_a checks (constants zeta(1-s), zeta(s)) at m=4096
        m = 4096; j = np.arange(1, m)
        Sa = float(np.sum(j**(s-1.0))); Sb = float(np.sum(j**(-s)))
        remA = Sa - (z1s + m**s/s - m**(s-1.0)/2 + (s-1.0)*m**(s-2.0)/12)
        remB = Sb - (zs + m**(1.0-s)/(1.0-s) - m**(-s)/2) if abs(1.0-s) > 1e-12 else float("nan")
        print("    S_{s-1} EM-residual at 4096: %+.2e ; S_{-s} EM-residual: %+.2e" % (remA, remB))

# ---------------------------------------------------------------- traj
COLS = [
    ("snap_sig0p75_nu0p246500_M8192.npz", 0.75),
    ("snap_sig1p25_nu0p136000_M4096.npz", 1.25),
    ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375),
    ("snap_sig1p50_nu0p098200_M8192.npz", 1.50),
    ("snap_sig1p75_nu0p071500_M4096.npz", 1.75),
]

def tailpk(c):
    a = np.abs(c); return float(a[-8:].max()/a.max())

def col_fits(z, sig):
    ts = z["snaps_t"]; cs = z["snaps_c"]
    rows = []
    for i in range(len(ts)):
        q = np.abs(cs[i])
        mmax_good = int(np.max(np.arange(1, len(q) + 1)[q > q.max()*1e-12]))
        if int(0.4*mmax_good) < 12:
            continue
        r = fit_slot(q, sig, -sig)
        rows.append((float(ts[i]), tailpk(cs[i]), r))
    return rows

def quad_model(tarr, yarr, t0, w):
    ipk = int(np.argmin(np.abs(tarr - t0)))
    lo, hi = max(0, ipk - w), min(len(tarr), ipk + w)
    dt = tarr[lo:hi] - t0
    Amat = np.vstack([np.ones_like(dt), dt, dt*dt]).T
    coef, *_ = np.linalg.lstsq(Amat, yarr[lo:hi], rcond=None)
    return coef  # y(t) = c0 + c1 (t-t0) + c2 (t-t0)^2

def mode_traj():
    print("===): the along-trajectory corrected identity vs e-hat(t) ===")
    print("corrected row:  Ddot = e [A B1 - nu] + D e zeta(s) + A D zeta(1-s) - D^2/2")
    print("==> e_true(t) = [Ddot + D^2/2 - A D zeta(1-s) - D e-hat zeta(s)] / [A B1 - nu]")
    print("(Ddot(t) from the SAME column's quadratic D(t) model; all inputs banked; no tuning)")
    for fn, sig in COLS:
        z = np.load(os.path.join(DIR, fn))
        nu = float(z["nu"]); tpk = float(z["t_peak"])
        zs, z1s, b1 = zeta(sig), zeta(1.0 - sig), B1(sig)
        rows = col_fits(z, sig)
        tarr = np.array([r[0] for r in rows])
        Darr = np.array([r[2]["D"] for r in rows])
        w = 6 if len(rows) > 15 else 3
        cD = quad_model(tarr, Darr, tpk, w)
        print("--- %s sigma=%.3f nu=%.4f  Ddot_pk(quad)=%+.6f  D-window +-%d rows" % (fn, sig, nu, cD[1], w))
        pr = []
        for (ti, tpi, r) in rows:
            Dd = cD[1] + 2.0*cD[2]*(ti - tpk)
            Av, Dv, Ev = r["A"], r["D"], r["E"]
            den = Av*b1 - nu
            e_unc = (Dd + 0.5*Dv*Dv - Av*Dv*z1s)/den
            e_cor = (Dd + 0.5*Dv*Dv - Av*Dv*z1s - Dv*Ev*zs)/den
            pr.append((ti, Dv, Ev, e_unc, e_cor))
        # pointwise table (compact): t, D, e-hat, e_unc/e-hat, e_cor/e-hat
        for ti, Dv, Ev, eu, ec in pr:
            print("    t=%.4f D=%+.5f  e-hat=%+.6e  e_unc/e-hat=%+.4f  e_cor/e-hat=%+.4f"
                  % (ti, Dv, Ev, eu/Ev, ec/Ev))
        # slopes: fit e vs D for measured and both predictions over the same rows
        Dv = np.array([p[1] for p in pr])
        for lab, ys in [("MEASURED e-hat", np.array([p[2] for p in pr])),
                        ("e_unc (no Dezeta)", np.array([p[3] for p in pr])),
                        ("e_cor (with Dezeta)", np.array([p[4] for p in pr]))]:
            Am = np.vstack([np.ones_like(Dv), Dv]).T
            c, *_ = np.linalg.lstsq(Am, ys, rcond=None)
            print("    slope d(%s)/dD = %+.5f   (e0 = %+.6e)" % (lab, c[1], c[0]))

# ---------------------------------------------------------------- proj
def mode_proj():
    print("===) ===")
    print("linearized LSQ: dp = (J^T J)^{-1} J^T (m^p / F) for unit coefficient of m^p added to F;")
    print("columns: dA, dD, de, ddelta per candidate power p in {sigma-2, -1, 1-sigma};")
    print("'required coeff' = measured peak deviation (e-hat - e_pred) / de-projection  [REPORT, labeled]")
    PEAKS = [
        ("snap_sig1p25_nu0p136000_M4096.npz", 1.25, +2.931074e-02),
        ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375, +3.187992e-02),
        ("snap_sig1p50_nu0p098200_M8192.npz", 1.50, +2.866862e-02),
        ("snap_sig1p75_nu0p071500_M4096.npz", 1.75, +1.388673e-02),
        ("sig1p90_nu0p060500_M2048.npz", 1.90, +4.902931e-03),
    ]
    for fn, sig, epred in PEAKS:
        z = np.load(os.path.join(DIR, fn))
        nu = float(z["nu"])
        q = np.abs(z["c_pk"])
        r = fit_slot(q, sig, -sig)
        M = len(q); mm = np.arange(1, M + 1).astype(float)
        good = q > q.max()*1e-12
        mmax_good = int(np.max(mm[good])); mhi = int(0.4*mmax_good)
        selv = (mm >= 8) & (mm <= mhi) & good & (q > 0)
        x = mm[selv]
        Av, Dv, Ev, dv = r["A"], r["D"], r["E"], r["d"]
        F = Av*x**(sig-1.0) + Dv + Ev*x**(-sig)
        # Jacobian of log-model wrt (A, D, E, delta)
        J = np.vstack([x**(sig-1.0)/F, 1.0/F, x**(-sig)/F, -x]).T
        JtJ = J.T @ J
        out = {}
        for p in (sig - 2.0, -1.0, 1.0 - sig):
            g = (x**p)/F     # d(log q) from unit m^p content
            dp = np.linalg.solve(JtJ, J.T @ g)
            out[p] = dp
        dev = r["E"] - epred
        print("--- %s sigma=%.3f  window [8, %d]  e-hat=%+.5e e_pred=%+.5e dev=%+.5e" % (fn, sig, mhi, r["E"], epred, dev))
        for p, dp in out.items():
            req = dev/dp[2] if dp[2] != 0 else float("nan")
            print("    p=%+.3f : dA=%+.4e dD=%+.4e de=%+.4e dd=%+.2e   required coeff X(p) = %+.5e  (X: F += X m^p)"
                  % (p, dp[0], dp[1], dp[2], dp[3], req))

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if m == "verify": mode_verify()
    elif m == "traj": mode_traj()
    elif m == "proj": mode_proj()
    else: print("modes: verify | traj | proj")
