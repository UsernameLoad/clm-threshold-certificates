#!/usr/bin/env python
# S26 — the e-row program (;.
# Zero new rides: banked snap series only.
#   repro — C-0a: s25_threeterm repro+open reproduced to every printed digit
#           (delegated: this mode just re-runs both and the caller diffs)
#   above —
#           sigma=1.5/1.75; report at 1.25) + the 12-row e-hat(t) columns
#   sub   —.75 closure on the banked 12-snapshot series
import os, sys
import numpy as np
from math import gamma, pi, sin
from s25_threeterm import fit_slot

DIR = os.path.dirname(os.path.abspath(__file__))
ZETA_QUARTER = -0.8132784  # zeta(1/4)

def B(a, b):  # Beta via Gamma, valid for the positive-argument uses below
    return gamma(a)*gamma(b)/gamma(a + b)

def B_s_1ms(sig):  # B(sigma, 1-sigma) = pi/sin(pi sigma) — signed continuation
    return pi/sin(pi*sig)

ABOVE = [  # (file, e_pred/nu from s26_auditchk §2, scored?)
    ("snap_sig1p25_nu0p136000_M4096.npz", 0.2155, False),
    ("snap_sig1p50_nu0p098200_M8192.npz", 0.2919, True),
    ("snap_sig1p75_nu0p071500_M4096.npz", 0.1942, True),
]

def tailpk(c):
    a = np.abs(c)
    return float(a[-8:].max()/a.max())

def fitrow(c, sig):
    q = np.abs(np.asarray(c))
    r = fit_slot(q, sig, -sig)
    return r

def mode_above():
    print("===; zero rides) ===")
    for fn, epred_over_nu, scored in ABOVE:
        z = np.load(os.path.join(DIR, fn))
        sig = float(z["sigma"]); nu = float(z["nu"])
        epred = epred_over_nu*nu
        print("--- %s  sigma=%.2f nu=%.6f  e_pred = %+.5e  [%s]"
              % (fn, sig, nu, epred, "SCORED" if scored else "REPORT-grade (C-3)"))
        cpk = z["c_pk"]; tp = tailpk(cpk)
        r = fitrow(cpk, sig)
        ratio = r["E"]/epred
        print("    PEAK: tail/pk=%.1e  A=%+.6e  D=%+.6e  e-hat=%+.6e  d=%.6f  rms=%.2e" %
              (tp, r["A"], r["D"], r["E"], r["d"], r["rms"]))
        print("    ==> sign(e-hat) = %s (predicted POSITIVE); e-hat/e_pred = %+.4f  (band [1/2, 2])"
              % ("POSITIVE" if r["E"] > 0 else "NEGATIVE", ratio))
        ts = z["snaps_t"]; cs = z["snaps_c"]; tags = z["snaps_tag"]
        print("    C-4 report — along-trajectory e-hat(t):")
        for i in range(len(ts)):
            tpi = tailpk(cs[i])
            ri = fitrow(cs[i], sig)
            flag = "" if tpi <= 1e-6 else "  [DIRTY tail — reported-not-scored]"
            print("      %-7s t=%.4f  tail=%.1e  A=%+.5e  D=%+.5e  e-hat=%+.5e  d=%.6f  rms=%.1e%s"
                  % (tags[i], ts[i], tpi, ri["A"], ri["D"], ri["E"], ri["d"], ri["rms"], flag))

def quad_deriv(ts, ys, t0):
    # least-squares quadratic y(t) = a + b (t-t0) + c (t-t0)^2 -> (dy/dt at t0, d2y/dt2)
    dt = np.asarray(ts) - t0
    Amat = np.vstack([np.ones_like(dt), dt, dt*dt]).T
    coef, *_ = np.linalg.lstsq(Amat, np.asarray(ys), rcond=None)
    return coef[1], 2.0*coef[2]

def mode_sub():
    fn = "snap_sig0p75_nu0p246500_M8192.npz"
    z = np.load(os.path.join(DIR, fn))
    sig = float(z["sigma"]); nu = float(z["nu"]); tpk = float(z["t_peak"])
    Bss = B(sig, sig); B1 = B_s_1ms(sig); zet = ZETA_QUARTER
    den = None
    print("===.75 (banked series; zero rides) ===")
    print("file %s  sigma=%.2f nu=%.6f t_peak=%.6f" % (fn, sig, nu, tpk))
    ts = z["snaps_t"]; cs = z["snaps_c"]; tags = [str(t) for t in z["snaps_tag"]]
    rows = []
    for i in range(len(ts)):
        tpi = tailpk(cs[i])
        ri = fitrow(cs[i], sig)
        rows.append((tags[i], float(ts[i]), tpi, ri))
    cpk = z["c_pk"]; rpk = fitrow(cpk, sig); tppk = tailpk(cpk)
    Alaw = 2.0*nu*sig/(2.0 - sig)
    den = rpk["A"]*B1 - nu   # denominator with the FIT's own A (report also law-A variant)
    den_law = Alaw*B1 - nu
    print("A_law = %.6f ; fit A(peak) = %.6f ; [A B - nu]: fit %.6f / law %.6f"
          % (Alaw, rpk["A"], den, den_law))
    print("D-1: e-hat(peak) = %+.6e  (tail %.1e)  sign %s (registered NEGATIVE)"
          % (rpk["E"], tppk, "NEGATIVE" if rpk["E"] < 0 else "POSITIVE"))
    # --- D-2: FD of the fitted D column at three stencils around t_pk
    tarr = np.array([r[1] for r in rows]); Darr = np.array([r[3]["D"] for r in rows])
    pairs = [(5, 6, "0.93"), (4, 7, "0.80"), (3, 8, "0.60")]
    print("D-2: Ddot at t_pk by FD of the FITTED D column (N27-correct filter):")
    fds = []
    for iu, idn, lab in pairs:
        fd = (Darr[idn] - Darr[iu])/(tarr[idn] - tarr[iu])
        mid = 0.5*(tarr[idn] + tarr[iu])
        fds.append(fd)
        print("    stencil %s pair: Ddot = %+.6f  (midpoint t=%.4f vs t_pk %.4f)" % (lab, fd, mid, tpk))
    kfd = abs(fds[0])
    spread = max(abs(f - fds[0]) for f in fds)
    print("    k_FD = |Ddot(0.93-stencil)| = %.6f   stencil spread = %.6f" % (kfd, spread))
    print("    registered center 0.82*nu^2 = %.6f ; band [0.025, 0.100] ; k_FD/nu^2 = %.4f"
          % (0.82*nu*nu, kfd/nu**2))
    # --- D-3: ddot(delta)|_min from quadratic on fitted delta-hat(t)
    darr = np.array([r[3]["d"] for r in rows])
    for lo, hi, lab in [(3, 9, "6 central"), (2, 10, "8 central")]:
        dd, d2 = quad_deriv(tarr[lo:hi], darr[lo:hi], tpk)
        print("D-3 (%s): ddot(delta)|_min = %+.6e ; k_FD/2 = %.6e ; ratio k_FD/(2 ddot) = %.4f"
              % (lab, d2, kfd/2.0, kfd/(2.0*d2) if d2 != 0 else float("nan")))
    # --- D-4: e_qs vs e-hat at the peak-adjacent snapshots (local quadratic Ddot)
    b_lin, _ = quad_deriv(tarr[3:9], Darr[3:9], tpk)   # smooth Ddot(t) model over 6 central
    # evaluate quadratic derivative at each snapshot time:
    dt6 = tarr[3:9] - tpk
    A6 = np.vstack([np.ones_like(dt6), dt6, dt6*dt6]).T
    coefD, *_ = np.linalg.lstsq(A6, Darr[3:9], rcond=None)
    def Ddot_at(t):
        return coefD[1] + 2.0*coefD[2]*(t - tpk)
    print("D-4: the identity e_qs = [Ddot + D^2/2 - D A zeta(1/4)]/[A B - nu] vs the SAME fit's e-hat:")
    for i in (5, 6):
        tag, ti, tpi, ri = rows[i]
        Dd = Ddot_at(ti)
        eqs = (Dd + 0.5*ri["D"]**2 - ri["D"]*ri["A"]*zet)/(ri["A"]*B1 - nu)
        print("    %-7s t=%.4f  Ddot(quad)=%+.6f  e_qs=%+.6e  e-hat=%+.6e  e_qs/e-hat=%.4f"
              % (tag, ti, Dd, eqs, ri["E"], eqs/ri["E"]))
    # --- D-5: the full table
    print("D-5 report — the 12-row table:")
    for tag, ti, tpi, ri in rows:
        Dd = Ddot_at(ti) if (tarr[3] <= ti <= tarr[8]) else float("nan")
        eqs = (Dd + 0.5*ri["D"]**2 - ri["D"]*ri["A"]*zet)/(ri["A"]*B1 - nu) if Dd == Dd else float("nan")
        print("    %-7s t=%.4f tail=%.1e  A=%+.5e D=%+.6f d=%.6f e-hat=%+.5e rms=%.1e  e_qs=%s"
              % (tag, ti, tpi, ri["A"], ri["D"], ri["d"], ri["E"], ri["rms"],
                 ("%+.5e" % eqs) if eqs == eqs else "  (outside quad window)"))

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "above"
    if m == "above": mode_above()
    elif m == "sub": mode_sub()
    else: print("modes: above | sub")
