#!/usr/bin/env python3
# s17_kappa.py — the two-instrument kappa(sigma) extraction
# (.md 2a/2b — the N26 spend).
#
# REGISTERED PREDICTION (2a, derived from the N21/N26 balance BEFORE any
# ride): kappa(sigma) = 1/sigma - B(sigma,sigma)/2 = 1/sigma -
# Gamma(sigma)^2/(2 Gamma(2 sigma));  kappa(2) = 5/12 EXACT (Schochet).
# Predicted: 0.4909875538 (1.25) / 0.4703171258 (1.5) / 0.4443466087 (1.75).
#
# Instruments (both REPRO-OBLIGATED before scoring new data):
#   leg A (mode fit): the banked s16_sigD estimator — q_m = (A m^{s-1} + D)
#     e^{-d m}, log-objective, window [8, 0.4*mmax_good], good-mask
#     q > 1e-12*max. Stage `repro` must reproduce s16_sigD.txt ALL SIX ROWS
#     to printed digits.
#   leg B (track slope): s15_kcurv code objects IMPORTED (local_slopes /
#     curvature_from_track, DTW = 0.02); its stage sig re-run on the same
#     files must reproduce the s15_kcurv_sig*.txt rows (run separately, see
#     the session log).
# kappa extraction (stage `kappa`): at each non-peak snapshot of a snap_*
# npz: D_i from leg A on c_i; ddelta/dt_i from leg B on the full track
# (delta_impl = (K_s nu/Sigma)^{1/s}) interpolated at t_i; kappa_i =
# -ddot/D_i. Consensus = median over snapshots with S/Speak in [0.25, 0.93]
# (both sides; 0.15 rows reported-excluded).
import os, sys
import numpy as np
from scipy.special import gamma as GAMMA, beta as BETA
from scipy.optimize import least_squares

import s15_kcurv as KC   # local_slopes, curvature_from_track, Ksig (DTW=0.02)

DIR = os.path.dirname(os.path.abspath(__file__))

def kappa_pred(s):
    return 1.0/s - BETA(s, s)/2.0

def fit_AD(q, sigma, wlo=8, wfrac=0.4):
    """The banked s16_sigD estimator: log-objective fit of
    (A m^{sigma-1} + D) e^{-d m} on m in [wlo, 0.4*mmax_good]."""
    M = len(q)
    mm = np.arange(1, M+1).astype(float)
    good = q > q.max()*1e-12
    mmax_good = int(np.max(mm[good]))
    mhi = int(wfrac*mmax_good)
    sel = (mm >= wlo) & (mm <= mhi) & good & (q > 0)
    x = mm[sel]; y = np.log(q[sel])
    # init from the laws (A_law, D=0, d from tail slope)
    d0 = -np.polyfit(x, y, 1)[0]
    A0 = np.exp(np.mean(y + d0*x - (sigma-1.0)*np.log(x)))
    def res(p):
        A, D, d = p
        f = A*x**(sigma-1.0) + D
        f = np.where(f > 1e-300, f, 1e-300)
        return np.log(f) - d*x - y
    out = least_squares(res, [A0, 0.0, d0], method="lm", xtol=1e-15,
                        ftol=1e-15, gtol=1e-15, max_nfev=20000)
    A, D, d = out.x
    rms = float(np.sqrt(np.mean(out.fun**2)))
    return dict(A=A, D=D, d=d, rms=rms, n=int(sel.sum()), mhi=mhi,
                mmax_good=mmax_good)

def stage_repro():
    """Leg-A repro obligation: reproduce s16_sigD.txt (all six rows)."""
    rows = [
        (1.10, "sig1p10_nu0p165000_M2048.npz"),
        (1.25, "sig1p25_nu0p136000_M4096.npz"),
        (1.50, "sig1p50_nu0p098200_M8192.npz"),
        (1.75, "sig1p75_nu0p071500_M4096.npz"),
        (1.90, "sig1p90_nu0p059700_M2048.npz"),
        (2.00, "sig2p00_nu0p051600.npz"),
    ]
    print("=== s17 leg-A repro: the banked s16_sigD estimator on the same npz ===")
    print("  target: s16_sigD.txt all rows to printed digits")
    for s, f in rows:
        z = np.load(os.path.join(DIR, f))
        nu = float(z["nu"]); Spk = float(z["peak"])
        q = np.abs(z["c_pk"])
        r = fit_AD(q, s)
        d_impl = (KC.Ksig(s)*nu/Spk)**(1.0/s)
        eta = abs(r["D"])/(r["A"]*(1.0/r["d"])**(s-1.0))
        A_law = 2.0*nu/BETA(s, s)
        print("s=%.2f nu=%.4f Sig=%9.4g: A=%.5g D=%+.4e d=%.6f rms=%.1e | "
              "eta=%.4f %s | d/d_impl-1=%+.2e | A/A_law=%.4f"
              % (s, nu, Spk, r["A"], r["D"], r["d"], r["rms"], eta,
                 "PASS" if eta <= 0.02 else "FAIL", r["d"]/d_impl - 1.0,
                 r["A"]/A_law))

def stage_kappa(files):
    print("=== s17 kappa(sigma): two-instrument extraction () ===")
    print("  prediction (2a): kappa = 1/s - B(s,s)/2 ; kappa(2) = 5/12")
    for f in files:
        z = np.load(os.path.join(DIR, f))
        s = float(z["sigma"]); nu = float(z["nu"]); Spk = float(z["peak"])
        t_pk = float(z["t_peak"])
        tr = z["track"]; t = tr[:, 0]; S = tr[:, 1]
        goodS = S > 0
        delta = (KC.Ksig(s)*nu/S[goodS])**(1.0/s)
        tm, dd = KC.local_slopes(t[goodS], delta)
        kp = kappa_pred(s)
        print("--- %s: sigma=%.2f nu=%.6f Sigma_pk=%.4e t_pk=%.4f  "
              "kappa_pred=%.7f" % (os.path.basename(f), s, nu, Spk, t_pk, kp))
        print("  %-8s %8s %9s %12s %12s %10s %8s" %
              ("snap", "t", "S/Spk", "D", "ddelta/dt", "kappa", "k/pred"))
        cons = []
        for tag, ti, Si, ci in zip(z["snaps_tag"], z["snaps_t"],
                                   z["snaps_S"], z["snaps_c"]):
            q = np.abs(ci)
            r = fit_AD(q, s)
            i0 = int(np.argmin(np.abs(tm - ti)))
            ddi = dd[i0]
            kap = -ddi/r["D"] if r["D"] != 0 else np.nan
            frac = Si/Spk
            inband = 0.25 - 1e-9 <= frac <= 0.93 + 1e-9
            if inband:
                cons.append(kap)
            print("  %-8s %8.4f %9.4f %+12.4e %+12.4e %10.5f %8.4f%s"
                  % (tag, ti, frac, r["D"], ddi, kap, kap/kp,
                     "" if inband else "  [excl]"))
        cons = np.array(cons)
        med = float(np.median(cons))
        up = cons[:len(cons)//2]; dn = cons[len(cons)//2:]
        kr = float(np.median(up)) if len(up) else np.nan
        kf = float(np.median(dn)) if len(dn) else np.nan
        print("  CONSENSUS kappa = %.5f  (pred %.5f, dev %+.2f%%)  "
              "rise-med %.5f fall-med %.5f (|r/f-1| = %.1f%%)  n=%d"
              % (med, kp, 100*(med/kp - 1.0), kr, kf,
                 100*abs(kr/kf - 1.0) if kf else np.nan, len(cons)))

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "repro":
        stage_repro()
    elif st == "kappa":
        stage_kappa(sys.argv[2].split(","))
    else:
        raise SystemExit("unknown stage: %s" % st)
