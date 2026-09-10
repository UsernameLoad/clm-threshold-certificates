#!/usr/bin/env python3
# s18_kdot.py -- the EXACT-DERIVATIVE (kappa, k) instrument
# (.md 2c -- the k(sigma) round).
#
# Idea: the hierarchy is closed and triangular, so at
# any banked snapshot the time derivative is EXACT:
#     cdot = NL(c) - nu m^sigma c          (hier_if.NL imported unchanged)
# The fitted triple (A, D, delta) is a smooth functional of the mode vector,
# so (Adot, Ddot, deltadot) follow by central-differencing the SAME banked
# estimator along the exact flow direction, on a FIXED window index set:
#     X[q +- h*qdot],   qdot_m = Re(conj(c_m) cdot_m)/|c_m|.
# No time integration, no snapshot spacing, no track interpolation -- the
# S17 leg-B (crude 2-point track slope) is removed entirely.
#
# REGISTERED CLAUSES (2c, before any fit): A-VAL (repro), A1 (the
# rise/fall asymmetry discriminator), A2 (consensus kappa within 2%),
# A3 (k/nu^2 within 3% of K = Y/kappa), A4 (|Adot|/(nu A) <= 0.05),
# A5 (P, report-grade), plus the h-independence guard (h and h/4).
#
# Stages:  repro  -- A-VAL: the six s16_sigD rows + the s17 D-columns
#          kdot:<files>  -- the measurement
import os, sys
import numpy as np
from scipy.special import gamma as GAMMA, beta as BETA, zeta as ZETA
from scipy.optimize import least_squares

import hier_if as HI          # NL (the ride's own nonlinearity)
import s15_kcurv as KC        # Ksig
import s17_kappa as S17       # fit_AD (the banked estimator), kappa_pred

DIR = os.path.dirname(os.path.abspath(__file__))


def window_of(q, wlo=8, wfrac=0.4):
    """The banked s16_sigD window rule, extracted so it can be FROZEN
    across the +-h perturbations (a directional derivative must not move
    its own window)."""
    M = len(q)
    mm = np.arange(1, M + 1).astype(float)
    good = q > q.max() * 1e-12
    mmax_good = int(np.max(mm[good]))
    mhi = int(wfrac * mmax_good)
    sel = (mm >= wlo) & (mm <= mhi) & good & (q > 0)
    return sel, mm, mmax_good, mhi


def fit_AD_sel(q, sigma, sel, mm, p0=None):
    """s17_kappa.fit_AD with an EXTERNALLY SUPPLIED index set (identical
    objective, identical initialisation rule, identical solver settings)."""
    x = mm[sel]
    y = np.log(q[sel])
    if p0 is None:
        d0 = -np.polyfit(x, y, 1)[0]
        A0 = np.exp(np.mean(y + d0 * x - (sigma - 1.0) * np.log(x)))
        p0 = [A0, 0.0, d0]

    def res(p):
        A, D, d = p
        f = A * x ** (sigma - 1.0) + D
        f = np.where(f > 1e-300, f, 1e-300)
        return np.log(f) - d * x - y

    out = least_squares(res, p0, method="lm", xtol=1e-15, ftol=1e-15,
                        gtol=1e-15, max_nfev=20000)
    A, D, d = out.x
    rms = float(np.sqrt(np.mean(out.fun ** 2)))
    return dict(A=A, D=D, d=d, rms=rms, n=int(sel.sum()))


def cdot_exact(c, nu, sigma):
    """The EXACT right-hand side of the banked hierarchy at this vector."""
    M = len(c)
    m = np.arange(1, M + 1).astype(float)
    msig = m ** float(sigma) if sigma != 2 else (np.arange(1, M + 1) ** 2).astype(float)
    return HI.NL(c) - nu * msig * c


def qdot_exact(c, nu, sigma):
    cd = cdot_exact(c, nu, sigma)
    q = np.abs(c)
    qd = np.zeros_like(q)
    nz = q > 0
    qd[nz] = (c[nz].real * cd[nz].real + c[nz].imag * cd[nz].imag) / q[nz]
    return q, qd


def derivs(c, nu, sigma, h):
    """Exact-flow central difference of the banked fit on a FROZEN window."""
    q, qd = qdot_exact(c, nu, sigma)
    sel, mm, mmax_good, mhi = window_of(q)
    base = fit_AD_sel(q, sigma, sel, mm)
    p0 = [base["A"], base["D"], base["d"]]
    qp = q + h * qd
    qm = q - h * qd
    if np.any(qp[sel] <= 0) or np.any(qm[sel] <= 0):
        return base, None, sel, mm, mmax_good, mhi
    fp = fit_AD_sel(qp, sigma, sel, mm, p0=p0)
    fm = fit_AD_sel(qm, sigma, sel, mm, p0=p0)
    d = dict(Adot=(fp["A"] - fm["A"]) / (2 * h),
             Ddot=(fp["D"] - fm["D"]) / (2 * h),
             ddot=(fp["d"] - fm["d"]) / (2 * h))
    return base, d, sel, mm, mmax_good, mhi


def outer_P(q, sigma, A, D, delta, kmax):
    """A5: the outer number P = sum_k [ q_k e^{delta k} - A k^{s-1} - D ]
    over the clean window (report-grade; no regularisation applied)."""
    k = np.arange(1, kmax + 1).astype(float)
    F = q[:kmax] * np.exp(delta * k)
    return float(np.sum(F - A * k ** (sigma - 1.0) - D))


# ---------------------------------------------------------------- stages
def stage_repro():
    """A-VAL: (i) the six s16_sigD rows through the frozen-window path,
    checked against s17_kappa.fit_AD (the banked estimator) itself."""
    rows = [(1.10, "sig1p10_nu0p165000_M2048.npz"),
            (1.25, "sig1p25_nu0p136000_M4096.npz"),
            (1.50, "sig1p50_nu0p098200_M8192.npz"),
            (1.75, "sig1p75_nu0p071500_M4096.npz"),
            (1.90, "sig1p90_nu0p059700_M2048.npz"),
            (2.00, "sig2p00_nu0p051600.npz")]
    print("=== s18 A-VAL (i): frozen-window path vs the banked s16_sigD estimator ===")
    worst = 0.0
    for s, f in rows:
        z = np.load(os.path.join(DIR, f))
        nu = float(z["nu"]); Spk = float(z["peak"])
        q = np.abs(z["c_pk"])
        ref = S17.fit_AD(q, s)
        sel, mm, mmax_good, mhi = window_of(q)
        new = fit_AD_sel(q, s, sel, mm)
        d_impl = (KC.Ksig(s) * nu / Spk) ** (1.0 / s)
        eta = abs(new["D"]) / (new["A"] * (1.0 / new["d"]) ** (s - 1.0))
        A_law = 2.0 * nu / BETA(s, s)
        rel = max(abs(new["A"] / ref["A"] - 1), abs(new["d"] / ref["d"] - 1),
                  abs(new["D"] - ref["D"]) / max(abs(ref["D"]), 1e-30))
        worst = max(worst, rel)
        print("s=%.2f nu=%.4f Sig=%9.4g: A=%.5g D=%+.4e d=%.6f rms=%.1e | "
              "eta=%.4f | d/d_impl-1=%+.2e | A/A_law=%.4f | vs-banked rel=%.1e"
              % (s, nu, Spk, new["A"], new["D"], new["d"], new["rms"], eta,
                 new["d"] / d_impl - 1.0, new["A"] / A_law, rel))
    print("  A-VAL(i) worst relative deviation from the banked estimator: %.2e" % worst)
    print("  VERDICT: %s" % ("PASS (identical to the banked path)" if worst < 1e-10
                             else "FAIL -- STOP AND DIAGNOSE"))


def stage_kdot(files, hs=(1e-3, 2.5e-4)):
    Yrec = {1.25: 1.6339, 1.50: 2.3334, 1.75: 3.1081, 2.00: 3.9654}
    print("=== s18 exact-derivative (kappa, k) instrument () ===")
    print("  kappa_pred = 1/s - B(s,s)/2 ; K_pred = Y(rec)/kappa_pred")
    print("  Y(rec) from s15_kcurv_sig*.txt deepest rows: %s" % Yrec)
    for f in files:
        z = np.load(os.path.join(DIR, f))
        s = float(z["sigma"]); nu = float(z["nu"])
        Spk = float(z["peak"]); t_pk = float(z["t_peak"])
        kp = S17.kappa_pred(s)
        Kpred = Yrec[round(s, 2)] / kp
        print("--- %s: sigma=%.2f nu=%.6f Sigma_pk=%.4e t_pk=%.4f "
              "kappa_pred=%.7f K_pred=%.5f"
              % (os.path.basename(f), s, nu, Spk, t_pk, kp, Kpred))
        print("  %-8s %8s %9s %12s %12s %10s %9s %10s" %
              ("snap", "t", "S/Spk", "D", "ddelta/dt", "kappa", "k/pred",
               "|Ddot|/nu2"))
        cons = []
        h = hs[0]
        for tag, ti, Si, ci in zip(z["snaps_tag"], z["snaps_t"],
                                   z["snaps_S"], z["snaps_c"]):
            base, d, sel, mm, mg, mhi = derivs(ci, nu, s, h)
            if d is None:
                print("  %-8s  [h too large: q+hqdot non-positive in window]" % tag)
                continue
            kap = -d["ddot"] / base["D"] if base["D"] != 0 else np.nan
            frac = Si / Spk
            inband = 0.25 - 1e-9 <= frac <= 0.93 + 1e-9
            if inband:
                cons.append(kap)
            print("  %-8s %8.4f %9.4f %+12.4e %+12.4e %10.5f %9.4f %10.4f%s"
                  % (tag, ti, frac, base["D"], d["ddot"], kap, kap / kp,
                     abs(d["Ddot"]) / nu ** 2, "" if inband else "  [excl]"))
        cons = np.array(cons)
        med = float(np.median(cons))
        up = cons[:len(cons) // 2]; dn = cons[len(cons) // 2:]
        kr = float(np.median(up)); kf = float(np.median(dn))
        print("  CONSENSUS kappa = %.5f  (pred %.5f, dev %+.2f%%)  "
              "rise-med %.5f fall-med %.5f (|r/f-1| = %.1f%%)  n=%d"
              % (med, kp, 100 * (med / kp - 1.0), kr, kf,
                 100 * abs(kr / kf - 1.0), len(cons)))
        # --- the peak vector: A3 / A4 / A5, at two step sizes
        cpk = z["c_pk"]
        for h in hs:
            base, d, sel, mm, mg, mhi = derivs(cpk, nu, s, h)
            if d is None:
                print("  PEAK h=%.1e : [h too large]" % h)
                continue
            k_meas = abs(d["Ddot"]) / nu ** 2
            ddd = kp * abs(d["Ddot"])          # delta-ddot = kappa*k (2b)
            P = outer_P(np.abs(cpk), s, base["A"], base["D"], base["d"],
                        min(mg, 4000))
            print("  PEAK h=%.1e: A=%.6g D=%+.3e d=%.6f | Ddot=%+.5e "
                  "k/nu2=%.4f (pred %.4f, dev %+.2f%%) | Adot/(nu A)=%+.4f | "
                  "ddelta2=%.5e vs Y nu2=%.5e (dev %+.2f%%) | P=%+.5e "
                  "(-A zeta(1-s)=%+.5e)"
                  % (h, base["A"], base["D"], base["d"], d["Ddot"],
                     k_meas, Kpred, 100 * (k_meas / Kpred - 1.0),
                     d["Adot"] / (nu * base["A"]), ddd,
                     Yrec[round(s, 2)] * nu ** 2,
                     100 * (ddd / (Yrec[round(s, 2)] * nu ** 2) - 1.0),
                     P, -base["A"] * ZETA(1.0 - s)))


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "repro":
        stage_repro()
    elif st.startswith("kdot"):
        stage_kdot(sys.argv[2].split(","))
    else:
        raise SystemExit("unknown stage: %s" % st)
