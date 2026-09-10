#!/usr/bin/env python3
# s18_kD.py -- the k(sigma) measurement from the D(t) column
# (',
# exact-derivative Ddot leg was refuted-as-designed in 3).
#
# k := |D'(t0)| where t0 is the zero of the banked estimator's D(t) column,
# obtained by a LOCAL polynomial fit through the n snapshots nearest the
# zero.  Finite-Delta_t differencing is the CORRECT filter here: it removes
# the fast residual-relaxation mode (rate nu m^sigma) that contaminates the
# instantaneous derivative (3, the exact-derivative trap).
#
# Independence: k comes from the D-column (a mode-fit observable); Y(sigma)
# comes from the s15_kcurv curvature of delta_impl(t) (a track observable).
# So Y = kappa*k/nu^2 is a TEST, not a definition.
#
# REGISTERED (3a, before any new fit): A'1 value (+-2% of Y/kappa),
# A'2 stability (<=1.5% across degree 1/2/3 x n = 4/6/8), A'3 the D-zero
# locus (|t0 - t_pk| <= 0.05 and t0 > t_pk at every sigma).
import os, sys
import numpy as np
from scipy.special import beta as BETA

import s17_kappa as S17     # fit_AD -- the banked estimator, unchanged

DIR = os.path.dirname(os.path.abspath(__file__))
YREC = {1.25: 1.6339, 1.50: 2.3334, 1.75: 3.1081, 2.00: 3.9654}


def column(z, s):
    """The banked estimator's (t, D) column at every snapshot."""
    t, D = [], []
    for ti, ci in zip(z["snaps_t"], z["snaps_c"]):
        r = S17.fit_AD(np.abs(ci), s)
        t.append(float(ti)); D.append(r["D"])
    return np.array(t), np.array(D)


def kslope(t, D, deg, n):
    """Local polynomial fit through the n points nearest the D-zero;
    return (t0, |D'(t0)|)."""
    i = int(np.argmin(np.abs(D)))
    # take n points centred on the sign change
    j = int(np.searchsorted(-D, 0.0))          # D is decreasing
    lo = max(0, j - n // 2); hi = min(len(t), lo + n)
    lo = max(0, hi - n)
    tt, dd = t[lo:hi], D[lo:hi]
    c = np.polyfit(tt - tt.mean(), dd, deg)
    p = np.poly1d(c)
    rts = [r.real for r in np.roots(c) if abs(r.imag) < 1e-9
           and tt.min() - tt.mean() - 0.3 <= r.real <= tt.max() - tt.mean() + 0.3]
    if not rts:
        return np.nan, np.nan
    r0 = min(rts, key=abs)
    return r0 + tt.mean(), abs(np.polyder(p)(r0))


def main(files):
    print("=== s18 k(sigma) from the D(t) column (') ===")
    print("  k := |D'(t0)| at the D-zero; K_pred = Y(rec)/kappa(sigma)")
    print("  Y(rec) (s15_kcurv_sig*.txt deepest rows): %s" % YREC)
    for f in files:
        z = np.load(os.path.join(DIR, f))
        s = float(z["sigma"]); nu = float(z["nu"])
        t_pk = float(z["t_peak"])
        kp = S17.kappa_pred(s)
        Kpred = YREC[round(s, 2)] / kp
        t, D = column(z, s)
        print("--- %s: sigma=%.2f nu=%.6f t_pk=%.4f kappa=%.7f K_pred=%.5f"
              % (os.path.basename(f), s, nu, t_pk, kp, Kpred))
        grid = []
        for deg in (1, 2, 3):
            for n in (4, 6, 8):
                if n < deg + 1:
                    continue
                t0, k = kslope(t, D, deg, n)
                if not np.isfinite(k):
                    continue
                grid.append((deg, n, t0, k / nu ** 2))
                print("   deg=%d n=%d: t0=%.4f (t0-t_pk=%+.4f)  k/nu2=%.4f  "
                      "dev=%+.2f%%" % (deg, n, t0, t0 - t_pk, k / nu ** 2,
                                       100 * (k / nu ** 2 / Kpred - 1.0)))
        vals = np.array([g[3] for g in grid])
        ref = [g for g in grid if g[0] == 2 and g[1] == 6]
        kdef = ref[0][3] if ref else float(np.median(vals))
        t0def = ref[0][2] if ref else np.nan
        spread = 100 * (vals.max() - vals.min()) / np.median(vals)
        print("   >> DEFAULT (deg2,n6): k/nu2 = %.4f  (K_pred %.5f, dev %+.2f%%)"
              "  | grid spread %.2f%% [A'2 bar 1.5%%]  | t0-t_pk = %+.4f "
              "[A'3 bar 0.05, sign +]" % (kdef, Kpred,
                                          100 * (kdef / Kpred - 1.0), spread,
                                          t0def - t_pk))
        print("   >> closure Y_implied = kappa*k/nu2 = %.5f vs Y(rec) = %.4f "
              "(dev %+.2f%%)" % (kp * kdef, YREC[round(s, 2)],
                                 100 * (kp * kdef / YREC[round(s, 2)] - 1.0)))


if __name__ == "__main__":
    main(sys.argv[1].split(","))
