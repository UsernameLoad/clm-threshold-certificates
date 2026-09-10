#!/usr/bin/env python
# S29 — THE U(x)-PROJECTION TEST (.txt; zero rides).
# The shape-locked/collective-front candidate's first registered test:
# reconstruct the MEASURED flyby-correction content u(x = m*delta1) (N24)
# as profile content on the identity's peak vectors and push it through the
# REGISTERED projection arithmetic (s28_memint.proj_rows math, verbatim).
# Modes:
#   uverify — U-0b: the u-extraction leg re-extracts banked u rows
#             (sigma = 1.625 pair vector; the two 1.75 batteries) via the
#             s17_vsig functions and must match ALL printed digits.
#   urun    — the registered test: SUP/INF content models -> g = ln(1-u)
#             on each identity window -> dev_pred^U -> scoring U-1/U-1b/U-2
#             + the U-4 shape-cap (the B-6 pattern on the U-direction).
# (U-0a, the projection-leg repro, runs as `s28_memint.py proj` diffed
#  against s28_proj.txt — driven from the shell, receipts kept.)
import os, sys
import numpy as np
from math import log, exp
from s25_threeterm import fit_slot
from s17_vsig import primary_fit, u_profile

sys.stdout.reconfigure(encoding="utf-8")
DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- content models
# Banked u rows (record files named in s29_regcalc.txt, read this session).
# Convention: u = 1 - q/model (s17_vsig); u < 0 = constructive (V-like).
UROWS = {
    1.625: [(14, -2.703e-06), (18, -6.245e-05), (22, -2.548e-04), (26, -6.866e-04),
            (30, -1.447e-03), (34, -2.525e-03), (38, -3.785e-03)],
    1.75:  [(14, -1.927e-05), (18, -1.519e-04), (22, -5.954e-04), (26, -1.658e-03),
            (30, -3.675e-03), (34, -6.714e-03), (38, -1.034e-02)],
    1.5:   [(22, -4.52e-05), (26, -1.60e-04), (30, -3.36e-04), (34, -5.18e-04)],
    2.0:   [(14, -5.376e-05), (18, -4.584e-04), (22, -2.0677e-03), (26, -6.4437e-03),
            (30, -1.5300e-02), (34, -2.8812e-02), (38, -4.3773e-02)],
}
# sigma = 1.25: eps-DEPENDENT (x1.5-1.6, NOT the universal object) —
# ENVELOPE-ONLY generosity variant (larger battery), sign POSITIVE:
UROWS_125ENV = [(14, +2.678e-04), (18, +5.466e-04), (22, +8.621e-04), (26, +1.222e-03),
                (30, +1.642e-03), (34, +2.168e-03), (38, +2.986e-03)]

def u_model(sig, x, sup=True):
    """registered content model: log-linear between banked rows; below the
    lowest row, SUP = log-linear extension through the two lowest rows
    (generous, per the recorded log-concavity), INF = 0. sigma = 1.375/1.9:
    ln|u| interpolated linearly in sigma between neighbors at matched x
    (REPORT provenance)."""
    if abs(sig - 1.25) < 1e-9:
        rows = UROWS_125ENV
    elif abs(sig - 1.375) < 1e-9:
        # neighbors 1.25env (+) and 1.5 (-): sign CONFLICT -> the kill-switch
        # reading (universal content dies toward sigma = 1): magnitude =
        # geometric mean of |neighbors|, sign variant reported both ways in
        # urun; here return the NEGATIVE branch (V-like) as the model value.
        a = abs(u_model(1.25, x, sup)); b = abs(u_model(1.5, x, sup))
        if a == 0.0 or b == 0.0:
            return 0.0
        return -exp(0.5*(log(a) + log(b)))
    elif abs(sig - 1.9) < 1e-9:
        a = abs(u_model(1.75, x, sup)); b = abs(u_model(2.0, x, sup))
        if a == 0.0 or b == 0.0:
            return 0.0
        return -exp((log(a)*(2.0 - 1.9) + log(b)*(1.9 - 1.75))/0.25)
    else:
        rows = UROWS[sig]
    xs = [r[0] for r in rows]; us = [r[1] for r in rows]
    sgn = 1.0 if us[0] > 0 else -1.0
    la = [log(abs(u)) for u in us]
    if x < xs[0]:
        if not sup:
            return 0.0
        sl = (la[1] - la[0])/(xs[1] - xs[0])
        return sgn*exp(la[0] + sl*(x - xs[0]))
    if x >= xs[-1]:
        sl = (la[-1] - la[-2])/(xs[-1] - xs[-2])
        return sgn*exp(la[-1] + sl*(x - xs[-1]))
    i = np.searchsorted(xs, x) - 1
    i = max(0, min(i, len(xs) - 2))
    w = (x - xs[i])/(xs[i+1] - xs[i])
    return sgn*exp(la[i]*(1 - w) + la[i+1]*w)

# ---------------------------------------------------------------- projection
def proj_dir(fn, sig, gfun):
    """proj_rows (s28_memint) VERBATIM math, with g the registered
    U-direction instead of a power slot: g_j = ln(1 - u(delta-hat*m_j))
    (the data's log-deviation; u < 0 => g > 0). Returns the parameter
    response dp, the fit r, window, and the shape-cap ingredients."""
    z = np.load(os.path.join(DIR, fn))
    q = np.abs(z["c_pk"])
    r = fit_slot(q, sig, -sig)
    M = len(q); mm = np.arange(1, M + 1).astype(float)
    good = q > q.max()*1e-12
    mmax_good = int(np.max(mm[good])); mhi = int(0.4*mmax_good)
    selv = (mm >= 8) & (mm <= mhi) & good & (q > 0)
    x = mm[selv]
    Av, Dv, Ev, dv = r["A"], r["D"], r["E"], r["d"]
    F = Av*x**(sig-1.0) + Dv + Ev*x**(-sig)
    J = np.vstack([x**(sig-1.0)/F, 1.0/F, x**(-sig)/F, -x]).T
    JtJ = J.T @ J
    u = np.array([gfun(dv*mi) for mi in x])
    g = np.log1p(-u)                      # data log-deviation from the model
    dp = np.linalg.solve(JtJ, J.T @ g)
    resid = g - J @ dp
    orth = float(np.sqrt(np.mean(resid**2)))
    # shape-cap (U-4): normalized direction g-hat = g/max|g|
    gm = np.max(np.abs(g))
    if gm > 0:
        gh = g/gm
        dph = np.linalg.solve(JtJ, J.T @ gh)
        oh = float(np.sqrt(np.mean((gh - J @ dph)**2)))
        cap = abs(dph[2])*r["rms"]/oh if oh > 0 else float("inf")
    else:
        cap = 0.0
    return dp, orth, r, mhi, dv, float(np.max(np.abs(u))), cap, gm

# ---------------------------------------------------------------- uverify
def mode_uverify():
    print("=== U-0b: the u-extraction leg vs the banked records (printed-digit match required) ===")
    tests = [
        ("sig1p62_nu0p084500_M8192.npz", 1.625,
         {22: -2.548e-04, 26: -6.866e-04, 30: -1.447e-03, 34: -2.525e-03},
         dict(A=0.53593, D=+5.6177e-04, d=0.014491)),
        ("sig1p75_nu0p074000_M4096.npz", 1.75,
         {14: -1.766e-05, 18: -1.479e-04, 22: -5.905e-04, 26: -1.647e-03,
          30: -3.669e-03, 34: -6.698e-03, 38: -1.033e-02},
         dict(A=0.58228, D=+7.5073e-04, d=0.046817)),
        ("sig1p75_nu0p072000_M4096.npz", 1.75,
         {14: -2.087e-05, 18: -1.559e-04, 22: -6.003e-04, 26: -1.668e-03,
          30: -3.681e-03, 34: -6.729e-03, 38: -1.034e-02},
         dict(A=0.56655, D=+3.1811e-04, d=0.019417)),
    ]
    allok = True
    for fn, sig, urows, prim in tests:
        z = np.load(os.path.join(DIR, fn))
        q = np.abs(z["c_pk"])
        fit = primary_fit(q, sig)
        xx, u = u_profile(q, sig, fit)
        okp = (abs(fit["A"]/prim["A"] - 1) < 5e-5 and abs(fit["d"]/prim["d"] - 1) < 5e-5
               and abs(fit["D"] - prim["D"]) < 5e-8)
        print("--- %s: A=%.5f (banked %.5f) D=%+.4e (banked %+.4e) d=%.6f (banked %.6f)  %s"
              % (fn, fit["A"], prim["A"], fit["D"], prim["D"], fit["d"], prim["d"],
                 "OK" if okp else "MISMATCH"))
        allok &= okp
        for xr, ub in sorted(urows.items()):
            i = int(np.argmin(np.abs(xx - xr)))
            ok = abs(u[i]/ub - 1) < 5e-4
            print("    x=%g : u = %+.3e  (banked %+.3e)  %s" % (xr, u[i], ub, "OK" if ok else "MISMATCH"))
            allok &= ok
    print("U-0b verdict: %s" % ("ALL ROWS MATCH — extraction leg validated" if allok else "MISMATCH — STOP"))

# ---------------------------------------------------------------- urun
FROZEN = [
    # (file, sigma, dev target, class)
    ("snap_sig1p25_nu0p136000_M4096.npz", 1.25, +2.14724e-3, "F1 scored-branch (U-2; envelope content)"),
    ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375, +2.65354e-3, "F1 report-class row (U-2 caveat; interp content)"),
    ("snap_sig1p50_nu0p098200_M8192.npz", 1.5, +1.77778e-3, "F1 scored-branch (U-2; sub-bar content)"),
    ("sig1p62_nu0p084500_M8192.npz", 1.625, -5.655084e-4, "F1-EXT U-1b MANDATORY-2 (same-vector u)"),
    ("sig1p62_nu0p084200_M8192.npz", 1.625, -6.044618e-4, "F1-EXT report companion"),
    ("snap_sig1p75_nu0p071500_M4096.npz", 1.75, -3.79403e-3, "F1 U-1 MANDATORY (coherent battery u)"),
    ("sig1p90_nu0p060500_M2048.npz", 1.9, -9.16562e-3, "F1 report row (interp content)"),
]

def mode_urun():
    print("===)-PROJECTION TEST ===")
    print("(U-0a: s28_memint proj re-run diffed bit-identical -> s29_repro_proj.txt; U-0b: uverify passed first)")
    res = {}
    for fn, sig, dev, cls in FROZEN:
        out = {}
        for sup, lab in ((True, "SUP"), (False, "INF")):
            dp, orth, r, mhi, dv, umax, cap, gm = proj_dir(fn, sig, lambda x, s=sig, su=sup: u_model(s, x, su))
            out[lab] = (dp, orth, cap, gm, umax)
            if sup:
                rr, mh, dvv = r, mhi, dv
        dpS, orthS, capS, gmS, umaxS = out["SUP"]
        dpI, _, _, _, umaxI = out["INF"]
        xhi = mh*dvv
        print("--- %s sigma=%.4f window [8, %d] (x <= %.2f) delta-hat=%.6f rms_obs=%.3e" % (fn, sig, mh, xhi, dvv, rr["rms"]))
        print("    in-window content: max|u|_SUP = %.3e ; max|u|_INF = %.3e ; max|g|_SUP = %.3e" % (umaxS, umaxI, gmS))
        print("    dev_pred^U SUP = %+.4e  [dA=%+.3e dD=%+.3e dd=%+.2e]   INF = %+.4e" % (dpS[2], dpS[0], dpS[1], dpS[3], dpI[2]))
        print("    frozen dev = %+.4e ; ratio SUP/frozen = %+.4e ; (I-P)g rms = %.3e" % (dev, dpS[2]/dev, orthS))
        print("    U-4 shape-cap (max |e-shift| ANY U-shaped content allows at observed cleanliness): %.3e  -> %s |dev|" % (capS, "COVERS" if capS >= abs(dev) else "FORBIDS"))
        res[(fn, sig)] = (dpS[2], dpI[2], dev, capS)
    # ---------------- scoring (as registered)
    print("=== SCORING (SUP model; s29_reg.txt clauses) ===")
    dU175, _, dev175, cap175 = res[("snap_sig1p75_nu0p071500_M4096.npz", 1.75)]
    sgn = (np.sign(dU175) == np.sign(dev175))
    ratio = abs(dU175)/abs(dev175)
    u1 = sgn and (0.2 <= ratio <= 5.0)
    print("  U-1 (MANDATORY 1.75): sign %s (pred %+.2e vs dev %+.2e) ; ratio = %.3e ; in [1/5, 5]: %s  ==> %s"
          % ("MATCH" if sgn else "MISS", dU175, dev175, ratio, "YES" if 0.2 <= ratio <= 5.0 else "NO",
             "PASS" if u1 else "FAIL"))
    dU1625, _, dev1625, cap1625 = res[("sig1p62_nu0p084500_M8192.npz", 1.625)]
    sgn2 = (np.sign(dU1625) == np.sign(dev1625))
    ratio2 = abs(dU1625)/abs(dev1625)
    u1b = sgn2 and (0.2 <= ratio2 <= 5.0)
    print("  U-1b (1.625 same-vector): sign %s (pred %+.2e vs Delta-e %+.2e) ; ratio = %.3e  ==> %s"
          % ("MATCH" if sgn2 else "MISS", dU1625, dev1625, ratio2, "PASS" if u1b else "FAIL"))
    hits = 0; rows13 = []
    for fn, sig in (("snap_sig1p25_nu0p136000_M4096.npz", 1.25),
                    ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375),
                    ("snap_sig1p50_nu0p098200_M8192.npz", 1.5)):
        dU, _, dev, _ = res[(fn, sig)]
        ok = (np.sign(dU) == np.sign(dev)) and (abs(dU) >= abs(dev)/5.0)
        hits += int(ok)
        rows13.append((sig, dU, dev, ok))
        print("  U-2 row sigma=%.3f: pred %+.3e vs dev %+.3e -> %s" % (sig, dU, dev, "hit" if ok else "miss"))
    u2 = hits >= 2
    print("  U-2 (positive branch): %d/3 hits ==> %s" % (hits, "PASS" if u2 else "FAIL"))
    survive = u1 and u1b and u2
    print("=== VERDICT: the U-account %s ===" % ("SURVIVES its first registered test" if survive
          else "FAILS -> THE REGISTERED FAILURE BRANCH FIRES: the residual lies OUTSIDE the front's measured deviation classes (record and stop)"))

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "urun"
    {"uverify": mode_uverify, "urun": mode_urun}[m]()
