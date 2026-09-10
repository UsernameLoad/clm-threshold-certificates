#!/usr/bin/env python
# S27 — the e-row program extensions (; s27_reg.txt).
# Zero new rides: banked vectors/series only.
# Modes:
#   ext    —.375 full 25-column + k_FD;
#            the two clean sigma=1.5 vectors; the two sigma=1.9 report rows)
#   slopes —)/dD linear fits per banked column (F2)
#   sub2   —.5 / 0.25
# Repro obligation (C-0a/D-6): the caller re-runs s26_erow.py above|sub and
# s25_threeterm repro|open and diffs against the S26/S25 records FIRST.
import os, sys
import numpy as np
from math import gamma, pi, sin
from scipy.optimize import least_squares, brentq
from s25_threeterm import fit_slot
import s17_kappa as sk

sys.stdout.reconfigure(encoding="utf-8")
DIR = os.path.dirname(os.path.abspath(__file__))

ZETA_HALF = -1.4603545088095868   # zeta(1/2)
ZETA_34   = -3.4412853651016576   # zeta(3/4)  [1 - sigma at sigma = 0.25... NOTE: zeta(1-sigma) at sigma=0.25 is zeta(0.75)]

def B(a, b): return gamma(a)*gamma(b)/gamma(a + b)
def B1(s):   return pi/sin(pi*s)          # B(sigma, 1-sigma), signed continuation

def tailpk(c):
    a = np.abs(c)
    return float(a[-8:].max()/a.max())

# ---------------------------------------------------------------- ext ()
EXT_PEAKS = [  # (file, e_pred from s27_regcalc.txt, label, scored?)
    ("snapt_sig1p38_nu0p115500_M4096.npz", +3.187992e-02, "C-1 sigma=1.375 (S18 direct k)", True),
    ("sig1p50_nu0p098500_M8192.npz",       +2.875621e-02, "C-2 sigma=1.5 clean-vector",     True),
    ("sig1p50_nu0p099000_M8192.npz",       +2.890218e-02, "C-2 sigma=1.5 clean-vector 2",   True),
    ("sig1p90_nu0p060500_M2048.npz",       +4.902931e-03, "C-3 sigma=1.9 (derived k) REPORT", False),
    ("sig1p90_nu0p059700_M2048.npz",       +4.838099e-03, "C-3 sigma=1.9 (derived k) REPORT 2", False),
]

def fitrow(c, sig):
    return fit_slot(np.abs(np.asarray(c)), sig, -sig)

def mode_ext():
    print("===; zero rides) ===")
    for fn, epred, lab, scored in EXT_PEAKS:
        z = np.load(os.path.join(DIR, fn))
        sig = float(z["sigma"]); nu = float(z["nu"])
        cpk = z["c_pk"]; tp = tailpk(cpk)
        r = fitrow(cpk, sig)
        cln = tp <= 1e-6
        tag = ("SCORED" if scored else "REPORT") + ("" if cln else " -> DOWNGRADED-to-report by C-0b (tail %.1e > 1e-6)" % tp)
        print("--- %s  [%s]" % (fn, lab))
        print("    sigma=%.3f nu=%.6f  e_pred=%+.6e  tail/pk=%.1e  [%s]" % (sig, nu, epred, tp, tag))
        print("    PEAK: A=%+.6e  D=%+.6e  e-hat=%+.6e  d=%.6f  rms=%.2e" %
              (r["A"], r["D"], r["E"], r["d"], r["rms"]))
        print("    ==> sign(e-hat) = %s (predicted POSITIVE); e-hat/e_pred = %+.4f  (band [1/2, 2])"
              % ("POSITIVE" if r["E"] > 0 else "NEGATIVE", r["E"]/epred))
        if "snaps_c" in z.files:
            ts = z["snaps_t"]; cs = z["snaps_c"]; tags = z["snaps_tag"]
            tpkt = float(z["t_peak"])
            rows = []
            print("    C-1c report — the %d-row e-hat(t) column:" % len(ts))
            for i in range(len(ts)):
                tpi = tailpk(cs[i]); ri = fitrow(cs[i], sig)
                rows.append((float(ts[i]), tpi, ri))
                flag = "" if tpi <= 1e-6 else "  [DIRTY tail — reported-not-scored]"
                print("      %-4s t=%.4f  tail=%.1e  A=%+.5e  D=%+.5e  e-hat=%+.5e  d=%.6f  rms=%.1e%s"
                      % (str(tags[i]), ts[i], tpi, ri["A"], ri["D"], ri["E"], ri["d"], ri["rms"], flag))
            # C-1d: k_FD by FD of the fitted D column at the three closest
            # symmetric-in-time bracketing pairs around t_peak
            tarr = np.array([x[0] for x in rows]); Darr = np.array([x[2]["D"] for x in rows])
            below = [i for i in range(len(tarr)) if tarr[i] < tpkt]
            above = [i for i in range(len(tarr)) if tarr[i] > tpkt]
            below.sort(key=lambda i: tpkt - tarr[i]); above.sort(key=lambda i: tarr[i] - tpkt)
            print("    C-1d: Ddot at t_pk by FD of the fitted D column (stencil trio):")
            fds = []
            for s_i in range(3):
                iu, idn = below[s_i], above[s_i]
                fd = (Darr[idn] - Darr[iu])/(tarr[idn] - tarr[iu])
                fds.append(fd)
                print("      stencil %d: Ddot = %+.6f  (t = %.4f .. %.4f)" % (s_i+1, fd, tarr[iu], tarr[idn]))
            kfd = abs(fds[0]); spread = max(abs(f - fds[0]) for f in fds)
            print("      k_FD = %.6f  spread = %.6f  k_FD/nu^2 = %.4f  vs banked k/nu^2 = 4.0969  ratio = %.4f  (C-1d band +-15%%)"
                  % (kfd, spread, kfd/nu**2, kfd/nu**2/4.0969))

def clean_rows_for_slope(rows):
    # deterministic criteria registered C-5: tail <= 1e-6 AND rms <= 1e-4
    return [(t, r) for (t, tp, r) in rows if tp <= 1e-6 and r["rms"] <= 1e-4]

SLOPE_FILES = [
    ("snap_sig1p25_nu0p136000_M4096.npz", 1.25),
    ("snap_sig1p50_nu0p098200_M8192.npz", 1.50),
    ("snap_sig1p75_nu0p071500_M4096.npz", 1.75),
    ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375),
    ("snap_sig0p75_nu0p246500_M8192.npz", 0.75),
]

def mode_slopes():
    print("===): d(e-hat)/dD by least-squares linear fit on CLEAN rows ===")
    print("(clean := tail <= 1e-6 and fit rms <= 1e-4, the registered deterministic criteria)")
    for fn, sig in SLOPE_FILES:
        z = np.load(os.path.join(DIR, fn))
        nu = float(z["nu"])
        ts = z["snaps_t"]; cs = z["snaps_c"]
        rows = []
        for i in range(len(ts)):
            tpi = tailpk(cs[i]); ri = fitrow(cs[i], sig)
            rows.append((float(ts[i]), tpi, ri))
        cl = clean_rows_for_slope(rows)
        if len(cl) >= 3:
            D = np.array([r["D"] for (_, r) in cl]); E = np.array([r["E"] for (_, r) in cl])
            Amat = np.vstack([np.ones_like(D), D]).T
            coef, *_ = np.linalg.lstsq(Amat, E, rcond=None)
            resid = E - Amat @ coef
            print("%-40s sigma=%.3f nu=%.4f  n_clean=%2d  e0=%+.5e  s=d(e-hat)/dD=%+.5f  rms(resid)=%.1e"
                  % (fn, sig, nu, len(cl), coef[0], coef[1], float(np.sqrt(np.mean(resid**2)))))
        else:
            print("%-40s sigma=%.3f nu=%.4f  n_clean=%2d  — EMPTY by the registered criterion (no scored slope)"
                  % (fn, sig, nu, len(cl)))
        # labeled NON-registered observation for theory use only: all-rows slope
        Da = np.array([r["D"] for (_, _, r) in rows]); Ea = np.array([r["E"] for (_, _, r) in rows])
        if len(Da) >= 3:
            Am = np.vstack([np.ones_like(Da), Da]).T
            ca, *_ = np.linalg.lstsq(Am, Ea, rcond=None)
            ra = Ea - Am @ ca
            print("%-40s   [ALL-ROWS observation, reported-not-scored: e0=%+.5e  s=%+.5f  rms=%.1e]"
                  % ("", ca[0], ca[1], float(np.sqrt(np.mean(ra**2)))))

# ---------------------------------------------------------------- sub2 ()
def fd_trio(tarr, arr, tpk):
    below = [i for i in range(len(tarr)) if tarr[i] < tpk]
    above = [i for i in range(len(tarr)) if tarr[i] > tpk]
    below.sort(key=lambda i: tpk - tarr[i]); above.sort(key=lambda i: tarr[i] - tpk)
    fds = []
    for s_i in range(min(3, len(below), len(above))):
        iu, idn = below[s_i], above[s_i]
        fds.append((arr[idn] - arr[iu])/(tarr[idn] - tarr[iu]))
    return fds

def quad_min(ts, ys, t0, lo, hi):
    dt = np.asarray(ts[lo:hi]) - t0
    Amat = np.vstack([np.ones_like(dt), dt, dt*dt]).T
    coef, *_ = np.linalg.lstsq(Amat, np.asarray(ys[lo:hi]), rcond=None)
    return 2.0*coef[2]

def mode_sub2():
    print("===; zero rides) ===")
    # ---------------- sigma = 0.5: the MERGED two-slot fit + decomposition
    fn = "snap_sig0p50_nu0p322000_M4096.npz"
    z = np.load(os.path.join(DIR, fn))
    sig = 0.5; nu = float(z["nu"]); tpk = float(z["t_peak"])
    ts = z["snaps_t"]; cs = z["snaps_c"]; tags = z["snaps_tag"]
    print("--- %s  sigma=%.2f nu=%.6f t_peak=%.6f  [D-1/D-2/D-3/D-4]" % (fn, sig, nu, tpk))
    rows = []
    print("    merged two-slot fit T m^{-1/2} + D per snapshot (fit_AD; the (A,e) collision — S25):")
    for i in range(len(ts)):
        tpi = tailpk(cs[i])
        q = np.abs(cs[i])
        mmax_good = int(np.max(np.arange(1, len(q) + 1)[q > q.max()*1e-12]))
        if int(0.4*mmax_good) < 12:
            print("      %-7s t=%.4f tail=%.1e  SKIPPED (front too shallow: mmax_good=%d, window empty)"
                  % (str(tags[i]), ts[i], tpi, mmax_good))
            continue
        r = sk.fit_AD(q, sig)      # A-slot m^{sigma-1} = m^{-1/2} == the MERGED slot T
        rows.append((float(ts[i]), tpi, r))
        print("      %-7s t=%.4f tail=%.1e  T=%+.6e  D=%+.6f  d=%.6f  rms=%.1e"
              % (str(tags[i]), ts[i], tpi, r["A"], r["D"], r["d"], r["rms"]))
    tarr = np.array([x[0] for x in rows]); Darr = np.array([x[2]["D"] for x in rows])
    darr = np.array([x[2]["d"] for x in rows])
    fds = fd_trio(tarr, Darr, tpk)
    kfd = abs(fds[0]); spread = max(abs(f - fds[0]) for f in fds)
    print("    D-column FD trio: Ddot = %s ; k_FD = %.6f (spread %.6f) ; k_FD/nu^2 = %.4f  [D-4 report]"
          % (", ".join("%+.5f" % f for f in fds), kfd, spread, kfd/nu**2))
    ipk = int(np.argmin(np.abs(tarr - tpk)))
    for w, lab in [(3, "6c"), (4, "8c")]:
        lo, hi = max(0, ipk - w), min(len(tarr), ipk + w)
        d2 = quad_min(tarr, darr, tpk, lo, hi)
        print("    ddot(delta)|_min (%s: rows %d..%d) = %+.6e ; Y = ddot/nu^2 = %.4f ; ratio k_FD/(2 ddot) = %.4f  [D-3 +-35%% / D-4]"
              % (lab, lo, hi - 1, d2, d2/nu**2, kfd/(2.0*d2) if d2 != 0 else float("nan")))
    # peak decomposition: T_pk = A + e, e = [Ddot_pk + D^2/2 - D A zeta(1/2)]/[A pi - nu]
    ipk = int(np.argmin(np.abs(tarr - tpk)))
    # use the two peak-adjacent rows' T, D (and the c_pk fit) — score on c_pk
    qpk = np.abs(z["c_pk"]); rpk = sk.fit_AD(qpk, sig); tppk = tailpk(z["c_pk"])
    Ddot_pk = -kfd
    Tpk, Dpk = rpk["A"], rpk["D"]
    def eq(A):
        return A + (Ddot_pk + 0.5*Dpk*Dpk - Dpk*A*ZETA_HALF)/(A*pi - nu) - Tpk
    # bracket: A in (nu/pi + eps, 2*Tpk)
    Alo = nu/pi*1.05; Ahi = 2.5*Tpk
    A_root = brentq(eq, Alo, Ahi, xtol=1e-14)
    e_dec = Tpk - A_root
    Alaw = 2.0*nu*sig/(2.0 - sig)
    print("    c_pk fit: T=%+.6e D=%+.6f d=%.6f tail=%.1e" % (Tpk, Dpk, rpk["d"], tppk))
    print("    D-1/D-2: decomposition at the peak with Ddot_pk = %+.6f:" % Ddot_pk)
    print("      A = %+.6e  e = %+.6e  sign(e) = %s (D-1 registered NEGATIVE)"
          % (A_root, e_dec, "NEGATIVE" if e_dec < 0 else "POSITIVE"))
    print("      A_law = %.6f ; merged-T deficit = %+.2f%% ; decomposed-A deficit = %+.2f%%  [D-2 report; free 2-term -21.50%%, S24 3-slot -9.11%%]"
          % (Alaw, 100*(Tpk/Alaw - 1), 100*(A_root/Alaw - 1)))
    print("      e/nu = %+.4f ; denominator [A pi - nu]/nu at decomposed A = %+.4f (robustly positive as registered)"
          % (e_dec/nu, (A_root*pi - nu)/nu))
    # ---------------- sigma = 0.25: constrained refit (report, D-5)
    fn = "snap_sig0p25_nu0p407000_M4096.npz"
    z = np.load(os.path.join(DIR, fn))
    sig = 0.25; nu = float(z["nu"]); tpk = float(z["t_peak"])
    ts = z["snaps_t"]; cs = z["snaps_c"]; tags = z["snaps_tag"]
    Bss = B(sig, sig); B1s = B1(sig)     # B(0.25, 0.75) = 4.442883
    zet = ZETA_34                        # zeta(1 - sigma) = zeta(0.75)
    print("--- %s  sigma=%.2f nu=%.6f t_peak=%.6f  [D-5 report]" % (fn, sig, nu, tpk))
    # first pass: two-term columns for Ddot
    rows = []
    for i in range(len(ts)):
        tpi = tailpk(cs[i])
        q = np.abs(cs[i])
        mmax_good = int(np.max(np.arange(1, len(q) + 1)[q > q.max()*1e-12]))
        if int(0.4*mmax_good) < 12:
            print("    %-7s t=%.4f SKIPPED (front too shallow: mmax_good=%d)" % (str(tags[i]), float(ts[i]), mmax_good))
            continue
        r = sk.fit_AD(q, sig)
        rows.append((float(ts[i]), tpi, r))
    tarr = np.array([x[0] for x in rows]); Darr = np.array([x[2]["D"] for x in rows])
    darr = np.array([x[2]["d"] for x in rows])
    fds = fd_trio(tarr, Darr, tpk)
    kfd = abs(fds[0]); spread = max(abs(f - fds[0]) for f in fds)
    print("    two-term D-column FD trio: Ddot = %s ; k_FD = %.6f (spread %.6f) ; k_FD/nu^2 = %.4f"
          % (", ".join("%+.5f" % f for f in fds), kfd, spread, kfd/nu**2))
    ipk = int(np.argmin(np.abs(tarr - tpk)))
    for w, lab in [(3, "6c"), (4, "8c")]:
        lo, hi = max(0, ipk - w), min(len(tarr), ipk + w)
        d2 = quad_min(tarr, darr, tpk, lo, hi)
        print("    ddot(delta)|_min (%s: rows %d..%d) = %+.6e ; Y = %.4f ; ratio k_FD/(2 ddot) = %.4f"
              % (lab, lo, hi - 1, d2, d2/nu**2, kfd/(2.0*d2) if d2 != 0 else float("nan")))
    # constrained refit on c_pk: e eliminated via the identity at the peak
    qpk = np.abs(z["c_pk"]); tppk = tailpk(z["c_pk"])
    Ddot_pk = -kfd
    M = len(qpk); mm = np.arange(1, M + 1).astype(float)
    good = qpk > qpk.max()*1e-12
    mmax_good = int(np.max(mm[good])); mhi = int(0.4*mmax_good)
    selv = (mm >= 8) & (mm <= mhi) & good & (qpk > 0)
    x = mm[selv]; y = np.log(qpk[selv])
    two = sk.fit_AD(qpk, sig)
    def e_of(Av, Dv):
        den = Av*B1s - nu
        if abs(den) < 1e-9: den = 1e-9 if den >= 0 else -1e-9
        return (Ddot_pk + 0.5*Dv*Dv - Dv*Av*zet)/den
    def res(p):
        Av, Dv, dv = p
        ev = e_of(Av, Dv)
        f = Av*x**(sig - 1.0) + Dv + ev*x**(-sig)
        f = np.where(f > 1e-300, f, 1e-300)
        return np.log(f) - dv*x - y
    out = least_squares(res, [two["A"], two["D"], two["d"]], method="lm",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=40000)
    Av, Dv, dv = out.x
    ev = e_of(Av, Dv)
    rmsv = float(np.sqrt(np.mean(out.fun**2)))
    Alaw = 2.0*nu*sig/(2.0 - sig)
    print("    c_pk tail=%.1e ; free two-term: A=%+.6e (deficit %+.2f%%) D=%+.6f d=%.6f rms=%.2e"
          % (tppk, two["A"], 100*(two["A"]/Alaw - 1), two["D"], two["d"], two["rms"]))
    print("    CONSTRAINED refit (e = identity(A, D; Ddot_pk)):")
    print("      A=%+.6e (deficit %+.2f%%)  D=%+.6f  e=%+.6e  d=%.6f  rms=%.2e"
          % (Av, 100*(Av/Alaw - 1), Dv, ev, dv, rmsv))
    print("      denominator [A B1 - nu]/nu at fit A = %+.4f  (sign fragility recorded in the registration)"
          % ((Av*B1s - nu)/nu))
    print("      [D-5 anchors: free two-term deficit -37.16%%, S24 3-slot -19.63%%, free e-slot crash -62.85%% (S25 G-2)]")

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "ext"
    if m == "ext": mode_ext()
    elif m == "slopes": mode_slopes()
    elif m == "sub2": mode_sub2()
    else: print("modes: ext | slopes | sub2")
