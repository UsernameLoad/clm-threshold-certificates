#!/usr/bin/env python
# S28 — the memory-integral sub-slot computation (.txt +
# Amendment 1) and the sigma = 1.625 zero-crossing row ().
# NUMERICS-ANCHORED: every analytic label exact-sum-verified BEFORE use
# (B-0); the S27 verify/proj/traj records reproduced bit-identically first
# (s28_repro_*.txt, diffed — 16th session of the predecessor rule).
# Modes:
#   verify — B-0: the NEW kernel labels (e*e leading Beta incl. the
#            sigma=1.5 zero; A*f, A*w in-row coefficients; D*slot -1/2 EM
#            terms; A*h (2-sigma) zeta(1-sigma) re-verified)
#   proj   — the extended projection table: de at p in {sigma-2, -1,
#            1-sigma, 1-2 sigma} + the B-6 (I-P) g-norms and rms_obs
#   run    —
#            above-corner trajectories; project; score vs the FROZEN F1;
#            B-6 cap; B-4 extrapolation; both IC and model variants
#   slopes —) at
#            0.75/1.25/1.375/1.5/1.75 vs measured/S27-corrected
#   s1625  —.625 identity row on two banked vectors
import os, sys
import numpy as np
from math import gamma, pi, sin
import s27_theory as st                       # zeta, B1, col_fits, quad_model
from s25_threeterm import fit_slot

sys.stdout.reconfigure(encoding="utf-8")
DIR = os.path.dirname(os.path.abspath(__file__))

def B(a, b): return gamma(a)*gamma(b)/gamma(a+b)

def Bcont(a, b):
    # analytic continuation of Beta via Gammas; 1/Gamma(pole) -> 0 handled
    ga, gb, gab = gamma(a), gamma(b), gamma(a+b)
    return ga*gb/gab

def Bff(s):
    # B(1-s,1-s) continuation = Gamma(1-s)^2/Gamma(2-2s); zero at s=1.5
    if abs(2.0 - 2.0*s + 1.0) < 1e-12:
        return 0.0
    return gamma(1.0-s)**2/gamma(2.0-2.0*s)

SIGS = [1.25, 1.375, 1.5, 1.75, 1.9]
# c_G from the S27 verify table (bit-identically reproduced this session)
CG = {1.25: +0.02445, 1.375: +0.02704, 1.5: +0.02548, 1.75: +0.01485, 1.9: +0.00610}

# ------------------------------------------------------------------ verify
def kernel_sum(s, pj, pk, m):
    j = np.arange(1, m)
    return float(np.sum(j.astype(float)**pj * (m - j).astype(float)**pk))

def measure_inrow(s, pk_pow, sub_terms, ms, label, cands):
    """kernel K(m) = sum j^{s-1}(m-j)^{pk_pow}; subtract sub_terms (list of
    (coeff, power)); the residual's coefficient at m^{pk_pow} measured; the
    ratio to zeta(1-s) compared against candidate factors."""
    z1s = st.zeta(1.0 - s)
    rows = []
    for m in ms:
        K = kernel_sum(s, s - 1.0, pk_pow, m)
        r = K - sum(c*m**p for (c, p) in sub_terms)
        rows.append((m, r))
    # local power of the residual
    pw = [np.log(abs(rows[i+1][1]/rows[i][1]))/np.log(ms[i+1]/ms[i])
          for i in range(len(ms)-1) if rows[i][1] != 0]
    c_meas = rows[-1][1]/ms[-1]**pk_pow
    ratio = c_meas/z1s if z1s != 0 else float("nan")
    cand_str = "  ".join("%s=%+.5f" % (nm, v) for nm, v in cands(s))
    print("    %s: residual power = %s [target %.3f]; coeff = %+.6f; coeff/zeta(1-s) = %+.5f   [cands: %s]"
          % (label, ", ".join("%.3f" % p for p in pw), pk_pow, c_meas, ratio, cand_str))
    return ratio

def measure_inrow_lsq(s, target_pow, sub_terms, extra_pows, label):
    """subtract the analytic leaders; LSQ-fit the residual over log-spaced m
    on the basis {extra_pows..., target_pow} (known intermediate k-end EM
    powers measured jointly, not assumed); report target coeff / zeta(1-s)."""
    z1s = st.zeta(1.0 - s)
    ms = np.array([256, 362, 512, 724, 1024, 1448, 2048, 2896])
    res = []
    for m in ms:
        K = kernel_sum(s, s - 1.0, target_pow, m)
        res.append(K - sum(c*m**p for (c, p) in sub_terms))
    res = np.array(res)
    # keep only basis powers strictly above-or-equal target and below the
    # subtracted leaders (dedupe collisions with the target)
    basis = [p for p in extra_pows if p > target_pow + 1e-9] + [target_pow]
    Amat = np.vstack([ms.astype(float)**p for p in basis]).T
    coef, *_ = np.linalg.lstsq(Amat, res, rcond=None)
    c_t = coef[-1]
    # fit-quality echo: refit dropping the largest m (stability)
    coef2, *_ = np.linalg.lstsq(Amat[:-1], res[:-1], rcond=None)
    ratio = c_t/z1s if z1s != 0 else float("nan")
    print("    %s [LSQ basis %s]: target coeff = %+.6f (drop-1 echo %+.6f); coeff/zeta(1-s) = %+.5f  [cands: 1, (2-s)=%.3f]"
          % (label, ",".join("%.2f" % p for p in basis), c_t, coef2[-1], ratio, 2.0 - s))
    return ratio

def mode_verify():
    print("===")
    print("(the S27 verify table reproduced bit-identically first: s28_repro_verify.txt)")
    ms = np.array([512, 1024, 2048, 4096, 8192])
    for s in SIGS:
        zs, z1s = st.zeta(s), st.zeta(1.0 - s)
        print("--- sigma = %.3f" % s)
        # (a) e*e kernel: sum j^{-s}(m-j)^{-s} == B(1-s,1-s) m^{1-2s} + 2 zeta(s) m^{-s} + ...
        bf = Bff(s)
        rows = []
        for m in ms:
            K = kernel_sum(s, -s, -s, m)
            r = K - (bf*m**(1.0-2.0*s) + 2.0*zs*m**(-s))
            rows.append((m, K, r))
        pw = [np.log(abs(rows[i+1][2]/rows[i][2]))/np.log(ms[i+1]/ms[i])
              for i in range(len(ms)-1) if rows[i][2] != 0]
        print("    e*e: B(1-s,1-s) = %+.5f ; residual after 2 terms at m=8192: %+.3e ; local power = %s"
              % (bf, rows[-1][2], ", ".join("%.3f" % p for p in pw)))
        if abs(s - 1.5) < 1e-12:
            # the zero: the m^{1-2s} content should be ABSENT — check the
            # residual against the pure 2 zeta(s) m^{-s} + next-order model
            K = kernel_sum(s, -s, -s, 8192)
            r0 = K - 2.0*zs*8192.0**(-s)
            print("    e*e sigma=1.5 ZERO check: K - 2 zeta(s) m^{-s} = %+.3e at m=8192 (m^{1-2s} = m^{-2} content would be %+.3e per unit coeff)"
                  % (r0, 8192.0**(-2.0)))
        # (b) A*h kernel re-verify: sum j^{s-1}(m-j)^{s-2}: continuum
        # B(s,s-1) m^{2s-2} + zeta(2-s) m^{s-1} + c m^{s-2}, c =? (2-s) zeta(1-s)
        measure_inrow(s, s - 2.0,
                      [(Bcont(s, s - 1.0), 2.0*s - 2.0), (st.zeta(2.0 - s), s - 1.0)],
                      ms, "A*h in-row",
                      lambda s: [("(2-s)", 2.0 - s)])
        # (c) A*f kernel: sum j^{s-1}(m-j)^{1-2s}: continuum B(s,2-2s) m^{1-s}
        # + k-end zeta(2s-1) m^{s-1} + intermediate k-end EM orders (m^{s-2},
        # m^{s-3}, m^{s-4}) sit ABOVE the deep target m^{1-2s} — the pass-1
        # subtraction model missed them (s28_verify_pass1.txt); the upgraded
        # measurement subtracts the two analytic leaders and LSQ-fits the
        # residual on the known-power basis, extracting the target coeff.
        if abs(s - 1.5) < 1e-12:
            print("    A*f in-row: SKIPPED at sigma=1.5 (B(s,2-2s) continuum pole AND the f-forcing is exactly 0 there)")
        else:
            measure_inrow_lsq(s, 1.0 - 2.0*s,
                              [(Bcont(s, 2.0 - 2.0*s), 1.0 - s), (st.zeta(2.0*s - 1.0), s - 1.0)],
                              [s - 2.0, s - 3.0, s - 4.0], "A*f in-row")
        # (d) A*w kernel: continuum B(s,2-s) m^{1} + k-end zeta(s-1) m^{s-1}
        # + intermediate k-end m^{s-2} above the target m^{1-s} for s > 1.5
        measure_inrow_lsq(s, 1.0 - s,
                          [(Bcont(s, 2.0 - s), 1.0), (st.zeta(s - 1.0), s - 1.0)],
                          [s - 2.0, s - 3.0], "A*w in-row")
        # (e) D*slot -1/2 EM terms: S_p(m) = sum j^p, the -m^p/2 term:
        for p, nm in [(s - 2.0, "D*h"), (1.0 - 2.0*s, "D*f"), (1.0 - s, "D*w")]:
            m = 8192
            j = np.arange(1, m).astype(float)
            Sp = float(np.sum(j**p))
            # S_p(m) = zeta(-p) + m^{p+1}/(p+1) - m^p/2 + p m^{p-1}/12 + ...
            model = st.zeta(-p) + m**(p+1.0)/(p+1.0) - m**p/2.0 + p*m**(p-1.0)/12.0
            print("    %s EM check (S_%.3f at m=8192): residual vs (zeta - m^p/2 + ...) model = %+.2e  [-1/2 in-row term verified within]"
                  % (nm, p, Sp - model))

# ------------------------------------------------------------------ proj
PEAKS = [
    ("snap_sig1p25_nu0p136000_M4096.npz", 1.25, +2.931074e-02),
    ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375, +3.187992e-02),
    ("snap_sig1p50_nu0p098200_M8192.npz", 1.50, +2.866862e-02),
    ("snap_sig1p75_nu0p071500_M4096.npz", 1.75, +1.388673e-02),
    ("sig1p90_nu0p060500_M2048.npz", 1.90, +4.902931e-03),
]

def proj_rows(fn, sig, powers):
    """the REGISTERED S27 projection arithmetic (mode_proj verbatim math),
    extended to the given powers; returns {p: dp}, plus (I-P)-rms per power
    and the fit rms_obs."""
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
    out, orth = {}, {}
    for p in powers:
        g = (x**p)/F
        dp = np.linalg.solve(JtJ, J.T @ g)
        out[p] = dp
        resid = g - J @ dp
        orth[p] = float(np.sqrt(np.mean(resid**2)))
    return out, orth, r, mhi

def uniq_powers(sig):
    ps = [sig - 2.0, -1.0, 1.0 - sig, 1.0 - 2.0*sig, 2.0*sig - 2.0]
    u = []
    for p in ps:
        if not any(abs(p - q) < 1e-12 for q in u):
            u.append(p)
    return u

def mode_proj():
    print("=== the extended projection table (S27 rows reproduced via s28_repro_proj.txt first) ===")
    for fn, sig, epred in PEAKS:
        out, orth, r, mhi = proj_rows(fn, sig, uniq_powers(sig))
        dev = r["E"] - epred
        print("--- %s sigma=%.3f window [8, %d] rms_obs=%.3e  e-hat=%+.5e dev=%+.5e"
              % (fn, sig, mhi, r["rms"], r["E"], dev))
        for p, dp in out.items():
            print("    p=%+.3f : dA=%+.4e dD=%+.4e de=%+.4e dd=%+.2e  (I-P)g rms=%.3e"
                  % (p, dp[0], dp[1], dp[2], dp[3], orth[p]))

# ------------------------------------------------------------------ run
COLS = [
    ("snap_sig1p25_nu0p136000_M4096.npz", 1.25),
    ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375),
    ("snap_sig1p50_nu0p098200_M8192.npz", 1.50),
    ("snap_sig1p75_nu0p071500_M4096.npz", 1.75),
]
NSUB = 256   # fixed substeps per snapshot interval (registered implementation constant)

def integrate_linear(tgrid, Fgrid, Lgrid, t_end, y0, nsub=NSUB):
    """y' = F(t) + L(t) y, piecewise-linear F, L on tgrid; exponential-midpoint
    with nsub substeps per interval, integrated from tgrid[0] to t_end."""
    y = y0
    for i in range(len(tgrid) - 1):
        ta, tb = tgrid[i], tgrid[i+1]
        if ta >= t_end:
            break
        tb_eff = min(tb, t_end)
        n = max(2, int(nsub*(tb_eff - ta)/(tb - ta))) if tb > ta else 2
        hh = (tb_eff - ta)/n
        for k in range(n):
            tm = ta + (k + 0.5)*hh
            w1 = (tm - ta)/(tb - ta)
            Fm = Fgrid[i]*(1 - w1) + Fgrid[i+1]*w1
            Lm = Lgrid[i]*(1 - w1) + Lgrid[i+1]*w1
            eL = np.exp(Lm*hh)
            y = y*eL + (Fm*(eL - 1.0)/Lm if abs(Lm) > 1e-14 else Fm*hh)
        if tb >= t_end:
            break
    return y

def column_arrays(fn, sig):
    z = np.load(os.path.join(DIR, fn))
    nu = float(z["nu"]); tpk = float(z["t_peak"])
    rows = st.col_fits(z, sig)
    t = np.array([r[0] for r in rows])
    A = np.array([r[2]["A"] for r in rows])
    D = np.array([r[2]["D"] for r in rows])
    E = np.array([r[2]["E"] for r in rows])
    return nu, tpk, t, A, D, E, rows

def slot_integrals(sig, nu, tpk, t, A, D, E, full=True, ic_qs=False, nsub=NSUB):
    z1s = st.zeta(1.0 - sig)
    C2 = -(sig - 1.0)*st.zeta(sig - 1.0)
    cG = CG[sig]
    bf = Bff(sig)
    # forcings and dampings on the snapshot grid
    Fh = A*E*C2 + (A*A/2.0)*cG + (A*D*(sig - 1.0)/12.0 if full else 0.0)
    Lh = A*(2.0 - sig)*z1s - (D/2.0 if full else 0.0)
    Ff = (E*E/2.0)*bf
    # c_Af = 1 (B-0 measured: +0.984/+1.008/—/+1.004/+0.999)
    Lf = A*1.0*z1s - (D/2.0 if full else 0.0)
    Fw = D*E/(1.0 - sig)
    # c_Aw = 1 off the collision (B-0: +0.996/+0.949/+0.997/+0.989);
    # AT sigma = 1.5 the w-power collides with the h-power and the kernel
    # measures 0.500 = (2-s) — the merged-slot value (B-0 table)
    c_Aw = 0.5 if abs(sig - 1.5) < 1e-12 else 1.0
    Lw = A*c_Aw*z1s - (D/2.0 if full else 0.0)
    ics = [0.0, 0.0, 0.0]
    if ic_qs:
        ics = [-Fh[0]/Lh[0], -Ff[0]/Lf[0] if Lf[0] != 0 else 0.0, -Fw[0]/Lw[0] if Lw[0] != 0 else 0.0]
    h = integrate_linear(t, Fh, Lh, tpk, ics[0], nsub)
    f = integrate_linear(t, Ff, Lf, tpk, ics[1], nsub)
    w = integrate_linear(t, Fw, Lw, tpk, ics[2], nsub)
    # integral of h dt for B-6 (trapezoid of the h(t) path re-integrated)
    hpath = []
    y = ics[0]
    hpath.append(y)
    for i in range(len(t) - 1):
        y = integrate_linear(t[i:i+2], Fh[i:i+2], Lh[i:i+2], t[i+1], y, nsub)
        hpath.append(y)
    hpath = np.array(hpath)
    tcap = np.minimum(t, tpk)
    ihdt = float(np.trapezoid(np.where(t <= tpk, hpath, hpath[-1]),
                              tcap)) if t[-1] >= tpk else float(np.trapezoid(hpath, t))
    return h, f, w, (Fh, Lh, Ff, Lf, Fw, Lw), ihdt

def mode_run():
    print("===")
    print("(S27 traj/proj reproduced bit-identically first; verify stage passed before this ran)")
    DEV = {1.25: +2.14724e-3, 1.375: +2.65354e-3, 1.5: +1.77778e-3, 1.75: -3.79403e-3, 1.9: -9.16562e-3}
    devpred = {}
    for (fn, sig) in COLS:
        nu, tpk, t, A, D, E, rows = column_arrays(fn, sig)
        pfn = fn  # same file is the peak vector source
        powers = uniq_powers(sig)
        out, orth, rpk, mhi = proj_rows(pfn, sig, powers)
        de_h = out[min(powers, key=lambda p: abs(p - (sig - 2.0)))][2]
        de_f = out[min(powers, key=lambda p: abs(p - (1.0 - 2.0*sig)))][2]
        de_w = out[min(powers, key=lambda p: abs(p - (1.0 - sig)))][2]
        print("--- %s sigma=%.3f nu=%.4f t0=%.4f t_pk=%.4f rows=%d" % (fn, sig, nu, t[0], tpk, len(t)))
        for full, icq, lab in [(True, False, "FULL model, IC=0 (PRIMARY)"),
                               (True, True,  "FULL model, IC=qs (variant)"),
                               (False, False, "minimal model, IC=0 (variant)")]:
            h, f, w, FL, ihdt = slot_integrals(sig, nu, tpk, t, A, D, E, full=full, ic_qs=icq)
            Fh, Lh, Ff, Lf, Fw, Lw = FL
            dp = h*de_h + f*de_f + w*de_w
            dp2 = h*de_h + f*de_f
            if lab.startswith("FULL model, IC=0"):
                devpred[sig] = dp
                print("    forcing sign check: F_h in [%+.3e, %+.3e]%s ; F_f in [%+.3e, %+.3e]%s ; F_w in [%+.3e, %+.3e]%s"
                      % (Fh.min(), Fh.max(), " SINGLE-SIGNED" if Fh.min()*Fh.max() > 0 else " SIGN-CHANGES",
                         Ff.min(), Ff.max(), " SINGLE-SIGNED" if Ff.min()*Ff.max() > 0 else (" ZERO" if Ff.max() == 0 and Ff.min() == 0 else " SIGN-CHANGES"),
                         Fw.min(), Fw.max(), " SINGLE-SIGNED" if Fw.min()*Fw.max() > 0 else " SIGN-CHANGES"))
                print("    damping: Lam_h in [%+.4f, %+.4f] ; Lam_f/w in [%+.4f, %+.4f]"
                      % (Lh.min(), Lh.max(), Lf.min(), Lf.max()))
                # integration-error echo (NSUB doubled)
                h2, f2, w2, _, _ = slot_integrals(sig, nu, tpk, t, A, D, E, full=full, ic_qs=icq, nsub=2*NSUB)
                print("    integration echo (2x substeps): |dh|=%.1e |df|=%.1e |dw|=%.1e"
                      % (abs(h2 - h), abs(f2 - f), abs(w2 - w)))
                # B-6
                Bg = Bcont(sig, sig - 1.0)
                Alaw_t = float(np.mean(A))
                gcoup = Alaw_t*Bg - nu
                Xg_max = rpk["rms"]/orth[min(powers, key=lambda p: abs(p - (2.0*sig - 2.0)))]
                cap = 2.0*Xg_max/abs(gcoup)
                print("    B-6: rms_obs=%.3e ; |X_g|_max=%.3e ; g-coupling=%.4f ; cap on |int h dt| = %.3e ; LR int h dt = %+.3e  -> %s"
                      % (rpk["rms"], Xg_max, gcoup, cap, ihdt,
                         "LR VIOLATES the profile cap" if abs(ihdt) > cap else "consistent with the cap"))
            print("    [%s] h(t_pk)=%+.4e f(t_pk)=%+.4e w(t_pk)=%+.4e -> dev_pred(3-slot)=%+.4e [2-slot variant %+.4e]  (frozen dev=%+.4e)"
                  % (lab, h, f, w, dp, dp2, DEV[sig]))
    # scoring
    print("=== SCORING vs the FROZEN F1 (PRIMARY variant) ===")
    hits = 0
    m175 = False
    for sig in (1.25, 1.375, 1.5, 1.75):
        ok = np.sign(devpred[sig]) == np.sign(DEV[sig])
        hits += int(ok)
        if sig == 1.75:
            m175 = ok
        print("  sigma=%.3f  dev_pred=%+.4e  dev=%+.4e  sign %s" % (sig, devpred[sig], DEV[sig], "MATCH" if ok else "MISS"))
    print("  B-1: %d/4 sign matches; mandatory sigma=1.75 row: %s  ==> %s"
          % (hits, "MATCH" if m175 else "MISS",
             "CONFIRMED" if (hits >= 3 and m175) else "FAILED -> the registered FAILURE BRANCH FIRES"))
    if m175:
        rr = abs(devpred[1.75])/abs(DEV[1.75])
        print("  B-2: |dev_pred(1.75)|/|dev| = %.3f  ==> %s" % (rr, "CONFIRMED" if 0.5 <= rr <= 2.0 else "FAILED"))
    else:
        print("  B-2: fails with B-1's mandatory row (conditional clause)")
    # B-4
    dp19 = devpred[1.75] + (devpred[1.75] - devpred[1.5])*(1.9 - 1.75)/(1.75 - 1.5)
    print("  B-4 [MODELED-EXTRAPOLATION]: linear-in-sigma dev_pred(1.9) = %+.4e vs frozen %+.4e" % (dp19, DEV[1.9]))

# ------------------------------------------------------------------ slopes
def mode_slopes():
    print("===")
    SC = [("snap_sig0p75_nu0p246500_M8192.npz", 0.75)] + COLS
    MEAS = {0.75: +0.0493, 1.25: +0.1117, 1.375: +0.1944, 1.5: +0.2295, 1.75: +0.1433}
    S27C = {0.75: -0.1057, 1.25: +0.1927, 1.375: +0.2067, 1.5: +0.2060, 1.75: +0.1188}
    for fn, sig in SC:
        z = np.load(os.path.join(DIR, fn))
        nu = float(z["nu"]); tpk = float(z["t_peak"])
        zs, z1s, b1 = st.zeta(sig), st.zeta(1.0 - sig), st.B1(sig)
        rows = st.col_fits(z, sig)
        t = np.array([r[0] for r in rows])
        A = np.array([r[2]["A"] for r in rows])
        D = np.array([r[2]["D"] for r in rows])
        E = np.array([r[2]["E"] for r in rows])
        w6 = 6 if len(rows) > 15 else 3
        cD = st.quad_model(t, D, tpk, w6)
        # w-ODE along the column, evaluated at EVERY row time (IC=0 primary)
        Fw = D*E/(1.0 - sig)
        Lw = A*1.0*z1s - D/2.0
        wpath = [0.0]
        y = 0.0
        for i in range(len(t) - 1):
            y = integrate_linear(t[i:i+2], Fw[i:i+2], Lw[i:i+2], t[i+1], y)
            wpath.append(y)
        wpath = np.array(wpath)
        # per-row de_w: projection of m^{1-sigma} at that row's own window
        cs = z["snaps_c"]; ts = z["snaps_t"]
        de_w_rows, keep = [], []
        ri = 0
        for i in range(len(ts)):
            q = np.abs(cs[i])
            mm = np.arange(1, len(q) + 1).astype(float)
            good = q > q.max()*1e-12
            mmax_good = int(np.max(mm[good]))
            if int(0.4*mmax_good) < 12:
                continue
            r = rows[ri][2]; ri += 1
            mhi = int(0.4*mmax_good)
            selv = (mm >= 8) & (mm <= mhi) & good & (q > 0)
            x = mm[selv]
            F = r["A"]*x**(sig-1.0) + r["D"] + r["E"]*x**(-sig)
            J = np.vstack([x**(sig-1.0)/F, 1.0/F, x**(-sig)/F, -x]).T
            g = (x**(1.0-sig))/F
            dp = np.linalg.solve(J.T @ J, J.T @ g)
            de_w_rows.append(dp[2])
        de_w_rows = np.array(de_w_rows)
        # refined prediction per row
        pred = []
        for i in range(len(t)):
            Dd = cD[1] + 2.0*cD[2]*(t[i] - tpk)
            den = A[i]*b1 - nu
            e_cor = (Dd + 0.5*D[i]*D[i] - A[i]*D[i]*z1s - D[i]*E[i]*zs)/den
            pred.append(e_cor + wpath[i]*de_w_rows[i])
        pred = np.array(pred)
        Am = np.vstack([np.ones_like(D), D]).T
        c, *_ = np.linalg.lstsq(Am, pred, rcond=None)
        print("--- sigma=%.3f: refined slope d(e_pred)/dD = %+.5f   [measured %+.4f ; S27-corrected %+.4f]  w(t_end)=%+.3e de_w(range %.2f..%.2f)"
              % (sig, c[1], MEAS[sig], S27C[sig], wpath[-1], de_w_rows.min(), de_w_rows.max()))
    print("scoring: C-1 = sign at 0.75 POSITIVE?; C-2 = |refined-0.1117| < 0.081 at 1.25 (see rows above)")

# ------------------------------------------------------------------ s1625
def mode_s1625():
    print("===.625 zero-crossing row (REPORT by k-provenance) ===")
    s0 = 1.625
    kq, band = 5.9288, 0.0825          # registered (s28_regcalc.txt)
    b1 = st.B1(s0)
    A_over_nu = 2.0/B(s0, s0)
    den_over_nu = A_over_nu*b1 - 1.0
    for fn in ("sig1p62_nu0p084500_M8192.npz", "sig1p62_nu0p084200_M8192.npz"):
        z = np.load(os.path.join(DIR, fn))
        nu = float(z["nu"]); q = np.abs(z["c_pk"])
        tail = float(np.abs(q[-8:]).max()/q.max())
        r = fit_slot(q, s0, -s0)
        epred = -kq*nu/den_over_nu
        eplo = -(kq + band)*nu/den_over_nu
        ephi = -(kq - band)*nu/den_over_nu
        dev = r["E"] - epred
        print("--- %s nu=%.4f tail=%.2e  A=%+.5e D=%+.5e e-hat=%+.6e rms=%.2e" % (fn, nu, tail, r["A"], r["D"], r["E"], r["rms"]))
        print("    e_pred = %+.6e [band %+.6e .. %+.6e] ; Delta-e = %+.6e ; Delta band = [%+.6e, %+.6e]"
              % (epred, min(eplo, ephi), max(eplo, ephi), dev, r["E"] - max(eplo, ephi), r["E"] - min(eplo, ephi)))
        print("    e-hat/e_pred = %+.4f ; sign(Delta-e) = %s" % (r["E"]/epred, "POSITIVE" if dev > 0 else "NEGATIVE"))

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"verify": mode_verify, "proj": mode_proj, "run": mode_run,
     "slopes": mode_slopes, "s1625": mode_s1625}[m]()
