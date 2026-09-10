#!/usr/bin/env python
# S32 — THE OPENED-TOWER JOINT FIT (.txt; zero rides).
# The S31 per-shape instrument (s30_g2c.fit5, one opened slot) generalized
# to the joint tower {h = m^(s-2), w = m^(1-s), f = m^(1-2s)} (h == w merged
# at sigma = 1.5): 7 parameters (A, D, e, c_h, c_w, c_f, delta), 6 at 1.5.
# Stages (deterministic):
#   check  — B-0b (the code-path obligation): fitN with ONE opened shape must
#            reproduce s30_g2c.fit5 (A, D, E, cg, d, rms) EXACTLY on every
#            ladder snapshot and every single vector (after B-0a: the
#            s31_eleg / s31_hout bit-identical re-runs).
#   tower  — the T-clauses on the ladders (letters per s32_reg.txt) + the
#            H-clauses (the outer-forced h account at AMPLITUDE level) +
#            the report rows (factor band, projection consistency,
#            suppression ratios, k').
#   single — the single-vector report rows (1.5625 / 1.625 / 1.9 + the F1
#            peak vectors): letters by gain / sigma_e only, unscored.
import os, sys, re
import numpy as np
from math import gamma, log
from s25_threeterm import fit_slot
from s30_dclose import FROZEN, KWIN, KWIN_REPORT, shape_vec
from s30_g2c import fit5, slope_lsq
import s28_memint as sm

sys.stdout.reconfigure(encoding="utf-8")
DIR = os.path.dirname(os.path.abspath(__file__))
ITMAX = 200          # registered implementation constant (fit5 uses 60; the tower may need more)
TOL = 1e-14          # fit5's convergence criterion, verbatim

def B(a, b):
    return gamma(a)*gamma(b)/gamma(a + b)

def epred_of(s):
    nu, dev, k, dkreq, Rabs, breq = FROZEN[s]
    Alaw = 2.0*nu/B(s, s)
    den = Alaw*B(s, 1.0 - s) - nu
    return -k*nu*nu/den

def tower_shapes(s):
    return ["h", "w", "f"] if abs(s - 1.5) > 1e-9 else ["w", "f"]

def fitN(q, s, gnames, r0, itmax=ITMAX):
    """joint (A, D, E, c_g[1..k], delta) log-space Gauss-Newton on the standard
    window, warm-started from the fit_slot solution r0 with every c = 0 —
    s30_g2c.fit5 VERBATIM for k = 1 (same window, same Jacobian column order,
    same damping 1e-12*trace/npar, same update, same criterion)."""
    M = len(q); mm = np.arange(1, M + 1).astype(float)
    good = q > q.max()*1e-12
    mhi = int(0.4*np.max(mm[good]))
    sel = (mm >= 8) & (mm <= mhi) & good & (q > 0)
    x = mm[sel]
    y = np.log(q[sel])
    A, D, E, d = r0["A"], r0["D"], r0["E"], r0["d"]
    k = len(gnames)
    cg = [0.0]*k
    G = [shape_vec(nm, s, x, d) for nm in gnames]
    npar = 4 + k
    dp = np.zeros(npar); nit = 0
    for it in range(itmax):
        F = A*x**(s - 1.0) + D + E*x**(-s)
        for j in range(k):
            F = F + cg[j]*G[j]
        if np.any(F <= 0):
            return None
        resid = y - (np.log(F) - d*x)
        J = np.vstack([x**(s - 1.0)/F, 1.0/F, x**(-s)/F] + [G[j]/F for j in range(k)] + [-x]).T
        JtJ = J.T @ J
        try:
            dp = np.linalg.solve(JtJ + 1e-12*np.eye(npar)*np.trace(JtJ)/npar, J.T @ resid)
        except np.linalg.LinAlgError:
            return None
        A += dp[0]; D += dp[1]; E += dp[2]
        for j in range(k):
            cg[j] += dp[3 + j]
        d += dp[3 + k]
        nit = it + 1
        if np.max(np.abs(dp)) < TOL:
            break
    rms = float(np.sqrt(np.mean(resid**2)))
    conv = bool(np.max(np.abs(dp)) < TOL)
    # diagnostics at the solution (post-update parameters)
    F = A*x**(s - 1.0) + D + E*x**(-s)
    for j in range(k):
        F = F + cg[j]*G[j]
    J = np.vstack([x**(s - 1.0)/F, 1.0/F, x**(-s)/F] + [G[j]/F for j in range(k)] + [-x]).T
    JtJ = J.T @ J
    try:
        cov = np.linalg.inv(JtJ)
        sig = rms*np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        sig = np.full(npar, np.inf)
    sv = np.linalg.svd(J, compute_uv=False)
    kraw = float(sv[0]/sv[-1]) if sv[-1] > 0 else float("inf")
    Jn = J/np.linalg.norm(J, axis=0)
    svn = np.linalg.svd(Jn, compute_uv=False)
    knorm = float(svn[0]/svn[-1]) if svn[-1] > 0 else float("inf")
    out = dict(A=A, D=D, E=E, d=d, rms=rms, nit=nit, conv=conv, dpmax=float(np.max(np.abs(dp))),
               sigA=float(sig[0]), sigD=float(sig[1]), sigE=float(sig[2]), sigd=float(sig[3 + k]),
               kraw=kraw, knorm=knorm, n=int(sel.sum()))
    for j, nm in enumerate(gnames):
        out["c_" + nm] = cg[j]
        out["sig_" + nm] = float(sig[3 + j])
    if k == 1:
        out["cg"] = cg[0]
    return out

def hout_column():
    """the S31 B-3 linear-response amplitude column, parsed from the banked
    (bit-identically re-run, B-0a) record s31_hout.txt."""
    txt = open(os.path.join(DIR, "s31_hout.txt"), encoding="utf-8").read()
    blocks = re.split(r"\n--- ", txt)
    col = {}
    for s in (1.25, 1.375, 1.5, 1.75):
        blk = [b for b in blocks if ("sigma=%.3f " % s) in b][0]
        hout = float(re.search(r"h_out\(t_pk\) = ([+-]\d\.\d+e[+-]\d+)", blk).group(1))
        hoo = float(re.search(r"outer-only h = ([+-]\d\.\d+e[+-]\d+)", blk).group(1))
        f28 = float(re.search(r"; f = ([+-]\d\.\d+e[+-]\d+)", blk).group(1))
        w28 = float(re.search(r"; w = ([+-]\d\.\d+e[+-]\d+)", blk).group(1))
        h28 = float(re.search(r"h\(t_pk\) = ([+-]\d\.\d+e[+-]\d+)", blk).group(1))
        col[s] = dict(h=hout, hoo=hoo, f=f28, w=w28, h28=h28)
    return col

def de_constants(sig):
    powers = sm.uniq_powers(sig)
    fn = [p[0] for p in sm.PEAKS if abs(p[1] - sig) < 1e-9][0]
    out, orth, r, mhi = sm.proj_rows(fn, sig, powers)
    de_h = out[min(powers, key=lambda p: abs(p - (sig - 2.0)))][2]
    de_w = out[min(powers, key=lambda p: abs(p - (1.0 - sig)))][2]
    de_f = out[min(powers, key=lambda p: abs(p - (1.0 - 2.0*sig)))][2]
    return de_h, de_w, de_f

LADDERS = [(s, KWIN[s][3], KWIN[s][0], KWIN[s][1], "scored") for s in (1.25, 1.50, 1.75)] + \
          [(1.375, KWIN_REPORT[1.375], None, None, "REPORT")]
REP = [("sig1p56_nu0p092000_M8192.npz", 1.5625, +8.405e-04),
       ("sig1p62_nu0p084500_M8192.npz", 1.625, -5.6551e-04),
       ("sig1p90_nu0p060500_M2048.npz", 1.9, -9.1656e-03),
       ("snap_sig1p25_nu0p136000_M4096.npz", 1.25, +2.1472e-03),
       ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375, +2.6535e-03),
       ("snap_sig1p50_nu0p098200_M8192.npz", 1.50, +1.7778e-03),
       ("snap_sig1p75_nu0p071500_M4096.npz", 1.75, -3.7940e-03)]

# ---------------------------------------------------------------- check (B-0b)
def mode_check():
    print("=== B-0b: the code-path obligation — fitN with ONE opened shape vs s30_g2c.fit5 (A, D, E, cg, d, rms), every ladder snapshot + every single vector ===")
    worst = 0.0
    for s, fn, h4, h2, cls in LADDERS:
        z = np.load(os.path.join(DIR, fn))
        snaps = np.abs(z["snaps_c"]); n = snaps.shape[0]
        base = [fit_slot(snaps[i], s, -s) for i in range(n)]
        shapes = ["h", "w", "f"] if abs(s - 1.5) > 1e-9 else ["w", "f"]
        for nm in shapes:
            wd = 0.0; nfail = 0
            for i in range(n):
                f5 = fit5(snaps[i], s, nm, base[i])
                fN = fitN(snaps[i], s, [nm], base[i], itmax=60)
                if (f5 is None) != (fN is None):
                    nfail += 1; continue
                if f5 is None:
                    continue
                for key in ("A", "D", "E", "cg", "d", "rms"):
                    a, b = f5[key], fN[key]
                    wd = max(wd, abs(a - b)/max(abs(a), abs(b), 1e-300))
            worst = max(worst, wd)
            print("  ladder sigma=%.4f shape %-2s: %d snapshots, worst relative difference over (A, D, E, cg, d, rms) = %.1e ; None-mismatches = %d  -> %s"
                  % (s, nm, n, wd, nfail, "IDENTICAL" if (wd == 0.0 and nfail == 0) else ("within 1e-13 (float-summation-order class)" if wd < 1e-13 and nfail == 0 else "MISMATCH — STOP")))
    for fn, s, dev in REP:
        z = np.load(os.path.join(DIR, fn)); q = np.abs(z["c_pk"]); base = fit_slot(q, s, -s)
        shapes = ["h", "w", "f"] if abs(s - 1.5) > 1e-9 else ["w", "f"]
        wd = 0.0
        for nm in shapes:
            f5 = fit5(q, s, nm, base); fN = fitN(q, s, [nm], base, itmax=60)
            if f5 is None or fN is None:
                print("  vector %s shape %s: None (%s / %s)" % (fn, nm, f5 is None, fN is None)); continue
            for key in ("A", "D", "E", "cg", "d", "rms"):
                a, b = f5[key], fN[key]
                wd = max(wd, abs(a - b)/max(abs(a), abs(b), 1e-300))
        worst = max(worst, wd)
        print("  vector %s sigma=%.5f: worst relative difference = %.1e -> %s" % (fn, s, wd, "IDENTICAL" if wd == 0.0 else ("within 1e-13" if wd < 1e-13 else "MISMATCH — STOP")))
    print(">> B-0b VERDICT: worst relative difference overall = %.1e -> %s" % (worst, "the one-slot code path REPRODUCES fit5 (the tower may consume fitN)" if worst < 1e-13 else "STOP"))

# ---------------------------------------------------------------- tower (T + H clauses)
def letter_T(identifiable, why, devp, dev, sigE):
    if not identifiable:
        return "DEGENERATE (%s)" % why
    bar = max(abs(dev)/5.0, 2.0*sigE)
    return "RESOLVED" if abs(devp) <= bar else "TOWER-ROBUST"

def mode_tower():
    print("===.txt) ===")
    print("(tower = {h = m^(s-2), w = m^(1-s), f = m^(1-2s)} opened JOINTLY, 7 parameters (6 at 1.5: h == w merged); log-space GN warm-started from the 4-slot fit;")
    print(" iz = the D-zero-adjacent snapshot of the TOWER D column; sigma_e' = rms_T sqrt((J^T J)^-1_ee) at iz; kappa = condition numbers of J (column-normalized / raw);")
    print(" dev' = e-hat' - e_pred (k'/k) ; identifiable iff converged on n4, gain_T >= 5%, sigma_e' <= |dev|/2, every resolved slot (|c| > 3 sigma_c) stable to 0.5 over n4, kappa_norm <= 1e12;")
    print(" RESOLVED iff |dev'| <= max(|dev|/5, 2 sigma_e') ; TOWER-ROBUST otherwise ; K-ROBUST iff |k'/k - 1| <= 2% (stencil-stable to 5%))")
    HC = hout_column()
    rows = {}
    for s, fn, h4, h2, cls in LADDERS:
        nu, dev, k, dkreq, Rabs, breq = FROZEN[s]
        epred = epred_of(s)
        ehat_frozen = epred + dev
        gn = tower_shapes(s)
        de_h, de_w, de_f = de_constants(s)
        z = np.load(os.path.join(DIR, fn))
        snaps = np.abs(z["snaps_c"]); ts = np.array(z["snaps_t"], dtype=float); n = snaps.shape[0]
        base = [fit_slot(snaps[i], s, -s) for i in range(n)]
        fits = [fitN(snaps[i], s, gn, base[i]) for i in range(n)]
        r5 = {nm: [fit5(snaps[i], s, nm, base[i]) for i in range(n)] for nm in gn}
        print("--- sigma=%.4f ladder %s [%s] (frozen dev = %+.4e, e-hat_4(frozen) = %+.4e, e_pred = %+.4e, k/nu2 = %.4f; tower %s; de_h/w/f = %+.4f/%+.4f/%+.4f)"
              % (s, fn, cls, dev, ehat_frozen, epred, k, "+".join(gn), de_h, de_w, de_f))
        nnone = sum(1 for f in fits if f is None)
        if nnone:
            print("    %d/%d tower fits returned None (F <= 0 or singular) — snapshots: %s" % (nnone, n, [i for i in range(n) if fits[i] is None]))
        okidx = [i for i in range(n) if fits[i] is not None]
        Dcol = np.array([fits[i]["D"] if fits[i] is not None else np.nan for i in range(n)])
        zc = [i for i in range(n - 1) if fits[i] is not None and fits[i + 1] is not None and np.sign(Dcol[i]) != np.sign(Dcol[i + 1])]
        if not zc:
            print("    no D-zero in the TOWER D column — the ladder is DEGENERATE (no D-zero)")
            rows[s] = dict(letter="DEGENERATE (no D-zero)", kletter="n/a")
            continue
        iz = int(zc[0])
        t0 = float(ts[iz] + (ts[iz + 1] - ts[iz])*Dcol[iz]/(Dcol[iz] - Dcol[iz + 1]))
        if h4 is not None:
            in4 = [i for i in range(n) if abs(ts[i] - t0) <= h4 + 1e-12]
            in2 = [i for i in range(n) if abs(ts[i] - t0) <= h2 + 1e-12]
        else:
            in4, in2 = [], []
        if len(in2) < 2 or len(in4) < 3:
            in2 = [iz, iz + 1]
            in4 = list(range(max(0, iz - 1), min(n, iz + 3)))
        conv4 = all(fits[i] is not None and fits[i]["conv"] for i in in4)
        if not conv4:
            print("    non-convergence inside the n4 window: %s" % [(i, None if fits[i] is None else (fits[i]["nit"], fits[i]["dpmax"])) for i in in4])
        in4 = [i for i in in4 if fits[i] is not None]; in2 = [i for i in in2 if fits[i] is not None]
        k4 = abs(slope_lsq(ts[in4], Dcol[in4])); k2 = abs(slope_lsq(ts[in2], Dcol[in2]))
        kR = (4.0*k2 - k4)/3.0
        stable = abs(k2/k4 - 1.0) <= 0.05
        kp = kR/(nu*nu)
        fz = fits[iz]; fz1 = fits[iz + 1]
        E4 = base[iz]["E"]; ET = fz["E"]; sigE = fz["sigE"]
        rms4 = base[iz]["rms"]; rmsT = fz["rms"]
        gainT = 1.0 - rmsT/rms4
        best5 = min(r5[nm][iz]["rms"] for nm in gn if r5[nm][iz] is not None)
        gainT5 = 1.0 - rmsT/best5
        # slot amplitudes, resolution, stability
        slot = {}
        for nm in gn:
            c = fz["c_" + nm]; sc = fz["sig_" + nm]
            col = np.array([fits[i]["c_" + nm] for i in in4])
            stab = float(np.std(col)/abs(np.mean(col))) if abs(np.mean(col)) > 0 else float("inf")
            resolved = abs(c) > 3.0*sc
            slot[nm] = (c, sc, stab, resolved, float(np.mean(col)))
        unstable = [nm for nm in gn if slot[nm][3] and slot[nm][2] > 0.5]
        why = []
        if not conv4: why.append("non-convergence in n4")
        if gainT < 0.05: why.append("gain %.1f%% < 5%%" % (100*gainT))
        if sigE > abs(dev)/2.0: why.append("sigma_e' %.1e > |dev|/2 = %.1e" % (sigE, abs(dev)/2.0))
        if unstable: why.append("unstable slots %s" % unstable)
        if fz["knorm"] > 1e12: why.append("kappa_norm %.1e > 1e12" % fz["knorm"])
        identifiable = (len(why) == 0)
        devp = ET - epred*(kp/k)
        devp_b = fz1["E"] - epred*(kp/k)
        let = letter_T(identifiable, "; ".join(why), devp, dev, sigE)
        klet = ("K-ROBUST" if abs(kp/k - 1.0) <= 0.02 else "K-NOT-ROBUST") + ("" if stable else " [k' stencil UNSTABLE > 5% — reported]")
        # report rows
        biasT = E4 - ET
        if abs(s - 1.5) < 1e-9:
            proj = -(de_w*slot["w"][0] + de_f*slot["f"][0])
        else:
            proj = -(de_h*slot["h"][0] + de_w*slot["w"][0] + de_f*slot["f"][0])
        print("    tower at iz=%d (t=%.4f, D-zero t0=%.4f; n4=%s n2=%s): converged=%s nit=%d dpmax=%.1e  rms_4=%.2e rms_T=%.2e gain_T=%+.1f%% (vs best single-shape %.2e: %+.1f%%)  kappa_norm=%.2e kappa_raw=%.2e"
              % (iz, ts[iz], t0, in4, in2, fz["conv"], fz["nit"], fz["dpmax"], rms4, rmsT, 100*gainT, best5, 100*gainT5, fz["knorm"], fz["kraw"]))
        print("    e-hat_4=%+.5e  e-hat'=%+.5e (sigma_e'=%.2e)  [echo iz+1: e-hat'=%+.5e]  shift = e-hat' - e-hat_4 = %+.4e (%+.2f |dev|)  e-hat'/e_pred = %+.4f (factor band [1/2, 2]: %s)"
              % (E4, ET, sigE, fz1["E"], ET - E4, (ET - E4)/abs(dev), ET/epred, "inside" if 0.5 <= ET/epred <= 2.0 else "OUTSIDE"))
        print("    k'/nu2 = %.4f (n4 %.4f / n2 %.4f, %s) ; k'/k - 1 = %+.4f  -> %s" % (kp, k4/(nu*nu), k2/(nu*nu), "stable" if stable else "UNSTABLE", kp/k - 1.0, klet))
        print("    dev' = e-hat' - e_pred k'/k = %+.4e  [echo iz+1: %+.4e]  (dev %+.4e ; |dev|/5 = %.1e ; 2 sigma_e' = %.1e ; bar = %.1e)  -> %s"
              % (devp, devp_b, dev, abs(dev)/5.0, 2*sigE, max(abs(dev)/5.0, 2*sigE), let))
        for nm in gn:
            c, sc, stab, resolved, cm = slot[nm]
            print("    slot %-2s: c' = %+.4e (sigma_c = %.1e, %s ; n4 mean %+.4e, std/|mean| = %.2f)" % (nm, c, sc, "RESOLVED" if resolved else "unresolved", cm, stab))
        print("    H-3 projection consistency: bias_T = e-hat_4 - e-hat' = %+.4e vs -sum(de_s c_s') = %+.4e (ratio %.3f)" % (biasT, proj, biasT/proj if proj != 0 else float("nan")))
        # H-clauses: the outer-forced account at amplitude level
        hc = HC[s]
        if abs(s - 1.5) < 1e-9:
            lin_h = hc["h"] + hc["w"]; c_h = slot["w"][0]; res_h = slot["w"][3]; lab = "merged h+w"
        else:
            lin_h = hc["h"]; c_h = slot["h"][0]; res_h = slot["h"][3]; lab = "h"
        sgn_ok = (np.sign(c_h) == np.sign(lin_h))
        ratio = c_h/lin_h
        print("    H (%s): tower c' = %+.4e vs linear-response h_out(t_pk) = %+.4e -> sign %s ; ratio c'/h_out = %+.4f (suppression factor s_h) ; amplitude %s"
              % (lab, c_h, lin_h, "MATCH" if sgn_ok else "MISS", ratio, "resolved" if res_h else "UNRESOLVED (|c| <= 3 sigma_c)"))
        if abs(s - 1.5) > 1e-9:
            print("    H-4: s_w = c_w'/w28 = %+.4f (c_w' %+.3e vs w28 %+.3e) ; s_f = c_f'/f28 = %s (c_f' %+.3e vs f28 %+.3e) ; outer-only h = %+.3e ; S28 self-forced h28 = %+.3e"
                  % (slot["w"][0]/hc["w"], slot["w"][0], hc["w"], ("%+.4f" % (slot["f"][0]/hc["f"])) if hc["f"] != 0 else "n/a (f-forcing 0)", slot["f"][0], hc["f"], hc["hoo"], hc["h28"]))
        else:
            print("    H-4: s_f = %s (c_f' %+.3e vs f28 %+.3e — the f-forcing is exactly 0 at 1.5) ; outer-only h = %+.3e ; S28 self-forced h28 = %+.3e"
                  % ("n/a", slot["f"][0], hc["f"], hc["hoo"], hc["h28"]))
        rows[s] = dict(letter=let, kletter=klet, identifiable=identifiable, devp=devp, sigE=sigE, gain=gainT,
                       c_h=c_h, lin_h=lin_h, sgn_ok=bool(sgn_ok), ratio=ratio, res_h=res_h, kp=kp)
    # ---- clause bookkeeping
    print()
    print("=== CLAUSE BOOKKEEPING (s32_reg.txt) ===")
    def L(s): return rows.get(s, {}).get("letter", "n/a")
    l175 = L(1.75); l125 = L(1.25); l150 = L(1.5)
    print("  T-1 letters: 1.25 = %s ; 1.5 = %s ; 1.75 (MANDATORY) = %s ; 1.375 (REPORT) = %s" % (l125, l150, l175, L(1.375)))
    nres = sum(1 for s in (1.25, 1.5) if L(s) == "RESOLVED")
    nrob = sum(1 for s in (1.25, 1.5) if L(s) == "TOWER-ROBUST")
    if l175 == "RESOLVED" and nres >= 1:
        head = "RESOLVED-WITHIN-INSTRUMENT (the residual program closes as resolved within the tower instrument)"
    elif l175 == "TOWER-ROBUST" and nrob >= 1:
        head = "TOWER-ROBUST-RESIDUAL DEFINED (dev'(sigma) is the new residual column; no derived candidate exists for it today)"
    elif l175.startswith("DEGENERATE"):
        head = "DEGENERATE at the mandatory row — the tower is not identifiable on this window: SCOPE LIMIT recorded, STOP"
    else:
        head = "MIXED — per-row letters carry the result (no substituted branch-firing): 1.75 %s ; 1.25 %s ; 1.5 %s" % (l175, l125, l150)
    print("  T-1 PROGRAM VERDICT: %s" % head)
    kl = [rows.get(s, {}).get("kletter", "n/a") for s in (1.25, 1.5, 1.75)]
    print("  T-2 k-leg: %s -> %s" % (kl, "K-ROBUST at every scored ladder" if all(x.startswith("K-ROBUST") for x in kl) else "the k-leg is NOT tower-robust at some row — e_pred' = e_pred k'/k used in dev' as registered"))
    r = rows.get(1.75, {})
    if r.get("identifiable") and r.get("res_h"):
        signs = [rows[s]["sgn_ok"] for s in (1.25, 1.375, 1.5, 1.75) if s in rows and "sgn_ok" in rows[s]]
        nhit = sum(1 for x in signs if x)
        prices = r["sgn_ok"] and (0.2 <= abs(r["ratio"]) <= 5.0)
        print("  H-1 (amplitude sign at the MANDATORY 1.75 row): %s ; signs over the four rows: %d/4" % ("MATCH" if r["sgn_ok"] else "MISS", nhit))
        print("  H-2 (amplitude ratio at 1.75): c_h'/h_out = %+.4f -> %s" % (r["ratio"], "PRICES-AT-AMPLITUDE (ratio in [1/5, 5])" if prices else ("sign-pass, NOT pricing: the suppression factor s_h(1.75) = %+.4f is MEASURED" % r["ratio"] if r["sgn_ok"] else "FAILS the mandatory sign — the outer-forced account is REFUTED at amplitude level")))
    else:
        print("  H-1/H-2: NOT SCOREABLE — the tower is %s at 1.75 (the H-clauses are conditional on an identifiable tower with a resolved h-amplitude)" % ("not identifiable" if not r.get("identifiable") else "identifiable but c_h' is unresolved (|c| <= 3 sigma_c)"))
    print("  the sigma = 1.59375 ride (): %s" % ("CONDITION MET ONLY IF the single-vector report rows at 1.5625/1.625 carry opposite-signed identifiable dev' — see stage single" if head.startswith("TOWER-ROBUST") else "INACTIVE (no tower-robust residual defined at the mandatory row)"))

# ---------------------------------------------------------------- single (report rows)
def mode_single():
    print("=== single-vector REPORT rows: the tower on the peak vectors (no ladder; letters by gain / sigma_e' only, unscored) ===")
    for fn, s, dev in REP:
        z = np.load(os.path.join(DIR, fn))
        q = np.abs(z["c_pk"])
        base = fit_slot(q, s, -s)
        gn = tower_shapes(s)
        fN = fitN(q, s, gn, base)
        epred = epred_of(s) if s in FROZEN else base["E"] - dev
        print("--- %s sigma=%.5f (dev = %+.4e ; e_pred = %+.4e ; e-hat_4 = %+.4e ; rms_4 = %.2e)" % (fn, s, dev, epred, base["E"], base["rms"]))
        if fN is None:
            print("    tower fit returned None"); continue
        gain = 1.0 - fN["rms"]/base["rms"]
        r5 = [fit5(q, s, nm, base) for nm in gn]
        best5 = min(f["rms"] for f in r5 if f is not None)
        devp = fN["E"] - epred
        ident = fN["conv"] and gain >= 0.05 and fN["sigE"] <= abs(dev)/2.0 and fN["knorm"] <= 1e12
        cls = ("IDENTIFIABLE-class: " + ("would-RESOLVE" if abs(devp) <= max(abs(dev)/5.0, 2*fN["sigE"]) else "would-be-TOWER-ROBUST")) if ident else "DEGENERATE-class"
        print("    tower: converged=%s nit=%d  e-hat'=%+.5e (sigma_e'=%.2e)  shift=%+.4e (%+.2f |dev|)  rms'=%.2e gain=%+.1f%% (vs best single-shape %.2e: %+.1f%%)  kappa_norm=%.2e  e-hat'/e_pred=%+.4f  dev'=%+.4e (bar %.1e) -> %s"
              % (fN["conv"], fN["nit"], fN["E"], fN["sigE"], fN["E"] - base["E"], (fN["E"] - base["E"])/abs(dev), fN["rms"], 100*gain, best5, 100*(1 - fN["rms"]/best5), fN["knorm"], fN["E"]/epred, devp, max(abs(dev)/5.0, 2*fN["sigE"]), cls))
        print("    slots: %s" % " ; ".join("c_%s = %+.4e (sigma %.1e, %s)" % (nm, fN["c_" + nm], fN["sig_" + nm], "resolved" if abs(fN["c_" + nm]) > 3*fN["sig_" + nm] else "unresolved") for nm in gn))

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"check": mode_check, "tower": mode_tower, "single": mode_single}[m]()
