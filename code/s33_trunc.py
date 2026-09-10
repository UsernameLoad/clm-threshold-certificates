#!/usr/bin/env python
# S33 — THE TRUNCATION-ORDER CONVERGENCE TEST (.txt; zero rides).
# The S32 labeled diagnostic's damped solver (s32_tdiag.lm_fit: scipy
# least_squares 'lm', the s25_threeterm.fit_slot machinery, xtol/ftol/gtol
# 1e-15) promoted to the REGISTERED instrument and generalized from the tower
# {h, w, f} to the order ladder 4 -> 5 (per shape) -> 7 -> 9 -> 11 (the §0 A-1
# lattice members: order 9 adds m^-1 and m^(2-2s), order 11 adds m^(s-3) and
# m^(-s-1); collisions merged as in s33_regcalc.txt; at sigma = 1.5 the order-11
# member m^(s-3) IS the e-slot and is merged into e).
# Stages (deterministic):
#   check  — B-0b: (i) fitLM at order 4 from the fit_AD warm start reproduces
#            s25_threeterm.fit_slot EXACTLY (A, D, E, d, rms, nfev) on every
#            ladder snapshot + single vector; (ii) fitLM at order 7 from the
#            4-slot warm start reproduces s32_tdiag.lm_fit EXACTLY (p, rms,
#            nfev, status) at every ladder snapshot + single vector.
#   probe  — the PRE-registration convergence probe on REPORT rows only (the
#            1.375 ladder at the S32 iz = 11 snapshot; the sig1p56 vector):
#            orders 7 / 9 / 11 — prints ONLY convergence diagnostics (status,
#            nfev, restart step, basin agreement, rms ratio, kappa_norm);
#            NO parameters, NO e-hat.
#   ladder — the T/H clauses on the four ladders (letters per s33_reg.txt).
#   single — the S-rows on the seven single vectors (report).
import os, sys
import numpy as np
from math import gamma
from scipy.optimize import least_squares
from s25_threeterm import fit_slot
import s17_kappa as sk
from s30_dclose import FROZEN, KWIN, KWIN_REPORT
from s30_g2c import fit5, slope_lsq
from s32_tower import hout_column, LADDERS, REP, epred_of
import s28_memint as sm

sys.stdout.reconfigure(encoding="utf-8")
DIR = os.path.dirname(os.path.abspath(__file__))
MAXNFEV = 40000          # the fit_slot / s32_tdiag constant
TOLS = dict(xtol=1e-15, ftol=1e-15, gtol=1e-15)
# the "converged" sub-clause thresholds (filled from the probe at registration — s33_reg.txt Amendment 1)
RESTART_MAX = 1.0        # max |dp_i|/sigma_i on a restart from the solution (Amendment 1: the probe's class — sub-sigma at orders 7/9, multi-sigma at 11)
BASIN_MAX = 1.0          # max |p_4slot-seeded - p_prev-seeded|/sigma_i (Amendment 1)
BASIN_RMS = 1e-3         # |rms_4slot-seeded / rms_prev-seeded - 1| (Amendment 1; the probe: <= 2e-4 at order 9, 8-35% at order 11)

# ---------------------------------------------------------------- the order ladder
def power_of(nm, s):
    if nm == "h":  return s - 2.0
    if nm == "w":  return 1.0 - s
    if nm == "f":  return 1.0 - 2.0*s
    if nm == "m1": return -1.0
    if nm == "w2": return 2.0 - 2.0*s
    if nm == "h2": return s - 3.0
    if nm == "eh": return -s - 1.0
    raise KeyError(nm)

def order_shapes(s, order):
    """the named shapes of the given order at sigma s, with the lattice collisions
    merged exactly as s33_regcalc.txt prints them (a merged member is DROPPED and
    listed in `merged`; a member coinciding with the e-slot m^-s is merged into e)."""
    base = [s - 1.0, 0.0, -s]
    names = {7: ["h", "w", "f"], 9: ["h", "w", "f", "m1", "w2"], 11: ["h", "w", "f", "m1", "w2", "h2", "eh"]}[order]
    kept, pw, merged = [], [], []
    for nm in names:
        p = power_of(nm, s)
        if any(abs(p - q) < 1e-12 for q in base):
            merged.append("%s->%s" % (nm, "e" if abs(p + s) < 1e-12 else ("A" if abs(p - (s - 1.0)) < 1e-12 else "D")))
            continue
        hit = [k for k, q in enumerate(pw) if abs(p - q) < 1e-12]
        if hit:
            merged.append("%s->%s" % (nm, kept[hit[0]]))
            continue
        kept.append(nm); pw.append(p)
    return kept, pw, merged

# ---------------------------------------------------------------- the damped solver (s32_tdiag.lm_fit generalized)
def window(q):
    M = len(q); mm = np.arange(1, M + 1).astype(float)
    good = q > q.max()*1e-12
    mhi = int(0.4*np.max(mm[good]))
    sel = (mm >= 8) & (mm <= mhi) & good & (q > 0)
    return mm[sel], np.log(q[sel]), mhi

def fitLM(q, s, powers, p0, restart=True):
    """joint (A, D, E, c_1..c_k, delta) damped least squares (scipy 'lm', the
    fit_slot machinery) from p0; residual VERBATIM s32_tdiag.lm_fit (G_j = x^p_j)."""
    x, y, mhi = window(q)
    k = len(powers)
    G = [x**p for p in powers]
    def res(p):
        A, D, E = p[0], p[1], p[2]; cg = p[3:3 + k]; d = p[3 + k]
        f = A*x**(s - 1.0) + D + E*x**(-s)
        for j in range(k):
            f = f + cg[j]*G[j]
        f = np.where(f > 1e-300, f, 1e-300)
        return np.log(f) - d*x - y
    out = least_squares(res, list(p0), method="lm", max_nfev=MAXNFEV, **TOLS)
    p = out.x
    rms = float(np.sqrt(np.mean(out.fun**2)))
    # analytic Jacobian at the solution -> formal sigmas, condition numbers (the S32 arithmetic)
    A, D, E = p[0], p[1], p[2]; cg = p[3:3 + k]; d = p[3 + k]
    F = A*x**(s - 1.0) + D + E*x**(-s)
    for j in range(k):
        F = F + cg[j]*G[j]
    npar = 4 + k
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
    r = dict(p=p, A=A, D=D, E=E, d=d, cg=np.array(cg), rms=rms, nfev=int(out.nfev), status=int(out.status),
             msg=out.message, sig=sig, sigE=float(sig[2]), kraw=kraw, knorm=knorm, n=len(x), powers=list(powers))
    if restart:
        out2 = least_squares(res, list(p), method="lm", max_nfev=MAXNFEV, **TOLS)
        dp = out2.x - p
        with np.errstate(divide="ignore", invalid="ignore"):
            r["restart_sig"] = float(np.max(np.abs(dp)/np.where(sig > 0, sig, np.inf)))
            r["restart_rel"] = float(np.max(np.abs(dp)/np.where(np.abs(p) > 0, np.abs(p), np.inf)))
        r["restart_nfev"] = int(out2.nfev)
    return r

def p0_from_slot(r0, k):
    return [r0["A"], r0["D"], r0["E"]] + [0.0]*k + [r0["d"]]

def fit_order(q, s, order, r0, prev=None):
    """the registered fit at `order`: the 4-slot-seeded solve (the instrument) +
    the previous-order-seeded solve (the basin check) — both from the SAME window."""
    names, pw, merged = order_shapes(s, order)
    k = len(pw)
    f = fitLM(q, s, pw, p0_from_slot(r0, k))
    f["names"] = names; f["merged"] = merged
    if prev is None and order == 7:
        # the order-7 basin seed = the LM order-5 fit with the tower's first shape (Amendment 1: non-vacuous at every order)
        prev = fitLM(q, s, [pw[0]], p0_from_slot(r0, 1), restart=False)
        prev["cg"] = np.array(prev["cg"]); prev["powers"] = [pw[0]]
    if prev is not None:
        kp = len(prev["powers"])
        p0b = list(prev["p"][:3]) + list(prev["cg"]) + [0.0]*(k - kp) + [prev["d"]]
        fb = fitLM(q, s, pw, p0b, restart=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["basin_sig"] = float(np.max(np.abs(fb["p"] - f["p"])/np.where(f["sig"] > 0, f["sig"], np.inf)))
        f["basin_rms"] = float(abs(fb["rms"]/f["rms"] - 1.0))
        f["basin_E"] = float(fb["E"])
    else:
        f["basin_sig"] = 0.0; f["basin_rms"] = 0.0; f["basin_E"] = float(f["E"])
    f["conv"] = bool(f["status"] in (1, 2, 3, 4) and f["nfev"] < MAXNFEV and f["restart_sig"] <= RESTART_MAX
                     and f["basin_sig"] <= BASIN_MAX and f["basin_rms"] <= BASIN_RMS)
    return f

def de_table(s, powers):
    """the projection responses de_p (s28_memint.proj_rows on the PEAKS vector at sigma s)."""
    fn = [p[0] for p in sm.PEAKS if abs(p[1] - s) < 1e-9][0]
    out, orth, r, mhi = sm.proj_rows(fn, s, powers)
    return {p: out[p][2] for p in powers}

# ---------------------------------------------------------------- check (B-0b)
def mode_check():
    from s32_tdiag import lm_fit
    from s32_tower import tower_shapes
    print("=== B-0b: the code-path obligations of the S33 instrument ===")
    print("(i) fitLM at ORDER 4 (k = 0) from the s17_kappa.fit_AD warm start [A, D, 0, d] vs s25_threeterm.fit_slot (A, D, E, d, rms, nfev): EXACT or STOP")
    print("(ii) fitLM at ORDER 7 (the tower, 4-slot warm start) vs s32_tdiag.lm_fit (p, rms, nfev, status): EXACT or STOP")
    worst4 = 0.0; worst7 = 0.0; nfevmis = 0
    def cmp4(q, s):
        two = sk.fit_AD(q, s)
        ref = fit_slot(q, s, -s)
        f = fitLM(q, s, [], [two["A"], two["D"], 0.0, two["d"]], restart=False)
        wd = 0.0
        for key in ("A", "D", "E", "d", "rms"):
            a, b = ref[key], f[key]
            wd = max(wd, abs(a - b)/max(abs(a), abs(b), 1e-300))
        return wd
    def cmp7(q, s, r0):
        gn = tower_shapes(s)
        ref = lm_fit(q, s, gn, r0)
        f = fitLM(q, s, [power_of(nm, s) for nm in gn], p0_from_slot(r0, len(gn)), restart=False)
        wd = float(np.max(np.abs(ref["p"] - f["p"])/np.maximum(np.abs(ref["p"]), 1e-300)))
        wd = max(wd, abs(ref["rms"] - f["rms"])/max(ref["rms"], 1e-300))
        mis = int(ref["nfev"] != f["nfev"] or ref["status"] != f["status"])
        return wd, mis
    for s, fn, h4, h2, cls in LADDERS:
        z = np.load(os.path.join(DIR, fn))
        snaps = np.abs(z["snaps_c"]); n = snaps.shape[0]
        w4 = 0.0; w7 = 0.0; mis = 0
        for i in range(n):
            w4 = max(w4, cmp4(snaps[i], s))
            r0 = fit_slot(snaps[i], s, -s)
            d7, m7 = cmp7(snaps[i], s, r0); w7 = max(w7, d7); mis += m7
        worst4 = max(worst4, w4); worst7 = max(worst7, w7); nfevmis += mis
        print("  ladder sigma=%.4f: %d snapshots — order-4 vs fit_slot worst rel diff = %.1e ; order-7 vs s32_tdiag.lm_fit worst rel diff = %.1e, nfev/status mismatches = %d" % (s, n, w4, w7, mis))
    for fn, s, dev in REP:
        z = np.load(os.path.join(DIR, fn)); q = np.abs(z["c_pk"])
        w4 = cmp4(q, s); r0 = fit_slot(q, s, -s); w7, m7 = cmp7(q, s, r0)
        worst4 = max(worst4, w4); worst7 = max(worst7, w7); nfevmis += m7
        print("  vector %s sigma=%.5f: order-4 worst rel diff = %.1e ; order-7 worst rel diff = %.1e, nfev/status mismatches = %d" % (fn, s, w4, w7, m7))
    print(">> B-0b VERDICT: order-4 worst = %.1e (%s) ; order-7 worst = %.1e, mismatches %d (%s)"
          % (worst4, "IDENTICAL" if worst4 == 0.0 else "NOT identical — STOP", worst7, nfevmis,
             "IDENTICAL" if (worst7 == 0.0 and nfevmis == 0) else "NOT identical — STOP"))

# ---------------------------------------------------------------- probe (pre-registration; REPORT rows; convergence diagnostics only)
def probe_one(label, q, s, r0):
    prev = None
    for order in (7, 9, 11):
        f = fit_order(q, s, order, r0, prev)
        names, pw, merged = f["names"], f["powers"], f["merged"]
        print("    %s order %2d (%d params; shapes %s%s): status=%d nfev=%d rms/rms_4=%.4f  restart: max|dp|/sigma=%.2e max|dp/p|=%.2e (nfev %d)  basin(prev-seeded): max|dp|/sigma=%.2e |drms|=%.2e  kappa_norm=%.2e kappa_raw=%.2e -> %s"
              % (label, order, 4 + len(pw), "+".join(names), (" ; merged " + ",".join(merged)) if merged else "", f["status"], f["nfev"], f["rms"]/r0["rms"],
                 f["restart_sig"], f["restart_rel"], f["restart_nfev"], f["basin_sig"], f["basin_rms"], f["knorm"], f["kraw"],
                 "converged (registered sub-clause)" if f["conv"] else "NOT-CONVERGED"))
        prev = f

def mode_probe():
    print("=== the PRE-registration convergence probe (REPORT rows only; convergence diagnostics only — no parameters, no e-hat) ===")
    print("(solver: scipy least_squares 'lm', xtol/ftol/gtol 1e-15, max_nfev 40000, from the 4-slot fit_slot solution with every c = 0; restart = a second solve from the solution; basin = the previous-order-seeded solve)")
    z = np.load(os.path.join(DIR, KWIN_REPORT[1.375]))
    snaps = np.abs(z["snaps_c"]); ts = np.array(z["snaps_t"], dtype=float)
    for i in (10, 11, 12):
        r0 = fit_slot(snaps[i], 1.375, -1.375)
        print("--- the 1.375 REPORT ladder snapshot %d (t=%.4f)" % (i, ts[i]))
        probe_one("1.375 snap %d" % i, snaps[i], 1.375, r0)
    for fn, s in (("sig1p56_nu0p092000_M8192.npz", 1.5625),):
        z = np.load(os.path.join(DIR, fn)); q = np.abs(z["c_pk"]); r0 = fit_slot(q, s, -s)
        print("--- the %s REPORT vector sigma=%.5f" % (fn, s))
        probe_one("%.4f vector" % s, q, s, r0)

# ---------------------------------------------------------------- ladder (the T/H clauses)
def analyze_ladder(s, fn, h4, h2, cls, HC):
    nu, dev, k, dkreq, Rabs, breq = FROZEN[s]
    epred = epred_of(s)
    z = np.load(os.path.join(DIR, fn))
    snaps = np.abs(z["snaps_c"]); ts = np.array(z["snaps_t"], dtype=float); n = snaps.shape[0]
    base = [fit_slot(snaps[i], s, -s) for i in range(n)]
    print("--- sigma=%.4f ladder %s [%s] (frozen dev = %+.4e, e_pred = %+.4e, k/nu2 = %.4f, |dev|/5 = %.2e, |dev|/2 = %.2e)"
          % (s, fn, cls, dev, epred, k, abs(dev)/5, abs(dev)/2))
    # order 5 report rows: fit5 (GN) vs LM order 5, per shape, at the 4-slot D-zero-adjacent snapshot
    D4 = np.array([b["D"] for b in base])
    iz4 = int(np.where(np.diff(np.sign(D4)) != 0)[0][0])
    from s32_tower import tower_shapes
    for nm in tower_shapes(s):
        g5 = fit5(snaps[iz4], s, nm, base[iz4])
        l5 = fitLM(snaps[iz4], s, [power_of(nm, s)], p0_from_slot(base[iz4], 1), restart=False)
        dsig = abs(g5["E"] - l5["E"])/l5["sigE"] if l5["sigE"] > 0 else float("inf")
        print("    order 5 (%s) at the 4-slot iz=%d: fit5 GN e-hat=%+.5e vs LM e-hat=%+.5e (|diff| = %.2e sigma_e; rms %.3e vs %.3e; LM nfev %d status %d) — same-optimum check %s"
              % (nm, iz4, g5["E"], l5["E"], dsig, g5["rms"], l5["rms"], l5["nfev"], l5["status"], "ok" if dsig < 1e-2 else "DIFFERS"))
    results = {}
    prevfits = None
    for order in (7, 9, 11):
        fits = []
        for i in range(n):
            fits.append(fit_order(snaps[i], s, order, base[i], None if prevfits is None else prevfits[i]))
        names, pw, merged = fits[0]["names"], fits[0]["powers"], fits[0]["merged"]
        Dcol = np.array([f["D"] for f in fits])
        zc = [i for i in range(n - 1) if np.sign(Dcol[i]) != np.sign(Dcol[i + 1])]
        if not zc:
            print("    order %d: no D-zero in the D column — DEGENERATE (no D-zero)" % order)
            results[order] = dict(ident=False, why="no D-zero"); prevfits = fits; continue
        iz = int(zc[0])
        t0 = float(ts[iz] + (ts[iz + 1] - ts[iz])*Dcol[iz]/(Dcol[iz] - Dcol[iz + 1]))
        if h4 is not None:
            in4 = [i for i in range(n) if abs(ts[i] - t0) <= h4 + 1e-12]
            in2 = [i for i in range(n) if abs(ts[i] - t0) <= h2 + 1e-12]
        else:
            in4, in2 = [], []
        if len(in2) < 2 or len(in4) < 3:
            in2 = [iz, iz + 1]; in4 = list(range(max(0, iz - 1), min(n, iz + 3)))
        conv4 = all(fits[i]["conv"] for i in in4)
        k4 = abs(slope_lsq(ts[in4], Dcol[in4])); k2 = abs(slope_lsq(ts[in2], Dcol[in2]))
        kR = (4.0*k2 - k4)/3.0; stable = abs(k2/k4 - 1.0) <= 0.05; kp = kR/(nu*nu)
        fz = fits[iz]; fz1 = fits[iz + 1]
        E4 = base[iz]["E"]; ET = fz["E"]; sigE = fz["sigE"]
        rms4 = base[iz]["rms"]; rmsT = fz["rms"]; gainT = 1.0 - rmsT/rms4
        rms_prev = prevfits[iz]["rms"] if prevfits is not None else rms4
        Erange = float(np.max([fits[i]["E"] for i in in4]) - np.min([fits[i]["E"] for i in in4]))
        slot = {}
        for j, nm in enumerate(names):
            c = fz["cg"][j]; sc = fz["sig"][3 + j]
            col = np.array([fits[i]["cg"][j] for i in in4])
            stab = float(np.std(col)/abs(np.mean(col))) if abs(np.mean(col)) > 0 else float("inf")
            slot[nm] = (c, sc, stab, abs(c) > 3.0*sc, float(np.mean(col)))
        unstable = [nm for nm in names if slot[nm][3] and slot[nm][2] > 0.5]
        why = []
        if not conv4: why.append("NOT-CONVERGED in n4 (%s)" % [(i, fits[i]["status"], fits[i]["nfev"], "%.1e" % fits[i]["restart_sig"], "%.1e" % fits[i]["basin_sig"], "%.1e" % fits[i]["basin_rms"]) for i in in4 if not fits[i]["conv"]])
        if rmsT > rms_prev*(1 + 1e-9): why.append("rms %.3e > previous order's %.3e (nested-model sanity: basin failure)" % (rmsT, rms_prev))
        if gainT < 0.05: why.append("gain %.1f%% < 5%%" % (100*gainT))
        if sigE > abs(dev)/2.0: why.append("sigma_e' %.1e > |dev|/2 = %.1e" % (sigE, abs(dev)/2.0))
        if unstable: why.append("unstable slots %s" % unstable)
        if fz["knorm"] > 1e12: why.append("kappa_norm %.1e > 1e12" % fz["knorm"])
        ident = (len(why) == 0)
        klet = ("K-ROBUST" if abs(kp/k - 1.0) <= 0.02 else "K-NOT-ROBUST") + ("" if stable else " [k' stencil UNSTABLE > 5%]")
        de = de_table(s, pw)
        proj = sum(de[p]*fz["cg"][j] for j, p in enumerate(pw))
        biasT = E4 - ET
        print("    ORDER %d (%d params; %s%s) at iz=%d (t=%.4f, D-zero t0=%.4f; n4=%s n2=%s): converged(n4)=%s [iz: status %d nfev %d restart %.1e basin %.1e/%.1e]  rms_4=%.2e rms_prev=%.2e rms=%.2e (gain vs 4-slot %+.1f%%, vs previous order %+.1f%%)  kappa_norm=%.2e kappa_raw=%.2e"
              % (order, 4 + len(pw), "+".join(names), (" ; merged " + ",".join(merged)) if merged else "", iz, ts[iz], t0, in4, in2, conv4, fz["status"], fz["nfev"], fz["restart_sig"], fz["basin_sig"], fz["basin_rms"],
                 rms4, rms_prev, rmsT, 100*gainT, 100*(1 - rmsT/rms_prev), fz["knorm"], fz["kraw"]))
        print("      e-hat_4=%+.5e  e-hat'=%+.5e (sigma_e'=%.2e; n4 range %.2e; echo iz+1 %+.5e; prev-seeded %+.5e)  shift vs 4-slot = %+.4e (%+.2f |dev|)  e-hat'/e_pred = %+.4f  k'/nu2 = %.4f (n4 %.4f / n2 %.4f) k'/k-1 = %+.4f -> %s"
              % (E4, ET, sigE, Erange, fz1["E"], fz["basin_E"], ET - E4, (ET - E4)/abs(dev), ET/epred, kp, k4/(nu*nu), k2/(nu*nu), kp/k - 1.0, klet))
        for nm in names:
            c, sc, stab, res, cm = slot[nm]
            print("      slot %-3s (m^%+.4f): c' = %+.4e (sigma_c = %.1e, %s ; n4 mean %+.4e, std/|mean| = %.2f)" % (nm, power_of(nm, s), c, sc, "RESOLVED" if res else "unresolved", cm, stab))
        print("      H-3 projection identity: e-hat_4 - e-hat' = %+.4e vs sum_p de_p c_p' = %+.4e (ratio %.3f)" % (biasT, proj, biasT/proj if proj != 0 else float("nan")))
        print("      identifiable = %s%s" % (ident, "" if ident else ("  [" + "; ".join(why) + "]")))
        results[order] = dict(ident=ident, why="; ".join(why), E=ET, sigE=sigE, Erange=Erange, kp=kp, klet=klet, slot=slot, names=names, iz=iz, rms=rmsT, gain=gainT, E1=fz1["E"])
        prevfits = fits
    # ---- the letters (s33_reg.txt T-1)
    idents = [o for o in (7, 9, 11) if results.get(o, {}).get("ident")]
    if len(idents) == 0:
        letter = "DEGENERATE (no identifiable order >= 7)"; plast = None
    elif len(idents) == 1:
        letter = "DEGENERATE (single identifiable order %d)" % idents[0]; plast = idents[0]
    else:
        pp, pl = idents[-2], idents[-1]
        bar = max(abs(dev)/5.0, 2.0*max(results[pp]["sigE"], results[pl]["sigE"]), 0.5*results[pl]["Erange"])
        move = results[pl]["E"] - results[pp]["E"]
        letter = ("CONVERGENT" if abs(move) <= bar else "WANDERING") + " (orders %d -> %d: e-hat' moves %+.4e vs bar %.2e = max(|dev|/5 %.2e, 2 max sigma_e' %.2e, n4 half-range %.2e))" % (pp, pl, move, bar, abs(dev)/5, 2*max(results[pp]["sigE"], results[pl]["sigE"]), 0.5*results[pl]["Erange"])
        plast = pl
    print("    >> T-1 LETTER at sigma=%.4f: %s" % (s, letter))
    row = dict(letter=letter.split(" ")[0], detail=letter, plast=plast, results=results)
    if row["letter"] == "CONVERGENT":
        r = results[plast]
        devp = r["E"] - epred*(r["kp"]/k)
        bar2 = max(abs(dev)/5.0, 2.0*r["sigE"])
        row["score"] = "RESOLVED" if abs(devp) <= bar2 else "TOWER-ROBUST"
        row["devp"] = devp; row["kp"] = r["kp"]; row["klet"] = r["klet"]
        print("    >> identity score at the converged order %d: dev' = e-hat' - e_pred k'/k = %+.4e (bar %.2e) -> %s ; k-leg %s ; e-hat'/e_pred = %+.4f"
              % (plast, devp, bar2, row["score"], r["klet"], r["E"]/epred))
        hc = HC.get(s)
        if hc is not None:
            if abs(s - 1.5) < 1e-9:
                lin_h = hc["h"] + hc["w"]; nm = "w" if "w" in r["names"] else "h"; lab = "merged h+w"
            else:
                lin_h = hc["h"]; nm = "h"; lab = "h"
            c_h, sc, stab, res_h, cm = r["slot"][nm]
            row["c_h"] = c_h; row["lin_h"] = lin_h; row["res_h"] = res_h; row["sgn_ok"] = bool(np.sign(c_h) == np.sign(lin_h)); row["ratio"] = c_h/lin_h
            print("    >> H (%s) at order %d: tower c' = %+.4e vs h_out = %+.4e -> sign %s ; ratio %+.4f ; amplitude %s"
                  % (lab, plast, c_h, lin_h, "MATCH" if row["sgn_ok"] else "MISS", row["ratio"], "resolved" if res_h else "UNRESOLVED"))
    return row

def mode_ladder():
    print("===.txt) ===")
    print("(orders 7 -> 9 -> 11 under the registered damped solver from the 4-slot warm start; iz = the D-zero-adjacent snapshot of THAT order's D column;")
    print(" identifiable iff converged on n4 (the registered sub-clause: status in {1,2,3,4}, nfev < 40000, restart <= %.0e sigma, basin <= %.0e sigma & |drms| <= %.0e), rms <= previous order's, gain >= 5%%, sigma_e' <= |dev|/2, resolved slots stable to 0.5, kappa_norm <= 1e12;" % (RESTART_MAX, BASIN_MAX, BASIN_RMS))
    print(" CONVERGENT iff >= 2 identifiable orders and |e-hat'(p_last) - e-hat'(p_prev)| <= max(|dev|/5, 2 max sigma_e', n4 half-range) ; WANDERING otherwise ; DEGENERATE iff < 2 identifiable orders)")
    HC = hout_column()
    rows = {}
    for s, fn, h4, h2, cls in LADDERS:
        rows[s] = analyze_ladder(s, fn, h4, h2, cls, HC)
    print()
    print("=== CLAUSE BOOKKEEPING (s33_reg.txt) ===")
    L = {s: rows[s]["letter"] for s in rows}
    print("  T-1 letters: 1.25 = %s ; 1.5 = %s ; 1.75 (MANDATORY) = %s ; 1.375 (REPORT) = %s" % (L[1.25], L[1.5], L[1.75], L[1.375]))
    l175 = L[1.75]; inter = [L[1.25], L[1.5]]
    if l175 == "CONVERGENT":
        head = "CONVERGENT-PROGRAM (the mandatory row settles: the identity is scored at the converged order; the H-clauses apply)"
    elif l175 == "WANDERING" or (l175 == "DEGENERATE" and "WANDERING" in inter):
        head = "WANDERING-CLOSURE (the dense-lattice non-measurability MEASURED%s: the residual program CLOSES on the record)" % (" at the interior rows, by scope at the mandatory row" if l175 == "DEGENERATE" else "")
    elif l175 == "DEGENERATE" and "CONVERGENT" in inter:
        head = "MIXED-INTERIOR-CONVERGENT (mandatory row scope-limited; interior converged values scored as REPORT rows; no closure)"
    else:
        head = "DEGENERATE-CLOSURE (no scored row identifiable at two orders: closure by scope)"
    print("  T-1 PROGRAM VERDICT: %s" % head)
    for s in (1.25, 1.5, 1.75, 1.375):
        r = rows[s]
        if r["letter"] == "CONVERGENT":
            print("  identity at sigma=%.3f (order %d): %s (dev' = %+.4e) ; T-2 k-leg: %s" % (s, r["plast"], r["score"], r["devp"], r["klet"]))
    kl = [rows[s].get("klet", "n/a (not CONVERGENT)") for s in (1.25, 1.5, 1.75)]
    print("  T-2 k-leg at the converged orders: %s" % kl)
    r = rows[1.75]
    if r["letter"] == "CONVERGENT" and r.get("res_h"):
        signs = [rows[s]["sgn_ok"] for s in (1.25, 1.375, 1.5, 1.75) if rows[s].get("sgn_ok") is not None]
        prices = r["sgn_ok"] and (0.2 <= abs(r["ratio"]) <= 5.0)
        print("  H-1 (amplitude sign at the MANDATORY 1.75 row): %s ; signs over the CONVERGENT rows: %d/%d" % ("MATCH" if r["sgn_ok"] else "MISS", sum(signs), len(signs)))
        print("  H-2 (amplitude ratio at 1.75): c_h'/h_out = %+.4f -> %s" % (r["ratio"], "PRICES-AT-AMPLITUDE (ratio in [1/5, 5])" if prices else ("sign-pass, NOT pricing: the suppression factor s_h(1.75) = %+.4f is MEASURED" % r["ratio"] if r["sgn_ok"] else "FAILS the mandatory sign — the outer-forced account is REFUTED at amplitude level")))
    else:
        print("  H-1/H-2: NOT SCOREABLE — the mandatory row is %s" % ("not CONVERGENT" if r["letter"] != "CONVERGENT" else "CONVERGENT but c_h' unresolved"))
    print("  the sigma = 1.59375 ride (): %s" % ("CONDITION MET ONLY IF the S-rows at 1.5625/1.625 carry opposite-signed identifiable dev' — see stage single" if (l175 == "CONVERGENT" and r.get("score") == "TOWER-ROBUST") else "INACTIVE"))

# ---------------------------------------------------------------- single (S-rows)
def mode_single():
    print("=== single-vector REPORT rows: the order ladder on the peak vectors (no ladder: no k', no n4; letters by convergence / gain / sigma_e' only, unscored) ===")
    for fn, s, dev in REP:
        z = np.load(os.path.join(DIR, fn)); q = np.abs(z["c_pk"]); r0 = fit_slot(q, s, -s)
        epred = epred_of(s) if s in FROZEN else r0["E"] - dev
        print("--- %s sigma=%.5f (dev = %+.4e ; e_pred = %+.4e ; e-hat_4 = %+.5e ; rms_4 = %.2e)" % (fn, s, dev, epred, r0["E"], r0["rms"]))
        prev = None; res = {}
        for order in (7, 9, 11):
            f = fit_order(q, s, order, r0, prev)
            gain = 1.0 - f["rms"]/r0["rms"]
            rms_prev = prev["rms"] if prev is not None else r0["rms"]
            why = []
            if not f["conv"]: why.append("NOT-CONVERGED (status %d nfev %d restart %.1e basin %.1e/%.1e)" % (f["status"], f["nfev"], f["restart_sig"], f["basin_sig"], f["basin_rms"]))
            if f["rms"] > rms_prev*(1 + 1e-9): why.append("rms > previous order")
            if gain < 0.05: why.append("gain %.1f%% < 5%%" % (100*gain))
            if f["sigE"] > abs(dev)/2: why.append("sigma_e' %.1e > |dev|/2 = %.1e" % (f["sigE"], abs(dev)/2))
            if f["knorm"] > 1e12: why.append("kappa_norm > 1e12")
            ident = len(why) == 0
            res[order] = dict(ident=ident, E=f["E"], sigE=f["sigE"])
            print("    order %2d (%d params; %s%s): status=%d nfev=%d restart %.1e basin %.1e/%.1e  rms=%.2e gain=%+.1f%% (vs previous order %+.1f%%)  e-hat'=%+.5e (sigma_e'=%.2e)  shift=%+.4e (%+.2f |dev|)  e-hat'/e_pred=%+.4f  dev'=%+.4e  kappa_norm=%.2e -> %s%s"
                  % (order, 4 + len(f["powers"]), "+".join(f["names"]), (" ; merged " + ",".join(f["merged"])) if f["merged"] else "", f["status"], f["nfev"], f["restart_sig"], f["basin_sig"], f["basin_rms"],
                     f["rms"], 100*gain, 100*(1 - f["rms"]/rms_prev), f["E"], f["sigE"], f["E"] - r0["E"], (f["E"] - r0["E"])/abs(dev), f["E"]/epred, f["E"] - epred, f["knorm"],
                     "IDENTIFIABLE-class" if ident else "DEGENERATE-class", "" if ident else " [" + "; ".join(why) + "]"))
            print("      slots: %s" % " ; ".join("c_%s = %+.4e (sigma %.1e, %s)" % (nm, f["cg"][j], f["sig"][3 + j], "resolved" if abs(f["cg"][j]) > 3*f["sig"][3 + j] else "unresolved") for j, nm in enumerate(f["names"])))
            prev = f
        idents = [o for o in (7, 9, 11) if res[o]["ident"]]
        if len(idents) < 2:
            let = "DEGENERATE-class (%d identifiable order(s))" % len(idents)
        else:
            pp, pl = idents[-2], idents[-1]
            bar = max(abs(dev)/5.0, 2.0*max(res[pp]["sigE"], res[pl]["sigE"]))
            mv = res[pl]["E"] - res[pp]["E"]
            let = ("CONVERGENT-class" if abs(mv) <= bar else "WANDERING-class") + " (orders %d -> %d: move %+.4e vs bar %.2e)" % (pp, pl, mv, bar)
            if abs(mv) <= bar:
                devp = res[pl]["E"] - epred
                let += " ; would-%s (dev' = %+.4e vs bar %.2e)" % ("RESOLVE" if abs(devp) <= max(abs(dev)/5, 2*res[pl]["sigE"]) else "be-TOWER-ROBUST", devp, max(abs(dev)/5, 2*res[pl]["sigE"]))
        print("    >> S-row letter: %s" % let)

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"check": mode_check, "probe": mode_probe, "ladder": mode_ladder, "single": mode_single}[m]()
