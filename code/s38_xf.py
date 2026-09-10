#!/usr/bin/env python
"""
s38_xf.py -- THE S38 SECONDARY + TERTIARY ANALYSIS (..C-5,
D-1..D-2;.  IMPORTS
s35_strip, s36_tc and s37_xf VERBATIM (certified S35: 45/45, 22/22; S36 and
S37: the line-for-line reproductions of their predecessors' tables) and adds
ONLY: the X6 column (by glob), the EXCLUSIVE cross-column letter with per-step
RESOLVED / UNRESOLVED flags and ONE common timing shift for all columns, the
strict-set report, the 13-mode guard on single-column courses, and the X5
N1024 pair entries.

Stages
  check   C-0(a): s37_xf.stage_x2() and s37_xf.stage_x1() re-run with stdout
          captured and DIFFED against s37_xf_x2.txt and the X-1 block of
          s37_xf_all.txt line for line (trailing blank lines stripped)
          BEFORE any new number.
  c0      C-0(b): the bit-identity receipt -- the fresh-tag sigma = 2 N512 row
          at X6 vs its banked partner (s35_strip.stage_repro VERBATIM); the
          seven X6 rows' in-ride identities.
  c1      C-1: the five-column letter at t_c (validity: the X6 rows' omega
          usable >= 0.5 and |M5| >= 13, else UNDEFINED(floor)).
  c2      C-2: the four-column table X3..X6 at the earliest common time t_c'.
  c3      C-3: the S37 X-1 / X-2 rows re-read under the exclusive rule (the
          same numbers; flags and the common shift added; report).
  c4      C-4: the strict-set robustness of the t_c four-column separation.
  c5      C-5: the sixth column's own rows.
  d1      D-1: the X5 N1024 pair -- the N-convergence of the X5 value at t_c.
  d2      D-2: the pair's common-window course on 52..76 (report).
  all     check + c0 + c1 + c2 + c3 + c4 + c5
"""
import io, os, sys, glob, re, math, contextlib
import numpy as np
import s35_strip as S
import s36_tc as T
import s37_xf as X7
from s35_strip import BSTAR, FLOOR, kd_index, sig_of, spectra_at, joint_mask, a_norm, bracket, column, keys_equal
from s36_tc import first_usable, cadence, window, usable_frac, COLS, TM, TC, rows_at
from s37_xf import strip_tail, sep_on, T2, MC

sys.stdout.reconfigure(encoding="utf-8")
KD = 24 * math.pi
XK = {"X2": 0.10, "X3": 0.15, "X4": 0.20, "X5": 0.25, "X6": 0.30}
M16 = list(range(52, 83, 2))         # the S37 16-mode set at t_c (52..82)
X6_FILES = [f for f in sorted(glob.glob("p3d_s*_x1705d_*_N512.npz"), key=sig_of) if sig_of(f) < 2.0]   # the column; the sigma = 2 receipt row excluded
COLS6 = dict(X7.COLS5)
if X6_FILES:
    COLS6["X6"] = ("X6 N512 (x1705d)", X6_FILES)
PAIR5 = sorted(glob.glob("p3d_s*_x1421d1024_*_N1024.npz"), key=sig_of)
BANKED2 = "p3_s2p000_n512x1705_k0p300000_N512.npz"
FRESH2 = "p3_s2p000_x1705d_k0p300000_N512.npz"
SIDE2 = "p3d_s2p000_x1705d_k0p300000_N512.npz"

# ------------------------------------------------------------------ helpers
def files_of(key, s):
    return [f for f in COLS6[key][1] if sig_of(f) == s][0]

def tm_of(key):
    return TM.get(key, min(float(np.load(f)["tcert"]) for f in COLS6[key][1]))

def common_set(keys, t, s_lo=0.875, s_hi=1.5):
    """the S37 common_set generalised to COLS6: the joint mask over the 2*len(keys) rows at t AND their bracketing records."""
    amps = []
    for k in keys:
        for s in (s_lo, s_hi):
            sp = spectra_at(files_of(k, s), t)
            amps += [sp["omega"], sp["omega_lo"], sp["omega_hi"]]
    return joint_mask(BSTAR, *amps)

def letter_excl(vals, flags):
    """THE EXCLUSIVE RULE (S38 C-1/C-2): the deciding statistic = max |dev from mean| at the +-10% bar (X-FLAT iff <= 10%);
    above it, X-ORDERED iff at least one step is RESOLVED and every RESOLVED step has the same direction; else MIXED."""
    v = np.asarray(vals, float)
    if np.any(np.isnan(v)):
        return "UNDEFINED(floor)", float("nan")
    dev = float(np.abs(v / v.mean() - 1).max())
    if dev <= 0.10:
        return "X-FLAT", dev
    res = [d for f, d in flags if f == "RESOLVED"]
    if res and all(d == res[0] for d in res):
        return "X-ORDERED", dev
    return "MIXED", dev

def cross_column_x(keys, t, label, s_lo=0.875, s_hi=1.5, score=True, M=None):
    """the S37 cross_column observer with (i) ONE common shift delta = the max half-cadence over the compared columns,
    (ii) per-step RESOLVED / UNRESOLVED flags (the two columns' [-delta, +delta] ranges disjoint or not), (iii) the exclusive letter,
    printed at -delta / 0 / +delta (the letter at 0 decides)."""
    if M is None:
        M = common_set(keys, t, s_lo, s_hi)
    delta = max(cadence(files_of(k, s)) for k in keys for s in (s_lo, s_hi)) / 2.0
    print("   the common fixed set M(%s = %.6f) over the %d rows (%g and %g at %s) and their brackets: %d modes%s ; validity |M| >= 13: %s ; common shift delta = %.2e (the max half-cadence over the columns)"
          % (label, t, 2 * len(keys), s_lo, s_hi, "/".join(keys), len(M), " (%d..%d)" % (M[0], M[-1]) if M else "", len(M) >= 13, delta))
    vals = {}; rng = {}; prods = {}
    for k in keys:
        f_lo, f_hi = files_of(k, s_lo), files_of(k, s_hi)
        sp_lo, sp_hi = spectra_at(f_lo, t), spectra_at(f_hi, t)
        v, n = sep_on(sp_hi, sp_lo, M)
        vl, nl = sep_on(spectra_at(f_hi, t - delta), spectra_at(f_lo, t - delta), M)
        vh, nh = sep_on(spectra_at(f_hi, t + delta), spectra_at(f_lo, t + delta), M)
        u_lo, u_hi = usable_frac(sp_lo["omega"]), usable_frac(sp_hi["omega"])
        vals[k] = (v if len(M) >= 13 else float("nan"), vl, vh); rng[k] = (min(vl, vh), max(vl, vh)); prods[k] = v * XK[k]
        print("   %s : omega S_bar(%g, %g) on M = %+.4f [-delta %+.4f, +delta %+.4f ; modes used %d/%d/%d] (row usable fractions of B*: %.2f / %.2f) -> product with X/k_d^2 = %+.4f"
              % (k, s_hi, s_lo, v, vl, vh, n, nl, nh, u_lo, u_hi, v * XK[k]))
    flags = []
    for a, b in zip(keys[:-1], keys[1:]):
        disjoint = rng[a][1] < rng[b][0] or rng[b][1] < rng[a][0]
        direction = "|S| DEC" if abs(vals[b][0]) < abs(vals[a][0]) else "|S| INC"
        flags.append(("RESOLVED" if disjoint else "UNRESOLVED", direction))
    print("   steps: %s" % " ; ".join("%s->%s %s %s (%.1f%%)" % (a, b, f, d, 100 * abs(vals[b][0] / vals[a][0] - 1)) for (a, b), (f, d) in zip(zip(keys[:-1], keys[1:]), flags)))
    v0 = np.array([vals[k][0] for k in keys]); vm = np.array([vals[k][1] for k in keys]); vp = np.array([vals[k][2] for k in keys])
    let, dev = letter_excl(v0, flags)
    def dev_of(v):
        return float(np.abs(v / v.mean() - 1).max()) if not np.any(np.isnan(v)) else float("nan")
    span = (v0.max() - v0.min()) / abs(v0.mean()) if not np.any(np.isnan(v0)) else float("nan")
    p = np.array([prods[k] for k in keys]); pdev = dev_of(p)
    if score:
        print("   LETTER (%s, %d columns %s on M, EXCLUSIVE): %s (deciding statistic max |dev from mean| = %.1f%% at the +-10%% bar; at -delta %.1f%%, at +delta %.1f%% -> letters %s / %s ; span %.1f%% ; values %s)"
              % (label, len(keys), "/".join(keys), let, 100 * dev, 100 * dev_of(vm), 100 * dev_of(vp), letter_excl(vm, flags)[0], letter_excl(vp, flags)[0], 100 * span, " / ".join("%+.4f" % x for x in v0)))
        print("   product test on M: %s (max |dev from mean| = %.1f%%; products %s)" % ("CONSTANT" if pdev <= 0.10 else "VARIES", 100 * pdev, " / ".join("%+.4f" % x for x in p)))
    else:
        print("   REPORT (%s): values %s ; max |dev from mean| %.1f%% (at -delta %.1f%%, +delta %.1f%%) ; span %.1f%% -> under the exclusive rule would read %s"
              % (label, " / ".join("%+.4f" % x for x in v0), 100 * dev, 100 * dev_of(vm), 100 * dev_of(vp), 100 * span, let))
    return M, {k: vals[k][0] for k in keys}, let, flags

# ------------------------------------------------------------------ C-0 (a)
def stage_check():
    print("== C-0(a): the reproduction obligation — s37_xf.stage_x2() and stage_x1() re-run today, stdout DIFFED against s37_xf_x2.txt and the X-1 block of s37_xf_all.txt ==")
    ok_all = True
    for name, stage, rec_lines in (("s37_xf_x2.txt", X7.stage_x2, strip_tail(open("s37_xf_x2.txt", encoding="utf-8").read().splitlines())),
                                   ("s37_xf_all.txt [X-1 block]", X7.stage_x1, None)):
        if rec_lines is None:
            allr = open("s37_xf_all.txt", encoding="utf-8").read().splitlines()
            i0 = next(i for i, l in enumerate(allr) if l.startswith("== X-1:")); i1 = next(i for i, l in enumerate(allr) if l.startswith("== X-3:"))
            rec_lines = strip_tail(allr[i0:i1])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stage()
        mine = strip_tail(buf.getvalue().splitlines())
        nd = sum(1 for a, b in zip(mine, rec_lines) if a != b) + abs(len(mine) - len(rec_lines))
        print("   %-28s record %2d lines | today %2d lines | differing lines %d -> %s" % (name, len(rec_lines), len(mine), nd, "REPRODUCED (every printed digit)" if nd == 0 else "NOT-REPRODUCED"))
        if nd:
            ok_all = False
            for a, b in zip(mine, rec_lines):
                if a != b:
                    print("      record: %s\n      today : %s" % (b[:170], a[:170])); break
    print("   C-0(a) LETTER: %s" % ("REPRODUCED" if ok_all else "NOT-REPRODUCED (STOP)"))
    return ok_all

# ------------------------------------------------------------------ C-0 (b)
def stage_c0():
    print("== C-0(b): the bit-identity receipt at X6 — the fresh-tag sigma = 2 N512 row vs its banked S21 partner (s35_strip.stage_repro VERBATIM); the seven X6 rows' in-ride identities ==")
    if not (os.path.exists(FRESH2) and os.path.exists(SIDE2)):
        print("   fresh row absent (%s / %s): nothing checked" % (FRESH2, SIDE2)); return
    letter = S.stage_repro(fresh=FRESH2, banked=BANKED2, side=SIDE2)
    print("   C-0(b) LETTER: %s" % ("BIT-IDENTICAL" if letter == "BIT-IDENTICAL" else letter))
    a = np.load(FRESH2); b = np.load(BANKED2)
    print("   printed observables fresh vs banked: verdict %s / %s ; t_cert %.6f / %.6f ; Om_cert_peak %.4f / %.4f ; g_cert %.4f / %.4f ; itx %.4f / %.4f ; records %d / %d"
          % (str(a["verdict"]), str(b["verdict"]), float(a["tcert"]), float(b["tcert"]), float(a["om_pk"]), float(b["om_pk"]), float(a["gcert"]), float(b["gcert"]), float(a["itx"]), float(b["itx"]), int(np.asarray(a["every_t"]).size), int(np.asarray(b["every_t"]).size)))
    if "X6" in COLS6:
        for f in COLS6["X6"][1]:
            z = np.load(f); p3 = str(z["source_p3"]); q = np.load(p3); N = int(q["N"])
            last_ok = np.array_equal(z["th_bot"][-1], q["th_final"][N])
            sup_ok = np.array_equal(np.asarray(z["sup_w_chk"]), np.asarray(q["sup_w"]))
            dx = np.asarray(z["delta_x_chk"]); sdx = np.asarray(q["delta_x"])
            dx_ok = np.array_equal(np.isnan(dx), np.isnan(sdx)) and np.array_equal(dx[~np.isnan(dx)], sdx[~np.isnan(sdx)])
            print("   X6 sigma=%-6g kappa FIELD %r  %-10s t_cert %.6f  Om_cert_peak %.4f  g_cert %.4f  itx %.4f  steps %d  records %d | align_ok %s ; last th_bot == th_final[Ny] %s ; sup_w bit-identical %s ; delta_x bit-identical %s (no banked partner at this sigma)"
                  % (sig_of(f), float(q["kappa"]), str(q["verdict"]), float(q["tcert"]), float(q["om_pk"]), float(q["gcert"]), float(q["itx"]), int(np.asarray(q["every_t"]).size), len(z["t"]), bool(z["align_ok"]), last_ok, sup_ok, dx_ok))

# ------------------------------------------------------------------ C-1
def stage_c1():
    print("== C-1: THE FIVE-COLUMN LETTER AT t_c = %.6f on ONE common set across the ten rows (0.875 / 1.5 at X2 .. X6) — the validity branch ==" % TC)
    if "X6" not in COLS6 or not all(s in [sig_of(f) for f in COLS6["X6"][1]] for s in (0.875, 1.5)):
        print("   X6 rows at 0.875 / 1.5 absent — nothing scored"); return
    keys = ["X2", "X3", "X4", "X5", "X6"]
    f6 = {s: files_of("X6", s) for s in (0.875, 1.5)}
    tm6 = tm_of("X6")
    u6 = [usable_frac(spectra_at(f6[s], TC)["omega"]) for s in (0.875, 1.5)]
    inside = all(float(np.load(f6[s])["t"][0]) <= TC <= float(np.load(f6[s])["tcert"]) for s in (0.875, 1.5))
    starts6 = [first_usable(f6[s], "omega") for s in (0.875, 1.5)]
    print("   validity: t_c inside the X6 rows' certified windows: %s (t_m(X6) = min t_cert = %.6f) ; X6 omega usable at t_c: %.2f / %.2f (0.875 / 1.5) ; bar >= 0.5: %s ; the X6 rows' omega window starts %.6f / %.6f (receipts §3 prior 0.00381-0.00383)"
          % (inside, tm6, u6[0], u6[1], all(u >= 0.5 for u in u6), starts6[0], starts6[1]))
    valid = inside and all(u >= 0.5 for u in u6)
    M, vals, let, flags = cross_column_x(keys, TC, "t_c", score=valid)
    if valid and len(M) >= 13:
        print("   C-1 LETTER: %s" % let)
    else:
        print("   C-1 LETTER: UNDEFINED(floor) — the labeled expectation (receipts §3); the four-column table at the earliest common time is C-2")

# ------------------------------------------------------------------ C-2
def stage_c2():
    print("== C-2: THE FOUR-COLUMN TABLE X3 / X4 / X5 / X6 AT THE EARLIEST COMMON TIME t_c' (the X6 window start) on ONE common set — EXCLUSIVE letters ==")
    if "X6" not in COLS6:
        print("   no X6 column present"); return
    keys = ["X3", "X4", "X5", "X6"]
    rows = [(k, s, files_of(k, s)) for k in keys for s in (0.875, 1.5)]
    starts = {(k, s): first_usable(f, "omega") for k, s, f in rows}
    tcp = max(starts.values())
    tends = {k: tm_of(k) for k in keys}
    inside = all(float(np.load(f)["t"][0]) <= tcp <= float(np.load(f)["tcert"]) for k, s, f in rows)
    print("   omega first-usable per row: %s" % " ; ".join("%s s=%g %.6f" % (k, s, v) for (k, s), v in sorted(starts.items())))
    print("   t_c' = max first-usable = %.6f ; t_m per column: %s ; t_c' <= min t_m (%.6f): %s ; t_c' inside every row's [t_first, t_cert]: %s ; t_m(X2) = %.6f < t_c': %s (X2 excluded by construction)"
          % (tcp, " ".join("%s %.6f" % (k, v) for k, v in tends.items()), min(tends.values()), tcp <= min(tends.values()), inside, TM["X2"], TM["X2"] < tcp))
    if not (tcp <= min(tends.values()) and inside):
        print("   C-2 LETTER: UNDEFINED(window) — no common time inside the four certified windows"); return
    M, vals, let, flags = cross_column_x(keys, tcp, "t_c'", score=True)
    if len(M) < 13:
        print("   C-2 LETTER: UNDEFINED(floor) — |M| < 13"); return
    print("   C-2 LETTER: %s" % let)
    print("   BESIDE (report): the S37 t_c = %.6f row X2..X5 on its 16-mode set: -0.8725 / -0.8688 / -0.8069 / -0.7346 (X-ORDERED as fired, the X2->X3 step unresolved — C-3); the receipts §4b rescaled priors at t_c' ~ -0.79 / -0.74 / -0.68 / -0.62" % TC)
    # the same table on the S37 13-mode set 52..76 (the X-4 form), for the set-robustness of the pattern
    print("   the same four columns at t_c' on the fixed 13-mode set 52..76 (report):")
    for k in keys:
        lo, hi = spectra_at(files_of(k, 0.875), tcp), spectra_at(files_of(k, 1.5), tcp)
        v, n = sep_on(hi, lo, MC)
        print("     %s: %+.4f (modes used %d/13; usable of B* %.2f / %.2f)" % (k, v, n, usable_frac(lo["omega"]), usable_frac(hi["omega"])))

# ------------------------------------------------------------------ C-3
def stage_c3():
    print("== C-3: THE S37 ROWS RE-READ UNDER THE EXCLUSIVE RULE with per-step flags and ONE common shift (the SAME numbers at the unshifted time; report, never a re-score) ==")
    print(" -- the S37 X-1 row: t2 = %.6f, X2 / X3 / X4 (the fired letter X-FLAT 8.7%%; the X-ORDERED clause also met) --" % T2)
    cross_column_x(["X2", "X3", "X4"], T2, "t2", score=False)
    print(" -- the S37 X-2 row: t_c = %.6f, X2 / X3 / X4 / X5 (the fired letter X-ORDERED 10.5%%) --" % TC)
    cross_column_x(["X2", "X3", "X4", "X5"], TC, "t_c", score=False)

# ------------------------------------------------------------------ C-4
def stage_c4():
    print("== C-4: THE STRICT-SET ROBUSTNESS of the t_c four-column separation (X2 / X3 / X4 / X5): the joint mask at f x the 1e-12 mask over the eight rows and brackets (report; the deciding statistic d = 1 - S_bar(X5)/S_bar(X2)) ==")
    keys = ["X2", "X3", "X4", "X5"]
    amps = []
    for k in keys:
        for s in (0.875, 1.5):
            sp = spectra_at(files_of(k, s), TC); amps += [sp["omega"], sp["omega_lo"], sp["omega_hi"]]
    for f in (10, 5, 3, 1):
        m = np.ones(len(BSTAR), bool)
        for a in amps:
            a = np.asarray(a); m &= a[BSTAR] > f * FLOOR * a[kd_index]
        Mf = [k for k, ok in zip(BSTAR, m) if ok]
        vals = {}
        for k in keys:
            lo, hi = spectra_at(files_of(k, 0.875), TC), spectra_at(files_of(k, 1.5), TC)
            vals[k] = sep_on(hi, lo, Mf)[0]
        d = 1 - vals["X5"] / vals["X2"]
        let = "PERSISTS" if d >= 0.12 else ("FLOOR-LINKED" if d <= 0.08 else "INCONCLUSIVE")
        print("   f = %2d x mask: set = %2d modes (%d..%d)%s ; S_bar(1.5, 0.875) = %s ; d = %.3f -> %s%s"
              % (f, len(Mf), Mf[0], Mf[-1], "" if len(Mf) >= 13 else " [below the 13-mode validity bar: UNDEFINED(set)]", " / ".join("%+.4f" % vals[k] for k in keys), d, let, " (the S37 X-4 / X-2 sets)" if f in (10, 1) else ""))

# ------------------------------------------------------------------ C-5
def stage_c5():
    print("== C-5: THE SIXTH COLUMN X6 = 0.30 k_d^2 (x1705d, N512): the certified observables vs the prior; the ordering at t_m(X6); the fixed-set course (13-mode guard) ==")
    if "X6" not in COLS6:
        print("   no X6 side files present"); return
    label, files = COLS6["X6"]
    oms = []; tcs = []
    for f in files:
        q = np.load(str(np.load(f)["source_p3"])); oms.append(float(q["om_pk"])); tcs.append(float(q["tcert"]))
        print("   sigma=%-6g %-10s t_cert %.6f  Om_cert_peak %.4f" % (sig_of(f), str(q["verdict"]), float(q["tcert"]), float(q["om_pk"])))
    oms = np.array(oms); spread = 100 * (oms.max() - oms.min()) / oms.mean()
    print("   (a) Om_cert_peak across the column: mean %.4f, spread %.3f%% -> prior 'sigma-flat to <= 0.6%%': %s ; mean within [220.14, 222.35] (+-0.5%% of the N512 partners' 221.2464): %s (ratio %.4f)"
          % (oms.mean(), spread, "CONSISTENT" if spread <= 0.6 else "INCONSISTENT", "CONSISTENT" if 220.14 <= oms.mean() <= 222.35 else "INCONSISTENT", oms.mean() / 221.2464))
    tm6 = min(tcs)
    print("   (b) the matched-time ordering at t_m(X6) = min t_cert = %.6f (the S35 G-2 instrument; REPORT):" % tm6)
    column(label + " @ t_m(X6)", files, tm6, score=False)
    print("   (c) the fixed-set time course (the S36 tcfix instrument with the 13-mode guard; REPORT):")
    sigs = sorted(sig_of(f) for f in files)
    for fld in ("omega", "theta"):
        ts = window(files, fld); grid = [ts + j * (tm6 - ts) / 8.0 for j in range(9)]
        d0 = dict(rows_at(files, ts)); M0 = joint_mask(BSTAR, *[d0[s][fld] for s in sigs])
        if len(M0) < 13:
            print("     %-6s window [%.6f, %.6f] M0 = %d modes -> UNDEFINED(set) (below the 13-mode validity bar; no course read)" % (fld, ts, tm6, len(M0))); continue
        vals = []
        for t in grid:
            d = dict(rows_at(files, t)); m = joint_mask(M0, d[sigs[0]][fld], d[sigs[-1]][fld])
            vals.append(abs(float((a_norm(d[sigs[-1]][fld], m) - a_norm(d[sigs[0]][fld], m)).mean())) if len(m) == len(M0) else float("nan"))
        v = np.array(vals); mono = all(v[j + 1] <= v[j] + 1e-9 for j in range(8))
        print("     %-6s window [%.6f, %.6f] M0 = %d modes (%d..%d): |S_col(%g, %g)| at the 9 samples: %s ; ratio start/t_m %.3f ; non-increasing in t: %s"
              % (fld, ts, tm6, len(M0), M0[0], M0[-1], sigs[-1], sigs[0], " ".join("%.4f" % x for x in v), v[0] / v[-1], "YES" if mono else "no"))
        letters = []
        for t in grid:
            let, det = S.ordering(rows_at(files, t), fld, BSTAR)
            letters.append(let if not let.startswith("UNDEFINED") else "UNDEFINED(floor)")
        print("     %-6s ORDER letters at the 9 samples: %s" % (fld, " ".join(l[:9] for l in letters)))
    # the common-set (52..76) course over the column's own omega window (the S37 X-4 form)
    ts = window(files, "omega"); grid = [ts + j * (tm6 - ts) / 8.0 for j in range(9)]
    vals = []
    for t in grid:
        d = dict(rows_at(files, t)); m = joint_mask(MC, d[sigs[0]]["omega"], d[sigs[-1]]["omega"])
        vals.append(abs(float((a_norm(d[sigs[-1]]["omega"], m) - a_norm(d[sigs[0]]["omega"], m)).mean())) if len(m) == len(MC) else float("nan"))
    v = np.array(vals)
    print("   (d) omega (%g -> %g) on the common set 52..76 over [%.6f, %.6f]: %s ; ratio start/t_m %.3f (X2 / X3 / X4 / X5 on the record: 1.773 / 2.125 / 2.166 / 1.701 over their own windows)"
          % (sigs[0], sigs[-1], ts, tm6, " ".join("%.4f" % x for x in v), v[0] / v[-1]))

# ------------------------------------------------------------------ D-1
def stage_d1():
    print("== D-1: THE X5 N1024 PAIR (x1421d1024) — the N-convergence of the X5 value at t_c = %.6f on the S37 16-mode set 52..82 ==" % TC)
    f1024 = {sig_of(f): f for f in PAIR5 if sig_of(f) in (0.875, 1.5)}
    if len(f1024) < 2:
        print("   N1024 rows present: %s — nothing scored" % sorted(f1024)); return
    f512 = {s: files_of("X5", s) for s in (0.875, 1.5)}
    for s in (0.875, 1.5):
        for N, f in (("N512", f512[s]), ("N1024", f1024[s])):
            q = np.load(str(np.load(f)["source_p3"]))
            print("   sigma=%-6g %-5s kappa FIELD %r  %-10s t_cert %.6f  Om_cert_peak %.4f  g_cert %.4f  itx %.4f  steps %d | align_ok %s" % (s, N, float(q["kappa"]), str(q["verdict"]), float(q["tcert"]), float(q["om_pk"]), float(q["gcert"]), float(q["itx"]), int(np.asarray(q["every_t"]).size), bool(np.load(f)["align_ok"])))
    u = {s: usable_frac(spectra_at(f1024[s], TC)["omega"]) for s in (0.875, 1.5)}
    inside = all(float(np.load(f1024[s])["t"][0]) <= TC <= float(np.load(f1024[s])["tcert"]) for s in (0.875, 1.5))
    print("   validity: t_c inside both N1024 certified windows: %s ; N1024 omega usable at t_c: %.2f / %.2f ; bar >= 0.5: %s" % (inside, u[0.875], u[1.5], all(x >= 0.5 for x in u.values())))
    delta = max(cadence(f) for f in list(f512.values()) + list(f1024.values())) / 2.0
    out = {}
    for N, fs in (("N512", f512), ("N1024", f1024)):
        v, n = sep_on(spectra_at(fs[1.5], TC), spectra_at(fs[0.875], TC), M16)
        vl, _ = sep_on(spectra_at(fs[1.5], TC - delta), spectra_at(fs[0.875], TC - delta), M16)
        vh, _ = sep_on(spectra_at(fs[1.5], TC + delta), spectra_at(fs[0.875], TC + delta), M16)
        out[N] = (v, vl, vh, n)
        print("   %-5s S_bar(1.5, 0.875) on 52..82 at t_c = %+.4f [common shift -delta %+.4f, +delta %+.4f ; delta %.2e ; modes used %d/16]" % (N, v, vl, vh, delta, n))
    r = out["N512"][0] / out["N1024"][0]; rl = out["N512"][1] / out["N1024"][1]; rh = out["N512"][2] / out["N1024"][2]
    rec = float(re.search(r"X5 : omega S_bar\(1\.5, 0\.875\) on M = ([+-]\d\.\d{4})", open("s37_xf_x2.txt", encoding="utf-8").read()).group(1))
    print("   the S37 record's N512 value on this set: %+.4f (today's recomputation %+.4f, equal to 4 dp: %s)" % (rec, out["N512"][0], abs(rec - out["N512"][0]) < 5e-5))
    dr = abs(r - 1)
    let = "N-CONVERGED" if dr <= 0.005 else ("DRIFTING" if dr <= 0.02 else "NOT-CONVERGED")
    if not (inside and all(x >= 0.5 for x in u.values()) and out["N1024"][3] >= 13):
        let = "UNDEFINED(floor)"
    print("   r_N = S_bar_512 / S_bar_1024 = %.4f (|r - 1| = %.2f%%) [at -delta %.4f, +delta %.4f] -> D-1 LETTER: %s (N-CONVERGED iff <= 0.5%% ; DRIFTING iff <= 2%% ; else NOT-CONVERGED)" % (r, 100 * dr, rl, rh, let))
    for s in (0.875, 1.5):
        a5 = a_norm(spectra_at(f512[s], TC)["omega"], M16); a10 = a_norm(spectra_at(f1024[s], TC)["omega"], M16)
        d = a10 - a5
        print("   D_N sigma=%-6g = max_{52..82} |a_1024 - a_512| = %.4f (mean %+.4f ; at k=52 %+.4f, k=82 %+.4f) [the X3 pair at its t_m: 0.0013 / 0.0015]" % (s, float(np.abs(d).max()), float(d.mean()), float(d[0]), float(d[-1])))

# ------------------------------------------------------------------ D-2
def stage_d2():
    print("== D-2: the pair's common-window course on 52..76 — the X5 N512 rows (0.875, 1.5) vs the N1024 pair on ONE window and ONE fixed set (report; the S37 X-3 form) ==")
    f1024 = {sig_of(f): f for f in PAIR5 if sig_of(f) in (0.875, 1.5)}
    if len(f1024) < 2:
        print("   N1024 rows absent — nothing printed"); return
    f512 = {s: files_of("X5", s) for s in (0.875, 1.5)}
    rows = {("N512", s): f512[s] for s in f512}; rows.update({("N1024", s): f1024[s] for s in f1024})
    t_start = max(first_usable(f, "omega") for f in rows.values())
    t_end = min([tm_of("X5")] + [float(np.load(f)["tcert"]) for f in rows.values()])
    print("   window [%.6f, %.6f] (start = max first-usable of the four rows; end = min(t_m(X5), t_cert)) ; fixed set 52..76 (13 modes)" % (t_start, t_end))
    grid = [t_start + j * (t_end - t_start) / 8.0 for j in range(9)]
    course = {}
    for N in ("N512", "N1024"):
        vals = []
        for t in grid:
            lo = spectra_at(rows[(N, 0.875)], t); hi = spectra_at(rows[(N, 1.5)], t)
            m = joint_mask(MC, lo["omega"], hi["omega"])
            vals.append(abs(float((a_norm(hi["omega"], m) - a_norm(lo["omega"], m)).mean())) if len(m) == len(MC) else float("nan"))
        course[N] = np.array(vals)
        print("   %-5s |S_bar(1.5, 0.875)| on 52..76 at the 9 samples: %s ; ratio start/end %.3f" % (N, " ".join("%.4f" % x for x in course[N]), course[N][0] / course[N][-1]))
    q = course["N512"] / course["N1024"]
    print("   N512/N1024 quotient at the 9 samples: %s ; max |q - 1| = %.2f%%" % (" ".join("%.3f" % x for x in q), 100 * float(np.nanmax(np.abs(q - 1)))))

if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("check", "all"):
        ok = stage_check(); print()
        if not ok and st == "all":
            sys.exit("C-0(a) NOT-REPRODUCED: STOP")
    for name, fn in (("c0", stage_c0), ("c1", stage_c1), ("c2", stage_c2), ("c3", stage_c3), ("c4", stage_c4), ("c5", stage_c5)):
        if st in (name, "all"):
            fn(); print()
    if st == "d1":
        stage_d1()
    if st == "d2":
        stage_d2()
