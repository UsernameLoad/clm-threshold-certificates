#!/usr/bin/env python
"""
s37_xf.py -- THE S37 SECONDARY (..X-5;
read with the file open).  IMPORTS s35_strip and s36_tc VERBATIM (certified
S35: 45/45 code-path check, 22/22 bit-identity; S36: the line-for-line
reproduction of both S35 tables) and adds ONLY the common-set cross-column
observer, the like-for-like N-window and the X5 column entries.

Stages
  check   X-0(a): s36_tc.stage_xdep() re-run with stdout captured and DIFFED
          against s36_tc_xdep.txt line for line (trailing blank lines
          stripped) BEFORE any new number.
  x1      X-1: the X-flatness of omega S_bar(1.5, 0.875) at the SECOND common
          time t2 = 0.0033 on ONE fixed common set M(t2) across the six rows;
          the t_c values restated on M(t2); the product test; theta at X2.
  x2      X-2: the FOURTH column at t_c = 0.003714 on ONE common set M4(t_c)
          across the eight rows (validity: X5 usable >= 0.5 and |M4| >= 13,
          else UNDEFINED(floor) + the report at the earliest common time).
  x3      X-3: the like-for-like N-comparison of the backward-growth ratio
          (X3 N512 (0.875, 1.375) vs the N1024 pair; one window, one set).
  x4      X-4: the S36 fixed-set ratios recomputed on the COMMON set 52..76
          over each column's own window (+ X5 if present); the t_c separation
          on the same set.
  x5      X-5: the fifth column's own rows -- (a) the certified observables
          vs the labeled prior + the alignment identities, (b) the matched-
          time sigma-ordering (the S35 G-2 instrument, report), (c) the
          fixed-set time course (the S36 tcfix instrument, report).
  all     check + x1 + x3 + x4   (x2 and x5 need the X5 side files)
"""
import io, os, sys, glob, re, math, contextlib
import numpy as np
import s35_strip as S
import s36_tc as T
from s35_strip import BSTAR, FLOOR, kd_index, sig_of, spectra_at, joint_mask, a_norm, bracket, column
from s36_tc import first_usable, cadence, window, usable_frac, COLS, TM, TC, rows_at, col_sep

sys.stdout.reconfigure(encoding="utf-8")
KD = 24 * math.pi
XK = {"X2": 0.10, "X3": 0.15, "X4": 0.20, "X5": 0.25}
T2 = 0.0033
MC = list(range(52, 77, 2))          # the common set 52..76 (13 modes) of receipts §7
X5_FILES = sorted(glob.glob("p3d_s*_x1421d_*_N512.npz"), key=sig_of)
COLS5 = dict(COLS)
if X5_FILES:
    COLS5["X5"] = ("X5 N512 (x1421d)", X5_FILES)

# ------------------------------------------------------------------ helpers
def strip_tail(lines):
    while lines and lines[-1].strip() == "":
        lines = lines[:-1]
    return lines

def files_of(key, s):
    return [f for f in COLS5[key][1] if sig_of(f) == s][0]

def sep_on(sp_hi, sp_lo, M, fld="omega"):
    """mean over M of a_hi - a_lo where both rows are above the floor; returns (mean, n_used)."""
    m = joint_mask(M, sp_lo[fld], sp_hi[fld])
    if len(m) < 2:
        return float("nan"), len(m)
    return float((a_norm(sp_hi[fld], m) - a_norm(sp_lo[fld], m)).mean()), len(m)

def xletter(v):
    v = np.asarray(v, float)
    if np.any(np.isnan(v)):
        return "UNDEFINED(floor)", float("nan")
    dev = float(np.abs(v / v.mean() - 1).max())
    if dev <= 0.10:
        return "X-FLAT", dev
    if (np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0)) and (v.max() - v.min()) / abs(v.mean()) > 0.10:
        return "X-ORDERED", dev
    return "MIXED", dev

def common_set(keys, t, s_lo=0.875, s_hi=1.5):
    """the joint mask over the 2*len(keys) rows at t AND at their bracketing records (the ordering() lo/hi convention)."""
    sps = []
    for k in keys:
        for s in (s_lo, s_hi):
            sps.append(spectra_at(files_of(k, s), t))
    amps = []
    for sp in sps:
        amps += [sp["omega"], sp["omega_lo"], sp["omega_hi"]]
    return joint_mask(BSTAR, *amps), sps

def cross_column(keys, t, label, s_lo=0.875, s_hi=1.5, score=True):
    M, sps = common_set(keys, t, s_lo, s_hi)
    print("   the common fixed set M(%s = %.6f) over the %d rows (%g and %g at %s) and their brackets: %d modes%s ; validity |M| >= 13: %s"
          % (label, t, 2 * len(keys), s_lo, s_hi, "/".join(keys), len(M), " (%d..%d)" % (M[0], M[-1]) if M else "", len(M) >= 13))
    vals = {}; prods = {}
    for k in keys:
        f_lo, f_hi = files_of(k, s_lo), files_of(k, s_hi)
        h = np.median([cadence(f_lo), cadence(f_hi)])
        sp_lo, sp_hi = spectra_at(f_lo, t), spectra_at(f_hi, t)
        v, n = sep_on(sp_hi, sp_lo, M)
        vl, nl = sep_on(spectra_at(f_hi, t - h / 2), spectra_at(f_lo, t - h / 2), M)
        vh, nh = sep_on(spectra_at(f_hi, t + h / 2), spectra_at(f_lo, t + h / 2), M)
        u_lo, u_hi = usable_frac(sp_lo["omega"]), usable_frac(sp_hi["omega"])
        vals[k] = v if len(M) >= 13 else float("nan"); prods[k] = v * XK[k]
        print("   %s : omega S_bar(%g, %g) on M = %+.4f [shift %+.4f, %+.4f ; modes used %d/%d/%d] (row usable fractions of B*: %.2f / %.2f) -> product with X/k_d^2 = %+.4f"
              % (k, s_hi, s_lo, v, min(vl, vh), max(vl, vh), n, nl, nh, u_lo, u_hi, v * XK[k]))
    v = np.array([vals[k] for k in keys]); p = np.array([prods[k] for k in keys])
    let, dev = xletter(v)
    if score:
        print("   LETTER (%s, %d columns %s on M): %s (max |dev from mean| = %.1f%%; values %s)" % (label, len(keys), "/".join(keys), let, 100 * dev, " / ".join("%+.4f" % x for x in v)))
        pdev = float(np.abs(p / p.mean() - 1).max()) if not np.any(np.isnan(p)) else float("nan")
        print("   product test on M: %s (max |dev from mean| = %.1f%%; products %s)" % ("CONSTANT" if pdev <= 0.10 else "VARIES", 100 * pdev, " / ".join("%+.4f" % x for x in p)))
    else:
        print("   REPORT (%s): values %s (max |dev from mean| = %.1f%% -> would read %s)" % (label, " / ".join("%+.4f" % x for x in v), 100 * dev, let))
    return M, vals, let

# ------------------------------------------------------------------ X-0 (a)
def stage_check():
    print("== X-0(a): the reproduction obligation — s36_tc.stage_xdep() re-run today, stdout DIFFED against s36_tc_xdep.txt ==")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.stage_xdep()
    mine = strip_tail(buf.getvalue().splitlines()); rec = strip_tail(open("s36_tc_xdep.txt", encoding="utf-8").read().splitlines())
    nd = sum(1 for a, b in zip(mine, rec) if a != b) + abs(len(mine) - len(rec))
    print("   s36_tc_xdep.txt  record %d lines | today %d lines | differing lines %d -> %s" % (len(rec), len(mine), nd, "REPRODUCED (every printed digit)" if nd == 0 else "NOT-REPRODUCED"))
    if nd:
        for a, b in zip(mine, rec):
            if a != b:
                print("      record: %s\n      today : %s" % (b[:170], a[:170])); break
    print("   X-0(a) LETTER: %s" % ("REPRODUCED" if nd == 0 else "NOT-REPRODUCED (STOP)"))
    return nd == 0

# ------------------------------------------------------------------ X-1
def stage_x1():
    print("== X-1: THE X-FLATNESS AT THE SECOND COMMON TIME t2 = %.6f on ONE fixed common set across the six rows (0.875 / 1.5 at X2 / X3 / X4) ==" % T2)
    keys = ["X2", "X3", "X4"]
    M, vals, let = cross_column(keys, T2, "t2")
    print("   BESIDE (report): the S36 t_c = %.6f values RESTATED on the same set M(t2):" % TC)
    for k in keys:
        v, n = sep_on(spectra_at(files_of(k, 1.5), TC), spectra_at(files_of(k, 0.875), TC), M)
        print("     %s @ t_c on M(t2): %+.4f (modes used %d/%d)" % (k, v, n, len(M)))
    vtc = [sep_on(spectra_at(files_of(k, 1.5), TC), spectra_at(files_of(k, 0.875), TC), M)[0] for k in keys]
    l2, d2 = xletter(vtc)
    print("     -> on M(t2) the t_c values read %s (max dev %.1f%%) — the R-2 letter on 26 modes was X-FLAT at 4.7%%" % (l2, 100 * d2))
    # theta at X2 only (report)
    sp_lo, sp_hi = spectra_at(files_of("X2", 0.875), T2), spectra_at(files_of("X2", 1.5), T2)
    mth = joint_mask(BSTAR, sp_lo["theta"], sp_hi["theta"])
    vth = float((a_norm(sp_hi["theta"], mth) - a_norm(sp_lo["theta"], mth)).mean()) if len(mth) >= 2 else float("nan")
    print("   theta at t2, X2 only (report): S_bar(1.5, 0.875) = %+.4f on %d/%d modes (usable %.2f / %.2f); X3 sigma >= 1.25 and every X4 row are FLOOR at t2 (receipts §3)"
          % (vth, len(mth), len(BSTAR), usable_frac(sp_lo["theta"]), usable_frac(sp_hi["theta"])))

# ------------------------------------------------------------------ X-2
def stage_x2():
    print("== X-2: THE FOURTH COLUMN AT t_c = %.6f on ONE common set across the eight rows (0.875 / 1.5 at X2 / X3 / X4 / X5) ==" % TC)
    if "X5" not in COLS5 or len(COLS5["X5"][1]) < 2:
        print("   X5 side files present: %d — the column is incomplete; nothing scored" % len(X5_FILES)); return
    f5 = {sig_of(f): f for f in COLS5["X5"][1]}
    if 0.875 not in f5 or 1.5 not in f5:
        print("   X5 rows at 0.875 / 1.5 missing: %s — nothing scored" % sorted(f5)); return
    keys = ["X2", "X3", "X4", "X5"]
    tm5 = min(float(np.load(f)["tcert"]) for f in COLS5["X5"][1])
    u5 = [usable_frac(spectra_at(f5[s], TC)["omega"]) for s in (0.875, 1.5)]
    inside = all(float(np.load(f5[s])["t"][0]) <= TC <= float(np.load(f5[s])["tcert"]) for s in (0.875, 1.5))
    print("   validity: t_c inside the X5 rows' certified windows: %s (t_m(X5) = min t_cert = %.6f) ; X5 omega usable at t_c: %.2f / %.2f (0.875 / 1.5) ; bar >= 0.5: %s"
          % (inside, tm5, u5[0], u5[1], all(u >= 0.5 for u in u5)))
    M, vals, let = cross_column(keys, TC, "t_c", score=(inside and all(u >= 0.5 for u in u5)))
    if not (inside and all(u >= 0.5 for u in u5) and len(M) >= 13):
        print("   X-2 LETTER: UNDEFINED(floor) — the four-column comparison is REPORTED at the earliest common time instead:")
        starts = {k: window(COLS5[k][1] if k != "X5" else [f5[0.875], f5[1.5]], "omega") for k in keys}
        tcp = max(starts.values()); tend = min([TM[k] for k in ("X2", "X3", "X4")] + [tm5])
        print("   omega window starts: %s ; t_c' = max = %.6f ; min t_m over the four columns = %.6f ; t_c' <= min t_m: %s"
              % (" ; ".join("%s %.6f" % (k, v) for k, v in starts.items()), tcp, tend, tcp <= tend))
        if tcp <= tend:
            cross_column(keys, tcp, "t_c'", score=False)
    else:
        print("   X-2 LETTER: %s" % let)
        print("   BESIDE (report): the three S36 t_c values on 26 modes were -0.9924 / -0.9933 / -0.9245 (X-FLAT, 4.7%%); on M4(t_c) above: %s" % " / ".join("%+.4f" % vals[k] for k in ("X2", "X3", "X4")))

# ------------------------------------------------------------------ X-3
def stage_x3():
    print("== X-3: THE LIKE-FOR-LIKE N-COMPARISON of the backward-growth ratio — X3 N512 (0.875, 1.375) vs the N1024 pair on ONE window and ONE fixed set (report) ==")
    f512 = {sig_of(f): f for f in COLS["X3"][1] if sig_of(f) in (0.875, 1.375)}
    f1024 = {sig_of(f): f for f in COLS["PAIR1024"][1] if sig_of(f) in (0.875, 1.375)}
    rows = {("N512", s): f512[s] for s in f512}; rows.update({("N1024", s): f1024[s] for s in f1024})
    t_start = max(first_usable(f, "omega") for f in rows.values())
    t_end = min([TM["X3"]] + [float(np.load(f)["tcert"]) for f in rows.values()])
    sp0 = {k: spectra_at(f, t_start) for k, f in rows.items()}
    Mcap = joint_mask(BSTAR, *[v["omega"] for v in sp0.values()])
    print("   window [%.6f, %.6f] ; M_cap = %d modes (%d..%d) ; the receipts §6 design: [0.002675, 0.004488], 13 modes 52..76" % (t_start, t_end, len(Mcap), Mcap[0], Mcap[-1]))
    grid = [t_start + j * (t_end - t_start) / 8.0 for j in range(9)]
    course = {}
    for N in ("N512", "N1024"):
        vals = []
        for t in grid:
            lo = spectra_at(rows[(N, 0.875)], t); hi = spectra_at(rows[(N, 1.375)], t)
            m = joint_mask(Mcap, lo["omega"], hi["omega"])
            vals.append(abs(float((a_norm(hi["omega"], m) - a_norm(lo["omega"], m)).mean())) if len(m) == len(Mcap) else float("nan"))
        course[N] = np.array(vals)
        mono = all(course[N][j + 1] <= course[N][j] + 1e-9 for j in range(8))
        print("   %-5s |S_bar(1.375, 0.875)| on M_cap at the 9 samples: %s ; ratio start/end %.3f ; non-increasing in t: %s"
              % (N, " ".join("%.4f" % x for x in course[N]), course[N][0] / course[N][-1], "YES" if mono else "no"))
    r5, r10 = course["N512"][0] / course["N512"][-1], course["N1024"][0] / course["N1024"][-1]
    print("   ratio quotient N512 / N1024 = %.3f ; the values' N512/N1024 quotient at the start %.3f and at the end %.3f" % (r5 / r10, course["N512"][0] / course["N1024"][0], course["N512"][-1] / course["N1024"][-1]))
    print("   restated beside (the S36 record): the pair's fixed-set ratio 3.599 over [0.002675, 0.005239] to ITS t_m (set 52..76); the X3 column's 2.123 for (0.875 -> 1.5) over [0.002748, 0.004488] (set 52..80)")

# ------------------------------------------------------------------ X-4
def stage_x4():
    print("== X-4: THE S36 FIXED-SET RATIOS ON THE COMMON SET 52..76 (13 modes) over each column's OWN window; the t_c separation on the same set (report) ==")
    rec = {"X2": 1.773, "X3": 2.123, "X4": 2.165}
    for key in [k for k in ("X2", "X3", "X4", "X5") if k in COLS5]:
        label, files = COLS5[key]
        tm = TM.get(key, min(float(np.load(f)["tcert"]) for f in files))
        sigs = sorted(sig_of(f) for f in files)
        ts = window(files, "omega"); grid = [ts + j * (tm - ts) / 8.0 for j in range(9)]
        vals = []
        for t in grid:
            d = dict(rows_at(files, t)); m = joint_mask(MC, d[sigs[0]]["omega"], d[sigs[-1]]["omega"])
            vals.append(abs(float((a_norm(d[sigs[-1]]["omega"], m) - a_norm(d[sigs[0]]["omega"], m)).mean())) if len(m) == len(MC) else float("nan"))
        v = np.array(vals); mono = all(v[j + 1] <= v[j] + 1e-9 for j in range(8))
        print("   %-22s omega (%g -> %g) on 52..76 over [%.6f, %.6f]: %s ; ratio start/t_m %.3f (the S36 record on its own M0: %s) ; non-increasing: %s"
              % (label, sigs[0], sigs[-1], ts, tm, " ".join("%.4f" % x for x in v), v[0] / v[-1], rec.get(key, "—"), "YES" if mono else "no"))
    print("   the t_c = %.6f separation S_bar(1.5, 0.875) on 52..76 (where the rows exist and are usable):" % TC)
    for key in [k for k in ("X2", "X3", "X4", "X5") if k in COLS5]:
        fs = {sig_of(f): f for f in COLS5[key][1]}
        if 0.875 in fs and 1.5 in fs:
            lo, hi = spectra_at(fs[0.875], TC), spectra_at(fs[1.5], TC)
            v, n = sep_on(hi, lo, MC)
            print("     %s: %+.4f (modes used %d/13; usable of B* %.2f / %.2f)" % (key, v, n, usable_frac(lo["omega"]), usable_frac(hi["omega"])))

# ------------------------------------------------------------------ X-5
def stage_x5():
    print("== X-5: THE FIFTH COLUMN X5 = 0.25 k_d^2 (x1421d, N512): the rows, the alignment identities, the certified observables vs the prior; the ordering; the time course ==")
    if not X5_FILES:
        print("   no X5 side files present"); return
    label, files = COLS5["X5"]
    oms = []
    for f in files:
        z = np.load(f); p3 = str(z["source_p3"]); a = np.load(p3)
        N = int(a["N"])
        last_ok = np.array_equal(z["th_bot"][-1], a["th_final"][N])
        sup_ok = np.array_equal(np.asarray(z["sup_w_chk"]), np.asarray(a["sup_w"]))
        dx = np.asarray(z["delta_x_chk"]); sdx = np.asarray(a["delta_x"])
        dx_ok = np.array_equal(np.isnan(dx), np.isnan(sdx)) and np.array_equal(dx[~np.isnan(dx)], sdx[~np.isnan(sdx)])
        oms.append(float(a["om_pk"]))
        print("   sigma=%-6g kappa FIELD %r  %-10s t_cert %.6f  Om_cert_peak %.2f  g_cert %.4f  itx %.4f  steps %d  records %d | align_ok %s ; last th_bot == th_final[Ny] %s ; sup_w bit-identical %s ; delta_x bit-identical %s"
              % (sig_of(f), float(a["kappa"]), str(a["verdict"]), float(a["tcert"]), float(a["om_pk"]), float(a["gcert"]), float(a["itx"]), int(np.asarray(a["every_t"]).size), len(z["t"]), bool(z["align_ok"]), last_ok, sup_ok, dx_ok))
    oms = np.array(oms); spread = 100 * (oms.max() - oms.min()) / oms.mean()
    print("   (a) Om_cert_peak across the column: mean %.2f, spread %.2f%% -> prior 'sigma-flat to <= 0.6%%': %s ; mean within [265.4, 269.4] (0 .. +1.5%% over the N256 partners' 265.43): %s (ratio to 265.43 = %.4f)"
          % (oms.mean(), spread, "CONSISTENT" if spread <= 0.6 else "INCONSISTENT", "CONSISTENT" if 265.4 <= oms.mean() <= 269.4 else "INCONSISTENT", oms.mean() / 265.43))
    tm5 = min(float(np.load(f)["tcert"]) for f in files)
    print("   (b) the matched-time ordering at t_m(X5) = min t_cert = %.6f (the S35 G-2 instrument; REPORT):" % tm5)
    column(label + " @ t_m(X5)", files, tm5, score=False)
    print("   (c) the fixed-set time course (the S36 tcfix instrument; REPORT):")
    sigs = sorted(sig_of(f) for f in files)
    for fld in ("omega", "theta"):
        ts = window(files, fld); grid = [ts + j * (tm5 - ts) / 8.0 for j in range(9)]
        d0 = dict(rows_at(files, ts)); M0 = joint_mask(BSTAR, *[d0[s][fld] for s in sigs])
        vals = []
        for t in grid:
            d = dict(rows_at(files, t)); m = joint_mask(M0, d[sigs[0]][fld], d[sigs[-1]][fld])
            vals.append(abs(float((a_norm(d[sigs[-1]][fld], m) - a_norm(d[sigs[0]][fld], m)).mean())) if len(m) == len(M0) else float("nan"))
        v = np.array(vals); mono = all(v[j + 1] <= v[j] + 1e-9 for j in range(8))
        print("     %-6s window [%.6f, %.6f] M0 = %d modes (%s..%s): |S_col| at the 9 samples: %s ; ratio start/t_m %.3f ; non-increasing in t: %s"
              % (fld, ts, tm5, len(M0), M0[0] if M0 else "-", M0[-1] if M0 else "-", " ".join("%.4f" % x for x in v), v[0] / v[-1] if len(v) and not np.isnan(v[0]) and not np.isnan(v[-1]) else float("nan"), "YES" if mono else "no"))
        # the ORDER letter at every sample (the S36 R-1 order instrument)
        letters = []
        for t in grid:
            let, det = S.ordering(rows_at(files, t), fld, BSTAR)
            letters.append(let if not let.startswith("UNDEFINED") else "UNDEFINED(floor)")
        print("     %-6s ORDER letters at the 9 samples: %s" % (fld, " ".join(l[:9] for l in letters)))

if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("check", "all"):
        ok = stage_check(); print()
        if not ok and st == "all":
            sys.exit("X-0(a) NOT-REPRODUCED: STOP")
    if st in ("x1", "all"):
        stage_x1(); print()
    if st in ("x3", "all"):
        stage_x3(); print()
    if st in ("x4", "all"):
        stage_x4(); print()
    if st == "x2":
        stage_x2()
    if st == "x5":
        stage_x5()
