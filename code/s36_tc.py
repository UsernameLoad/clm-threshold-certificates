#!/usr/bin/env python
"""
s36_tc.py -- THE ZERO-COMPUTE REPORT ROWS ON THE BANKED SIDE FILES (S36..R-6) and THE TERTIARY'S ANALYSIS (..U-2);'s fit / mask /
interpolation / ordering routines VERBATIM (certified S35: 45/45 code-path
check, 22/22 bit-identity) and adds only the fixed-grid time loop and the new
observers.

Stages
  check   R-0: the reproduction obligation -- stage_cols() and stage_npair()
          of s35_strip re-run with stdout captured and DIFFED line by line
          against s35_strip_cols.txt and s35_strip_npair.txt (every printed
          digit) BEFORE any new number is printed.
  tc      R-1: the time course on the fixed 9-point grids, every row at the
          SAME grid time, like-for-like timing shifts (all rows together by
          +- half a record cadence).
  xdep    R-2: the X-dependence at the common time t_c = 0.003714 (+ the
          own-t_m restatement and the product test).
  dnk     R-3: the N-difference (a_1024 - a_512)(k) on B* as a linear fit.
  wall    R-4 / R-5 / R-6: the wall-omega and far-wall-theta discriminators,
          the location of the spectral maxima.
  u0      U-0: the extended-stash rides vs their banked partners (every key)
          and vs the S35 side files (every key); the alignment identities.
  u1      U-1 / U-2: LAYER / WALL-CONFINED / FIELD-WIDE at sigma = 1.375 (the
          0.875 control beside), and the argmax-row course in time.
  all     check + tc + xdep + dnk + wall
"""
import io, os, sys, glob, re, math, contextlib
import numpy as np
import s35_strip as S
from s35_strip import (BSTAR, B512, FLOOR, kd_index, sig_of, wall_amp_from_row, spectra_at, bracket, a_norm,
                       joint_mask, ordering, pure_exp, lsq, keys_equal)

sys.stdout.reconfigure(encoding="utf-8")
KD = 24 * math.pi
XK = {"X2": 0.10, "X3": 0.15, "X4": 0.20}                     # X / k_d^2
TM = {"X2": 0.003714, "X3": 0.004488, "X4": 0.005078}          # the S35 literals (= field minima to <= 5e-7)
TC = 0.003714                                                  # the common time (receipts §3)
COLS = {"X2": ("X2 N512 (x568d)", sorted(glob.glob("p3d_s*_x568d_*_N512.npz"), key=sig_of)),
        "X3": ("X3 N512 (x853d)", sorted(glob.glob("p3d_s*_x853d_*_N512.npz"), key=sig_of)),
        "X4": ("X4 N512 (x1137d)", sorted(glob.glob("p3d_s*_x1137d_*_N512.npz"), key=sig_of)),
        "PAIR1024": ("X3 N1024 pair (x853d1024)", sorted(glob.glob("p3d_s*_x853d1024_*_N1024.npz"), key=sig_of))}

# ------------------------------------------------------------------ helpers
def usable_frac(amp):
    amp = np.asarray(amp)
    return float(np.mean([amp[k] > FLOOR * amp[kd_index] for k in BSTAR]))

def first_usable(side, fld):
    z = np.load(side); t = np.asarray(z["t"])
    for i in range(len(t)):
        amp = wall_amp_from_row(z["th_bot"][i]) if fld == "theta" else np.asarray(z["Ew"][i])
        if usable_frac(amp) >= 0.5:
            return float(t[i])
    return float("nan")

def cadence(side):
    t = np.asarray(np.load(side)["t"])
    return float(np.median(np.diff(t)))

def window(files, fld):
    return max(first_usable(f, fld) for f in files)

def col_sep(rows, fld, s_lo, s_hi):
    """S_bar(s_hi, s_lo) on the joint mask of the two rows; (mean, usable fraction)."""
    d = dict(rows)
    bm = joint_mask(BSTAR, d[s_lo][fld], d[s_hi][fld])
    if len(bm) < 2:
        return float("nan"), len(bm) / len(BSTAR)
    return float((a_norm(d[s_hi][fld], bm) - a_norm(d[s_lo][fld], bm)).mean()), len(bm) / len(BSTAR)

def rows_at(files, t):
    return sorted(((sig_of(f), spectra_at(f, t)) for f in files), key=lambda q: q[0])

def obs_at(side, tm, key, transform=lambda v, i, z: np.asarray(v, float)):
    """log-linear interpolation of transform(z[key][i]) between the bracketing records (spectra_at's rule)."""
    z = np.load(side); t = np.asarray(z["t"]); i, j = bracket(t, tm)
    w = (tm - t[i]) / (t[j] - t[i]) if t[j] > t[i] else 0.0
    lo = np.maximum(transform(z[key][i], i, z), 1e-300); hi = np.maximum(transform(z[key][j], j, z), 1e-300)
    return np.exp((1 - w) * np.log(lo) + w * np.log(hi)), i, j, w

# ------------------------------------------------------------------ R-0
def stage_check():
    print("== R-0: the reproduction obligation — s35_strip.stage_cols() / stage_npair() re-run today, stdout DIFFED against the S35 records ==")
    ok_all = True
    for fn, stage in (("s35_strip_cols.txt", S.stage_cols), ("s35_strip_npair.txt", S.stage_npair)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stage()
        # trailing empty lines stripped on both sides: the S35 main printed a blank line AFTER each stage
        # (s35_strip.py: `stage_cols(); print()`), which the captured stage output cannot contain
        def strip_tail(lines):
            while lines and lines[-1].strip() == "":
                lines = lines[:-1]
            return lines
        mine = strip_tail(buf.getvalue().splitlines())
        rec = strip_tail(open(fn, encoding="utf-8").read().splitlines())
        nd = sum(1 for a, b in zip(mine, rec) if a != b) + abs(len(mine) - len(rec))
        print("   %-22s record %4d lines | today %4d lines | differing lines %d -> %s" % (fn, len(rec), len(mine), nd, "REPRODUCED (every printed digit)" if nd == 0 else "NOT-REPRODUCED"))
        if nd:
            ok_all = False
            for a, b in zip(mine, rec):
                if a != b:
                    print("      record: %s\n      today : %s" % (b[:160], a[:160])); break
    print("   R-0 LETTER: %s" % ("REPRODUCED" if ok_all else "NOT-REPRODUCED (STOP)"))
    return ok_all

# ------------------------------------------------------------------ R-1
def magnitude_letter(vals, tol):
    """vals = |S_bar| at the grid samples in INCREASING t (last = t_m); tol = the like-for-like timing uncertainty."""
    v = np.asarray(vals, float)
    if np.any(np.isnan(v)):
        return "UNDEFINED(floor)"
    ratio = v[0] / v[-1]
    nonincr = all(v[j + 1] <= v[j] + tol for j in range(len(v) - 1))
    nondecr = all(v[j + 1] >= v[j] - tol for j in range(len(v) - 1))
    if ratio >= 1.10 and nonincr:
        return "GROWING-BACKWARD"
    if np.all(np.abs(v / v[-1] - 1.0) <= 0.10):
        return "FLAT"
    if ratio <= 0.90 and nondecr:
        return "SHRINKING-BACKWARD"
    return "NON-MONOTONE-IN-t"

def stage_tc():
    print("== R-1: THE TIME COURSE of the sigma-separation on B*, every row at the SAME grid time; like-for-like timing shift +- h/2 ==")
    for key, (label, files) in COLS.items():
        tm = TM.get(key, min(float(np.load(f)["tcert"]) for f in files))
        h = np.median([cadence(f) for f in files])
        sigs = sorted(sig_of(f) for f in files)
        for fld in ("omega", "theta"):
            ts = window(files, fld)
            grid = [ts + j * (tm - ts) / 8.0 for j in range(9)]
            print(" -- %s / %s : window [%.6f, %.6f] (t_m = %.6f, cadence h = %.2e, shift +- %.2e); column pair (%.4g -> %.4g) --" % (label, fld, ts, tm, tm, h, h / 2, sigs[0], sigs[-1]))
            print("    %-9s %-42s %-9s %-24s %-7s %-6s" % ("t", "ORDER letter (model-free, floor rule)", "S_col", "[shift -h/2, +h/2]", "usable", "dlnS/dlnt"))
            letters = []; mags = []; tols = []; prev = None
            for t in grid:
                rows = rows_at(files, t)
                let, det = ordering(rows, fld, BSTAR)
                sc, use = col_sep(rows, fld, sigs[0], sigs[-1])
                scl, _ = col_sep(rows_at(files, t - h / 2), fld, sigs[0], sigs[-1])
                sch, _ = col_sep(rows_at(files, t + h / 2), fld, sigs[0], sigs[-1])
                rng = (min(scl, sch), max(scl, sch))
                dl = ""
                if prev is not None and not (np.isnan(sc) or np.isnan(prev[1])):
                    dl = "%+.2f" % ((math.log(abs(sc)) - math.log(abs(prev[1]))) / (math.log(t) - math.log(prev[0])))
                short = let if not let.startswith("UNDEFINED") else "UNDEFINED(floor)"
                print("    %.6f %-42s %+8.4f  [%+8.4f, %+8.4f]  %4.2f   %s" % (t, short, sc, rng[0], rng[1], use, dl))
                letters.append(short); mags.append(abs(sc) if use >= 0.5 else float("nan")); tols.append(abs(rng[1] - rng[0]) / 2 if not np.isnan(sc) else 0.0)
                prev = (t, sc)
            # adjacent pairs at the window start and at t_m (descriptive)
            for tt, nm in ((grid[0], "start"), (grid[-1], "t_m")):
                rows = rows_at(files, tt); _, det = ordering(rows, fld, BSTAR)
                print("    adjacent-pair S_bar at %s: %s" % (nm, " / ".join("%+.4f (%.4g->%.4g, use %.2f, %s)" % (m, s1, s2, use, flag) for s1, s2, m, sd, neg, pos, lo, hi, flag, use in det)))
            tol = max(tols) if tols else 0.0
            mag = magnitude_letter(mags, tol)
            defined = [l for l in letters if l != "UNDEFINED(floor)"]
            if len(defined) == 0 or (len(defined) == 1 and letters[-1] != "UNDEFINED(floor)" and all(l == "UNDEFINED(floor)" for l in letters[:-1])):
                order = "UNDEFINED(floor)"
            elif all(l == "MONOTONE" for l in defined):
                order = "ORDER-STABLE" + ("" if len(defined) == len(letters) else " (defined at %d of 9 samples)" % len(defined))
            else:
                order = "ORDER-BREAKS"
            ratio = mags[0] / mags[-1] if not (np.isnan(mags[0]) or np.isnan(mags[-1])) else float("nan")
            print("    R-1 LETTERS (%s, %s): MAGNITUDE %s (|S_col(start)|/|S_col(t_m)| = %.3f; timing tol %.4f) ; ORDER %s\n" % (label, fld, mag, ratio, tol, order))

# ------------------------------------------------------------------ labeled DIAGNOSTIC on R-1 (re-scores nothing)
def stage_tcfix():
    print("== LABELED DIAGNOSTIC on R-1 (re-scores nothing): the column separation on a FIXED mode set M0 = the joint mask of all rows at the window start, at every grid sample ==")
    print("   (the registered letter's band mean runs over the joint mask AT EACH SAMPLE, which grows from ~half of B* at the window start to all of B*; here the mode set is frozen)")
    for key, (label, files) in COLS.items():
        tm = TM.get(key, min(float(np.load(f)["tcert"]) for f in files))
        h = np.median([cadence(f) for f in files]); sigs = sorted(sig_of(f) for f in files)
        for fld in ("omega", "theta"):
            ts = window(files, fld); grid = [ts + j * (tm - ts) / 8.0 for j in range(9)]
            rows0 = rows_at(files, ts); d0 = dict(rows0)
            M0 = joint_mask(BSTAR, *[d0[s][fld] for s in sigs])
            vals = []
            for t in grid:
                d = dict(rows_at(files, t)); m = joint_mask(M0, d[sigs[0]][fld], d[sigs[-1]][fld])
                vals.append(float((a_norm(d[sigs[-1]][fld], m) - a_norm(d[sigs[0]][fld], m)).mean()) if len(m) == len(M0) else float("nan"))
            v = np.abs(np.array(vals)); mono = all(v[j + 1] <= v[j] + 1e-9 for j in range(len(v) - 1))
            print("   %-28s %-6s M0 = %d modes (%d..%d): |S_col| on M0 at the 9 samples: %s ; ratio start/t_m %.3f ; non-increasing in t: %s"
                  % (label, fld, len(M0), M0[0], M0[-1], " ".join("%.4f" % x for x in v), v[0] / v[-1], "YES" if mono else "no"))

# ------------------------------------------------------------------ R-2
def stage_xdep():
    print("== R-2: THE X-DEPENDENCE of S_bar(1.5, 0.875) — at each column's OWN t_m (the S35 record restated) and at the COMMON time t_c = %.6f ==" % TC)
    own = {}; com = {}
    for key in ("X2", "X3", "X4"):
        label, files = COLS[key]
        h = np.median([cadence(f) for f in files])
        for tag, t in (("own t_m", TM[key]), ("common t_c", TC)):
            rows = rows_at(files, t)
            so, uo = col_sep(rows, "omega", 0.875, 1.5)
            st, ut = col_sep(rows, "theta", 0.875, 1.5)
            sol, _ = col_sep(rows_at(files, t - h / 2), "omega", 0.875, 1.5); soh, _ = col_sep(rows_at(files, t + h / 2), "omega", 0.875, 1.5)
            print("    %s @ %-10s t = %.6f : omega S_bar(1.5, 0.875) = %+.4f [shift %+.4f, %+.4f] (usable %.2f) -> product with X/k_d^2 = %+.4f | theta S_bar(1.5, 0.875) = %+.4f (usable %.2f%s)"
                  % (key, tag, t, so, min(sol, soh), max(sol, soh), uo, so * XK[key], st, ut, "" if ut >= 0.5 else " -> FLOOR"))
            (own if tag == "own t_m" else com)[key] = (so, so * XK[key], st, ut)
    v = np.array([com[k][0] for k in ("X2", "X3", "X4")]); p = np.array([com[k][1] for k in ("X2", "X3", "X4")])
    dev = np.abs(v / v.mean() - 1).max(); pdev = np.abs(p / p.mean() - 1).max()
    if dev <= 0.10:
        let = "X-FLAT"
    elif (np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0)) and (v.max() - v.min()) / abs(v.mean()) > 0.10:
        let = "X-ORDERED"
    else:
        let = "MIXED"
    print("    R-2 LETTER (omega S_bar(1.5, 0.875) at t_c across X2/X3/X4): %s (max |dev from mean| = %.1f%%; values %s)" % (let, 100 * dev, " / ".join("%+.4f" % x for x in v)))
    print("    R-2 product test at t_c: %s (max |dev from mean| = %.1f%%; products %s)" % ("CONSTANT" if pdev <= 0.10 else "VARIES", 100 * pdev, " / ".join("%+.4f" % x for x in p)))
    po = np.array([own[k][1] for k in ("X2", "X3", "X4")])
    print("    (the own-t_m products, restated: %s ; spread %.1f%% — the A-2 receipt)" % (" / ".join("%+.4f" % x for x in po), 100 * (po.max() - po.min()) / abs(po.mean())))

# ------------------------------------------------------------------ R-3
def pair_files():
    f512 = {sig_of(f): f for f in COLS["X3"][1] if sig_of(f) in (0.875, 1.375)}
    f1024 = {sig_of(f): f for f in COLS["PAIR1024"][1] if sig_of(f) in (0.875, 1.375)}
    return f512, f1024

def stage_dnk():
    tm = TM["X3"]
    print("== R-3: the N-difference d(k) = a_1024(k) - a_512(k) on B* at t_m = %.6f as a LINEAR FIT d = c0 + c1 k (report; formal sigmas printed) ==" % tm)
    f512, f1024 = pair_files()
    for fld in ("theta", "omega"):
        for s in (0.875, 1.375):
            a = spectra_at(f512[s], tm); b = spectra_at(f1024[s], tm)
            bm = joint_mask(BSTAR, a[fld], b[fld])
            d = a_norm(b[fld], bm) - a_norm(a[fld], bm); kk = np.array(bm, float)
            A = np.vstack([np.ones_like(kk), kk]).T
            c, rms, r2 = lsq(A, d)
            cov = np.linalg.inv(A.T @ A) * rms ** 2 * len(kk) / max(len(kk) - 2, 1)
            print("    %-5s sigma=%.4g : slope %+.5f ± %.5f per mode, intercept %+.4f ± %.4f, R2 %.4f, rms %.4f (n=%d) ; D_N = max|d| = %.4f" % (fld, s, c[1], math.sqrt(cov[1, 1]), c[0], math.sqrt(cov[0, 0]), r2, rms, len(kk), float(np.abs(d).max())))

# ------------------------------------------------------------------ R-4 / R-5 / R-6
def stage_wall():
    tm = TM["X3"]
    print("== R-4 / R-5 / R-6: the wall-omega and far-wall-theta discriminators and the maxima's location, the X3 pair at t_m = %.6f ==" % tm)
    f512, f1024 = pair_files()
    for s in (0.875, 1.375):
        out = {}
        for N, f in ((512, f512[s]), (1024, f1024[s])):
            wall, i, j, w = obs_at(f, tm, "Ew_wall", lambda v, i, z: np.asarray(v, float) / max(float(z["Ew_max"][i]), 1e-300))
            top, _, _, _ = obs_at(f, tm, "th_top", lambda v, i, z: wall_amp_from_row(v))
            bot, _, _, _ = obs_at(f, tm, "th_bot", lambda v, i, z: wall_amp_from_row(v))
            ew, _, _, _ = obs_at(f, tm, "Ew")
            out[N] = dict(wall=wall, top=top, bot=bot, ew=ew, ampratio=float(top[kd_index] / max(bot[kd_index], 1e-300)), rec=(i, j, w))
        # R-4
        bm = joint_mask(BSTAR, out[512]["wall"], out[1024]["wall"])
        dwall = a_norm(out[1024]["wall"], bm) - a_norm(out[512]["wall"], bm); DN = float(np.abs(dwall).max())
        let = "FIELD-CONVERGED" if DN <= 0.05 else ("PARTIAL" if DN <= 0.30 else "NOT-CONVERGED")
        print("    R-4 sigma=%.4g omega AT THE WALL: D_N on B* = %.4f (mean %+.4f; at k=%d %+.4f, k=%d %+.4f; usable %d/%d) -> %s" % (s, DN, dwall.mean(), bm[0], dwall[0], bm[-1], dwall[-1], len(bm), len(BSTAR), let))
        # R-5
        ar = (out[512]["ampratio"], out[1024]["ampratio"])
        if min(ar) < 1e-8:
            print("    R-5 sigma=%.4g theta at the FAR wall: amp_top(k_d)/amp_bot(k_d) = %.2e / %.2e (N512 / N1024) -> UNDEFINED(amp)" % (s, ar[0], ar[1]))
        else:
            bm2 = joint_mask(BSTAR, out[512]["top"], out[1024]["top"])
            if len(bm2) < len(BSTAR) / 2:
                print("    R-5 sigma=%.4g theta at the FAR wall: amp ratio %.2e / %.2e ; usable %d/%d -> UNDEFINED(floor)" % (s, ar[0], ar[1], len(bm2), len(BSTAR)))
            else:
                dtop = a_norm(out[1024]["top"], bm2) - a_norm(out[512]["top"], bm2); DN2 = float(np.abs(dtop).max())
                let2 = "FIELD-CONVERGED" if DN2 <= 0.05 else ("PARTIAL" if DN2 <= 0.30 else "NOT-CONVERGED")
                print("    R-5 sigma=%.4g theta at the FAR wall: amp_top(k_d)/amp_bot(k_d) = %.2e / %.2e ; D_N on B* = %.4f (usable %d/%d) -> %s" % (s, ar[0], ar[1], DN2, len(bm2), len(BSTAR), let2))
        # R-6
        for N in (512, 1024):
            r = out[N]["wall"][BSTAR] / np.maximum(out[N]["ew"][BSTAR], 1e-300)
            print("    R-6 sigma=%.4g N=%-4d Ew_wall/(Ew*Ew_max) on B*: min %.4f  median %.4f  max %.4f  (1 = the wall row carries the max-over-y omega spectrum)" % (s, N, r.min(), np.median(r), r.max()))

# ------------------------------------------------------------------ U-0
def sig_of_e(f):
    """s35_strip.sig_of knows the p3_/p3d_ prefixes only; the p3e_ side files carry the same tag rule."""
    return sig_of(f.replace("p3e_", "p3d_", 1))

def e_files():
    e512 = {sig_of_e(f): f for f in glob.glob("p3e_s*_x853e_*_N512.npz")}
    e1024 = {sig_of_e(f): f for f in glob.glob("p3e_s*_x853e1024_*_N1024.npz")}
    return e512, e1024

def stage_u0():
    print("== U-0: the extended-stash rides vs their banked partners (EVERY key) and vs the S35 side files (EVERY key); the alignment identities ==")
    nbad = 0; n = 0
    for fresh_tag, banked_tag, s35_tag in (("x853e", "x853n", "x853d"), ("x853e1024", "x853n1024", "x853d1024")):
        for f in sorted(glob.glob("p3_s*_%s_k*_N*.npz" % fresh_tag), key=sig_of):
            b = f.replace("_%s_" % fresh_tag, "_%s_" % banked_tag)
            a = np.load(f); z = np.load(b)
            ok1, bad1 = keys_equal(a, z)
            d_new = f.replace("p3_", "p3d_", 1); d_old = d_new.replace("_%s_" % fresh_tag, "_%s_" % s35_tag)
            zn = np.load(d_new); zo = np.load(d_old)
            ok2, bad2 = keys_equal(zn, zo)
            bad2 = [k for k in bad2 if k != "source_p3"]      # the provenance string names the fresh tag by design
            ok2 = len(bad2) == 0
            e = np.load(f.replace("p3_", "p3e_", 1)); N = int(a["N"])
            ok3 = bool(e["align_ok"]) and np.array_equal(e["th_bot"][-1], a["th_final"][N]) and np.array_equal(e["th_bot"], zn["th_bot"])
            n += 1; nbad += 0 if (ok1 and ok2 and ok3) else 1
            print("   sigma=%-6.4g %-10s %-10s t_cert %.6f Om %.4f g %.4f itx %.4f it %d | p3 keys equal to the banked row: %s%s | p3d keys equal to the S35 side file: %s%s | extended stash aligned: %s"
                  % (sig_of(f), fresh_tag, str(a["verdict"]), float(a["tcert"]), float(a["om_pk"]), float(a["gcert"]), float(a["itx"]), int(np.asarray(a["every_t"]).size),
                     ok1, "" if ok1 else " (differs: %s)" % ",".join(bad1), ok2, "" if ok2 else " (differs: %s)" % ",".join(bad2), ok3))
    print("   %d rows, %d non-identical -> U-0 LETTER: %s" % (n, nbad, "BIT-IDENTICAL x%d" % n if (nbad == 0 and n == 4) else "NOT-IDENTICAL (STOP)" if nbad else "INCOMPLETE (%d of 4 rows)" % n))
    return nbad == 0 and n == 4

# ------------------------------------------------------------------ U-1 / U-2
def stage_u1():
    tm = TM["X3"]
    print("== U-1 / U-2: where the N512 theta defect lives — the X3 pair at t_m = %.6f on B* ==" % tm)
    e512, e1024 = e_files(); f512, f1024 = pair_files()
    res = {}
    for s in (0.875, 1.375):
        vals = {}
        obs = {}
        for N, fe, fd in ((512, e512[s], f512[s]), (1024, e1024[s], f1024[s])):
            obs[(N, "wall")] = obs_at(fd, tm, "th_bot", lambda v, i, z: wall_amp_from_row(v))[0]
            obs[(N, "max-over-y")] = obs_at(fe, tm, "Eth")[0]
            obs[(N, "y=0.05")] = obs_at(fe, tm, "th_y005", lambda v, i, z: wall_amp_from_row(v))[0]
            obs[(N, "y=0.01")] = obs_at(fe, tm, "th_y001", lambda v, i, z: wall_amp_from_row(v))[0]
        for nm in ("wall", "max-over-y", "y=0.05", "y=0.01"):
            bm = joint_mask(BSTAR, obs[(512, nm)], obs[(1024, nm)])
            d = a_norm(obs[(1024, nm)], bm) - a_norm(obs[(512, nm)], bm)
            vals[nm] = float(np.abs(d).max()) if len(bm) >= len(BSTAR) / 2 else float("nan")
            print("    sigma=%.4g %-11s D_N = %.4f (mean %+.4f; usable %d/%d)" % (s, nm, vals[nm], d.mean() if len(bm) else float("nan"), len(bm), len(BSTAR)))
        res[s] = vals
    v = res[1.375]
    if v["wall"] > 0.30 and v["max-over-y"] <= 0.05 and v["y=0.05"] <= 0.05:
        let = "WALL-CONFINED"
    elif v["max-over-y"] > 0.30 and v["y=0.05"] > 0.30:
        let = "FIELD-WIDE"
    elif v["max-over-y"] > 0.30 and v["y=0.05"] <= 0.05:
        let = "LAYER"
    else:
        let = "MIXED"
    print("    U-1 LETTER (sigma = 1.375): %s   [control sigma = 0.875: wall %.4f, max-over-y %.4f, y=0.05 %.4f, y=0.01 %.4f]" % (let, res[0.875]["wall"], res[0.875]["max-over-y"], res[0.875]["y=0.05"], res[0.875]["y=0.01"]))
    # U-2: the argmax row course
    print("    U-2: the theta x-spectrum's argmax row on B* (min / median / max over the 26 modes; the wall row = N) at the R-1 theta grid times of the X3 column and the pair:")
    for s in (0.875, 1.375):
        for N, fe in ((512, e512[s]), (1024, e1024[s])):
            z = np.load(fe); t = np.asarray(z["t"]); iy = np.asarray(z["iy_th"]); iyw = np.asarray(z["iy_w"])
            ts = max(first_usable(f512[s], "theta"), first_usable(f1024[s], "theta"))
            grid = [ts + j * (tm - ts) / 8.0 for j in range(9)] + [float(z["tcert"])]
            line = []
            for g in grid:
                i = int(np.clip(np.searchsorted(t, g, side="right") - 1, 0, len(t) - 1))
                rows = iy[i][BSTAR]; roww = iyw[i][BSTAR]
                line.append("t=%.6f th %d/%d/%d w %d" % (t[i], rows.min(), int(np.median(rows)), rows.max(), int(np.median(roww))))
            print("      sigma=%.4g N=%-4d: %s" % (s, N, " | ".join(line)))

if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("check", "all"):
        ok = stage_check(); print()
        if not ok and st == "all":
            sys.exit("R-0 NOT-REPRODUCED: STOP")
    if st in ("tc", "all"):
        stage_tc(); print()
    if st in ("xdep", "all"):
        stage_xdep(); print()
    if st in ("dnk", "all"):
        stage_dnk(); print()
    if st in ("wall", "all"):
        stage_wall(); print()
    if st == "tcfix":
        stage_tcfix()
    if st == "u0":
        stage_u0()
    if st == "u1":
        stage_u1()
