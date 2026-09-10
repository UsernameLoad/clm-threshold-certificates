#!/usr/bin/env python
"""
s39_layer.py -- THE THETA-LAYER PROBE'S ANALYSIS on the S39 profile stash
(..U-3;.
IMPORTS s35_strip's keys_equal / bracket / wall_amp_from_row VERBATIM and adds
only the grid-profile observers.

Stages
  u0   U-0: every fresh p3 (x853f / x853f1024) vs its banked x853n / x853n1024
       partner (every key); the fresh p3d vs the S35 x853d side file (every key,
       source_p3 excluded); the fresh p3e vs the S36 x853e side file (every key,
       source_p3 / source_p3d excluded); the profile identities recorded in-ride
       (max_y pth == Eth[cols], argmax == iy_th[cols], the grid rows at y = 0.01 /
       0.05 vs the p3e Chebyshev rows) re-read from the p3f file; the last th_bot
       == th_final[Ny].
  u1   U-1: the layer letter at t_m = 0.004488, sigma = 1.375 (the 0.875 control
       beside); the timing spread at the bracketing records.
  u2   U-2: the thickness (e-fold / half depths) at t_m, both N, both sigma, theta
       and omega; the three-time course (descriptive).
  u3   U-3: the absolute-floor content extent at t_m.
  all  u0 + u1 + u2 + u3
"""
import sys, glob, math
import numpy as np
from s35_strip import keys_equal, bracket, BSTAR, FLOOR, kd_index, sig_of

sys.stdout.reconfigure(encoding="utf-8")
TM3 = 0.004488
DBAR = 0.30
UMIN = 13
FILES = {(0.875, 512): "p3f_s0p875_x853f_k19p414335_N512.npz",
         (1.375, 512): "p3f_s1p375_x853f_k2p235846_N512.npz",
         (0.875, 1024): "p3f_s0p875_x853f1024_k19p414335_N1024.npz",
         (1.375, 1024): "p3f_s1p375_x853f1024_k2p235846_N1024.npz"}
SHOW_Y = [0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]


# ------------------------------------------------------------------ U-0
def stage_u0():
    print("== U-0: the profile-stash rides vs their banked partners (EVERY key of p3 / p3d / p3e) and the in-ride profile identities ==")
    n = 0; nbad = 0
    for (fresh, banked, s35, s36) in (("x853f", "x853n", "x853d", "x853e"), ("x853f1024", "x853n1024", "x853d1024", "x853e1024")):
        for f in sorted(glob.glob("p3_s*_%s_k*_N*.npz" % fresh), key=sig_of):
            b = f.replace("_%s_" % fresh, "_%s_" % banked)
            a = np.load(f); z = np.load(b)
            ok1, bad1 = keys_equal(a, z)
            d_new = f.replace("p3_", "p3d_", 1); d_old = d_new.replace("_%s_" % fresh, "_%s_" % s35)
            ok2, bad2 = keys_equal(np.load(d_new), np.load(d_old)); bad2 = [k for k in bad2 if k != "source_p3"]; ok2 = len(bad2) == 0
            e_new = f.replace("p3_", "p3e_", 1); e_old = e_new.replace("_%s_" % fresh, "_%s_" % s36)
            ok3, bad3 = keys_equal(np.load(e_new), np.load(e_old)); bad3 = [k for k in bad3 if k not in ("source_p3", "source_p3d")]; ok3 = len(bad3) == 0
            pf = np.load(f.replace("p3_", "p3f_", 1)); N = int(a["N"])
            ok4 = bool(pf["align_ok"]) and bool(pf["eth_identity"]) and bool(pf["iy_identity"]) and float(pf["row001_reldiff"]) <= 1e-12 and float(pf["row005_reldiff"]) <= 1e-12 \
                and np.array_equal(pf["th_bot"][-1], a["th_final"][N]) and np.array_equal(pf["th_bot"], np.load(e_new)["th_bot"])
            n += 1; nbad += 0 if (ok1 and ok2 and ok3 and ok4) else 1
            print("   sigma=%-6.4g %-10s %-10s t_cert %.6f Om %.4f g %.4f itx %.4f it %d | p3 == banked: %s%s | p3d == S35 side: %s%s | p3e == S36 side: %s%s | profile identities: %s (Eth %s, iy %s, rows %.1e / %.1e)"
                  % (sig_of(f), fresh, str(a["verdict"]), float(a["tcert"]), float(a["om_pk"]), float(a["gcert"]), float(a["itx"]), int(np.asarray(a["every_t"]).size),
                     ok1, "" if ok1 else " (differs: %s)" % ",".join(bad1), ok2, "" if ok2 else " (differs: %s)" % ",".join(bad2),
                     ok3, "" if ok3 else " (differs: %s)" % ",".join(bad3), ok4, bool(pf["eth_identity"]), bool(pf["iy_identity"]), float(pf["row001_reldiff"]), float(pf["row005_reldiff"])))
    print("   %d rows, %d non-identical -> U-0 LETTER: %s" % (n, nbad, "BIT-IDENTICAL x%d" % n if (nbad == 0 and n == 4) else "NOT-IDENTICAL (STOP)" if nbad else "INCOMPLETE (%d of 4 rows)" % n))
    return nbad == 0 and n == 4


# ------------------------------------------------------------------ helpers
def grid_amp_at(pf, tm, key="gth"):
    """|mode| on the grid rows at tm: the log-linear interpolation of the two bracketing records (the S35 rule)."""
    z = np.load(pf); t = np.asarray(z["t"]); i, j = bracket(t, tm)
    w = (tm - t[i]) / (t[j] - t[i]) if t[j] > t[i] else 0.0
    lo = np.maximum(np.abs(np.asarray(z[key][i])), 1e-300); hi = np.maximum(np.abs(np.asarray(z[key][j])), 1e-300)
    amp = np.exp((1 - w) * np.log(lo) + w * np.log(hi))
    return dict(amp=amp, lo=lo, hi=hi, t_lo=float(t[i]), t_hi=float(t[j]), w=float(w), yg=np.asarray(z["yg"]), N=int(z["N"]), sigma=float(z["sigma"]), tcert=float(z["tcert"]))


def dn_profile(A5, A10):
    """per grid row: u(y) = modes above the row's own-driver mask at BOTH N; D_N(y) defined iff u >= UMIN."""
    yg = A5["yg"]; assert np.array_equal(yg, A10["yg"])
    a5 = A5["amp"]; a10 = A10["amp"]
    out = []
    for g in range(len(yg)):
        m5 = a5[g, 1:] > FLOOR * a5[g, 0]; m10 = a10[g, 1:] > FLOOR * a10[g, 0]
        u = m5 & m10; cnt = int(u.sum())
        if cnt >= UMIN:
            d = np.log(a10[g, 1:][u] / a10[g, 0]) - np.log(a5[g, 1:][u] / a5[g, 0])
            out.append((float(yg[g]), cnt, float(np.abs(d).max()), float(d.mean())))
        else:
            out.append((float(yg[g]), cnt, float("nan"), float("nan")))
    return out


def letter_from(prof):
    defined = [(y, c, d, m) for (y, c, d, m) in prof if not math.isnan(d)]
    n_int = sum(1 for (y, c, d, m) in defined if y > 0)
    d0 = next((d for (y, c, d, m) in defined if y == 0.0), float("nan"))
    if n_int < 3 or math.isnan(d0):
        return "UNDEFINED(floor)", d0, float("nan"), float("nan")
    yL = max(y for (y, c, d, m) in defined)
    over = [y for (y, c, d, m) in defined if y <= yL and d > DBAR]
    yD = max(over) if over else float("nan")
    if d0 <= DBAR:
        L = "NONE"
    elif (not math.isnan(yD)) and yD >= 0.5 * yL:
        L = "LAYER"
    elif math.isnan(yD) or yD < 0.1 * yL:
        L = "WALL-CONFINED"
    else:
        L = "MIXED"
    return L, d0, yD, yL


def show_profile(prof, label):
    print("   %s: y | u(y) modes usable at both N | D_N(y) = max|a_1024 - a_512| | mean(a_1024 - a_512)" % label)
    for (y, c, d, m) in prof:
        if any(abs(y - s) <= 1e-12 * max(1.0, s) for s in SHOW_Y) or (not math.isnan(d) and d > DBAR and y > 0 and y in [p[0] for p in prof if not math.isnan(p[2])][-2:]):
            print("      y=%-9.3g u=%2d  D_N=%s  mean=%s" % (y, c, ("%.4f" % d) if not math.isnan(d) else "  nan ", ("%+.4f" % m) if not math.isnan(m) else "  nan "))


# ------------------------------------------------------------------ U-1
def stage_u1():
    print("== U-1: the layer letter at t_m = %g on B* (per-row own-driver normalisation; D_N defined iff >= %d modes usable at both N; the bar %.2f) ==" % (TM3, UMIN, DBAR))
    res = {}
    for s in (1.375, 0.875):
        A5 = grid_amp_at(FILES[(s, 512)], TM3); A10 = grid_amp_at(FILES[(s, 1024)], TM3)
        print("   sigma=%g: N512 records bracket [%.6f, %.6f] w=%.3f ; N1024 [%.6f, %.6f] w=%.3f ; t_cert %.6f / %.6f" % (s, A5["t_lo"], A5["t_hi"], A5["w"], A10["t_lo"], A10["t_hi"], A10["w"], A5["tcert"], A10["tcert"]))
        prof = dn_profile(A5, A10)
        show_profile(prof, "sigma=%g at t_m (interpolated)" % s)
        L, d0, yD, yL = letter_from(prof)
        n_def = sum(1 for p in prof if not math.isnan(p[2]))
        # the timing spread: both bracketing records
        spread = []
        for tag in ("lo", "hi"):
            B5 = dict(A5); B5["amp"] = A5[tag]; B10 = dict(A10); B10["amp"] = A10[tag]
            Lb, d0b, yDb, yLb = letter_from(dn_profile(B5, B10)); spread.append((tag, Lb, d0b, yDb, yLb))
        print("   sigma=%g: definable rows %d ; D_N(0) = %.4f ; y_L = %.3e ; y_D = %s ; y_D/y_L = %s -> %s%s"
              % (s, n_def, d0, yL, ("%.3e" % yD) if not math.isnan(yD) else "nan", ("%.3f" % (yD / yL)) if not (math.isnan(yD) or math.isnan(yL)) else "nan",
                 ("U-1 LETTER: " if s == 1.375 else "control: ") + L, "" if s == 1.375 else " (expected D_N <= 0.05 wherever defined)"))
        print("   sigma=%g: at the bracketing records: " % s + " ; ".join("%s -> %s (D_N(0) %.4f, y_D %s, y_L %.3e)" % (tag, Lb, d0b, ("%.3e" % yDb) if not math.isnan(yDb) else "nan", yLb) for (tag, Lb, d0b, yDb, yLb) in spread))
        res[s] = (L, d0, yD, yL)
    return res


# ------------------------------------------------------------------ U-2 / U-3
def depths(A, key_amp, frac):
    """per B* mode: the smallest grid y (> 0) at which |mode| <= frac * |mode at the wall|; nan if never within the grid; usable = above the wall mask."""
    amp = key_amp; yg = A["yg"]
    out = []
    for c in range(1, amp.shape[1]):
        wall = amp[0, c]
        if not (wall > FLOOR * amp[0, 0]):
            out.append(float("nan")); continue
        idx = np.where(amp[1:, c] <= frac * wall)[0]
        out.append(float(yg[1 + idx[0]]) if len(idx) else float("nan"))
    return np.array(out)


def stage_u2():
    print("== U-2 (report): the layer's thickness at t_m = %g -- per-mode e-fold and half depths from the wall value on the fixed grid; medians over the usable modes; the N-ratio ==" % TM3)
    med = {}
    for s in (1.375, 0.875):
        for fld, key in (("theta", "gth"), ("omega", "gw")):
            for N in (512, 1024):
                A = grid_amp_at(FILES[(s, N)], TM3, key)
                ye = depths(A, A["amp"], 1.0 / math.e); yh = depths(A, A["amp"], 0.5)
                ok = ~np.isnan(ye)
                med[(s, fld, N)] = (float(np.median(ye[ok])) if ok.any() else float("nan"), float(np.median(yh[~np.isnan(yh)])) if (~np.isnan(yh)).any() else float("nan"))
                print("   sigma=%-6g %-5s N=%-5d usable modes %2d/26 : y_e median %.3e (min %.3e, max %.3e) ; y_h median %.3e ; y_e at k=52/76/102: %s"
                      % (s, fld, N, int(ok.sum()), med[(s, fld, N)][0], float(np.nanmin(ye)) if ok.any() else float("nan"), float(np.nanmax(ye)) if ok.any() else float("nan"),
                         med[(s, fld, N)][1], " / ".join("%.2e" % ye[[0, 12, 25][i]] for i in range(3))))
            r = med[(s, fld, 512)][0] / med[(s, fld, 1024)][0] if med[(s, fld, 1024)][0] > 0 else float("nan")
            print("   sigma=%-6g %-5s : N-ratio of the median e-fold depth y_e(512)/y_e(1024) = %.3f ; half-depth ratio %.3f" % (s, fld, r, med[(s, fld, 512)][1] / med[(s, fld, 1024)][1]))
    print("   the three-time course of the median theta e-fold depth (descriptive; fixed times, no form):")
    for s in (1.375, 0.875):
        for N in (512, 1024):
            vals = []
            for tt in (0.0036, 0.0040, TM3):
                A = grid_amp_at(FILES[(s, N)], tt, "gth"); ye = depths(A, A["amp"], 1.0 / math.e); ok = ~np.isnan(ye)
                vals.append("t=%.4f y_e=%.3e (%d modes)" % (tt, float(np.median(ye[ok])) if ok.any() else float("nan"), int(ok.sum())))
            print("      sigma=%-6g N=%-5d : %s" % (s, N, " ; ".join(vals)))


def stage_u3():
    print("== U-3 (report): the absolute-floor content extent y_A at t_m = %g -- the largest grid y at which >= %d B* modes exceed 1e-12 x the WALL's driver amplitude ==" % (TM3, UMIN))
    for s in (1.375, 0.875):
        for N in (512, 1024):
            A = grid_amp_at(FILES[(s, N)], TM3, "gth"); amp = A["amp"]; yg = A["yg"]
            wall_kd = amp[0, 0]
            cnt = (amp[:, 1:] > FLOOR * wall_kd).sum(axis=1)
            yA13 = float(yg[cnt >= UMIN].max()) if (cnt >= UMIN).any() else float("nan")
            yA1 = float(yg[cnt >= 1].max()) if (cnt >= 1).any() else float("nan")
            # the row-normalised definable extent for comparison
            cntr = np.array([int((amp[g, 1:] > FLOOR * amp[g, 0]).sum()) for g in range(len(yg))])
            yR13 = float(yg[cntr >= UMIN].max()) if (cntr >= UMIN).any() else float("nan")
            print("   sigma=%-6g N=%-5d : y_A(>=13 modes above 1e-12 x wall driver) = %.3e ; y_A(>=1 mode) = %.3e ; [own-driver mask, >= 13 modes: %.3e] ; modes at the wall %d/26"
                  % (s, N, yA13, yA1, yR13, int(cnt[0])))


# ------------------------------------------------------------------ U-1 under ONE common shift (the S38 rule; a REPORT added after the letter fired -- the letter at 0 stands)
def stage_u1s():
    print("== U-1 robustness under ONE common timing shift (the S38 rule): both N interpolated at t_m -+ delta, delta = half the coarsest record cadence among the four rows; the letter at 0 decides (as fired) ==")
    cad = []
    for key, f in FILES.items():
        t = np.asarray(np.load(f)["t"]); cad.append(float(np.median(np.diff(t))))
    delta = 0.5 * max(cad)
    print("   record cadences (median dt) N512/N1024: %s ; common shift delta = %.3e" % (", ".join("%.2e" % c for c in cad), delta))
    for s in (1.375, 0.875):
        for tt, tag in ((TM3 - delta, "-delta"), (TM3, "0"), (TM3 + delta, "+delta")):
            A5 = grid_amp_at(FILES[(s, 512)], tt); A10 = grid_amp_at(FILES[(s, 1024)], tt)
            prof = dn_profile(A5, A10); L, d0, yD, yL = letter_from(prof)
            inside = tt <= min(A5["tcert"], A10["tcert"])
            means = {y: m for (y, c, d, m) in prof}
            print("   sigma=%-6g %-7s t=%.6f (inside both certified windows: %s): D_N(0) = %.4f ; y_L = %.3e ; y_D = %s ; y_D/y_L = %s -> %s ; D_N at y = 1e-4 / 2e-3 / 3e-3 / 5e-3: %s ; mean diff at the same rows: %s"
                  % (s, tag, tt, inside, d0, yL, ("%.3e" % yD) if not math.isnan(yD) else "nan", ("%.3f" % (yD / yL)) if not (math.isnan(yD) or math.isnan(yL)) else "nan", L,
                     " / ".join(("%.3f" % d) if not math.isnan(d) else "nan" for (y, c, d, m) in prof if any(abs(y - q) <= 1e-12 for q in (1e-4, 2e-3, 3e-3, 5e-3))),
                     " / ".join(("%+.3f" % m) if not math.isnan(m) else "nan" for (y, c, d, m) in prof if any(abs(y - q) <= 1e-12 for q in (1e-4, 2e-3, 3e-3, 5e-3)))))


def depths_interp(A, amp, frac):
    """the e-fold / half depth with the crossing log-interpolated in y between the two grid rows that bracket it (a continuous refinement of the registered on-grid value; report)."""
    yg = A["yg"]; out = []
    for c in range(1, amp.shape[1]):
        wall = amp[0, c]
        if not (wall > FLOOR * amp[0, 0]):
            out.append(float("nan")); continue
        idx = np.where(amp[1:, c] <= frac * wall)[0]
        if not len(idx):
            out.append(float("nan")); continue
        g = 1 + idx[0]
        if g <= 1:
            out.append(float(yg[g])); continue
        y1, y2 = yg[g - 1], yg[g]; a1, a2 = math.log(amp[g - 1, c]), math.log(amp[g, c]); target = math.log(frac * wall)
        if y1 <= 0 or a1 == a2:
            out.append(float(y2)); continue
        w = (a1 - target) / (a1 - a2)
        out.append(float(math.exp(math.log(y1) + w * (math.log(y2) - math.log(y1)))))
    return np.array(out)


def stage_u2i():
    print("== U-2 refinement (report): the e-fold depth with the crossing log-interpolated between grid rows (the registered on-grid value quantises to the grid's 7.5% step) ==")
    for s in (1.375, 0.875):
        for fld, key in (("theta", "gth"), ("omega", "gw")):
            med = {}
            for N in (512, 1024):
                A = grid_amp_at(FILES[(s, N)], TM3, key); ye = depths_interp(A, A["amp"], 1.0 / math.e); ok = ~np.isnan(ye)
                med[N] = float(np.median(ye[ok])); print("   sigma=%-6g %-5s N=%-5d : y_e (interpolated) median %.3e (min %.3e, max %.3e) ; at k=52/76/102: %s"
                                                          % (s, fld, N, med[N], float(np.nanmin(ye)), float(np.nanmax(ye)), " / ".join("%.3e" % ye[i] for i in (0, 12, 25))))
            print("   sigma=%-6g %-5s : N-ratio (interpolated medians) y_e(512)/y_e(1024) = %.3f" % (s, fld, med[512] / med[1024]))


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st == "u1s":
        stage_u1s(); sys.exit(0)
    if st == "u2i":
        stage_u2i(); sys.exit(0)
    if st in ("u0", "all"):
        ok = stage_u0()
        if st == "all" and not ok:
            print("U-0 not BIT-IDENTICAL x4 -> STOP (no U-1 letter is read)"); sys.exit(1)
    if st in ("u1", "all"):
        stage_u1()
    if st in ("u2", "all"):
        stage_u2()
    if st in ("u3", "all"):
        stage_u3()
