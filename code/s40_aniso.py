#!/usr/bin/env python
"""
s40_aniso.py -- THE ANISOTROPIC-RESOLUTION DISCRIMINATOR'S ANALYSIS on the S40 p3g stash
(..U-3;.  IMPORTS s39_layer's
grid_amp_at / dn_profile / depths (and through it s35_strip's keys_equal / bracket) VERBATIM.

Stages
  g0   U-0: the square p3g rides (tag x853g, Nx = Ny = 512, both sigma) vs the S39 x853f files --
       every S39 key of p3 / p3d / p3e / p3f equal (the source_* strings excluded; the three
       added record fields Nx / Ny / grid reported as extras).
  g1   U-1: the discriminator letter at t_m = 0.004488, sigma = 1.375: the wall-row difference
       of each anisotropic row against the square 1024^2 reference on the fixed grid
       (own-driver normalisation; >= 13 usable modes), d_A (512x1024: coarse x, fine y) and
       d_B (1024x512: fine x, coarse y); the share r = d_B/(d_A + d_B); the letter; the
       square 512^2 defect recomputed beside (must reproduce 0.5724); the mean beside the max.
  g2   U-2 (report): the theta / omega e-fold depths of the anisotropic rows against the two
       square rows; the D profile through the layer; the control sigma = 0.875.
  all  g0 + g1 + g2
"""
import sys, glob, math
import numpy as np
from s35_strip import keys_equal, bracket, BSTAR, FLOOR, kd_index
from s39_layer import grid_amp_at, dn_profile, depths, TM3, UMIN

sys.stdout.reconfigure(encoding="utf-8")
DTOP = 0.20          # validity: at least one anisotropic row must carry this much of the defect
RLO, RHI = 0.2, 0.8  # the share bars
SQ = {(0.875, 512): "p3f_s0p875_x853f_k19p414335_N512.npz", (1.375, 512): "p3f_s1p375_x853f_k2p235846_N512.npz",
      (0.875, 1024): "p3f_s0p875_x853f1024_k19p414335_N1024.npz", (1.375, 1024): "p3f_s1p375_x853f1024_k2p235846_N1024.npz"}
AN = {(0.875, "A"): "p3f_s0p875_x853g_k19p414335_N512x1024.npz", (0.875, "B"): "p3f_s0p875_x853g_k19p414335_N1024x512.npz",
      (1.375, "A"): "p3f_s1p375_x853g_k2p235846_N512x1024.npz", (1.375, "B"): "p3f_s1p375_x853g_k2p235846_N1024x512.npz"}
GRID = {"A": "512x1024 (coarse x, fine y)", "B": "1024x512 (fine x, coarse y)"}


def keys_equal_common(a, z, ignore=("source_p3", "source_p3d", "source_p3e")):
    """every key of z (the S39 file) present in a and equal; extras in a reported."""
    missing = [k for k in z.files if k not in a.files]
    extras = [k for k in a.files if k not in z.files]
    bad = []
    for k in z.files:
        if k in ignore or k in missing:
            continue
        x = a[k]; y = z[k]
        if x.dtype.kind in "US":
            eq = str(x) == str(y)
        else:
            eq = x.shape == y.shape and (np.array_equal(x, y) or (np.issubdtype(x.dtype, np.floating) and np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)])))
        if not eq:
            bad.append(k)
    return (len(bad) == 0 and len(missing) == 0), bad, missing, extras


def stage_g0():
    print("== U-0: the p3g square rides (x853g, Nx = Ny = 512) vs the S39 x853f files -- every S39 key of p3 / p3d / p3e / p3f ==")
    n = 0; nbad = 0
    for s, kt in ((0.875, "k19p414335"), (1.375, "k2p235846")):
        st = ("s%s" % ("%.3f" % s).replace(".", "p"))
        row_ok = True; parts = []
        for pre in ("p3", "p3d", "p3e", "p3f"):
            fa = "%s_%s_x853g_%s_N512.npz" % (pre, st, kt); fz = "%s_%s_x853f_%s_N512.npz" % (pre, st, kt)
            a = np.load(fa); z = np.load(fz)
            ok, bad, missing, extras = keys_equal_common(a, z)
            row_ok &= ok
            parts.append("%s: %s%s%s%s" % (pre, ok, "" if not bad else " (differs: %s)" % ",".join(bad), "" if not missing else " (missing: %s)" % ",".join(missing), "" if not extras else " [extras: %s]" % ",".join(extras)))
        a = np.load("p3_%s_x853g_%s_N512.npz" % (st, kt))
        print("   sigma=%-6.4g 512x512 %s t_cert %.6f Om %.4f g %.4f itx %.4f it %d | %s" % (s, str(a["verdict"]), float(a["tcert"]), float(a["om_pk"]), float(a["gcert"]), float(a["itx"]), int(np.asarray(a["every_t"]).size), " | ".join(parts)))
        n += 1; nbad += 0 if row_ok else 1
    print("   %d rows, %d non-identical -> U-0 LETTER: %s" % (n, nbad, "BIT-IDENTICAL x%d" % n if (nbad == 0 and n == 2) else "NOT-IDENTICAL (STOP)" if nbad else "INCOMPLETE"))
    return nbad == 0 and n == 2


def wall_diff(Aref, Arow):
    """the wall row (y = 0): u = modes above the own-driver mask at both rows; D = max |log a_ref - log a_row|; the mean; the rms."""
    a = Aref["amp"][0]; b = Arow["amp"][0]
    m = (a[1:] > FLOOR * a[0]) & (b[1:] > FLOOR * b[0]); u = int(m.sum())
    if u < UMIN:
        return u, float("nan"), float("nan"), float("nan")
    d = np.log(a[1:][m] / a[0]) - np.log(b[1:][m] / b[0])
    return u, float(np.abs(d).max()), float(d.mean()), float(math.sqrt(float((d * d).mean())))


def windows(files, tm):
    tc = {k: float(np.load(f)["tcert"]) for k, f in files.items()}
    inside = all(tm <= v for v in tc.values())
    return tc, inside


def stage_g1():
    print("== U-1: the discriminator letter at t_m = %g, sigma = 1.375, on B* at the wall row of the fixed grid (own-driver normalisation; >= %d modes usable; reference = the square 1024^2 row) ==" % (TM3, UMIN))
    out = {}
    for s in (1.375, 0.875):
        files = {"512": SQ[(s, 512)], "1024": SQ[(s, 1024)], "A": AN[(s, "A")], "B": AN[(s, "B")]}
        tc, inside = windows(files, TM3)
        t_use = TM3 if inside else min(tc.values())
        print("   sigma=%g: t_cert 512^2 %.6f | 1024^2 %.6f | A %.6f | B %.6f ; t_m inside every window: %s -> the letter read at t = %.6f%s"
              % (s, tc["512"], tc["1024"], tc["A"], tc["B"], inside, t_use, "" if inside else "  (FALLBACK t_f = min t_cert)"))
        R = grid_amp_at(files["1024"], t_use); C = grid_amp_at(files["512"], t_use); A = grid_amp_at(files["A"], t_use); B = grid_amp_at(files["B"], t_use)
        for lab, X in (("512^2 (the S39 defect, must reproduce 0.5724 at t_m)", C), ("A = 512x1024 (coarse x, fine y) -> d_A", A), ("B = 1024x512 (fine x, coarse y) -> d_B", B)):
            u, dmax, dmean, drms = wall_diff(R, X)
            print("      vs %-52s : u=%2d  D=%.4f  mean=%+.4f  rms=%.4f  [records bracket %.6f..%.6f w=%.3f]" % (lab, u, dmax, dmean, drms, X["t_lo"], X["t_hi"], X["w"]))
        uA, dA, mA, _ = wall_diff(R, A); uB, dB, mB, _ = wall_diff(R, B); uC, dC, mC, _ = wall_diff(R, C)
        if s == 1.375:
            if math.isnan(dA) or math.isnan(dB):
                L = "UNDEFINED(floor)"; r = float("nan")
            elif max(dA, dB) < DTOP:
                L = "NEITHER"; r = dB / (dA + dB) if (dA + dB) > 0 else float("nan")
            else:
                r = dB / (dA + dB)
                L = "Y-RESOLUTION" if r >= RHI else "X-RESOLUTION" if r <= RLO else "MIXED"
            print("   sigma=%g: d_A = %.4f, d_B = %.4f, d_A + d_B = %.4f vs the square defect %.4f ; the share r = d_B/(d_A + d_B) = %s -> U-1 LETTER: %s   (bars: validity max(d_A, d_B) >= %.2f ; r >= %.1f Y / <= %.1f X / between MIXED)"
                  % (s, dA, dB, dA + dB, dC, ("%.3f" % r) if not math.isnan(r) else "nan", L, DTOP, RHI, RLO))
            # the mean beside the max (the S39 rule): the same share on the means
            rm = mB / (mA + mB) if (mA + mB) != 0 else float("nan")
            print("   sigma=%g: the same share on the MEAN differences: mean_A = %+.4f, mean_B = %+.4f -> r_mean = %s (report)" % (s, mA, mB, ("%.3f" % rm) if not math.isnan(rm) else "nan"))
            out[s] = (L, dA, dB, r)
        else:
            print("   sigma=%g (control): d_A = %.4f, d_B = %.4f, the square 512^2 defect %.4f -> expected <= 0.05 everywhere (report): %s" % (s, dA, dB, dC, "as expected" if max(dA, dB, dC) <= 0.05 else "NOT as expected"))
            out[s] = ("control", dA, dB, float("nan"))
    return out


def stage_g2():
    print("== U-2 (report): the layer thickness (e-fold depth medians over the usable B* modes; on-grid) of the anisotropic rows against the square rows, and the D profile through the layer ==")
    for s in (1.375, 0.875):
        files = {"512^2": SQ[(s, 512)], "1024^2": SQ[(s, 1024)], "A 512x1024": AN[(s, "A")], "B 1024x512": AN[(s, "B")]}
        tc, inside = windows(files, TM3); t_use = TM3 if inside else min(tc.values())
        for fld, key in (("theta", "gth"), ("omega", "gw")):
            vals = []
            for lab, f in files.items():
                X = grid_amp_at(f, t_use, key); ye = depths(X, X["amp"], 1.0 / math.e); ok = ~np.isnan(ye)
                vals.append("%s %.3e (%d)" % (lab, float(np.median(ye[ok])) if ok.any() else float("nan"), int(ok.sum())))
            print("   sigma=%-6g %-5s y_e medians (usable modes): %s" % (s, fld, " ; ".join(vals)))
        R = grid_amp_at(files["1024^2"], t_use)
        for lab in ("512^2", "A 512x1024", "B 1024x512"):
            X = grid_amp_at(files[lab], t_use); prof = dn_profile(X, R)
            rows = [(y, c, d, m) for (y, c, d, m) in prof if not math.isnan(d) and any(abs(y - s0) <= 1e-12 * max(1.0, s0) for s0 in (0.0, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3))]
            print("   sigma=%-6g D(%s vs 1024^2) through the layer: %s" % (s, lab, " ; ".join("y=%.0e u=%d D=%.3f mean=%+.3f" % r for r in rows)))


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("g0", "all"):
        ok = stage_g0()
        if st == "all" and not ok:
            print("U-0 not BIT-IDENTICAL x2 -> STOP (no U-1 letter is read)"); sys.exit(1)
    if st in ("g1", "all"):
        stage_g1()
    if st in ("g2", "all"):
        stage_g2()


# ------------------------------------------------------------------ U-1 under ONE common shift (the S38 rule; a REPORT added after the letter fired -- the letter at 0 stands as fired)
def stage_g1s():
    print("== U-1 robustness under ONE common timing shift (report; the letter at 0 decides, as fired): every row interpolated at t_m -+ delta, delta = half the coarsest record cadence among the four sigma = 1.375 rows ==")
    files = {"512": SQ[(1.375, 512)], "1024": SQ[(1.375, 1024)], "A": AN[(1.375, "A")], "B": AN[(1.375, "B")]}
    cad = {k: float(np.median(np.diff(np.asarray(np.load(f)["t"])))) for k, f in files.items()}
    delta = 0.5 * max(cad.values())
    print("   record cadences (median dt): %s ; common shift delta = %.3e" % (", ".join("%s %.2e" % (k, v) for k, v in cad.items()), delta))
    tc = {k: float(np.load(f)["tcert"]) for k, f in files.items()}
    for sh, lab in ((-delta, "-delta"), (0.0, "0"), (delta, "+delta")):
        t = TM3 + sh
        inside = all(t <= v for v in tc.values())
        R = grid_amp_at(files["1024"], t); C = grid_amp_at(files["512"], t); A = grid_amp_at(files["A"], t); B = grid_amp_at(files["B"], t)
        uC, dC, mC, _ = wall_diff(R, C); uA, dA, mA, _ = wall_diff(R, A); uB, dB, mB, _ = wall_diff(R, B)
        r = dB / (dA + dB) if (dA + dB) > 0 else float("nan"); rm = mB / (mA + mB) if (mA + mB) != 0 else float("nan")
        L = "UNDEFINED" if (uA < UMIN or uB < UMIN) else "NEITHER" if max(dA, dB) < DTOP else "Y-RESOLUTION" if r >= RHI else "X-RESOLUTION" if r <= RLO else "MIXED"
        print("   %-7s t=%.6f (inside every window: %s): square d_C = %.4f ; d_A = %.4f (mean %+.4f) ; d_B = %.4f (mean %+.4f) ; r = %.3f ; r_mean = %.3f -> %s" % (lab, t, inside, dC, dA, mA, dB, mB, r, rm, L))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "g1s":
    stage_g1s()
