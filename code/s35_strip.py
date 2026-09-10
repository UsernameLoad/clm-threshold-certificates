#!/usr/bin/env python
"""
s35_strip.py -- THE MATCHED-TIME / MATCHED-BAND STRIP INSTRUMENT (S35
G-1..G-7;.

Stages
  check   the code-path check (G-0 iv): the fit routines run on the banked
          STOP fields reproduce s34_e3d.txt's V0 / V3 / V5 values to every
          printed digit at every banked column.
  repro   the G-0 (ii)/(iii) reproduction obligation: the fresh-tag npz vs the
          banked npz key by key; the side file's last wall row vs th_final;
          the side file's Ew re-fitted by the solver's strip code vs the
          saved delta_x / strip_r2 series.
  g7      every re-ridden row vs its banked partner (observables + every key).
  cols    G-2 / G-4 / G-5 / G-6: the model-free object and the fits at the
          matched times for every column that has side files.
  npair   G-3: the X3 N-pair {0.875, 1.375} x {N512, N1024} at t_m = 0.004488.
  all     check + g7 + cols + npair
Conventions (fitrow / s34_e3d VERBATIM where they overlap): the theta driver
wall = the row with the larger mean |theta| (index Ny in every banked row);
amp(k) = |rfft(row - mean)|/N; the band = even modes 2..int(0.4*(N//2)) with
the 1e-12 amplitude mask (relative to the driver mode); B* = modes 52..102
(the N512 upper half by count), B*_q = 78..102; the omega spectrum = the
parent's normalised max-over-y |rfft_x w| (the side file's Ew); the
matched-time spectrum = the log-linear interpolation of the two bracketing
records, with both records also fitted (the timing spread).
Floor rule (added after the first scoring run crashed on an under-populated
band at 0.8 t_m — s35_strip_cols_run1_crash.txt): a fit needs >= 3 (pure
exponential) / >= 4 (two-shape) modes above the mask, else NaN; the
model-free ordering uses only the modes above the mask in BOTH rows at all
three time samples, prints the usable fraction of B*, and the letter is
UNDEFINED(floor) if under half of B* is usable for some adjacent pair.
"""
import os, sys, glob, re, math
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
KD = 24 * math.pi
LX = 1.0 / 6.0
kd_index = 2
BSTAR = list(range(52, 103, 2))       # the matched physical band (26 even modes)
BSTARQ = list(range(78, 103, 2))      # its upper quarter (13 modes)
B512 = list(range(2, 103, 2))         # fitrow's N512 band (51 modes)
FLOOR = 1e-12

# ------------------------------------------------------------------ fits
def lsq(A, yy):
    coef, res, rk, sv = np.linalg.lstsq(A, yy, rcond=None)
    r = yy - A @ coef
    rms = float(np.sqrt((r ** 2).mean()))
    ss = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - float((r ** 2).sum()) / ss if ss > 0 else float("nan")
    return coef, rms, r2

def masked(amp, modes):
    amp = np.asarray(amp)
    return [k for k in modes if amp[k] > FLOOR * amp[kd_index]]

def pure_exp(amp, modes):
    """ln amp = a - delta k on the mode list; returns delta, rms, R2, formal sigma_delta, n, k_c (NaN if < 3 modes above the mask)."""
    band = masked(amp, modes)
    n = len(band)
    if n < 3:
        return dict(delta=float("nan"), rms=float("nan"), r2=float("nan"), sd=float("nan"), n=n, kc=float("nan"))
    kk = np.array(band, float); yy = np.log(np.asarray(amp)[band])
    A = np.vstack([np.ones_like(kk), -kk]).T
    c, rms, r2 = lsq(A, yy)
    cov = np.linalg.inv(A.T @ A) * rms ** 2 * n / max(n - 2, 1)
    return dict(delta=float(c[1]), rms=rms, r2=r2, sd=math.sqrt(cov[1, 1]), n=n, kc=float(kk.mean()))

def two_shape(amp, modes):
    """ln amp = a + p ln k - delta k with formal sigma_delta, sigma_p, corr(p, delta) (s34_e3d V0 VERBATIM; NaN if < 4 modes)."""
    band = masked(amp, modes)
    n = len(band)
    if n < 4:
        return dict(delta=float("nan"), p=float("nan"), rms=float("nan"), r2=float("nan"), sd=float("nan"), sp=float("nan"), corr=float("nan"), n=n)
    kk = np.array(band, float); yy = np.log(np.asarray(amp)[band])
    A0 = np.vstack([np.ones_like(kk), np.log(kk), -kk]).T
    c, rms, r2 = lsq(A0, yy)
    cov = np.linalg.inv(A0.T @ A0) * rms ** 2 * n / max(n - 3, 1)
    sd, sp = math.sqrt(cov[2, 2]), math.sqrt(cov[1, 1])
    return dict(delta=float(c[2]), p=float(c[1]), rms=rms, r2=r2, sd=sd, sp=sp, corr=cov[1, 2] / (sd * sp), n=n)

def own_band(N):
    top = int(0.4 * (N // 2))
    return list(range(kd_index, top + 1, kd_index))

def s34_variants(amp, N):
    """the s34_e3d V0 / V3 / V5 numbers on the field's own band (by count)."""
    band = masked(amp, own_band(N))
    n = len(band); h = n // 2; q = (3 * n) // 4
    return two_shape(amp, band)["delta"], pure_exp(amp, band[h:])["delta"], pure_exp(amp, band[q:])["delta"]

def wall_amp_from_row(row):
    row = np.asarray(row, float)
    F = np.fft.rfft(row - row.mean()) / row.size
    return np.abs(F)

def wall_amp_final(z):
    th = z["th_final"]
    wall = 0 if abs(th[0]).mean() > abs(th[-1]).mean() else -1
    return wall_amp_from_row(th[wall]), wall

def solver_strip(Ew, N):
    """bsq_solver.diagnostics' strip fit VERBATIM on the normalised Ew (own band [ncut/2, ncut), physical kx)."""
    ncut = int((2.0 / 3.0) * (N // 2))
    kx = (2.0 * np.pi / LX) * np.arange(N // 2 + 1)
    k1, k2 = ncut // 2, ncut
    seg = Ew[k1:k2]; msk = seg > 1e-12
    if msk.sum() >= 8:
        kk = kx[k1:k2][msk]; ll = np.log(seg[msk])
        A_ = np.vstack([kk, np.ones_like(kk)]).T
        sol, res, *_ = np.linalg.lstsq(A_, ll, rcond=None)
        ss = ((ll - ll.mean()) ** 2).sum()
        return -sol[0], 1.0 - (res[0] / ss if len(res) and ss > 0 else np.nan)
    return np.nan, np.nan

def sig_of(fn):
    m = re.search(r"p3d?_s(\d)p(\d+)_", fn)
    s = float("%s.%s" % (m.group(1), m.group(2)))
    return 0.9375 if s == 0.938 else s        # the %.3f tag (s35_regcalc §A-3)

# ------------------------------------------------------------------ matched-time spectra from a side file
def bracket(t, tm):
    i = int(np.searchsorted(t, tm, side="right") - 1)
    i = max(0, min(i, len(t) - 2))
    return i, i + 1

def spectra_at(side, tm):
    """dict(theta = amp(k) at tm (log-linear interp), theta_lo/hi at the bracketing records, omega = ..., t_lo, t_hi, w, ...)"""
    z = np.load(side)
    t = np.asarray(z["t"]); i, j = bracket(t, tm)
    wgt = (tm - t[i]) / (t[j] - t[i]) if t[j] > t[i] else 0.0
    def interp(lo, hi):
        lo = np.maximum(np.asarray(lo, float), 1e-300); hi = np.maximum(np.asarray(hi, float), 1e-300)
        return np.exp((1 - wgt) * np.log(lo) + wgt * np.log(hi))
    th_lo = wall_amp_from_row(z["th_bot"][i]); th_hi = wall_amp_from_row(z["th_bot"][j])
    Ew_lo = np.asarray(z["Ew"][i]); Ew_hi = np.asarray(z["Ew"][j])
    return dict(theta=interp(th_lo, th_hi), theta_lo=th_lo, theta_hi=th_hi,
                omega=interp(Ew_lo, Ew_hi), omega_lo=Ew_lo, omega_hi=Ew_hi,
                t_lo=float(t[i]), t_hi=float(t[j]), w=float(wgt), N=int(z["N"]), tcert=float(z["tcert"]), verdict=str(z["verdict"]),
                sigma=float(z["sigma"]))

def a_norm(amp, modes):
    amp = np.asarray(amp)
    return np.log(amp[modes] / amp[kd_index])

def describe_row(lab, sp, N):
    ob = own_band(N); oh = ob[len(ob) // 2:]
    for fld in ("theta", "omega"):
        amp = sp[fld]
        f1 = pure_exp(amp, BSTAR); f2 = pure_exp(amp, BSTARQ); f3 = pure_exp(amp, oh); f4 = two_shape(amp, B512)
        f1lo = pure_exp(sp[fld + "_lo"], BSTAR)["delta"]; f1hi = pure_exp(sp[fld + "_hi"], BSTAR)["delta"]
        extra = ""
        if fld == "omega":
            dx, r2x = solver_strip(amp, N)
            extra = " | solver own-band delta_x %.4e (R2 %.3f)" % (dx, r2x)
        print("      %-5s %s: F1(B*) %.5f ±%.5f R2 %.4f n=%d [timing %.5f..%.5f] | F2(B*_q) %.5f ±%.5f R2 %.4f n=%d | F3(own upper half %d..%d) %.5f R2 %.4f | F4 two-shape delta %.5f ±%.5f p %+.3f ±%.3f corr %+.3f R2 %.4f n=%d%s"
              % (fld, lab, f1["delta"], f1["sd"], f1["r2"], f1["n"], np.nanmin([f1lo, f1hi]), np.nanmax([f1lo, f1hi]), f2["delta"], f2["sd"], f2["r2"], f2["n"],
                 oh[0], oh[-1], f3["delta"], f3["r2"], f4["delta"], f4["sd"], f4["p"], f4["sp"], f4["corr"], f4["r2"], f4["n"], extra))

def joint_mask(band, *amps):
    m = np.ones(len(band), bool)
    for a in amps:
        a = np.asarray(a); m &= a[band] > FLOOR * a[kd_index]
    return [k for k, ok in zip(band, m) if ok]

def ordering(rows, fld, band):
    """rows: list of (sigma, sp) sorted by sigma; returns letter, per-pair details on the model-free object."""
    det = []; flags = []; floor = False
    for (s1, sp1), (s2, sp2) in zip(rows[:-1], rows[1:]):
        bm = joint_mask(band, sp1[fld], sp2[fld], sp1[fld + "_lo"], sp2[fld + "_lo"], sp1[fld + "_hi"], sp2[fld + "_hi"])
        use = len(bm) / len(band)
        if len(bm) < 2:
            det.append((s1, s2, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), "FLOOR", use)); flags.append("FLOOR"); floor = True; continue
        S = a_norm(sp2[fld], bm) - a_norm(sp1[fld], bm)
        Slo = a_norm(sp2[fld + "_lo"], bm) - a_norm(sp1[fld + "_lo"], bm)
        Shi = a_norm(sp2[fld + "_hi"], bm) - a_norm(sp1[fld + "_hi"], bm)
        neg = float((S < 0).mean()); pos = float((S > 0).mean())
        if use < 0.5:
            flag = "FLOOR"; floor = True
        elif neg >= 0.9 and (Slo < 0).mean() >= 0.9 and (Shi < 0).mean() >= 0.9:
            flag = "NEG"
        elif pos >= 0.9 and (Slo > 0).mean() >= 0.9 and (Shi > 0).mean() >= 0.9:
            flag = "POS"
        else:
            flag = "MIX"
        det.append((s1, s2, float(S.mean()), float(S.std()), neg, pos, float(min(Slo.mean(), Shi.mean())), float(max(Slo.mean(), Shi.mean())), flag, use))
        flags.append(flag)
    if floor:
        letter = "UNDEFINED(floor: under half of B* above the 1e-12 mask for some adjacent pair)"
    elif all(f == "NEG" for f in flags):
        letter = "MONOTONE"
    elif any(f == "POS" for f in flags):
        letter = "NON-MONOTONE"
    else:
        letter = "MIXED"
    return letter, det

def column(label, sides, tm, score=True):
    rows = sorted(((sig_of(f), spectra_at(f, tm)) for f in sides), key=lambda q: q[0])
    N = rows[0][1]["N"]
    print(" -- %s : %d rows, N = %d, t_m = %.6f (bracketing records t_lo/t_hi per row below) --" % (label, len(rows), N, tm))
    for s, sp in rows:
        print("    sigma=%-6.4g  %-10s t_cert %.6f  records [%.6f, %.6f] w=%.3f" % (s, sp["verdict"], sp["tcert"], sp["t_lo"], sp["t_hi"], sp["w"]))
        describe_row("sigma=%.4g" % s, sp, N)
    letters = {}
    for fld in ("theta", "omega"):
        letter, det = ordering(rows, fld, BSTAR)
        print("    MODEL-FREE ORDERING on B* (%s): adjacent pairs sigma'->sigma: S_bar (k-scatter) frac<0 frac>0 [timing range] usable flag" % fld)
        for s1, s2, m, sd, neg, pos, lo, hi, flag, use in det:
            print("      %.4g -> %.4g : S_bar %+.4f (std %.4f)  frac<0 %.2f  frac>0 %.2f  [%+.4f, %+.4f]  usable %.2f of B*  %s" % (s1, s2, m, sd, neg, pos, lo, hi, use, flag))
        s0, sp0 = rows[0]
        def sep(sp, spref):
            bm = joint_mask(BSTAR, spref[fld], sp[fld])
            return float((a_norm(sp[fld], bm) - a_norm(spref[fld], bm)).mean()) if len(bm) >= 2 else float("nan")
        seps = [(s, sep(sp, sp0)) for s, sp in rows[1:]]
        print("      S_bar(sigma, %.4g) over the column: %s" % (s0, " / ".join("%+.4f (%.4g)" % (v, s) for s, v in seps)))
        print("    LETTER (%s, %s): %s%s" % (label, fld, letter, "" if score else "   [REPORT]"))
        letters[fld] = letter
    return rows, letters

# ------------------------------------------------------------------ stages
def stage_check():
    print("== G-0 (iv) the code-path check: the fit routines on the banked STOP fields vs s34_e3d.txt (V0 / V3 / V5 on the own band) ==")
    txt = open("s34_e3d.txt", encoding="utf-8").read()
    blocks = re.findall(r" -- (.+?) \((\d+) rows.*?\n(.*?)(?=\n -- |\nReading rule)", txt, flags=re.S)
    cols = {"X2 N512": ["p3_s0p875_x568n_k12p942890_N512.npz", "p3_s0p938_x568n_k9p878618_N512.npz", "p3_s1p000_x568n_k7p539822_N512.npz",
                        "p3_s1p125_x568n_k4p392290_N512.npz", "p3_s1p250_x568n_k2p558709_N512.npz", "p3_s1p375_x568n_k1p490564_N512.npz",
                        "p3_s1p500_x568n_k0p868322_N512.npz"],
            "X2 N1024": sorted(glob.glob("p3_s*_n1024x568_*_N1024.npz"), key=sig_of),
            "X3 N512": sorted(glob.glob("p3_s*_x853n_*_N512.npz"), key=sig_of),
            "X3 N1024": sorted(glob.glob("p3_s*_x853n1024_*_N1024.npz"), key=sig_of),
            "X4 N512": sorted(glob.glob("p3_s*_x1137n_*_N512.npz"), key=sig_of),
            "X1 N512": sorted(glob.glob("p3_s*_n512x426_*_N512.npz"), key=sig_of),
            "X4 N256 (the S20 family; mixed verdicts)": ["p3_s0p500_k130p939798_N256.npz", "p3_s0p750_k44p435635_N256.npz", "p3_s1p000_k15p079645_N256.npz", "p3_s1p500_k1p736643_N256.npz", "p3_s2p000_k0p200000_N256.npz"],
            "X5 N256 (the S20 family; all SATURATED at one stop time)": ["p3_s0p500_k163p674748_N256.npz", "p3_s0p750_k55p544543_N256.npz", "p3_s1p000_k18p849556_N256.npz", "p3_s1p500_k2p170804_N256.npz", "p3_s2p000_k0p250000_N256.npz"]}
    nrow = 0; nbad = 0
    for name, nr, body in blocks:
        files = cols.get(name.strip())
        if files is None:
            print("   (block '%s' has no file list here — skipped)" % name); continue
        recs = re.findall(r"sigma=(\S+)\s+V0 delta=([-\d.]+).*?V3\(upper half, pure exp\) delta=([-\d.]+).*?V5\(upper quarter, pure exp\) delta=([-\d.]+)", body)
        by = {}
        for f in files:
            z = np.load(f); amp, _ = wall_amp_final(z); by[sig_of(f)] = s34_variants(amp, int(z["N"]))
        for s, v0, v3, v5 in recs:
            s = float(s); s = 0.9375 if s == 0.938 else s
            mine = by.get(s)
            if mine is None:
                print("   %s sigma=%g: no file" % (name, s)); nbad += 1; continue
            good = ("%.5f" % mine[0] == v0) and ("%.5f" % mine[1] == v3) and ("%.5f" % mine[2] == v5)
            nrow += 1; nbad += 0 if good else 1
            if not good:
                print("   MISMATCH %s sigma=%g: mine V0 %.5f V3 %.5f V5 %.5f vs s34 %s %s %s" % (name, s, mine[0], mine[1], mine[2], v0, v3, v5))
    print("   %d rows compared, %d mismatches -> %s" % (nrow, nbad, "CODE-PATH CHECK PASSED (every printed digit)" if nbad == 0 else "CODE-PATH CHECK FAILED"))

def keys_equal(a, z):
    """every key of npz a equal to npz z (arrays exact, NaN-aware; strings exact); returns (allok, bad_keys)."""
    if set(a.files) != set(z.files):
        return False, ["<key sets differ>"]
    bad = []
    for k in sorted(z.files):
        x = a[k]; y = z[k]
        if x.dtype.kind in "US":
            eq = str(x) == str(y)
        else:
            eq = x.shape == y.shape and (np.array_equal(x, y) or (np.issubdtype(x.dtype, np.floating) and np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)])))
        if not eq:
            bad.append(k)
    return len(bad) == 0, bad

def stage_repro(fresh="p3_s1p125_x853d_k6p588434_N512.npz", banked="p3_s1p125_x853n_k6p588434_N512.npz", side="p3d_s1p125_x853d_k6p588434_N512.npz"):
    print("== G-0 (ii)/(iii) the reproduction obligation: %s (fresh tag) vs %s (banked) ==" % (fresh, banked))
    a = np.load(fresh); b = np.load(banked)
    print("   key sets equal: %s (%d keys)" % (set(a.files) == set(b.files), len(b.files)))
    allok, bad = keys_equal(a, b)
    for k in bad:
        print("   key %-12s NOT EQUAL" % k)
    print("   every key equal: %s" % allok)
    z = np.load(side); N = int(z["N"])
    last_ok = np.array_equal(z["th_bot"][-1], a["th_final"][N])
    dx = np.array([solver_strip(E, N)[0] for E in z["Ew"]]); r2 = np.array([solver_strip(E, N)[1] for E in z["Ew"]])
    sdx = np.asarray(a["delta_x"]); sr2 = np.asarray(a["strip_r2"])
    def eqnan(u, v):
        return np.array_equal(np.isnan(u), np.isnan(v)) and np.array_equal(u[~np.isnan(u)], v[~np.isnan(v)])
    dx_ok = eqnan(dx, sdx); r2_ok = eqnan(r2, sr2)
    n_ok = (len(z["t"]) == len(a["t"])) and bool(z["align_ok"])
    print("   side file: last th_bot == th_final[Ny]: %s ; Ew re-fitted by the solver's strip code == saved delta_x: %s ; == saved strip_r2: %s ; record count + in-ride alignment: %s"
          % (last_ok, dx_ok, r2_ok, n_ok))
    letter = "BIT-IDENTICAL" if (allok and last_ok and dx_ok and r2_ok and n_ok) else "NOT-IDENTICAL (STOP)"
    print("   G-0 LETTER: %s" % letter)
    return letter

def stage_g7():
    """G-7 (report): every re-ridden row vs its banked partner — the printed observables AND every saved key."""
    print("== G-7: every fresh-tag row vs its banked partner (verdict / t_cert / Omega_cert_peak / g_cert / itx / steps, then EVERY key) ==")
    pairs = [("x568d", "x568n"), ("x853d", "x853n"), ("x1137d", "x1137n"), ("x853d1024", "x853n1024")]
    nrow = 0; nbad = 0
    for fresh_tag, banked_tag in pairs:
        for f in sorted(glob.glob("p3_s*_%s_k*_N*.npz" % fresh_tag), key=sig_of):
            b = f.replace("_%s_" % fresh_tag, "_%s_" % banked_tag)
            if not os.path.exists(b):
                print("   %-46s banked partner %s MISSING" % (f, b)); nbad += 1; continue
            a = np.load(f); z = np.load(b)
            allok, bad = keys_equal(a, z)
            side = f.replace("p3_", "p3d_", 1)
            sz = np.load(side) if os.path.exists(side) else None
            side_ok = (sz is not None) and bool(sz["align_ok"]) and np.array_equal(sz["th_bot"][-1], a["th_final"][int(a["N"])])
            nrow += 1; nbad += 0 if (allok and side_ok) else 1
            print("   sigma=%-6.4g %-9s %-10s t_cert %.6f Om %.4f g %.4f itx %.4f it %d | every key equal to the banked row: %s%s | side file aligned: %s"
                  % (sig_of(f), fresh_tag, str(a["verdict"]), float(a["tcert"]), float(a["om_pk"]), float(a["gcert"]), float(a["itx"]), int(np.asarray(a["every_t"]).size),
                     allok, "" if allok else " (differs: %s)" % ",".join(bad), side_ok))
    print("   %d rows compared, %d non-identical -> %s" % (nrow, nbad, "EVERY RE-RIDDEN ROW BIT-IDENTICAL TO ITS BANKED PARTNER" if nbad == 0 else "NON-IDENTICAL ROWS NAMED ABOVE"))

def stage_cols():
    print("== G-2 / G-4 / G-5 / G-6: the model-free object and the fits at the matched times ==")
    plan = [("X2 N512 (x568d)", sorted(glob.glob("p3d_s*_x568d_*_N512.npz"), key=sig_of), 0.003714, 0.002972, True),
            ("X3 N512 (x853d)", sorted(glob.glob("p3d_s*_x853d_*_N512.npz"), key=sig_of), 0.004488, 0.003590, True),
            ("X4 N512 (x1137d; MIXED verdicts, report)", sorted(glob.glob("p3d_s*_x1137d_*_N512.npz"), key=sig_of), 0.005078, 0.004063, False)]
    for label, sides, tm, tm2, score in plan:
        if len(sides) < 3:
            print(" -- %s: only %d side files present — skipped" % (label, len(sides))); continue
        rows, L1 = column(label + " @ t_m", sides, tm, score)
        rows2, L2 = column(label + " @ 0.8 t_m (G-5)", sides, tm2, False)
        print("    G-5 time robustness (%s): theta %s -> %s = %s ; omega %s -> %s = %s\n" % (label, L1["theta"], L2["theta"], "SAME" if L1["theta"] == L2["theta"] else "DIFFERENT",
                                                                                          L1["omega"], L2["omega"], "SAME" if L1["omega"] == L2["omega"] else "DIFFERENT"))

def stage_npair():
    print("== G-3: N-convergence at matched time AND matched band — the X3 pair {0.875, 1.375} x {N512, N1024} at t_m = 0.004488 ==")
    tm = 0.004488
    f512 = {s: f for s, f in ((sig_of(f), f) for f in glob.glob("p3d_s*_x853d_*_N512.npz")) if s in (0.875, 1.375)}
    f1024 = {s: f for s, f in ((sig_of(f), f) for f in glob.glob("p3d_s*_x853d1024_*_N1024.npz")) if s in (0.875, 1.375)}
    if len(f512) < 2 or len(f1024) < 2:
        print("   side files present: N512 %s, N1024 %s — the pair is incomplete; nothing scored" % (sorted(f512), sorted(f1024))); return
    sp = {(s, 512): spectra_at(f512[s], tm) for s in f512}; sp.update({(s, 1024): spectra_at(f1024[s], tm) for s in f1024})
    for (s, N), v in sorted(sp.items()):
        print("    sigma=%.4g N=%d  %-10s t_cert %.6f  records [%.6f, %.6f] w=%.3f" % (s, N, v["verdict"], v["tcert"], v["t_lo"], v["t_hi"], v["w"]))
        describe_row("sigma=%.4g N=%d" % (s, N), v, N)
    for fld in ("theta", "omega"):
        print("   -- %s --" % fld)
        for s in (0.875, 1.375):
            bm = joint_mask(BSTAR, sp[(s, 512)][fld], sp[(s, 1024)][fld])
            a5 = a_norm(sp[(s, 512)][fld], bm); a10 = a_norm(sp[(s, 1024)][fld], bm)
            d = a10 - a5; DN = float(np.abs(d).max())
            dlo = float(np.abs(a_norm(sp[(s, 1024)][fld + "_lo"], bm) - a_norm(sp[(s, 512)][fld + "_lo"], bm)).max())
            dhi = float(np.abs(a_norm(sp[(s, 1024)][fld + "_hi"], bm) - a_norm(sp[(s, 512)][fld + "_hi"], bm)).max())
            let = "FIELD-CONVERGED" if DN <= 0.05 else ("PARTIAL" if DN <= 0.30 else "NOT-CONVERGED")
            print("    G-3a sigma=%.4g: D_N = max|a_1024 - a_512| on B* = %.4f (mean %+.4f; at k=%d %+.4f, k=%d %+.4f; timing %.4f..%.4f; usable %d/%d modes) -> %s"
                  % (s, DN, d.mean(), bm[0], d[0], bm[-1], d[-1], min(dlo, dhi), max(dlo, dhi), len(bm), len(BSTAR), let))
        bm4 = joint_mask(BSTAR, sp[(1.375, 512)][fld], sp[(0.875, 512)][fld], sp[(1.375, 1024)][fld], sp[(0.875, 1024)][fld])
        S512 = a_norm(sp[(1.375, 512)][fld], bm4) - a_norm(sp[(0.875, 512)][fld], bm4)
        S1024 = a_norm(sp[(1.375, 1024)][fld], bm4) - a_norm(sp[(0.875, 1024)][fld], bm4)
        n512 = float((S512 < 0).mean()); n1024 = float((S1024 < 0).mean())
        if n512 >= 0.9 and n1024 >= 0.9:
            rN = float(S512.mean() / S1024.mean())
            let = "CONVERGED" if abs(rN - 1) <= 0.10 else ("DRIFTING" if abs(rN - 1) <= 0.50 else "NOT-CONVERGED")
        else:
            rN = float("nan"); let = "UNDEFINED"
        # the timing spread of r_N from the bracketing records
        rlo = float((a_norm(sp[(1.375, 512)][fld + "_lo"], bm4) - a_norm(sp[(0.875, 512)][fld + "_lo"], bm4)).mean() / (a_norm(sp[(1.375, 1024)][fld + "_lo"], bm4) - a_norm(sp[(0.875, 1024)][fld + "_lo"], bm4)).mean())
        rhi = float((a_norm(sp[(1.375, 512)][fld + "_hi"], bm4) - a_norm(sp[(0.875, 512)][fld + "_hi"], bm4)).mean() / (a_norm(sp[(1.375, 1024)][fld + "_hi"], bm4) - a_norm(sp[(0.875, 1024)][fld + "_hi"], bm4)).mean())
        print("    G-3b S_bar_512(1.375, 0.875) = %+.4f (std %.4f, frac<0 %.2f) ; S_bar_1024 = %+.4f (std %.4f, frac<0 %.2f) ; r_N = %.3f [timing %.3f..%.3f] (usable %d/%d modes) -> %s"
              % (S512.mean(), S512.std(), n512, S1024.mean(), S1024.std(), n1024, rN, min(rlo, rhi), max(rlo, rhi), len(bm4), len(BSTAR), let))
        if fld == "omega":
            for s in (0.875, 1.375):
                d5, r5 = solver_strip(sp[(s, 512)]["omega"], 512); d10, r10 = solver_strip(sp[(s, 1024)]["omega"], 1024)
                print("    G-3c sigma=%.4g: solver own-band delta_x at t_m: N512 %.4e (R2 %.3f) | N1024 %.4e (R2 %.3f) | ratio %.3f  [report; expectation 1.00–1.10]" % (s, d5, r5, d10, r10, d5 / d10))

if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("check", "all"):
        stage_check(); print()
    if st == "repro":
        stage_repro()
    if st in ("g7", "all"):
        stage_g7(); print()
    if st in ("cols", "all"):
        stage_cols(); print()
    if st in ("npair", "all"):
        stage_npair()
