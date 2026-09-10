#!/usr/bin/env python3
# s15_kcurv.py — the sigma-family excursion-bottom curvature
# instrument (N25's registered gamma^2 nu^2 test).
#
# Instrument definition (the S14 pattern, generalized to sigma):
#   track (t, Sigma) -> delta_impl(t) = (K_sigma*nu/Sigma)^(1/sigma),
#   K_sigma = 2*Gamma(2 sigma)/Gamma(sigma)   [N21 amplitude law; K_2 = 12,
#   K_1 = 2, K_1.5 = 4.51353, K_1.25 = 2.93318, K_1.75 = 7.23202-class]
#   -> windowed local slopes ddelta/dt (centered, full width DTW = 0.02)
#   -> linear fit of ddelta/dt vs (t - t_pk) over |t - t_pk| <= W for
#      W in {0.03, 0.10}: slope = ddot-delta at the bottom.
# At sigma = 2 this is EXACTLY the S14 kcurv instrument up to the rate-law
# conversion D_rate = -(12/5)*ddelta/dt, k = (12/5)*ddot-delta; stage
# `validate` must reproduce s14_kcurv.txt digit-class BEFORE any new-sigma
# run is scored (instrument-continuity discipline).
#
# Stages:
#   validate                     — reproduce the S14 sigma=2 table rows
#   sig:<sigma>:<npz1,npz2,...>  — score sigma-family tracks (patched stage_sig npz)
#   exact1:<nu1,nu2,...>         — closed-form sigma=1 instrument arithmetic
#                                  (Sakajo r(t) = (t/2)e^{-nu t}; exact Sigma =
#                                  e^{-nu t}/(1-r)) — the instrument-vs-truth
#                                  bias at finite eps, exact anchor delta-ddot
#                                  = nu^2 at the true bottom.
import os, sys
import numpy as np
from scipy.special import gamma as GAMMA

DIR = os.path.dirname(os.path.abspath(__file__))
DTW = 0.02      # full width of the local-slope window (S14 "Delta t = 0.02-class")

def Ksig(sigma):
    return 2.0 * GAMMA(2.0 * sigma) / GAMMA(sigma)

def local_slopes(t, y, halfw=DTW / 2.0):
    """Centered linear-fit slope of y(t) in [t_i - halfw, t_i + halfw] at each
    track point with >= 4 points in window; returns (t_mid, slope)."""
    ts, sl = [], []
    j0 = 0
    n = len(t)
    for i in range(n):
        lo, hi = t[i] - halfw, t[i] + halfw
        a = np.searchsorted(t, lo, side="left")
        b = np.searchsorted(t, hi, side="right")
        if b - a >= 4:
            cf = np.polyfit(t[a:b], y[a:b], 1)
            ts.append(t[i]); sl.append(cf[0])
    return np.array(ts), np.array(sl)

def curvature_from_track(t, S, nu, sigma, t_pk):
    """Returns dict with ddot-delta estimates at W = 0.03 and 0.10 plus the
    rise/fall asymmetry at 0.10 and diagnostics."""
    K = Ksig(sigma)
    good = S > 0
    t = t[good]; S = S[good]
    delta = (K * nu / S) ** (1.0 / sigma)
    tm, dd = local_slopes(t, delta)
    out = {}
    for W in (0.03, 0.10):
        sel = np.abs(tm - t_pk) <= W
        if sel.sum() >= 6:
            cf = np.polyfit(tm[sel] - t_pk, dd[sel], 1)
            out["W%.2f" % W] = cf[0]
            out["n%.2f" % W] = int(sel.sum())
            if W == 0.10:
                sr = sel & (tm <= t_pk)
                sf = sel & (tm >= t_pk)
                if sr.sum() >= 4 and sf.sum() >= 4:
                    kr = np.polyfit(tm[sr] - t_pk, dd[sr], 1)[0]
                    kf = np.polyfit(tm[sf] - t_pk, dd[sf], 1)[0]
                    out["asym"] = (kf - kr) / cf[0]
                    out["k_rise"] = kr; out["k_fall"] = kf
        else:
            out["W%.2f" % W] = np.nan
            out["n%.2f" % W] = int(sel.sum())
    # D_rate zero at peak (H2-class diagnostic): ddelta/dt at t_pk
    if len(tm):
        i0 = np.argmin(np.abs(tm - t_pk))
        out["dd_at_pk"] = dd[i0]
    return out

def stage_validate():
    NUSTAR = 0.0505674
    files = [
        ("P5a", "peak_nu0p050650_M8192.npz"),
        ("P5b", "peak_nu0p050700_M8192.npz"),
        ("P5c", "peak_nu0p050750_M8192.npz"),
        ("s7a", "peak_nu0p050800.npz"),
        ("s7b", "peak_nu0p050900.npz"),
        ("s7c", "peak_nu0p051000.npz"),
        ("s7d", "peak_nu0p051200.npz"),
        ("s7e", "peak_nu0p051600.npz"),
        ("s7f", "peak_nu0p052200.npz"),
        ("s7g", "peak_nu0p053000.npz"),
        ("s7h", "peak_nu0p054000.npz"),
    ]
    print("=== s15 validate: reproduce s14_kcurv (sigma=2, D_rate-slope k = (12/5)*ddot-delta) ===")
    print("  %-4s %-9s %10s %11s %11s %8s %8s %8s" %
          ("tag", "nu", "eps", "k(pm0.03)", "k(pm0.10)", "asym10", "k/nu", "k/nu2"))
    for tag, f in files:
        p = os.path.join(DIR, f)
        if not os.path.exists(p):
            print("  %-4s MISSING %s" % (tag, f)); continue
        z = np.load(p, allow_pickle=True)
        nu = float(z["nu"]); tr = z["track"]; t_pk = float(z["t_peak"])
        r = curvature_from_track(tr[:, 0], tr[:, 1], nu, 2.0, t_pk)
        k3 = (12.0 / 5.0) * r.get("W0.03", np.nan)
        k10 = (12.0 / 5.0) * r.get("W0.10", np.nan)
        print("  %-4s %-9.6f %10.3e %+11.5f %+11.5f %+8.3f %8.4f %8.3f" %
              (tag, nu, nu - NUSTAR, k3, k10, r.get("asym", np.nan),
               k3 / nu if np.isfinite(k3) else np.nan,
               k3 / nu**2 if np.isfinite(k3) else np.nan))
    print("  [target: s14_kcurv.txt rows digit-class]")

def stage_sig(sigma, files):
    NUSTAR = {1.0: 0.18393972, 1.25: 0.13510, 1.5: 0.098066, 1.75: 0.07062,
              2.0: 0.0505674}
    print("=== s15 sig-curvature: sigma=%.2f  K_sigma=%.5f ===" % (sigma, Ksig(sigma)))
    print("  %-30s %-9s %10s %12s %12s %8s %8s" %
          ("file", "nu", "eps", "dd(pm0.03)", "dd(pm0.10)", "asym10", "dd/nu2"))
    for f in files:
        p = os.path.join(DIR, f)
        if not os.path.exists(p):
            print("  MISSING %s" % f); continue
        z = np.load(p, allow_pickle=True)
        if "track" not in z:
            print("  %-30s NO TRACK (pre-patch npz)" % f); continue
        nu = float(z["nu"]); tr = z["track"]; t_pk = float(z["t_peak"])
        r = curvature_from_track(tr[:, 0], tr[:, 1], nu, sigma, t_pk)
        d3 = r.get("W0.03", np.nan); d10 = r.get("W0.10", np.nan)
        ns = NUSTAR.get(sigma, np.nan)
        print("  %-30s %-9.6f %10.3e %+12.6f %+12.6f %+8.3f %8.4f  [n=%d/%d dd_pk=%+.2e]" %
              (os.path.basename(f), nu, nu - ns, d3, d10,
               r.get("asym", np.nan),
               d3 / nu**2 if np.isfinite(d3) else np.nan,
               r.get("n0.03", 0), r.get("n0.10", 0), r.get("dd_at_pk", np.nan)))

def stage_exact1(nus):
    """Closed-form sigma=1 arithmetic: r = (t/2)e^{-nu t}, Sigma =
    e^{-nu t}/(1-r) EXACT (Sakajo Lemma 3 geometric solution). The
    INSTRUMENT's delta_impl = 2 nu/Sigma; true delta = -ln r. Both the true
    bottom curvature (exact nu^2 at t = 1/nu) and the instrument's
    ddot-delta_impl at the Sigma-peak are computed to machine precision by
    dense finite differences on the closed form + the same windowed-slope
    instrument as the numerical stage (apples-to-apples)."""
    print("=== s15 exact1: sigma=1 closed-form instrument arithmetic ===")
    NUSTAR = 1.0 / (2.0 * np.e)
    print("  nu* = 1/(2e) = %.8f ; true ddot-delta(bottom) = nu^2 EXACT (t=1/nu)" % NUSTAR)
    print("  %-9s %10s %14s %14s %10s" %
          ("nu", "eps", "dd_impl(0.03)", "dd_impl(0.10)", "dd/nu2"))
    for nu in nus:
        # dense closed-form track around the Sigma-peak
        tgrid = np.linspace(0.2 / nu, 2.5 / nu, 400001)
        r = (tgrid / 2.0) * np.exp(-nu * tgrid)
        S = np.exp(-nu * tgrid) / (1.0 - r)
        ipk = np.argmax(S)
        t_pk = tgrid[ipk]
        rr = curvature_from_track(tgrid, S, nu, 1.0, t_pk)
        d3 = rr.get("W0.03", np.nan)
        print("  %-9.6f %10.3e %+14.6e %+14.6e %10.4f" %
              (nu, nu - NUSTAR, d3, rr.get("W0.10", np.nan), d3 / nu**2))

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "validate":
        stage_validate()
    elif st.startswith("sig:"):
        parts = st.split(":")
        stage_sig(float(parts[1]), parts[2].split(","))
    elif st.startswith("exact1:"):
        stage_exact1([float(x) for x in st.split(":")[1].split(",")])
    else:
        raise SystemExit("unknown stage: %s" % st)
