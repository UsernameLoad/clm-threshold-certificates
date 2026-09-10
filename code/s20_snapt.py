"""S20 -- the time-triggered snapshot ladder with an EXPLICIT bar.

hier_if.stage_snapt uses bar = 1.7*cap with cap = nu*(M/2.7)^sigma, which
DEGENERATES as sigma -> 0 exactly as stage_sig's does (the reason S19 wrote
s19_sigsub.py): at sigma = 0.20, M = 8192, nu = 0.424 the bar is 3.60 while
the excursion peak is 7.15, so the ride stops before the peak and the ladder
returns 7 of 25 snapshots..

This script is stage_snapt's loop copied VERBATIM (same dt rule, same NL,
same E1/E2, same stop conditions, same observers, same output keys) with the
bar as an explicit argument.  Predecessor-reproduction obligation: stage
`repro` re-rides the banked (sigma = 0.25, nu = 0.407, M = 4096) ladder and
must reproduce snapt_sig0p25_nu0p407000_M4096.npz digit for digit.

Stages:
  repro
  run:<sigma>:<nu>:<M>:<t0>:<t1>:<n>[:bar]
"""
import os
import sys
import time

import numpy as np

import hier_if
from hier_if import NL, DIR


def run_snapt(sigma, nu, M, t0, t1, n, bar=None, tag=None):
    hier_if.FFT_NL = True
    print("stage snapt(explicit bar): sigma=%.3f nu=%.6f M=%d t in [%.4f, %.4f]"
          " n=%d" % (sigma, nu, M, t0, t1, n))
    tt0 = time.time()
    targets = list(np.linspace(t0, t1, n))
    it = 0
    snaps = []
    cap = nu * (M / 2.7) ** float(sigma)
    if bar is None:
        bar = 1.7 * cap
    m2 = np.arange(1, M + 1).astype(float) ** float(sigma) if sigma != 2 else \
        (np.arange(1, M + 1) ** 2).astype(float)
    c = np.zeros(M, complex)
    c[0] = -1j
    t = 0.0
    dt_prev = -1.0
    E1 = E2 = None
    peak = (0.0, 0.0)
    c_pk = c.copy()
    tailmax = 0.0
    nstep = 0
    track = []
    verdict = "UNDET(Tmax)"
    Tmax = 400.0
    while t < Tmax:
        S = np.sum(np.abs(c))
        if S > peak[0]:
            peak = (S, t)
            c_pk = c.copy()
        while it < len(targets) and t >= targets[it]:
            snaps.append(("t%02d" % it, t, S, c.copy()))
            it += 1
        tailmax = max(tailmax, abs(c[M - 1]))
        track.append((t, S))
        if S >= bar:
            verdict = "BLOWUP-bar(%.1e)" % bar
            break
        if S <= 1e-3 and t > 1.0:
            verdict = "DECAY"
            break
        if not np.isfinite(S):
            verdict = "NAN"
            break
        dt = min(0.02, 0.05 / max(S, 1e-12), Tmax - t)
        if dt != dt_prev:
            E1 = np.exp(-nu * m2 * dt)
            E2 = np.exp(-nu * m2 * dt / 2.0)
            dt_prev = dt
        k1 = NL(c)
        U2 = E2 * (c + 0.5 * dt * k1); k2 = NL(U2)
        U3 = E2 * c + 0.5 * dt * k2;   k3 = NL(U3)
        U4 = E1 * c + dt * E2 * k3;    k4 = NL(U4)
        c = E1 * c + (dt / 6.0) * (E1 * k1 + 2.0 * E2 * (k2 + k3) + k4)
        t += dt
        nstep += 1
    clean = (tailmax / max(peak[0], 1e-300)) <= 1e-8
    print("  nu=%.6f: %-18s Sigma_peak=%.6e t_peak=%.4f tail/pk=%.1e cap=%.2e"
          " bar=%.2e%s [%d steps, %.0f s]"
          % (nu, verdict, peak[0], peak[1], tailmax / max(peak[0], 1e-300),
             cap, bar, "" if clean else "  [tail > 1e-8: INDICATIVE]",
             nstep, time.time() - tt0), flush=True)
    print("  time-triggered snapshots: %d of %d" % (it, n))
    fn = os.path.join(DIR, "s20snapt_sig%s_nu%s_M%d.npz"
                      % (("%.2f" % sigma).replace(".", "p"),
                         ("%.6f" % nu).replace(".", "p"), M)) if tag is None \
        else os.path.join(DIR, tag)
    np.savez(fn, sigma=sigma, nu=nu, M=M, peak=peak[0], t_peak=peak[1],
             tailmax=tailmax, verdict=verdict, c_pk=c_pk, cap=cap, bar=bar,
             track=np.array(track),
             snaps_tag=np.array([s[0] for s in snaps]),
             snaps_t=np.array([s[1] for s in snaps]),
             snaps_S=np.array([s[2] for s in snaps]),
             snaps_c=np.array([s[3] for s in snaps]))
    print("  saved %s" % os.path.basename(fn))
    return fn


def stage_repro():
    print("stage repro: the banked sigma=0.25 nu=0.407 M=4096 ladder must come"
          " back digit for digit (snapt_sig0p25_nu0p407000_M4096.npz)")
    ref = np.load(os.path.join(DIR, "snapt_sig0p25_nu0p407000_M4096.npz"))
    fn = run_snapt(0.25, 0.407, 4096, 3.0, 12.0, 25,
                   bar=1.7 * 0.407 * (4096 / 2.7) ** 0.25,
                   tag="s20snapt_repro.npz")
    new = np.load(fn)
    ok = True
    for k in ("peak", "t_peak", "tailmax"):
        a, b = float(ref[k]), float(new[k])
        hit = ("%.10e" % a) == ("%.10e" % b)
        ok = ok and hit
        print("   %-8s recorded %.10e   recomputed %.10e   %s"
              % (k, a, b, "MATCH" if hit else "MISMATCH"))
    dc = np.abs(ref["snaps_c"] - new["snaps_c"]).max()
    dt_ = np.abs(ref["snaps_t"] - new["snaps_t"]).max()
    print("   snapshots: %d vs %d ; max|dc| = %.1e ; max|dt| = %.1e"
          % (len(ref["snaps_t"]), len(new["snaps_t"]), dc, dt_))
    ok = ok and (dc == 0.0) and (dt_ == 0.0) and \
        len(ref["snaps_t"]) == len(new["snaps_t"])
    print("   REPRO: %s" % ("PASS -- bit-identical"
                            if ok else "FAIL -- STOP AND DIAGNOSE"))
    return ok


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "repro":
        stage_repro()
    else:
        p = st.split(":")
        run_snapt(float(p[1]), float(p[2]), int(p[3]), float(p[4]),
                  float(p[5]), int(p[6]),
                  bar=float(p[7]) if len(p) > 7 else None)
