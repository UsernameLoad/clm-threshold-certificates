#!/usr/bin/env python3
# gclm_exc.py — excursion-law universality test
# at general a (complex viscous hierarchy; a=0 machinery generalized).
#   b_m' = -nu m^2 b_m + i sum_{j=1}^{m-1} (a(m-j)/j - 1) b_j b_{m-j},  b_1(0) = -i
# Fast NL via two convolutions:  sum_j (a(m-j)/j - 1) b_j b_{m-j}
#   = a*conv(b/j, j*b)_m - conv(b, b)_m.
# Lawson-IFRK4 (exact stiff diagonal), rate-adaptive dt = alpha/Sigma.
# Stages: v | peak:<a>:<nu>[:M] | scan:<a>:<nu1,nu2,...>[:M] | loci:<a>
import os, sys, time
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

FFT_NL = False   # S11: opt-in FFT convolution (stage suffix ':fft'); default
                 # False so existing records reproduce bit-identically.
                 # Validation: stage nlv (mirrors hier_if's, s11_nlv.txt).

def make_nl(a, M):
    j = np.arange(1, M + 1).astype(float)
    def NL(b):
        if FFT_NL:
            from scipy.fft import fft, ifft
            n = 2*M
            Fu = fft(b/j, n); Fv = fft(j*b, n); Fb = fft(b, n)
            c1 = ifft(Fu*Fv)
            c2 = ifft(Fb*Fb)
            out = np.zeros(M, complex)
            out[1:] = 1j*(a*c1[:M-1] - c2[:M-1])
            return out
        u = b/j
        v = j*b
        c1 = np.convolve(u, v)[:M]
        c2 = np.convolve(b, b)[:M]
        out = np.zeros(M, complex)
        out[1:] = 1j*(a*c1[:M-1] - c2[:M-1])
        return out
    return NL

def ifrk4(a, nu, M, Tmax, alpha=0.05, dt_max=0.02, hi=1e6, lo=1e-3, rec=25):
    NL = make_nl(a, M)
    m2 = (np.arange(1, M + 1)**2).astype(float)
    b = np.zeros(M, complex); b[0] = -1j
    t = 0.0; dt_prev = -1.0; E1 = E2 = None
    peak = (0.0, 0.0); tailmax = 0.0; track = []
    nstep = 0; verdict = "UNDET(Tmax)"
    while t < Tmax:
        S = np.sum(np.abs(b))
        if S > peak[0]:
            peak = (S, t)
        tailmax = max(tailmax, abs(b[M-1]))
        if nstep % rec == 0:
            track.append((t, S))
        if S >= hi:
            verdict = "BLOWUP-bar(%.0e)" % hi; break
        if S <= lo and t > 1.0:
            verdict = "DECAY"; break
        dt = min(dt_max, alpha/max(S, 1e-12), Tmax - t)
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(b)
        U2 = E2*(b + 0.5*dt*k1); k2 = NL(U2)
        U3 = E2*b + 0.5*dt*k2;   k3 = NL(U3)
        U4 = E1*b + dt*E2*k3;    k4 = NL(U4)
        b = E1*b + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt; nstep += 1
        if not np.isfinite(S):
            verdict = "NAN"; break
    return dict(b=b, t=t, track=np.array(track), peak=peak, tailmax=tailmax,
                nsteps=nstep, verdict=verdict)

def stage_v():
    """Validate vs DOP853 at (a=-0.5, nu=0.05), and the a=0 reduction vs
    hier_if's real-q machinery (|b_m| must match q_m at a=0)."""
    from scipy.integrate import solve_ivp
    print("stage v: complex-hierarchy validation"); t0 = time.time()
    a, nu, M, T = -0.5, 0.05, 96, 0.3
    NL = make_nl(a, M)
    m2 = (np.arange(1, M + 1)**2).astype(float)
    def f(t, y):
        b = y[:M] + 1j*y[M:]
        db = -nu*m2*b + NL(b)
        return np.concatenate([db.real, db.imag])
    y0 = np.zeros(2*M); y0[M] = -1.0
    sol = solve_ivp(f, (0, T), y0, method="DOP853", rtol=1e-11, atol=1e-13)
    bd = sol.y[:M, -1] + 1j*sol.y[M:, -1]
    R = ifrk4(a, nu, M, T, alpha=0.02, dt_max=0.005, hi=1e12, lo=-1.0)
    print("  V1 a=-0.5 nu=0.05 t=%.1f: max|b_IF - b_DOP853| = %.3e (Sigma=%.4g)"
          % (T, np.max(np.abs(R["b"] - bd)), np.sum(np.abs(R["b"]))))
    # V2 (REWRITTEN S10): the a=0 reduction check with MATCHED conventions.
    # hier_if carries c = 2*w-hat (half-coefficient hierarchy); gclm_exc carries
    # b = w-hat of the amplitude-doubled datum (unit coefficient). The exact
    # same-nu relation is |b_m| = 2^{m-1} |c_m| (scale covariance of the
    # convolution; S10 log 1d). Short horizon so both sides are finite; the
    # original V2 (compare |b| to |c| raw, ride nu=0.052 to t=6) was miscoded:
    # mismatched conventions AND the b-side blows up there (nu*_b(a=0)=0.101) —
    # it rode a p=2 blowup toward the 1e12 bar at O(sqrt(bar)) step cost.
    import hier_if as hf
    Mv, Tv, nuv = 48, 0.5, 0.05
    Rq = hf.ifrk4_run(nuv, Mv, Tv, alpha=0.02, dt_max=0.001, sum_stop_hi=1e12, sum_stop_lo=-1.0)
    Rb = ifrk4(0.0, nuv, Mv, Tv, alpha=0.02, dt_max=0.001, hi=1e12, lo=-1.0)
    fac = 2.0**np.arange(Mv)
    print("  V2 a=0 reduction (matched: |b_m| vs 2^(m-1)|c_m|, nu=%.2f t=%.1f M=%d): max rel diff = %.3e"
          % (nuv, Tv, Mv,
             np.max(np.abs(np.abs(Rb["b"]) - fac*np.abs(Rq["c"]))/np.maximum(fac*np.abs(Rq["c"]), 1e-300))))
    print("[%.0f s]" % (time.time() - t0))

def stage_peak(a, nu, M=1024, Tmax=400.0):
    t0 = time.time()
    R = ifrk4(a, nu, M, Tmax)
    print("  a=%.2f nu=%.6f M=%d: %-14s Sigma_peak=%.6e t_peak=%.4f tail/pk=%.1e [%d steps, %.0f s]"
          % (a, nu, M, R["verdict"], R["peak"][0], R["peak"][1],
             R["tailmax"]/max(R["peak"][0], 1e-300), R["nsteps"], time.time() - t0))
    tag = ("a%s_nu%s" % (("%.2f" % a).replace(".", "p").replace("-", "m"),
                          ("%.6f" % nu).replace(".", "p")))
    np.savez(os.path.join(DIR, "exc_%s.npz" % tag), a=a, nu=nu, M=M,
             peak=R["peak"][0], t_peak=R["peak"][1], verdict=R["verdict"],
             tailmax=R["tailmax"], track=R["track"])
    return R

def stage_scan(a, nus, M=1024):
    print("stage scan: a=%.2f M=%d" % (a, M))
    for nu in nus:
        stage_peak(a, nu, M)

def ifrk4_peakmode(a, nu, M, Tmax, alpha=0.05, dt_max=0.02, hi=1e6, lo=1e-3):
    """Same integrator as ifrk4 (verbatim step), additionally snapshotting the
    full mode vector at the running Sigma-peak. S10 addition; ifrk4 untouched."""
    NL = make_nl(a, M)
    m2 = (np.arange(1, M + 1)**2).astype(float)
    b = np.zeros(M, complex); b[0] = -1j
    t = 0.0; dt_prev = -1.0; E1 = E2 = None
    peak = (0.0, 0.0); b_pk = b.copy(); nstep = 0; verdict = "UNDET(Tmax)"
    while t < Tmax:
        S = np.sum(np.abs(b))
        if S > peak[0]:
            peak = (S, t); b_pk = b.copy()
        if S >= hi:
            verdict = "BLOWUP-bar(%.0e)" % hi; break
        if S <= lo and t > 1.0:
            verdict = "DECAY"; break
        dt = min(dt_max, alpha/max(S, 1e-12), Tmax - t)
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(b)
        U2 = E2*(b + 0.5*dt*k1); k2 = NL(U2)
        U3 = E2*b + 0.5*dt*k2;   k3 = NL(U3)
        U4 = E1*b + dt*E2*k3;    k4 = NL(U4)
        b = E1*b + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt; nstep += 1
        if not np.isfinite(S):
            verdict = "NAN"; break
    return dict(b_pk=b_pk, peak=peak, nsteps=nstep, verdict=verdict)

def stage_peakm(a, nu, M=1024, Tmax=400.0):
    """Pole-structure discriminator at the excursion peak (N19 prediction ii):
    fit ln|b_m| = ln(C m + D) + m ln r on the settled bulk; report the log-slope
    sigma (simple pole: 0; double pole: 1) as in the a=0 stage `mode`."""
    t0 = time.time()
    R = ifrk4_peakmode(a, nu, M, Tmax)
    S, tp = R["peak"]
    q = np.abs(R["b_pk"])
    print("  peakm a=%.2f nu=%.6f M=%d: %s Sigma_peak=%.6e t_peak=%.4f [%d steps, %.0f s]"
          % (a, nu, M, R["verdict"], S, tp, R["nsteps"], time.time() - t0))
    # settled bulk: modes where q is well above floor and below the front cap
    mm = np.arange(1, M + 1).astype(float)
    good = q > q.max()*1e-12
    mlo, mhi = 8, int(0.6*np.max(mm[good]))
    sel = (mm >= mlo) & (mm <= mhi) & good
    x = mm[sel]; y = np.log(q[sel])
    # local ln-slope of the prefactor: fit y = c0 + sigma*ln m + m*ln r  (3-param)
    A = np.vstack([np.ones_like(x), np.log(x), x]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    c0, sig, lnr = coef
    resid = y - A @ coef
    # split-halves stability
    h = len(x)//2
    cA, *_ = np.linalg.lstsq(A[:h], y[:h], rcond=None)
    cB, *_ = np.linalg.lstsq(A[h:], y[h:], rcond=None)
    print("    prefactor exponent sigma = %.3f (halves %.3f / %.3f)  ln r = %.5f  rms = %.2e  [m in %d..%d]"
          % (sig, cA[1], cB[1], lnr, np.sqrt(np.mean(resid**2)), mlo, mhi))
    print("    [double pole: sigma -> 1; simple pole: sigma -> 0]  delta = -ln r = %.5f  Sigma*delta^2 = %.4g  12nu = %.4g"
          % (-lnr, S*lnr*lnr, 12*nu))
    tag = ("a%s_nu%s_M%d" % (("%.2f" % a).replace(".", "p").replace("-", "m"),
                             ("%.6f" % nu).replace(".", "p"), M))
    np.savez(os.path.join(DIR, "peakm_%s.npz" % tag), a=a, nu=nu, M=M,
             b_pk=R["b_pk"], peak=S, t_peak=tp, sigma=sig, lnr=lnr)
    return R

def ifrk4_snap(a, nu, M, Tmax, S_peak_est, alpha=0.05, dt_max=0.02, hi=1e6, lo=1e-3):
    """S11 additive: same integrator step VERBATIM as ifrk4/ifrk4_peakmode,
    snapshotting the full mode vector at Sigma-level triggers on the rise AND
    the fall (levels x S_peak_est from a prior pass), plus the running peak."""
    LEV = (0.3, 0.5, 0.7, 0.85, 0.95, 0.99)
    NL = make_nl(a, M)
    m2 = (np.arange(1, M + 1)**2).astype(float)
    b = np.zeros(M, complex); b[0] = -1j
    t = 0.0; dt_prev = -1.0; E1 = E2 = None
    peak = (0.0, 0.0); b_pk = b.copy(); nstep = 0
    snaps = []                     # (label, t, S, b)
    rise_next = 0; fall_next = len(LEV) - 1
    S_prev = 0.0
    while t < Tmax:
        S = np.sum(np.abs(b))
        if S > peak[0]:
            peak = (S, t); b_pk = b.copy()
        if rise_next < len(LEV) and S >= LEV[rise_next]*S_peak_est and S > S_prev:
            snaps.append(("r%.2f" % LEV[rise_next], t, S, b.copy()))
            rise_next += 1
        if (fall_next >= 0 and S < S_prev and peak[0] >= 0.995*S_peak_est
                and S <= LEV[fall_next]*S_peak_est):
            snaps.append(("f%.2f" % LEV[fall_next], t, S, b.copy()))
            fall_next -= 1
        if S >= hi or (S <= lo and t > 1.0) or not np.isfinite(S):
            break
        S_prev = S
        dt = min(dt_max, alpha/max(S, 1e-12), Tmax - t)
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(b)
        U2 = E2*(b + 0.5*dt*k1); k2 = NL(U2)
        U3 = E2*b + 0.5*dt*k2;   k3 = NL(U3)
        U4 = E1*b + dt*E2*k3;    k4 = NL(U4)
        b = E1*b + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt; nstep += 1
    snaps.append(("peak", peak[1], peak[0], b_pk))
    snaps.sort(key=lambda s: s[1])
    return snaps, peak, nstep

def fit_cdr(q, mlo=8, frac=0.6):
    """Constrained (C, D, r) fit of q_m = (C m + D) r^m on the settled bulk.
    S11 PATCH (instrument correction, on the record in the session log): the
    original version solved (C, D) by LSQ in LINEAR space (large-m dominated)
    while reporting log-space rms -- an inconsistent objective that inflated D
    (a=-0.5 peak: D = 0.096 at mlo=8, drifting to 0.022 at mlo=64; the
    log-objective refit gives D ~ 0.016 with C/nu unchanged to 4 digits).
    Now: full log-residual nonlinear fit. C and delta are robust to this
    change at the 1e-4 level; D (and anything built on it) is the sensitive
    output and carries the log-correction caveat (see stage peakm2 notes).
    Returns (C, D, lnr, rms_log, mhi)."""
    from scipy.optimize import least_squares
    M = len(q)
    mm = np.arange(1, M + 1).astype(float)
    good = q > q.max()*1e-12
    mhi = int(frac*np.max(mm[good]))
    sel = (mm >= mlo) & (mm <= mhi) & good
    x = mm[sel]; y = np.log(q[sel])
    A3 = np.vstack([np.ones_like(x), np.log(x), x]).T
    c3, *_ = np.linalg.lstsq(A3, y, rcond=None)
    C0 = np.exp(c3[0]); lnr0 = c3[2]
    def resid(p):
        C, D, lnr = p
        arg = C*x + D
        bad = arg <= 0
        r = np.log(np.where(bad, 1.0, arg)) + lnr*x - y
        r[bad] = 1e3
        return r
    sol = least_squares(resid, [C0, 0.0, lnr0])
    C, D, lnr = sol.x
    rms = np.sqrt(np.mean(sol.fun**2))
    return C, D, lnr, rms, mhi

def stage_peakm2(a, nu, M=1024, Tmax=400.0):
    """S11 instrument upgrade (registered S10, ledger N19): through-peak
    constrained (C, D, r) battery at general a. Measures, per snapshot:
    C/nu (residue law 6/(1-2a) in this b-convention), D(t) (zero-crossing
    displacement from the peak -- exact at a=0 by the rate law; OPEN at a!=0),
    delta(t) = -ln r, and pairwise k_hat = -delta_dot/D (a=0 derived value:
    5/12 = 0.4167; general-a s^-3 balance open).
    PREDICTIONS (registered before first run, S11):
      C/nu -> 6/(1-2a) at all snapshots (a=-0.5: 3.0; a=+0.2: 10.0);
      sigma-equivalent double pole throughout; D crosses zero NEAR the peak,
      displacement = the measurement (no prediction -- exploratory: if the
      a=+0.2 delta-independent 1.5% residue residual is D-zero displacement,
      expect a nonzero shift there and ~zero at a=-0.5... measured k(a)
      values are new numbers either way."""
    t0 = time.time()
    # pass 1: peak estimate (reuse saved peakm npz if present)
    tag = ("a%s_nu%s_M%d" % (("%.2f" % a).replace(".", "p").replace("-", "m"),
                             ("%.6f" % nu).replace(".", "p"), M))
    f1 = os.path.join(DIR, "peakm_%s.npz" % tag)
    if os.path.exists(f1):
        S_est = float(np.load(f1)["peak"])
        print("  peakm2 pass1: reusing %s  Sigma_peak=%.6e" % (os.path.basename(f1), S_est))
    else:
        R1 = ifrk4_peakmode(a, nu, M, Tmax)
        S_est = R1["peak"][0]
        print("  peakm2 pass1: Sigma_peak=%.6e t_peak=%.4f [%d steps]"
              % (S_est, R1["peak"][1], R1["nsteps"]))
    # pass 2: snapshot battery
    snaps, peak, nstep = ifrk4_snap(a, nu, M, Tmax, S_est)
    t_pk = peak[1]
    print("  peakm2 a=%.2f nu=%.6f M=%d: %d snapshots, peak Sigma=%.6e at t=%.4f [%d steps, %.0f s]"
          % (a, nu, M, len(snaps), peak[0], t_pk, nstep, time.time() - t0))
    print("    %-6s %9s %12s %10s %10s %10s %8s" %
          ("snap", "t-t_pk", "Sigma", "C/nu", "D", "delta", "rmslog"))
    rows = []
    for (lab, t, S, bv) in snaps:
        C, D, lnr, rms, mhi = fit_cdr(np.abs(bv))
        rows.append((lab, t, S, C, D, -lnr, rms))
        print("    %-6s %9.4f %12.5e %10.4f %10.5f %10.6f %8.1e" %
              (lab, t - t_pk, S, C/nu, D, -lnr, rms))
    # D zero crossing (linear interpolation between sign changes)
    tz = None
    for i in range(len(rows) - 1):
        D1, D2 = rows[i][4], rows[i+1][4]
        if D1 == 0 or D1*D2 < 0:
            t1, t2 = rows[i][1], rows[i+1][1]
            tz = t1 + (t2 - t1)*abs(D1)/(abs(D1) + abs(D2))
            break
    # pairwise k_hat = -delta_dot / D(mid)
    ks = []
    for i in range(len(rows) - 1):
        t1, t2 = rows[i][1], rows[i+1][1]
        d1, d2 = rows[i][5], rows[i+1][5]
        Dm = 0.5*(rows[i][4] + rows[i+1][4])
        if abs(Dm) > 1e-6 and t2 > t1:
            ks.append((0.5*(t1 + t2) - t_pk, -(d2 - d1)/(t2 - t1)/Dm))
    if tz is not None:
        w95 = [r[1] for r in rows if r[0] in ("r0.95", "f0.95")]
        width = (w95[1] - w95[0]) if len(w95) == 2 else np.nan
        print("    D-ZERO at t-t_pk = %+.5f   (peak width t[f95]-t[r95] = %.5f => displacement/width = %+.3f)"
              % (tz - t_pk, width, (tz - t_pk)/width if np.isfinite(width) and width > 0 else np.nan))
    print("    k_hat = -delta_dot/D by pair midpoint (a=0 derived: 0.4167):")
    for (tm, k) in ks:
        print("      t-t_pk = %+8.4f   k_hat = %8.4f" % (tm, k))
    np.savez(os.path.join(DIR, "peakm2_%s.npz" % tag), a=a, nu=nu, M=M,
             labels=[r[0] for r in rows], t=[r[1] for r in rows],
             S=[r[2] for r in rows], C=[r[3] for r in rows],
             D=[r[4] for r in rows], delta=[r[5] for r in rows],
             rms=[r[6] for r in rows], t_peak=t_pk, S_peak=peak[0],
             tz=tz if tz is not None else np.nan,
             k_pairs=np.array(ks) if ks else np.zeros((0, 2)),
             b_snaps=np.array([s[3] for s in snaps]))   # S11: raw vectors for offline refits

def stage_loci(a, numax=None):
    """gamma and t_peak-locus analysis on saved decay-side points at this a.
    numax (S10 addition): exclude far-field points nu > numax from the fits
    (global fits over a wide nu range are dragged by pre-asymptotic points;
    the near-field fit is the asymptotic instrument)."""
    import glob
    from scipy.optimize import curve_fit
    pat = os.path.join(DIR, "exc_a%s_nu*.npz" % ("%.2f" % a).replace(".", "p").replace("-", "m"))
    rows = []
    for f in sorted(glob.glob(pat)):
        z = np.load(f, allow_pickle=True)
        if str(z["verdict"]).startswith("DECAY") and float(z["tailmax"])/float(z["peak"]) < 2e-6:
            if numax is None or float(z["nu"]) <= numax:
                rows.append((float(z["nu"]), float(z["peak"]), float(z["t_peak"])))
    rows.sort()
    if len(rows) < 4:
        print("  only %d clean decay points — need more" % len(rows)); return
    nus = np.array([r[0] for r in rows]); S = np.array([r[1] for r in rows]); T = np.array([r[2] for r in rows])
    print("  %d clean points: nu in [%.5f, %.5f], Sigma_peak %.3g..%.3g" % (len(rows), nus[0], nus[-1], S.min(), S.max()))
    for i in range(len(rows) - 1):
        g = np.log(S[i]/S[i+1])/np.log((nus[i+1])/(nus[i]))  # placeholder; proper below
    def mS(nu, ns, C, g):
        return C*(nu - ns)**(-g)
    pS, _ = curve_fit(mS, nus, S, p0=(nus[0]*0.98, 1e-3, 2.0), maxfev=200000)
    print("  Sigma_peak fit: nu* = %.6f  gamma = %.4f  C = %.4g" % (pS[0], pS[2], pS[1]))
    gl = [np.log(S[i]/S[i+1])/np.log((nus[i+1]-pS[0])/(nus[i]-pS[0])) for i in range(len(rows)-1)]
    print("  gamma_loc (using fitted nu*): " + " ".join("%.3f" % g for g in gl))
    def mT(S_, t0, c, p):
        return t0 - c*S_**(-1.0/p)
    pT, _ = curve_fit(mT, S, T, p0=(T.max()+0.05, 5.0, 2.0), maxfev=200000)
    resT = T - mT(S, *pT)
    print("  peak-time locus: t0 = %.4f  p = %.4f  rms = %.2e" % (pT[0], pT[2], np.sqrt(np.mean(resT**2))))
    eps = nus - pS[0]
    ce = np.polyfit(np.log(eps), np.log(pT[0] - T), 1)
    print("  (t0 - t_peak) ~ eps^q: q = %.4f   [time-shift scenario: q = 1, p = 2, gamma = 2]" % ce[0])

if __name__ == "__main__":
    st = sys.argv[1]
    if st.endswith(":fft"):        # S12: same opt-in suffix as hier_if;
        FFT_NL = True              # default path untouched (records reproduce)
        st = st[:-4]
        print("[FFT_NL enabled]")
    if st == "v":
        stage_v()
    elif st.startswith("peak:"):
        p_ = st.split(":")
        stage_peak(float(p_[1]), float(p_[2]), int(p_[3]) if len(p_) > 3 else 1024)
    elif st.startswith("scan:"):
        p_ = st.split(":")
        stage_scan(float(p_[1]), [float(x) for x in p_[2].split(",")],
                   int(p_[3]) if len(p_) > 3 else 1024)
    elif st.startswith("peakm2:"):
        p_ = st.split(":")
        stage_peakm2(float(p_[1]), float(p_[2]), int(p_[3]) if len(p_) > 3 else 1024)
    elif st.startswith("peakm:"):
        p_ = st.split(":")
        stage_peakm(float(p_[1]), float(p_[2]), int(p_[3]) if len(p_) > 3 else 1024)
    elif st.startswith("loci:"):
        p_ = st.split(":")
        stage_loci(float(p_[1]), float(p_[2]) if len(p_) > 2 else None)
    elif st == "nlv":
        rng = np.random.default_rng(7)
        print("stage nlv: FFT-NL validation (b-world; predicted rel <= 1e-13)")
        for a_, M in ((-0.5, 2048), (0.2, 4096), (-0.5, 8192)):
            b = (rng.standard_normal(M) + 1j*rng.standard_normal(M))/np.sqrt(M)
            NLd = make_nl(a_, M)
            FFT_NL = False; x = NLd(b)
            FFT_NL = True;  y = NLd(b)
            FFT_NL = False
            print("  a=%+.1f M=%5d: rel sup diff %.2e"
                  % (a_, M, np.abs(x - y).max()/np.abs(x).max()))
    else:
        raise SystemExit("unknown stage: %s" % st)
