#!/usr/bin/env python3
# hier_if.py — exponential (Lawson-IFRK4)
# integrator for the a=0 viscous q-hierarchy, and the excursion-peak program
# for the excursion-proof determination of nu*(0)  (ledger N4/N15).
#
# System (S02, exact for the real cos-x problem at a=0 by triangularity):
#   c_m' = -nu m^2 c_m + (1/2) sum_{j=1}^{m-1} c_j c_{m-j},  c_1(0) = -i.
# Lawson-IFRK4 removes the stiff diagonal exactly (per-step decaying factors
# e^{-nu m^2 dt}); step size is rate-adaptive on the nonlinear scale Sum|c|.
# Near-separatrix runs track the metastable excursion; Sigma_peak(nu) diverges
# as nu -> nu*+ and its fit locates the separatrix without any certification
# bar (turning the N15 failure mode into the measurement).
# Stages:  ifv | peak:<nu> | scan | fit
import os, sys, time
import numpy as np
import clm_viscous as cv

DIR = os.path.dirname(os.path.abspath(__file__))

FFT_NL = False   # S11: opt-in FFT convolution for M >= 8192 rides (stage suffix
                 # ':fft'). Default False so every existing record reproduces
                 # bit-identically. Validation: stage nlv (predicted floors in
                 # its docstring).

def NL(c):
    M = len(c)
    if FFT_NL:
        from scipy.fft import fft, ifft
        F = fft(c, 2*M)
        conv = ifft(F*F)
        out = np.zeros(M, complex)
        out[1:] = 0.5*conv[:M-1]
        return out
    conv = np.convolve(c, c)
    out = np.zeros(M, complex)
    out[1:] = 0.5*conv[:M-1]
    return out

def ifrk4_run(nu, M, Tmax, alpha=0.05, dt_max=0.02, sum_stop_hi=1e6,
              sum_stop_lo=1e-3, record_every=25, sigma=2, keep_cpk=False):
    """Lawson(4) with exact diagonal propagation. Returns dict with track of
    (t, Sum|c|), peak info, tail maximum, and verdict.
    S11: optional dissipation order sigma (diagonal m^sigma nu); sigma=2 takes
    the historical integer-squaring path -> bit-identical to every record.
    S12: keep_cpk=True snapshots the mode vector at the running peak (returned
    as 'c_pk'); default False leaves every historical path untouched."""
    if sigma == 2:
        m2 = (np.arange(1, M+1)**2).astype(float)
    else:
        m2 = np.arange(1, M+1).astype(float)**float(sigma)
    c = np.zeros(M, complex); c[0] = -1j
    t = 0.0
    dt_prev = -1.0
    E1 = E2 = None
    track = []
    peak = (0.0, 0.0)          # (Sigma_peak, t_peak)
    c_pk = None
    tailmax = 0.0
    nstep = 0
    verdict = "UNDET(Tmax)"
    while t < Tmax:
        S = np.sum(np.abs(c))
        if S > peak[0]:
            peak = (S, t)
            if keep_cpk:
                c_pk = c.copy()
        tailmax = max(tailmax, abs(c[M-1]))
        if nstep % record_every == 0:
            track.append((t, S))
        if S >= sum_stop_hi:
            verdict = "BLOWUP-bar(%.0e)" % sum_stop_hi
            break
        if S <= sum_stop_lo and t > 1.0:
            verdict = "DECAY"
            break
        dt = min(dt_max, alpha/max(S, 1e-12), Tmax - t)
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt)
            E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(c)
        U2 = E2*(c + 0.5*dt*k1)
        k2 = NL(U2)
        U3 = E2*c + 0.5*dt*k2
        k3 = NL(U3)
        U4 = E1*c + dt*E2*k3
        k4 = NL(U4)
        c = E1*c + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt
        nstep += 1
    return dict(c=c, t=t, track=np.array(track), peak=peak, tailmax=tailmax,
                nsteps=nstep, verdict=verdict, c_pk=c_pk)

# ---------------- stages ----------------
def stage_ifv():
    """Validation battery for the integrator (arXiv-grade):
    V1 exact series at nu=0.25 (float64-safe regime, S02-verified vs mpmath);
    V2 mpmath exact series per-mode at nu=0.05, m<=24, t<=1.2;
    V3 DOP853 cross-check mid-excursion at nu=0.054;
    V4 alpha-halving self-convergence at nu=0.052 (excursion peak value)."""
    print("stage ifv: Lawson-IFRK4 validation"); t0 = time.time()
    # V1: nu=0.25, compare c_m(t=1) vs exact exponential-sum series
    M = 90
    R = ifrk4_run(0.25, M, 1.0, alpha=0.02, dt_max=0.01,
                  sum_stop_hi=1e12, sum_stop_lo=-1.0)
    B25 = cv.exact_coeffs(0.25, M)
    cex = cv.eval_exact(B25, 0.25, R["t"])
    print("V1  nu=0.25 t=1: max|c - exact series| = %.3e   (%d steps)"
          % (np.max(np.abs(R["c"] - cex)), R["nsteps"]))
    # V2: nu=0.05 per-mode m<=24 vs 40-dps mpmath series
    Bmp = cv.mp_exact_coeffs(0.05, 24)
    R = ifrk4_run(0.05, 64, 1.2, alpha=0.02, dt_max=0.01,
                  sum_stop_hi=1e12, sum_stop_lo=-1.0)
    cmp_ = cv.mp_eval(Bmp, 0.05, R["t"])
    print("V2  nu=0.05 t=1.2: per-mode m<=24 vs mpmath = %.3e"
          % np.max(np.abs(R["c"][:24] - cmp_)))
    # V3: nu=0.054 to t=10 (early excursion) vs DOP853 on the same system
    from scipy.integrate import solve_ivp
    M3 = 192
    m2 = (np.arange(1, M3+1)**2).astype(float)
    def f(t, y):
        c = y[:M3] + 1j*y[M3:]
        dc = -0.054*m2*c + NL(c)
        return np.concatenate([dc.real, dc.imag])
    y0 = np.zeros(2*M3); y0[M3] = -1.0
    sol = solve_ivp(f, (0, 10.0), y0, method="DOP853", rtol=1e-12, atol=1e-14)
    cd = sol.y[:M3, -1] + 1j*sol.y[M3:, -1]
    R = ifrk4_run(0.054, M3, 10.0, alpha=0.02, dt_max=0.01,
                  sum_stop_hi=1e12, sum_stop_lo=-1.0)
    print("V3  nu=0.054 t=10: max|c_IF - c_DOP853| = %.3e  (Sum=%.3e)"
          % (np.max(np.abs(R["c"] - cd)), np.sum(np.abs(R["c"]))))
    # V4: alpha-halving at nu=0.052: peak value self-convergence
    p = {}
    for al in (0.05, 0.025, 0.0125):
        R = ifrk4_run(0.052, 768, 400.0, alpha=al)
        p[al] = R["peak"]
        print("V4  nu=0.052 alpha=%.4f: verdict=%-12s Sigma_peak=%.6e at t=%.2f  tailmax=%.1e  [%d steps]"
              % (al, R["verdict"], R["peak"][0], R["peak"][1], R["tailmax"], R["nsteps"]))
    print("[%.0f s]" % (time.time() - t0))

def stage_peak(nu, M=1024, Tmax=400.0, bar=None):
    """S12 additions (records untouched): optional blowup bar (default keeps
    the historical 1e6); filenames carry _M<M> for M != 1024 (the S11 filename
    rule); the peak mode-vector snapshot is saved for deep-point profile fits;
    the clean cap nu(M/2.7)^2 is reported and DECAY peaks above it are flagged
    INDICATIVE (N15 tail discipline)."""
    t0 = time.time()
    if bar is None:
        bar = 1e6
    R = ifrk4_run(nu, M, Tmax, sum_stop_hi=bar, keep_cpk=True)
    cap = nu*(M/2.7)**2
    dirty = (R["verdict"] == "DECAY") and (R["peak"][0] > cap)
    print("  nu=%.6f M=%d: %-16s Sigma_peak=%.6e t_peak=%.2f t_end=%.1f tailmax=%.2e  cap=%.2e%s  [%d steps, %.0f s]"
          % (nu, M, R["verdict"], R["peak"][0], R["peak"][1], R["t"],
             R["tailmax"], cap, "  [PEAK>CAP: INDICATIVE]" if dirty else "",
             R["nsteps"], time.time() - t0), flush=True)
    tagM = "" if M == 1024 else "_M%d" % M
    fn = os.path.join(DIR, "peak_nu%s%s.npz" % (("%.6f" % nu).replace(".", "p"), tagM))
    if os.path.exists(fn):                      # never overwrite a record
        k = 1
        while os.path.exists(fn + ".bak%d" % k):
            k += 1
        os.rename(fn, fn + ".bak%d" % k)
        print("  [existing %s backed up to .bak%d]" % (os.path.basename(fn), k))
    np.savez(fn,
             nu=nu, M=M, peak=R["peak"][0], t_peak=R["peak"][1], verdict=R["verdict"],
             tailmax=R["tailmax"], track=R["track"], bar=bar, cap=cap,
             c_pk=(R["c_pk"] if R["c_pk"] is not None else np.zeros(1, complex)))
    return R

def stage_scan():
    """Sigma_peak(nu) from the decay side, descending until verdicts stop
    being DECAY (or fronts hit the truncation)."""
    print("stage scan: excursion-peak program (M=1024)")
    for nu in (0.0540, 0.0530, 0.0522, 0.0516, 0.0512, 0.0510,
               0.0509, 0.0508, 0.0507, 0.0506, 0.0505, 0.0503, 0.0500):
        R = stage_peak(nu)
        if not R["verdict"].startswith("DECAY"):
            print("  [scan stops: first non-decay verdict]")
            break

def stage_fit():
    """Fit Sigma_peak ~ C (nu - nu*)^(-gamma) over the DECAY points with
    front-clean tails; report nu*, gamma, and fit quality."""
    import glob
    from scipy.optimize import curve_fit
    rows = []
    for f in sorted(glob.glob(os.path.join(DIR, "peak_nu*.npz"))):
        d = np.load(f, allow_pickle=True)
        if not str(d["verdict"]).startswith("DECAY"):
            continue
        nu, pk, tm = float(d["nu"]), float(d["peak"]), float(d["tailmax"])
        if tm > 1e-6*pk:
            print("   [excluded: nu=%.6f tail-contaminated (tailmax/peak=%.1e)]" % (nu, tm/pk))
            continue
        rows.append((nu, pk, tm))
    rows.sort()
    nus = np.array([r[0] for r in rows])
    pks = np.array([r[1] for r in rows])
    print("stage fit: %d decay-side points, nu in [%.4f, %.4f], peaks %.3g..%.3g"
          % (len(rows), nus.min(), nus.max(), pks.min(), pks.max()))
    for nu, pk, tm in rows:
        print("   nu=%.6f  Sigma_peak=%.6e  tailmax=%.2e" % (nu, pk, tm))
    def model(nu, nustar, C, g):
        return C*(nu - nustar)**(-g)
    p0 = (nus.min() - 1e-4, 1.0, 1.0)
    try:
        p, cov = curve_fit(model, nus, pks, p0=p0, maxfev=100000)
        perr = np.sqrt(np.diag(cov))
        print("  fit Sigma_peak = C (nu-nu*)^-gamma :")
        print("    nu* = %.6f +- %.1e   gamma = %.4f +- %.3f   C = %.4g"
              % (p[0], perr[0], p[2], perr[2], p[1]))
        res = pks - model(nus, *p)
        print("    relative rms residual: %.2e" % np.sqrt(np.mean((res/pks)**2)))
    except Exception as e:
        print("  fit failed:", e)

def stage_gam2():
    """Mechanism test for gamma=2 (ledger N16): threshold-scaling theory gives
    Sigma_peak ~ eps^(-mu/lambda_u), eps ~ nu-nu*, where mu = growth rate of
    the critical (threshold) trajectory's Sigma and lambda_u = unstable
    eigenvalue (rate at which adjacent-nu trajectories separate). gamma = 2
    <=> mu = 2*lambda_u. Both rates are measured here from saved tracks:
    mu from the common climb of the two nearest-threshold clean runs, lambda_u
    from the exponential growth of their separation on the same stretch."""
    import glob
    d = {}
    for f in glob.glob(os.path.join(DIR, "peak_nu*.npz")):
        z = np.load(f, allow_pickle=True)
        if str(z["verdict"]).startswith("DECAY"):
            d[round(float(z["nu"]), 6)] = z["track"]
    nus = sorted(d)
    print("stage gam2: rate measurements from tracks; decay-side nus:", nus)
    nu1, nu2 = nus[0], nus[1]              # two nearest-threshold clean runs
    t1, s1 = d[nu1][:, 0], d[nu1][:, 1]
    t2, s2 = d[nu2][:, 0], d[nu2][:, 1]
    # common time grid on the climb (before either peak)
    tp1 = t1[np.argmax(s1)]; tp2 = t2[np.argmax(s2)]
    tmax = min(tp1, tp2)
    grid = np.linspace(2.0, tmax - 0.3, 400)
    S1 = np.interp(grid, t1, s1)
    S2 = np.interp(grid, t2, s2)
    sep = np.abs(S1 - S2)
    # restrict to the stretch where separation is exponential and Sigma climbing
    m = (sep > 1e-8*S1) & (S1 > 3.0)
    mu = np.polyfit(grid[m], np.log(0.5*(S1[m] + S2[m])), 1)[0]
    lam = np.polyfit(grid[m], np.log(sep[m]), 1)[0]
    print("  window t in [%.2f, %.2f] (peaks at %.2f/%.2f)" % (grid[m][0], grid[m][-1], tp1, tp2))
    print("  mu (climb rate of Sigma)        = %.4f" % mu)
    print("  lambda_u (separation rate)      = %.4f" % lam)
    print("  gamma candidates: mu/lam = %.4f ; lam/mu = %.4f   (measured gamma -> 2)" % (mu/lam, lam/mu))

def _load_tracks(tailtol=2e-6):
    """Decay-side runs, sorted by nu; clean = tail at least ~6 orders inside
    truncation (tailfrac < tailtol). 0.0508 (M=2048, tailfrac 3.6e-6) rides the
    boundary: included under the default tol with its flag printed."""
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(DIR, "peak_nu*.npz"))):
        z = np.load(f, allow_pickle=True)
        if not str(z["verdict"]).startswith("DECAY"):
            continue
        tr = np.asarray(z["track"], float)
        rows.append(dict(nu=round(float(z["nu"]), 6), peak=float(z["peak"]),
                         t_peak=float(z["t_peak"]),
                         tailfrac=float(z["tailmax"])/float(z["peak"]), track=tr))
    rows.sort(key=lambda r: r["nu"])
    for r in rows:
        r["clean"] = r["tailfrac"] < tailtol
        tr = r["track"]
        i = int(np.argmax(tr[:, 1]))
        r["tt"], r["ss"] = tr[:i+1, 0], tr[:i+1, 1]     # climb portion
    return rows

def _logS(r, tq):
    """log Sigma of track r at query times tq (linear interp of log S in t)."""
    return np.interp(tq, r["tt"], np.log(r["ss"]))

def _plaw_fit(tt, ss, t0):
    """log S = log A - p log(t0-t): returns (p, A, rms of log-residual)."""
    X, Y = np.log(t0 - tt), np.log(ss)
    c = np.polyfit(X, Y, 1)
    return -c[0], np.exp(c[1]), np.sqrt(np.mean((Y - np.polyval(c, X))**2))

def _plaw_fit_free(tt, ss, t0grid):
    """Same, with t0 free: scan+parabolic refine on the rms(t0) curve."""
    rms = np.array([_plaw_fit(tt, ss, t0)[2] for t0 in t0grid])
    i = int(np.argmin(rms))
    if 0 < i < len(t0grid) - 1:
        d = t0grid[1] - t0grid[0]
        t0 = t0grid[i] + 0.5*d*(rms[i-1] - rms[i+1])/(rms[i-1] - 2*rms[i] + rms[i+1])
    else:
        t0 = t0grid[i]
    p, A, e = _plaw_fit(tt, ss, t0)
    return p, A, t0, e

def stage_pfit(nustar=0.05057):
    """S07: power-law decomposition of the excursion-divergence law (N16).
    Peel-off scaling frame: critical trajectory Sigma*(t) = A (t0-t)^-p;
    linear deviation delta(t) = B eps (t0-t)^-kappa, eps = nu - nu*;
    peel-off (delta ~ Sigma*) gives Sigma_peak ~ eps^-gamma, gamma = p/(kappa-p),
    so the measured gamma -> 2 requires kappa = 3p/2.  Two scenarios to
    discriminate: (I) p=1 (inviscid-CLM-like climb) with ANOMALOUS kappa=3/2;
    (II) p=2 with kappa=3 = p+1 = the universal TIME-SHIFT mode.
    Model corollaries used as cross-checks:
      t_peak = t0 - [(kappa B eps)/(p A)]^{1/(kappa-p)}   (linear in eps iff kappa-p = ... exponent 1)
      Sigma_peak (t0-t_peak)^p = A (1 - p/kappa)."""
    from scipy.optimize import curve_fit
    rows = _load_tracks()
    print("stage pfit: nu* = %.5f (N16); tracks:" % nustar)
    for r in rows:
        print("   nu=%.6f  peak=%.6e t_peak=%.4f tail/pk=%.1e  %s"
              % (r["nu"], r["peak"], r["t_peak"], r["tailfrac"],
                 "CLEAN" if r["clean"] else "excluded (truncation-biased peak)"))
    cl = [r for r in rows if r["clean"]]

    # ---- A) peak-locus fit: t_peak = t0 - c*Sigma_peak^(-1/p) --------------
    S = np.array([r["peak"] for r in cl]); T = np.array([r["t_peak"] for r in cl])
    print("A) peak locus (all %d clean peaks; fixed-p linear fits of t_peak vs S^(-1/p)):" % len(S))
    for p in (1.0, 1.5, 2.0, 2.5, 3.0):
        X = S**(-1.0/p)
        c = np.polyfit(X, T, 1)
        res = T - np.polyval(c, X)
        print("   p=%.2f: t0=%.4f  c=%.4g   rms(t_peak resid)=%.2e" % (p, c[1], -c[0], np.sqrt(np.mean(res**2))))
    def mA(S, t0, c, p):
        return t0 - c*S**(-1.0/p)
    pA, _ = curve_fit(mA, S, T, p0=(T.max()+0.05, 5.0, 2.0), maxfev=200000)
    resA = T - mA(S, *pA)
    print("   free-p:  t0=%.4f  c=%.4g  p=%.4f   rms=%.2e" % (pA[0], pA[1], pA[2], np.sqrt(np.mean(resA**2))))
    S5, T5 = S[:5], T[:5]     # five nearest-threshold clean peaks
    pA5, _ = curve_fit(mA, S5, T5, p0=(T.max()+0.05, 5.0, 2.0), maxfev=200000)
    print("   free-p, 5 nearest peaks: t0=%.4f  c=%.4g  p=%.4f  rms=%.2e"
          % (pA5[0], pA5[1], pA5[2], np.sqrt(np.mean(T5 - mA(S5, *pA5)))))
    t0A = pA5[0]
    # eps-linearity of the peak locus (kappa-p exponent test): t0 - t_peak vs eps
    eps = np.array([r["nu"] - nustar for r in cl])
    ce = np.polyfit(np.log(eps[:6]), np.log(t0A - T[:6]), 1)
    print("   (t0 - t_peak) ~ eps^q on 6 nearest: q = %.4f  (q = p/... = gamma/p; scenario II predicts 1)" % ce[0])

    # ---- B) climb fits for p ----------------------------------------------
    print("B) climb fits, Sigma* ~ A (t0-t)^-p:")
    t0grid = np.arange(8.16, 8.40, 0.002)
    for r in cl[:3]:
        for f in (0.03, 0.1, 0.3):
            m = (r["ss"] >= 30.0) & (r["ss"] <= f*r["peak"])
            if m.sum() < 12:
                continue
            p1, A1, e1 = _plaw_fit(r["tt"][m], r["ss"][m], t0A)
            p2, A2, t02, e2 = _plaw_fit_free(r["tt"][m], r["ss"][m], t0grid)
            print("   single nu=%.4f  S in [30, %.2f pk]: t0 fixed %.4f -> p=%.4f A=%.3g (rms %.1e); t0 free %.4f -> p=%.4f A=%.3g"
                  % (r["nu"], f, t0A, p1, A1, e1, t02, p2, A2))
    # nu*-Richardson extrapolation to the critical trajectory (kills the O(eps) deviation)
    print("   Richardson (eliminates O(eps); windows relative to the smaller-eps track's peak):")
    trips = [(cl[0], cl[1]), (cl[1], cl[2]), (cl[0], cl[1], cl[2])]
    for tr_ in trips:
        r1 = tr_[0]
        e_ = [r["nu"] - nustar for r in tr_]
        for f in (0.1, 0.3, 0.6):
            m = (r1["ss"] >= 30.0) & (r1["ss"] <= f*r1["peak"])
            tq = r1["tt"][m]
            tq = tq[tq <= tr_[-1]["t_peak"] - 0.005]
            if len(tq) < 12:
                continue
            Svals = [np.exp(_logS(r, tq)) for r in tr_]
            if len(tr_) == 2:
                Sstar = (e_[1]*Svals[0] - e_[0]*Svals[1])/(e_[1] - e_[0])
                lab = "pair (%.4f,%.4f)" % (tr_[0]["nu"], tr_[1]["nu"])
            else:
                # 2nd-order: Lagrange extrapolation to eps=0 through three tracks
                L = [np.prod([e_[j] for j in range(3) if j != i]) /
                     np.prod([e_[j] - e_[i] for j in range(3) if j != i]) for i in range(3)]
                Sstar = sum(L[i]*Svals[i] for i in range(3))
                lab = "triple 2nd-order"
            ok = Sstar > 0
            p1, A1, e1 = _plaw_fit(tq[ok], Sstar[ok], t0A)
            p2, A2, t02, e2 = _plaw_fit_free(tq[ok], Sstar[ok], t0grid)
            print("   %s f=%.1f: t0 fixed -> p=%.4f A=%.3g (rms %.1e); t0 free %.4f -> p=%.4f A=%.3g"
                  % (lab, f, p1, A1, e1, t02, p2, A2))

    # ---- C) separation fits for kappa, NEAR-PEAK window --------------------
    print("C) adjacent-pair separations sep ~ B*dnu*(t0-t)^-kappa, near-peak window u in [1.5,10]*u_pk(nu2):")
    for i in range(len(cl) - 1):
        r1, r2 = cl[i], cl[i+1]
        upk2 = t0A - r2["t_peak"]
        tq = r1["tt"][(t0A - r1["tt"] >= 1.5*upk2) & (t0A - r1["tt"] <= 10.0*upk2)]
        tq = tq[(tq >= r2["tt"][0]) & (tq <= r2["tt"][-1])]
        if len(tq) < 12:
            print("   pair (%.4f,%.4f): too few usable points" % (r1["nu"], r2["nu"]))
            continue
        S1 = np.exp(_logS(r1, tq)); S2 = np.exp(_logS(r2, tq))
        sep = S1 - S2
        m = sep > 0
        k1, C1, e1 = _plaw_fit(tq[m], sep[m], t0A)
        k2, C2, t02, e2 = _plaw_fit_free(tq[m], sep[m], t0grid)
        Bhat = C1/(r2["nu"] - r1["nu"])
        print("   pair (%.4f,%.4f) [%d pts]: t0 fixed -> kappa=%.4f B=%.4g (rms %.1e); t0 free %.4f -> kappa=%.4f"
              % (r1["nu"], r2["nu"], m.sum(), k1, Bhat, e1, t02, k2))

    # ---- D0) per-track near-peak shape test with (p,kappa)=(2,3) FIXED ----
    print("D0) near-peak shape fits, (p,kappa)=(2,3) FIXED: Sigma = A u^-2 - Beps u^-3, u=t0-t,")
    print("    window u in [0.8, 8]*u_pk; params (A, Beps, t0) per track; model peak checks:")
    from scipy.optimize import least_squares
    for r in cl:
        upk = t0A - r["t_peak"]
        m = (t0A - r["tt"] >= 0.8*upk) & (t0A - r["tt"] <= 8.0*upk)
        if m.sum() < 12:
            print("   nu=%.4f: too few points" % r["nu"])
            continue
        tt, ss = r["tt"][m], r["ss"][m]
        def res(q):
            A, Be, t0 = q
            u = t0 - tt
            v = A/u**2 - Be/u**3
            return np.log(np.maximum(v, 1e-12)) - np.log(ss)
        q0 = (3.0*r["peak"]*upk**2, 2.0*(3.0*r["peak"]*upk**2)*upk/3.0, t0A)
        sol = least_squares(res, q0, method="lm", max_nfev=20000)
        A_, Be_, t0_ = sol.x
        rms = np.sqrt(np.mean(sol.fun**2))
        ups = 1.5*Be_/A_          # model peak position
        Smod = A_/ups**2 - Be_/ups**3
        eps = r["nu"] - nustar
        print("   nu=%.4f [%3d pts]: A=%.4g  B=Beps/eps=%.4g  t0=%.4f  rms(log)=%.2e | model peak: S=%.4g (obs %.4g), u*=%.4f (obs %.4f)"
              % (r["nu"], m.sum(), A_, Be_/eps, t0_, rms, Smod, r["peak"], ups, t0_ - r["t_peak"]))

    # ---- D) global near-peak normal-form fit, (p,kappa) FREE ---------------
    print("D) global near-peak fit  Sigma_nu(t) = A(t0-t)^-p - B eps (t0-t)^-kappa, u in [0.8,8]*u_pk:")
    TT, SS, EE = [], [], []
    for r in cl:
        upk = t0A - r["t_peak"]
        m = (t0A - r["tt"] >= 0.8*upk) & (t0A - r["tt"] <= 8.0*upk)
        TT.append(r["tt"][m]); SS.append(r["ss"][m])
        EE.append(np.full(int(m.sum()), r["nu"] - nustar))
    TT, SS, EE = map(np.concatenate, (TT, SS, EE))
    def mD(X, logA, t0, p, logB, kap):
        t, e = X
        u = np.maximum(t0 - t, 1e-6)
        v = np.exp(logA)*u**(-p) - np.exp(logB)*e*u**(-kap)
        return np.log(np.maximum(v, 1e-12))
    try:
        pD, _ = curve_fit(mD, (TT, EE), np.log(SS), p0=(np.log(120.), t0A, 2.0, np.log(12000.), 3.0),
                          maxfev=400000)
        rmsD = np.sqrt(np.mean((np.log(SS) - mD((TT, EE), *pD))**2))
        A_, t0_, p_, B_, k_ = np.exp(pD[0]), pD[1], pD[2], np.exp(pD[3]), pD[4]
        print("   [%d pts] A=%.4g  t0=%.4f  p=%.4f  B=%.4g  kappa=%.4f   rms(log)=%.2e"
              % (len(SS), A_, t0_, p_, B_, k_, rmsD))
        if k_ != p_:
            print("   implied gamma = p/(kappa-p) = %.4f   (S05 direct: gamma_loc -> 1.990)" % (p_/(k_ - p_)))
        print("E) consistency: Sigma_peak (t0-t_peak)^p / [A(1-p/kappa)] (should be 1):")
        for r in cl:
            val = r["peak"]*(t0_ - r["t_peak"])**p_/(A_*(1 - p_/k_))
            print("   nu=%.4f: %.4f" % (r["nu"], val))
    except RuntimeError as e:
        print("   global free-(p,kappa) fit DID NOT CONVERGE (%s) — expected: no settled" % e)
        print("   power law exists in the accessible window (see stage lam); the locus")
        print("   analysis in section A carries the decomposition instead.")

def _local_rate(tt, lnS, W=9):
    """lambda(t_i) = d lnS/dt by local linear regression over +-W points."""
    lam = np.full(len(tt), np.nan)
    for i in range(W, len(tt) - W):
        sl = slice(i - W, i + W + 1)
        lam[i] = np.polyfit(tt[sl], lnS[sl], 1)[0]
    return lam

def stage_lam(nustar=0.05057):
    """S07, parameter-free local-rate analysis (no t0 assumed).
    If Sigma ~ A (t0-t)^-p then 1/lambda(t) = (t0-t)/p: LINEAR in t, slope
    -1/p, root t0.  If instead the climb is exponential-attractor-like,
    1/lambda flattens to a constant.  Applied to every decay track's climb.
    Truncation honesty: triangularity makes modes 1..M exact, so a track's
    Sigma_M is trustworthy until the front nears M; S05 margin calibration
    (tail/pk 4e-9 at m*=M/5.0, 1.3e-7 at M/3.8, 3.6e-6 at M/2.7) gives the
    conservative cap  Sigma <= nu (M/2.7)^2  [m* = sqrt(Sigma/nu)].
    The near-threshold 0.0506/0.0507 tracks peel off at u ~ 160.9 eps
    (0.005/0.021), far below their caps' u — their capped climbs sample the
    CRITICAL trajectory nearly deviation-free: the best shape data we own."""
    rows = _load_tracks(tailtol=np.inf)
    M_of = {0.0508: 2048, 0.0509: 2048, 0.0510: 2048}
    print("stage lam: local rate lambda = d lnSigma/dt; 1/lambda linear <=> power law")
    for r in rows:
        M = M_of.get(r["nu"], 1024)
        cap = min(0.05*(M/2.7)**2, 0.8*r["peak"])
        tt, ss = r["tt"], r["ss"]
        m = ss >= 20.0
        tt, ss = tt[m], ss[m]
        lam = _local_rate(tt, np.log(ss))
        ok = np.isfinite(lam) & (lam > 0) & (ss <= cap)
        # table at Sigma levels
        levels = [30, 100, 300, 1000, 3000, 10000, 30000]
        out = []
        for L in levels:
            if L > cap or L > ss[ok].max():
                continue
            i = np.argmin(np.abs(np.log(ss) - np.log(L)) + 1e9*(~ok))
            out.append("S=%-6g t=%.3f 1/lam=%.4f" % (L, tt[i], 1.0/lam[i]))
        print("  nu=%.4f (M=%d, cap=%.3g): %s" % (r["nu"], M, cap, " | ".join(out)))
        # late-window linear fit of 1/lambda vs t: Sigma in [cap/30, cap]
        m2 = ok & (ss >= cap/30.0) & (ss <= cap)
        if m2.sum() >= 12:
            c = np.polyfit(tt[m2], 1.0/lam[m2], 1)
            resid = 1.0/lam[m2] - np.polyval(c, tt[m2])
            print("      1/lam fit on S in [%.3g, %.3g] (%d pts): slope=%.4f -> p_hat=%.3f, root t0_hat=%.4f, rms=%.2e"
                  % (cap/30.0, cap, m2.sum(), c[0], -1.0/c[0], -c[1]/c[0], np.sqrt(np.mean(resid**2))))
    # separations of the two nearest pairs on their trustworthy stretches
    print("  separations (same instrument; kappa if linear):")
    bynu = {r["nu"]: r for r in rows}
    for n1, n2 in [(0.0506, 0.0507), (0.0507, 0.0508), (0.0509, 0.051),
                   (0.051, 0.0512)]:
        if n1 not in bynu or n2 not in bynu:
            continue
        r1, r2 = bynu[n1], bynu[n2]
        cap = min(0.05*(M_of.get(n1, 1024)/2.7)**2, 0.05*(M_of.get(n2, 1024)/2.7)**2,
                  0.8*r2["peak"])
        tq = r1["tt"][(r1["ss"] >= 20.0) & (r1["ss"] <= cap)]
        tq = tq[(tq >= r2["tt"][0]) & (tq <= r2["tt"][-1])]
        S1 = np.exp(_logS(r1, tq)); S2 = np.exp(_logS(r2, tq))
        sep = S1 - S2
        m = (sep > 1e-3*S1) & (sep < 0.3*S1) & (S2 <= cap)
        if m.sum() < 40:
            print("   pair (%.4f,%.4f): too few points (%d)" % (n1, n2, m.sum()))
            continue
        tq2, sep2 = tq[m], sep[m]
        lam = _local_rate(tq2, np.log(sep2))
        ok = np.isfinite(lam) & (lam > 0)
        c = np.polyfit(tq2[ok][-200:], 1.0/lam[ok][-200:], 1)
        print("   pair (%.4f,%.4f) [%d pts, sep/S1 %.3f..%.3f]: late 1/lam slope=%.4f -> kappa_hat=%.3f, t0_hat=%.4f"
              % (n1, n2, m.sum(), (sep2/np.exp(_logS(bynu[n1], tq2)))[0],
                 (sep2/np.exp(_logS(bynu[n1], tq2)))[-1], c[0], -1.0/c[0], -c[1]/c[0]))

def stage_blow(nu, M=2048, Tmax=40.0, bar=None):
    """Blowup-side run (nu < nu*): ride to the bar, save the track.
    Modes 1..M stay exact (triangularity); Sigma_M is trustworthy while the
    front is inside the cap nu (M/2.7)^2 — so the bar defaults to just above
    that cap (riding further has no analysis value; S07 economics lesson).
    Purpose: the T*(nu) locus and the blowup-side local exponent via the
    1/lambda instrument (stage tstar)."""
    t0 = time.time()
    if bar is None:
        bar = 1.7*0.05*(M/2.7)**2
    R = ifrk4_run(nu, M, Tmax, sum_stop_hi=bar)
    print("  nu=%.6f M=%d: %-16s Sigma_end=%.6e t_end=%.4f tailmax=%.2e  [%d steps, %.0f s]"
          % (nu, M, R["verdict"], np.sum(np.abs(R["c"])), R["t"],
             R["tailmax"], R["nsteps"], time.time() - t0))
    # S11 ops note: the historical flat name blow_nu<nu>.npz collided with the
    # S10 M=4096 records when the M=16384 rides reused the same nu values
    # (caught mid-flight; S10 set backed up to blow_s10_M4096/ before any
    # landing). M now in the filename; keep it there.
    np.savez(os.path.join(DIR, "blow_nu%s_M%d.npz"
                          % (("%.6f" % nu).replace(".", "p"), M)),
             nu=nu, M=M, t_end=R["t"], verdict=R["verdict"],
             tailmax=R["tailmax"], track=R["track"], fft_nl=FFT_NL)
    return R

def stage_tstar(nustar=0.05057):
    """Analyze blow_nu*.npz: per run, 1/lambda fit on the truncation-clean top
    stretch -> local exponent p' and root T*; bar-crossing times as the
    bar-continuation cross-check; then the T*(nu) locus vs the decay-side
    t_peak locus (t0 = 8.2104, c' = 161)."""
    import glob
    print("stage tstar: blowup-side locus (nu* = %.5f)" % nustar)
    res = []
    for f in sorted(glob.glob(os.path.join(DIR, "blow_nu*.npz"))):
        z = np.load(f, allow_pickle=True)
        nu, M = float(z["nu"]), int(z["M"])
        tr = np.asarray(z["track"], float)
        tt, ss = tr[:, 0], tr[:, 1]
        cap = 0.05*(M/2.7)**2
        m = (ss >= 20.0) & (ss <= cap)
        lam = _local_rate(tt[m], np.log(ss[m]))
        ok = np.isfinite(lam) & (lam > 0)
        t2, il = tt[m][ok], 1.0/lam[ok]
        s2 = ss[m][ok]
        # per-Sigma-level table
        lev = [100, 1000, 10000, 25000]
        tab = []
        for L in lev:
            if L > s2.max():
                continue
            i = int(np.argmin(np.abs(np.log(s2) - np.log(L))))
            tab.append("S=%-6g t=%.3f 1/lam=%.4f" % (L, t2[i], il[i]))
        # top-stretch linear fit of 1/lambda
        m2 = s2 >= cap/30.0
        c = np.polyfit(t2[m2], il[m2], 1)
        # bar crossings
        bars = {}
        for L in (1e3, 1e4, cap):
            j = np.argmax(ss >= L)
            bars[L] = tt[j] if ss[j] >= L else np.nan
        print("  nu=%.6f (M=%d, cap=%.3g): %s" % (nu, M, cap, " | ".join(tab)))
        print("      top-stretch 1/lam: slope=%.4f -> p'=%.3f, root T*=%.4f; bar crossings t(1e3)=%.4f t(1e4)=%.4f t(cap)=%.4f"
              % (c[0], -1.0/c[0], -c[1]/c[0], bars[1e3], bars[1e4], bars[cap]))
        res.append((nu, -c[1]/c[0], bars))
    if len(res) >= 3:
        nus = np.array([r[0] for r in res])
        Ts = np.array([r[1] for r in res])
        cfit = np.polyfit(nus, Ts, 1)
        print("  locus fit T*(nu) linear: slope=%.1f  T*(nu*)=%.4f   [decay side: slope -161, t0=8.2104]"
              % (cfit[0], np.polyval(cfit, nustar)))
        for L in (1e3, 1e4):
            Tb = np.array([r[2][L] for r in res])
            if np.all(np.isfinite(Tb)):
                cb = np.polyfit(nus, Tb, 1)
                print("  locus fit t(bar %.0e) linear: slope=%.1f  at nu*: %.4f" % (L, cb[0], np.polyval(cb, nustar)))

def stage_mode(nu=0.0506, M=1024):
    """S07 double-pole discriminator (N16 mechanism candidates): capture the
    mode vector q_m = |c_m| at Sigma triggers on the near-critical climb and
    fit  ln q_m = a + sigma ln m + m ln r  on the settled bulk.
    Predictions: simple pole -> sigma = 0; DOUBLE pole -> sigma = 1;
    quasi-steady (q1/nu)^m/(m!)^2 -> strong negative curvature (no linear fit).
    Fit window: m in [4, m_hi], m_hi = last m with q_m >= 1e-6 * max_m q_m
    (bulk, front excluded); curvature honesty: rms + split-half sigma drift."""
    print("stage mode: nu=%.4f M=%d — pole-structure discriminator" % (nu, M)); t0 = time.time()
    m2 = (np.arange(1, M+1)**2).astype(float)
    c = np.zeros(M, complex); c[0] = -1j
    t = 0.0
    dt_prev = -1.0; E1 = E2 = None
    # NOTE: modes m <= M are exact regardless of the front's truncation status
    # (triangularity) — deep snapshots are valid for the bulk (C m + D) r^m
    # fits even where Sigma_M underestimates Sigma_true. Deep triggers extend
    # the D(t) measurement toward the peak (the D -> const question).
    trig = [100.0, 300.0, 1000.0, 3000.0, 6500.0, 15000.0, 40000.0, 90000.0]
    ti = 0
    snaps = []
    while ti < len(trig) and t < 400.0:
        S = np.sum(np.abs(c))
        if S >= trig[ti]:
            snaps.append((t, S, np.abs(c).copy()))
            ti += 1
            continue
        dt = min(0.02, 0.05/max(S, 1e-12))
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(c)
        U2 = E2*(c + 0.5*dt*k1)
        k2 = NL(U2)
        U3 = E2*c + 0.5*dt*k2
        k3 = NL(U3)
        U4 = E1*c + dt*E2*k3
        k4 = NL(U4)
        c = E1*c + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt
    np.savez(os.path.join(DIR, "mode_nu%s.npz" % ("%.6f" % nu).replace(".", "p")),
             nu=nu, M=M, snaps_t=[s[0] for s in snaps], snaps_S=[s[1] for s in snaps],
             q=np.array([s[2] for s in snaps]))
    for (ts, S, q) in snaps:
        mm = np.arange(1, M+1)
        qmax = q.max()
        hi = np.where(q >= 1e-6*qmax)[0]
        m_hi = hi[-1] + 1 if len(hi) else M
        sel = (mm >= 4) & (mm <= m_hi)
        X = np.column_stack([np.ones(sel.sum()), np.log(mm[sel]), mm[sel]])
        Y = np.log(q[sel])
        coef, res, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        rms = np.sqrt(np.mean((Y - X@coef)**2))
        # split-half drift
        half = mm[sel] <= 0.5*(4 + m_hi)
        c1, _, _, _ = np.linalg.lstsq(X[half], Y[half], rcond=None)
        c2, _, _, _ = np.linalg.lstsq(X[~half], Y[~half], rcond=None)
        print("  t=%.3f S=%.4g m_hi=%d: sigma=%.3f (halves %.3f/%.3f)  r=%.5f  rms=%.3f"
              % (ts, S, m_hi, coef[1], c1[1], c2[1], np.exp(coef[2]), rms))
    print("  [prediction: double pole sigma->1, simple pole sigma->0]")
    print("[%.0f s]" % (time.time() - t0))

def stage_modepk(nu=0.0506, M=1024):
    """S08: through-peak mode snapshots — the D<0 post-peak test (N16 corollary:
    delta_dot = -(5/12)D forces D(t_peak) = 0 and D < 0 on the receding side).
    Captures on the rise at [4e4, 9e4, 1.5e5], near the peak (S falling to
    0.98*runmax), and on the fall at [1.5e5, 9e4, 4e4, 1.5e4, 6.5e3, 3e3].
    Modes m <= M are exact regardless of front truncation (triangularity)."""
    print("stage modepk: nu=%.4f M=%d — through-peak pole tracking" % (nu, M)); t0 = time.time()
    m2 = (np.arange(1, M+1)**2).astype(float)
    c = np.zeros(M, complex); c[0] = -1j
    t = 0.0
    dt_prev = -1.0; E1 = E2 = None
    up = [4e4, 9e4, 1.5e5]; dn = [1.5e5, 9e4, 4e4, 1.5e4, 6.5e3, 3e3]
    iu = idn = 0
    runmax = 0.0; got_peak = False
    snaps = []
    while idn < len(dn) and t < 400.0:
        S = np.sum(np.abs(c))
        runmax = max(runmax, S)
        falling = S < 0.98*runmax
        if not falling and iu < len(up) and S >= up[iu]:
            snaps.append((t, S, "rise", np.abs(c).copy())); iu += 1
        elif falling and not got_peak:
            snaps.append((t, S, "peak", np.abs(c).copy())); got_peak = True
        elif falling and S <= dn[idn]:
            while idn < len(dn) and S <= dn[idn]:
                idn += 1
            snaps.append((t, S, "fall", np.abs(c).copy()))
        dt = min(0.02, 0.05/max(S, 1e-12))
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(c)
        U2 = E2*(c + 0.5*dt*k1)
        k2 = NL(U2)
        U3 = E2*c + 0.5*dt*k2
        k3 = NL(U3)
        U4 = E1*c + dt*E2*k3
        k4 = NL(U4)
        c = E1*c + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt
    np.savez(os.path.join(DIR, "modepk_nu%s.npz" % ("%.6f" % nu).replace(".", "p")),
             nu=nu, M=M, snaps_t=[s[0] for s in snaps], snaps_S=[s[1] for s in snaps],
             phase=[s[2] for s in snaps], q=np.array([s[3] for s in snaps]))
    print("  captured %d snapshots (runmax Sigma_M = %.4g)  [%.0f s]"
          % (len(snaps), runmax, time.time() - t0))
    # (C, D, r) fits, D sign free
    from scipy.optimize import curve_fit
    for (ts, S, ph, q) in snaps:
        mm = np.arange(1, M+1).astype(float)
        hi = np.where(q >= 1e-6*q.max())[0]
        m_hi = hi[-1] + 1 if len(hi) else M
        sel = (mm >= 8) & (mm <= m_hi)
        def f(m, C, D, lnr):
            return np.log(np.maximum(C*m + D, 1e-300)) + m*lnr
        p0 = (12*nu, 0.0, -np.sqrt(12*nu/S))
        try:
            p, _ = curve_fit(f, mm[sel], np.log(q[sel]), p0=p0, maxfev=100000)
            rms = np.sqrt(np.mean((np.log(q[sel]) - f(mm[sel], *p))**2))
            print("  %-4s t=%.4f S=%-9.4g: C=%.4f (C/12nu=%.3f)  D=%+.4f  delta=%.6f  rms=%.4f"
                  % (ph, ts, S, p[0], p[0]/(12*nu), p[1], -p[2], rms))
        except RuntimeError:
            print("  %-4s t=%.4f S=%-9.4g: fit failed" % (ph, ts, S))

def stage_pole(nu=0.0506):
    """S07 pole-dynamics closure test (N16). Matched-asymptotics results
    (derived this session, hand algebra): writing Q near its singularity
    s = x + i delta as Q = -C/s^2 + i D/s + E + ..., the balances give
      s^-4:  C = 12 nu                      (parameter-free residue law)
      s^-3:  delta_dot = -(5/12) D          (approach speed slaved to the
                                             simple-pole admixture)
      s^-2:  E = -D^2/(144 nu).
    Test on the saved mode snapshots: fit q_m = (C m + D) r^m (bulk window),
    then compare (i) C vs 12nu, (ii) finite-difference delta_dot vs -(5/12)D."""
    from scipy.optimize import curve_fit
    z = np.load(os.path.join(DIR, "mode_nu%s.npz" % ("%.6f" % nu).replace(".", "p")),
                allow_pickle=True)
    qs = z["q"]; Ts = np.array(z["snaps_t"], float); Ss = np.array(z["snaps_S"], float)
    print("stage pole: residue law and rate law tests (nu=%.4f, 12nu = %.4f)" % (nu, 12*nu))
    ds, Ds = [], []
    for (ts, Sv, q) in zip(Ts, Ss, qs):
        mm = np.arange(1, len(q)+1).astype(float)
        hi = np.where(q >= 1e-6*q.max())[0]
        m_hi = hi[-1] + 1 if len(hi) else len(q)
        sel = (mm >= 4) & (mm <= m_hi)
        def f(m, C, D, lnr):
            return np.log(np.maximum(C*m + D, 1e-300)) + m*lnr
        p0 = (12*nu, 1.0, -np.sqrt(0.61/Sv))
        p, _ = curve_fit(f, mm[sel], np.log(q[sel]), p0=p0, maxfev=100000)
        C_, D_, r_ = p[0], p[1], np.exp(p[2])
        rms = np.sqrt(np.mean((np.log(q[sel]) - f(mm[sel], *p))**2))
        ds.append(-p[2]); Ds.append(D_)
        print("  t=%.3f S=%-7.4g: C=%.4f (C/12nu=%.3f)  D=%.4f  delta=%.5f  rms=%.3f"
              % (ts, Sv, C_, C_/(12*nu), D_, -p[2], rms))
    ds = np.array(ds); Ds = np.array(Ds)
    print("  rate law delta_dot = -(5/12) D:")
    for i in range(len(Ts)-1):
        dd = (ds[i+1] - ds[i])/(Ts[i+1] - Ts[i])
        Dm = 0.5*(Ds[i] + Ds[i+1])
        print("   t=%.2f..%.2f: delta_dot(FD) = %+.5f   -(5/12)D = %+.5f   ratio = %.3f"
              % (Ts[i], Ts[i+1], dd, -(5.0/12.0)*Dm, dd/(-(5.0/12.0)*Dm)))

def stage_nlv():
    """S11 FFT-NL validation (predicted before first run):
    N1 FFT-vs-direct NL on random complex data, M in {1024, 2048, 4096, 8192}:
       rel sup diff <= 1e-13 (FFT convolution rounding ~ eps*log M with unit-
       scale data).
    N2 trajectory: ifrk4_run nu=0.054, M=2048, Tmax=6 both paths:
       |dSigma|/Sigma at end <= 1e-10 (1e-15-level seeds amplified by the
       excursion's local growth e^{~7}).
    N3 timing at M=8192 (expect >= 30x)."""
    import time as _t
    global FFT_NL
    rng = np.random.default_rng(7)
    print("stage nlv: FFT-NL validation")
    for M in (1024, 2048, 4096, 8192):
        c = (rng.standard_normal(M) + 1j*rng.standard_normal(M))/np.sqrt(M)
        FFT_NL = False; a_ = NL(c)
        FFT_NL = True;  b_ = NL(c)
        FFT_NL = False
        d = np.abs(a_ - b_).max()/max(np.abs(a_).max(), 1e-300)
        print("  N1 M=%5d: rel sup diff %.2e" % (M, d))
    FFT_NL = False
    R1 = ifrk4_run(0.054, 2048, 6.0)
    FFT_NL = True
    R2 = ifrk4_run(0.054, 2048, 6.0)
    FFT_NL = False
    S1 = np.sum(np.abs(R1["c"])); S2 = np.sum(np.abs(R2["c"]))
    print("  N2 trajectory nu=0.054 M=2048 T=6: Sigma %.10e vs %.10e  rel %.2e  (peaks %.6e/%.6e)"
          % (S1, S2, abs(S1 - S2)/S1, R1["peak"][0], R2["peak"][0]))
    M = 8192
    c = (rng.standard_normal(M) + 1j*rng.standard_normal(M))/np.sqrt(M)
    t0 = _t.perf_counter()
    for _ in range(3): NL(c)
    td = (_t.perf_counter() - t0)/3
    FFT_NL = True
    NL(c)
    t0 = _t.perf_counter()
    for _ in range(20): NL(c)
    tf = (_t.perf_counter() - t0)/20
    FFT_NL = False
    print("  N3 M=8192: direct %.1f ms  fft %.2f ms  speedup %.0fx" % (td*1e3, tf*1e3, td/tf))

def stage_sig(sigma, nus, M=1024, Tmax=400.0):
    """S11: sigma-generalized excursion probe (the P(sigma) interpolation
    program; N19 prediction iii). For each nu: ride, report verdict +
    Sigma_peak + t_peak + tail; on the DEEPEST clean decay point also fit the
    peak-mode profile (pole-order discriminator sigma_prefactor: simple pole
    -> 0, double pole -> 1). Anchors: sigma=2 reproduces the certified world
    bit-identically; sigma=1 has Sakajo's closed form (nu* = 1/(2e) =
    0.1839..., gamma = 1, simple pole).
    S12 PATCH (the S11 stage-design note executed): the blowup bar now SCALES
    WITH THE CLEAN CAP nu(M/2.7)^sigma (bar = 1.7x cap, the stage_blow ratio;
    the old fixed 1e6 sat 200x past the sigma=1.5 M=4096 cap -- verdicts far
    outside instrument reach). DECAY peaks above the cap are flagged
    INDICATIVE (N15 tail discipline); the deepest-decay pole fit now uses the
    deepest CLEAN decay. Filenames carry _M<M> for M != 1024 (S11 rule);
    existing records are never overwritten (auto .bak)."""
    print("stage sig: sigma=%.3f M=%d" % (sigma, M))
    best = None
    for nu in nus:
        t0 = time.time()
        cap = nu*(M/2.7)**float(sigma)
        bar = 1.7*cap
        # peak-mode variant: snapshot the mode vector at the running peak
        m2 = np.arange(1, M+1).astype(float)**float(sigma) if sigma != 2 else \
            (np.arange(1, M+1)**2).astype(float)
        c = np.zeros(M, complex); c[0] = -1j
        t = 0.0; dt_prev = -1.0; E1 = E2 = None
        peak = (0.0, 0.0); c_pk = c.copy(); tailmax = 0.0; nstep = 0
        track = []   # S15 additive patch: (t, Sigma) every step, for the
                     # N25 sigma-family curvature instrument (s15_kcurv.py).
                     # Print output and all existing npz keys UNCHANGED.
        verdict = "UNDET(Tmax)"
        while t < Tmax:
            S = np.sum(np.abs(c))
            if S > peak[0]:
                peak = (S, t); c_pk = c.copy()
            tailmax = max(tailmax, abs(c[M-1]))
            track.append((t, S))
            if S >= bar:
                verdict = "BLOWUP-bar(%.1e)" % bar; break
            if S <= 1e-3 and t > 1.0:
                verdict = "DECAY"; break
            if not np.isfinite(S):
                verdict = "NAN"; break
            dt = min(0.02, 0.05/max(S, 1e-12), Tmax - t)
            if dt != dt_prev:
                E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
                dt_prev = dt
            k1 = NL(c)
            U2 = E2*(c + 0.5*dt*k1); k2 = NL(U2)
            U3 = E2*c + 0.5*dt*k2;   k3 = NL(U3)
            U4 = E1*c + dt*E2*k3;    k4 = NL(U4)
            c = E1*c + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
            t += dt; nstep += 1
        clean = (peak[0] <= cap)
        print("  nu=%.6f: %-18s Sigma_peak=%.6e t_peak=%.4f tail/pk=%.1e cap=%.2e%s [%d steps, %.0f s]"
              % (nu, verdict, peak[0], peak[1],
                 tailmax/max(peak[0], 1e-300), cap,
                 "" if clean else "  [PEAK>CAP: INDICATIVE]",
                 nstep, time.time() - t0), flush=True)
        if verdict == "DECAY" and clean and (best is None or peak[0] > best[1][0]):
            best = (nu, peak, c_pk)
        tagM = "" if M == 1024 else "_M%d" % M
        fn = os.path.join(DIR, "sig%s_nu%s%s.npz"
                          % (("%.2f" % sigma).replace(".", "p"),
                             ("%.6f" % nu).replace(".", "p"), tagM))
        if os.path.exists(fn):
            k = 1
            while os.path.exists(fn + ".bak%d" % k):
                k += 1
            os.rename(fn, fn + ".bak%d" % k)
            print("  [existing %s backed up to .bak%d]" % (os.path.basename(fn), k))
        np.savez(fn,
                 sigma=sigma, nu=nu, M=M, peak=peak[0], t_peak=peak[1],
                 tailmax=tailmax, verdict=verdict, c_pk=c_pk, cap=cap, bar=bar,
                 track=np.array(track))   # S15: full-resolution track
    if best is not None:
        nu, peak, c_pk = best
        q = np.abs(c_pk)
        mm = np.arange(1, M + 1).astype(float)
        good = q > q.max()*1e-12
        mhi = int(0.6*np.max(mm[good]))
        sel = (mm >= 8) & (mm <= mhi) & good
        x = mm[sel]; y = np.log(q[sel])
        A = np.vstack([np.ones_like(x), np.log(x), x]).T
        cf, *_ = np.linalg.lstsq(A, y, rcond=None)
        h = len(x)//2
        cA, *_ = np.linalg.lstsq(A[:h], y[:h], rcond=None)
        cB, *_ = np.linalg.lstsq(A[h:], y[h:], rcond=None)
        print("  pole-order discriminator at nu=%.6f (deepest decay): sigma_prefactor = %.3f"
              " (halves %.3f/%.3f)  delta = %.5f   [simple pole -> 0, double -> 1]"
              % (nu, cf[1], cA[1], cB[1], -cf[2]))

def stage_snap(sigma, nu, M, Sref, Tmax=400.0):
    """S17 additive stage (): stage_sig's EXACT integrator loop (same
    dt rule, same NL, same E1/E2, same bar/cap, same stop conditions) plus
    snapshot OBSERVERS only — mode-vector snapshots at trigger levels
    {0.15, 0.25, 0.40, 0.60, 0.80, 0.93}*Sref on the rise and again on the
    fall, plus the running peak (c_pk), for the two-instrument kappa(sigma)
    extraction (s17_kappa.py). Sref = the RECORDED Sigma_peak of the same
    (sigma, nu, M) ride; the trajectory itself must be bit-identical to the
    stage_sig record (post-edit obligation (ii) in.
    Output: snap_sig<s>_nu<nu>_M<M>.npz with snaps_t/S/c + the sig keys."""
    print("stage snap: sigma=%.3f nu=%.6f M=%d Sref=%.6e" % (sigma, nu, M, Sref))
    t0 = time.time()
    FR = (0.15, 0.25, 0.40, 0.60, 0.80, 0.93)
    Lup = [f*Sref for f in FR]
    Ldn = [f*Sref for f in reversed(FR)]
    iu = 0; idn = 0
    snaps = []   # (tag, t, S, c.copy())
    cap = nu*(M/2.7)**float(sigma)
    bar = 1.7*cap
    m2 = np.arange(1, M+1).astype(float)**float(sigma) if sigma != 2 else \
        (np.arange(1, M+1)**2).astype(float)
    c = np.zeros(M, complex); c[0] = -1j
    t = 0.0; dt_prev = -1.0; E1 = E2 = None
    peak = (0.0, 0.0); c_pk = c.copy(); tailmax = 0.0; nstep = 0
    track = []
    S_prev = 0.0
    verdict = "UNDET(Tmax)"
    while t < Tmax:
        S = np.sum(np.abs(c))
        if S > peak[0]:
            peak = (S, t); c_pk = c.copy()
        # observers (no effect on the trajectory)
        while iu < len(Lup) and S >= Lup[iu]:
            snaps.append(("up%.2f" % FR[iu], t, S, c.copy())); iu += 1
        if iu == len(Lup) and idn < len(Ldn) and S <= Ldn[idn] and S_prev > Ldn[idn]:
            snaps.append(("dn%.2f" % tuple(reversed(FR))[idn], t, S, c.copy())); idn += 1
        S_prev = S
        tailmax = max(tailmax, abs(c[M-1]))
        track.append((t, S))
        if S >= bar:
            verdict = "BLOWUP-bar(%.1e)" % bar; break
        if S <= 1e-3 and t > 1.0:
            verdict = "DECAY"; break
        if not np.isfinite(S):
            verdict = "NAN"; break
        dt = min(0.02, 0.05/max(S, 1e-12), Tmax - t)
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(c)
        U2 = E2*(c + 0.5*dt*k1); k2 = NL(U2)
        U3 = E2*c + 0.5*dt*k2;   k3 = NL(U3)
        U4 = E1*c + dt*E2*k3;    k4 = NL(U4)
        c = E1*c + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt; nstep += 1
    clean = (peak[0] <= cap)
    print("  nu=%.6f: %-18s Sigma_peak=%.6e t_peak=%.4f tail/pk=%.1e cap=%.2e%s [%d steps, %.0f s]"
          % (nu, verdict, peak[0], peak[1],
             tailmax/max(peak[0], 1e-300), cap,
             "" if clean else "  [PEAK>CAP: INDICATIVE]",
             nstep, time.time() - t0), flush=True)
    print("  snapshots: %d up + %d dn (+peak)" % (iu, idn))
    for tag, ts, Ss, _ in snaps:
        print("    %-8s t=%.4f Sigma=%.6e" % (tag, ts, Ss))
    fn = os.path.join(DIR, "snap_sig%s_nu%s_M%d.npz"
                      % (("%.2f" % sigma).replace(".", "p"),
                         ("%.6f" % nu).replace(".", "p"), M))
    if os.path.exists(fn):
        k = 1
        while os.path.exists(fn + ".bak%d" % k):
            k += 1
        os.rename(fn, fn + ".bak%d" % k)
        print("  [existing %s backed up to .bak%d]" % (os.path.basename(fn), k))
    np.savez(fn,
             sigma=sigma, nu=nu, M=M, Sref=Sref, peak=peak[0], t_peak=peak[1],
             tailmax=tailmax, verdict=verdict, c_pk=c_pk, cap=cap, bar=bar,
             track=np.array(track),
             snaps_tag=np.array([s[0] for s in snaps]),
             snaps_t=np.array([s[1] for s in snaps]),
             snaps_S=np.array([s[2] for s in snaps]),
             snaps_c=np.array([s[3] for s in snaps]))

def stage_snapt(sigma, nu, M, t0, t1, n, Sref=None):
    """S18 additive stage ('): stage_sig's EXACT integrator loop (same
    dt rule, same NL, same E1/E2, same bar/cap, same stop conditions) plus
    TIME-TRIGGERED snapshot observers -- n mode vectors at equally spaced
    times on [t0, t1], for the dense D(t) column near the excursion peak.
    Observers only: the trajectory must be bit-identical to the stage_sig /
    stage_snap record of the same (sigma, nu, M).
    Output: snapt_sig<s>_nu<nu>_M<M>.npz with snaps_t/S/c + the sig keys."""
    print("stage snapt: sigma=%.3f nu=%.6f M=%d t in [%.4f, %.4f] n=%d"
          % (sigma, nu, M, t0, t1, n))
    tt0 = time.time()
    targets = list(np.linspace(t0, t1, n))
    it = 0
    snaps = []
    cap = nu*(M/2.7)**float(sigma)
    bar = 1.7*cap
    m2 = np.arange(1, M+1).astype(float)**float(sigma) if sigma != 2 else         (np.arange(1, M+1)**2).astype(float)
    c = np.zeros(M, complex); c[0] = -1j
    t = 0.0; dt_prev = -1.0; E1 = E2 = None
    peak = (0.0, 0.0); c_pk = c.copy(); tailmax = 0.0; nstep = 0
    track = []
    verdict = "UNDET(Tmax)"
    Tmax = 400.0
    while t < Tmax:
        S = np.sum(np.abs(c))
        if S > peak[0]:
            peak = (S, t); c_pk = c.copy()
        while it < len(targets) and t >= targets[it]:
            snaps.append(("t%02d" % it, t, S, c.copy())); it += 1
        tailmax = max(tailmax, abs(c[M-1]))
        track.append((t, S))
        if S >= bar:
            verdict = "BLOWUP-bar(%.1e)" % bar; break
        if S <= 1e-3 and t > 1.0:
            verdict = "DECAY"; break
        if not np.isfinite(S):
            verdict = "NAN"; break
        dt = min(0.02, 0.05/max(S, 1e-12), Tmax - t)
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(c)
        U2 = E2*(c + 0.5*dt*k1); k2 = NL(U2)
        U3 = E2*c + 0.5*dt*k2;   k3 = NL(U3)
        U4 = E1*c + dt*E2*k3;    k4 = NL(U4)
        c = E1*c + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt; nstep += 1
    clean = (peak[0] <= cap)
    print("  nu=%.6f: %-18s Sigma_peak=%.6e t_peak=%.4f tail/pk=%.1e cap=%.2e%s [%d steps, %.0f s]"
          % (nu, verdict, peak[0], peak[1],
             tailmax/max(peak[0], 1e-300), cap,
             "" if clean else "  [PEAK>CAP: INDICATIVE]",
             nstep, time.time() - tt0), flush=True)
    print("  time-triggered snapshots: %d of %d" % (it, n))
    fn = os.path.join(DIR, "snapt_sig%s_nu%s_M%d.npz"
                      % (("%.2f" % sigma).replace(".", "p"),
                         ("%.6f" % nu).replace(".", "p"), M))
    if os.path.exists(fn):
        k = 1
        while os.path.exists(fn + ".bak%d" % k):
            k += 1
        os.rename(fn, fn + ".bak%d" % k)
        print("  [existing %s backed up to .bak%d]" % (os.path.basename(fn), k))
    np.savez(fn,
             sigma=sigma, nu=nu, M=M, peak=peak[0], t_peak=peak[1],
             tailmax=tailmax, verdict=verdict, c_pk=c_pk, cap=cap, bar=bar,
             track=np.array(track),
             snaps_tag=np.array([s[0] for s in snaps]),
             snaps_t=np.array([s[1] for s in snaps]),
             snaps_S=np.array([s[2] for s in snaps]),
             snaps_c=np.array([s[3] for s in snaps]))

if __name__ == "__main__":
    st = sys.argv[1]
    if st.endswith(":fft"):
        FFT_NL = True
        st = st[:-4]
        print("[FFT_NL enabled]")
    if st == "ifv":     stage_ifv()
    elif st == "nlv":   stage_nlv()
    elif st.startswith("sig:"):
        parts = st.split(":")
        stage_sig(float(parts[1]), [float(x) for x in parts[2].split(",")],
                  M=int(parts[3]) if len(parts) > 3 else 1024)
    elif st.startswith("snapt:"):
        parts = st.split(":")
        stage_snapt(float(parts[1]), float(parts[2]), int(parts[3]),
                    float(parts[4]), float(parts[5]), int(parts[6]))
    elif st.startswith("snap:"):
        parts = st.split(":")
        stage_snap(float(parts[1]), float(parts[2]), int(parts[3]),
                   float(parts[4]))
    elif st.startswith("peak:"):
        parts = st.split(":")
        stage_peak(float(parts[1]), M=int(parts[2]) if len(parts) > 2 else 1024,
                   bar=float(parts[3]) if len(parts) > 3 else None)
    elif st == "scan":  stage_scan()
    elif st == "fit":   stage_fit()
    elif st == "gam2":  stage_gam2()
    elif st == "pfit":  stage_pfit()
    elif st == "lam":   stage_lam()
    elif st.startswith("blow:"):
        parts = st.split(":")
        stage_blow(float(parts[1]), M=int(parts[2]) if len(parts) > 2 else 2048)
    elif st == "tstar": stage_tstar()
    elif st.startswith("modepk"):
        parts = st.split(":")
        stage_modepk(float(parts[1]) if len(parts) > 1 else 0.0506,
                     int(parts[2]) if len(parts) > 2 else 1024)
    elif st.startswith("mode"):
        parts = st.split(":")
        stage_mode(float(parts[1]) if len(parts) > 1 else 0.0506,
                   int(parts[2]) if len(parts) > 2 else 1024)
    elif st == "pole":  stage_pole()
    else: raise SystemExit("unknown stage: %s" % st)
