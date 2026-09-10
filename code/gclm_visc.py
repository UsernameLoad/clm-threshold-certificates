#!/usr/bin/env python3
# gclm_visc.py — viscous gCLM (sigma=2)
#     w_t + a*u*w_x = u_x*w + nu*w_xx   on S^1 (2*pi torus), u_x = H w, mean zero
# Target: the quantitative phase boundary nu*(a) from w0 = cos x —
# nu*(a) = threshold viscosity separating finite-time blowup from decay.
# Literature status (checked S03/S04): Chen Nonlinearity 33 (2020) qualitative;
# SLSA arXiv:2411.01891 exact solutions at sigma in {0,1} only; no sigma=2
# quantitative map found. Novelty ledger entry N4.
#
# Exact structure used for validation:
#  - nu=0 reduces exactly to gclm_scan.rhs_g (S03-verified);
#  - a=0 reduces exactly to clm_viscous.rhs_w (S02-verified), with the S02
#    exponential-sum EXACT solution as a real-data anchor;
#  - complex positive-mode hierarchy generalizes with the diagonal term:
#        b_m' = -nu*m^2*b_m + i*sum_{j<m} (a*(m-j)/j - 1) b_j b_{m-j}
#    (quasi-exact reference at every (a, nu));
#  - exact amplitude-rescaling symmetry: w_A(x,t) = A*w(x, A*t) solves the
#    equation with nu_A = A*nu  =>  machine-precision pipeline test, and
#    nu*(A*cos x) = A*nu*(cos x) exactly.
# Deterministic. Stages:
#   vu | vc0 | vcc | amp | probe:<a> | bis:<a>[:lo:hi] | vfig
import os, sys, time
import numpy as np
import clm_viscous as cv
import gclm_scan as gs

DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- rhs and integrators ----------------
def rhs_gv(w, a, nu, ops):
    """Viscous gCLM rhs, dealiased; viscous term handled exactly as S02 rhs_w."""
    x, k, hmult, mask, K = ops
    wh = np.fft.fft(w)
    absk = np.abs(k); absk[0] = 1.0
    u  = np.real(np.fft.ifft(-wh/absk * (np.abs(k) > 0)))
    ux = np.real(np.fft.ifft(hmult*wh))
    wx = np.real(np.fft.ifft(1j*k*wh*mask))
    ph = np.fft.fft(-a*u*wx + w*ux); ph[~mask] = 0.0
    return np.real(np.fft.ifft(ph - nu*(k**2)*wh*mask))

def rhs_gvc(w, a, nu, ops):
    """Complex-field variant (positive-mode class)."""
    x, k, hmult, mask, K = ops
    wh = np.fft.fft(w)
    absk = np.abs(k); absk[0] = 1.0
    u  = np.fft.ifft(-wh/absk * (np.abs(k) > 0))
    ux = np.fft.ifft(hmult*wh)
    wx = np.fft.ifft(1j*k*wh*mask)
    ph = np.fft.fft(-a*u*wx + w*ux); ph[~mask] = 0.0
    return np.fft.ifft(ph - nu*(k**2)*wh*mask)

def rk4_v(w, dt, a, nu, ops, rhs):
    k1 = rhs(w, a, nu, ops)
    k2 = rhs(w + 0.5*dt*k1, a, nu, ops)
    k3 = rhs(w + 0.5*dt*k2, a, nu, ops)
    k4 = rhs(w + dt*k3, a, nu, ops)
    return w + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def hier_rhs_v(b, a, nu):
    m2 = (np.arange(1, len(b)+1)**2).astype(float)
    return -nu*m2*b + gs.hier_rhs(b, a)

def dop853_hier_v(a, nu, M, teval, rtol=1e-12, atol=1e-14):
    from scipy.integrate import solve_ivp
    def f(t, y):
        db = hier_rhs_v(y[:M] + 1j*y[M:], a, nu)
        return np.concatenate([db.real, db.imag])
    y0 = np.zeros(2*M); y0[0] = 1.0
    sol = solve_ivp(f, (0.0, max(teval)), y0, t_eval=teval,
                    method="DOP853", rtol=rtol, atol=atol)
    return [sol.y[:M, i] + 1j*sol.y[M:, i] for i in range(len(teval))]

def dop853_real_v(a, nu, M, teval, rtol=1e-12, atol=1e-14):
    from scipy.integrate import solve_ivp
    m2 = (np.arange(1, M+1)**2).astype(float)
    def f(t, y):
        c = y[:M] + 1j*y[M:]
        dc = gs.galerkin_rhs_full(c, a, M) - nu*m2*c
        return np.concatenate([dc.real, dc.imag])
    y0 = np.zeros(2*M); y0[M] = 0.0; y0[0] = 0.5          # cos x
    sol = solve_ivp(f, (0.0, max(teval)), y0, t_eval=teval,
                    method="DOP853", rtol=rtol, atol=atol)
    return [sol.y[:M, i] + 1j*sol.y[M:, i] for i in range(len(teval))]

# ---------------- classified run driver ----------------
def pde_run_gv(N, dt, a, nu, Tmax, checkpoints=(), amp0=1.0,
               track_every=100, spec_every=400,
               blow_cert=100.0, decay_abs=0.4, decay_frac=0.5):
    """Run until a certified outcome or Tmax. Certifications (relative to the
    data's own amplitude scale amp0):
      BLOWUP:  sup >= blow_cert*amp0 while the last strip measure (if any) was
               resolved (deltaK >= 12)  ['BLOWUP (resolved growth x%g)'];
               amplitude/nonfinite guard as fallback.
      DECAY:   sup < decay_abs*amp0 AND sup < decay_frac*peak.
      Otherwise Tmax -> UNDETERMINED (or strip death; classified by caller)."""
    ops = cv.make_ops(N)
    x = ops[0]; K = ops[4]
    w = amp0*np.cos(x)
    cp = {int(round(t/dt)): t for t in checkpoints}
    snaps, track, spec = {}, [], []
    strict_h = None
    last_dK = None
    peak = amp0
    nsteps = int(round(Tmax/dt))
    stop_reason = "Tmax"
    t = 0.0
    for n in range(1, nsteps+1):
        w = rk4_v(w, dt, a, nu, ops, rhs_gv)
        t = n*dt
        if n in cp:
            snaps[cp[n]] = w.copy()
        if n % track_every == 0:
            s = float(np.max(np.abs(w)))
            track.append((t, s))
            peak = max(peak, s)
            if s >= blow_cert*amp0:
                stop_reason = ("BLOWUP-certified (sup>=%gx, strip %s)"
                               % (blow_cert, "ok" if (last_dK is None or last_dK >= 12) else "DEAD"))
                break
            if s < decay_abs*amp0 and s < decay_frac*peak:
                stop_reason = "DECAY-certified (sup<%.2g and <%.2g*peak)" % (decay_abs*amp0, decay_frac)
                break
        if n % spec_every == 0:
            d = cv.strip_fit(w, K)
            if d is not None:
                dK = d*K
                spec.append((t, d))
                last_dK = dK
                if strict_h is None and dK < 32:
                    strict_h = t
                if dK < 12:
                    stop_reason = "strip collapsed (delta*K<12)"
                    break
        if not np.isfinite(w).all() or np.max(np.abs(w)) > 1e6*amp0:
            stop_reason = "amplitude blowup / nonfinite"
            break
    return dict(x=x, snaps=snaps, track=np.array(track), spec=np.array(spec),
                t_stop=t, stop=stop_reason, strict_h=strict_h, K=K,
                peak=peak, w_final=w)

def classify(R, amp0=1.0):
    """Map a finished run to BLOWUP / DECAY / UNDET, with S03 refit discipline
    for strip-death stops."""
    st = R["stop"]
    tr = R["track"]
    if st.startswith("BLOWUP-certified") or st.startswith("amplitude"):
        return "BLOWUP"
    if st.startswith("DECAY-certified"):
        return "DECAY"
    growth = (tr[-1, 1]/amp0) if len(tr) else 1.0
    if st.startswith("strip collapsed"):
        return "BLOWUP" if growth > 10.0 else "UNDET(strip)"
    # Tmax: look at the trailing slope
    if len(tr) > 20:
        tw = tr[:, 0] >= tr[-1, 0] - 2.0
        slope = np.polyfit(tr[tw, 0], tr[tw, 1], 1)[0]
        if slope < 0 and tr[-1, 1] < R["peak"]*0.9:
            return "UNDET(decaying)"
        if slope > 0:
            return "UNDET(growing)"
    return "UNDET"

# ---------------- stages ----------------
def stage_vu():
    print("stage vu: viscous unit tests"); t0 = time.time()
    N = 256
    ops = cv.make_ops(N)
    x, k, hmult, mask, K = ops
    rng = np.random.default_rng(4)
    wh = np.zeros(N, complex)
    for m in range(1, 41):
        c = rng.normal() + 1j*rng.normal()
        wh[m], wh[-m] = c, np.conj(c)
    w = np.real(np.fft.ifft(wh))
    # VU1: nu=0 reduces exactly to gclm_scan.rhs_g
    print("VU1 |rhs_gv(nu=0) - rhs_g|_inf, a=0.6        : %.3e"
          % np.max(np.abs(rhs_gv(w, 0.6, 0.0, ops) - gs.rhs_g(w, 0.6, ops))))
    # VU2: a=0 reduces exactly to S02 rhs_w
    print("VU2 |rhs_gv(a=0) - cv.rhs_w|_inf, nu=0.17    : %.3e"
          % np.max(np.abs(rhs_gv(w, 0.0, 0.17, ops) - cv.rhs_w(w, 0.17, k, hmult, mask))))
    # VU3: mean conservation with viscosity
    for a, nu in ((0.7, 0.05), (-0.5, 0.2)):
        print("VU3 |mean(rhs_gv)|, a=%+.1f nu=%.2f           : %.3e"
              % (a, nu, abs(np.mean(rhs_gv(w, a, nu, ops)))))
    # VU4: viscous hierarchy vs complex FFT rhs
    for a, nu in ((0.5, 0.05), (1.0, 0.25), (-0.5, 0.1)):
        M = 20
        b = (rng.normal(size=M) + 1j*rng.normal(size=M)) * 0.5**np.arange(M)
        wc = np.fft.ifft(np.concatenate([[0], b, np.zeros(N-M-1)])*N)
        r_fft = (np.fft.fft(rhs_gvc(wc, a, nu, ops))/N)[1:M+1]
        print("VU4 hierarchy-v vs FFT rhs, a=%+.1f nu=%.2f    : %.3e"
              % (a, nu, np.max(np.abs(r_fft - hier_rhs_v(b, a, nu)))))
    # VU5: real Galerkin (convolution + viscous diag) vs FFT rhs
    c40 = (np.fft.fft(w)/N)[1:41]
    m2 = (np.arange(1, 41)**2).astype(float)
    r_conv = gs.galerkin_rhs_full(c40, 0.6, 40) - 0.13*m2*c40
    r_fft = (np.fft.fft(rhs_gv(w, 0.6, 0.13, ops))/N)[1:41]
    print("VU5 convolution-v vs FFT rhs, a=0.6 nu=0.13  : %.3e" % np.max(np.abs(r_conv - r_fft)))
    print("[%.1f s]" % (time.time() - t0))

def stage_vc0():
    print("stage vc0: a=0 real-data anchor vs S02 exact viscous series"); t0 = time.time()
    N, dt = 1024, 2.5e-5
    cps = (0.5, 1.0, 1.2)
    # nu=0.25: float64 exact series (S02-verified vs mpmath to 4e-17)
    B25 = cv.exact_coeffs(0.25, 90)
    R = pde_run_gv(N, dt, 0.0, 0.25, 1.2 + dt, cps, blow_cert=1e9, decay_abs=-1)
    for t in cps:
        w1 = R["snaps"][t]
        c3 = cv.eval_exact(B25, 0.25, t)
        d = np.max(np.abs(w1 - cv.coeffs_to_w(c3, N)))
        print("  nu=0.25 t=%.1f  |PDE - exact series|_inf: %.3e   (S02 floor ~1e-14)" % (t, d))
    # nu=0.05: mpmath series per-mode at m<=24 (S02-verified path)
    Bmp = cv.mp_exact_coeffs(0.05, 24)
    R = pde_run_gv(N, dt, 0.0, 0.05, 1.2 + dt, cps, blow_cert=1e9, decay_abs=-1)
    for t in cps:
        cm = cv.mp_eval(Bmp, 0.05, t)
        cp_ = cv.pde_coeffs(R["snaps"][t], 24)
        print("  nu=0.05 t=%.1f  per-mode m<=24 vs mpmath : %.3e   (S02 floor ~5e-15)"
              % (t, np.max(np.abs(cm - cp_))))
    print("[%.1f s]" % (time.time() - t0))

def stage_vcc():
    print("stage vcc: viscous hierarchy vs complex PDE, (a,nu) grid"); t0 = time.time()
    N, dt, M = 256, 1e-4, 48
    teval = (0.3, 0.6)
    ops = cv.make_ops(N)
    for a in (0.5, 1.0):
        for nu in (0.05, 0.25):
            bh = dop853_hier_v(a, nu, M, teval)
            w = np.exp(1j*ops[0])
            cpn = {int(round(t/dt)): i for i, t in enumerate(teval)}
            outs = {}
            for n in range(1, int(round(max(teval)/dt)) + 1):
                w = rk4_v(w, dt, a, nu, ops, rhs_gvc)
                if n in cpn:
                    outs[cpn[n]] = (np.fft.fft(w)/N)[1:M+1]
            errs = " ".join("t=%.1f %.2e" % (teval[i], np.max(np.abs(outs[i] - bh[i])))
                            for i in sorted(outs))
            print("  a=%.1f nu=%.2f : %s" % (a, nu, errs))
    print("[%.1f s]" % (time.time() - t0))

def stage_amp():
    """Exact rescaling symmetry: w_A(x,t) = A*w(x,At) with nu_A = A*nu.
    Run (amp0=1, nu) and (amp0=2, 2*nu); compare 2*w(., 2t) vs w_A(., t)."""
    print("stage amp: exact amplitude-rescaling symmetry test"); t0 = time.time()
    N, dt, a, nu = 512, 5e-5, 0.5, 0.06
    T = 1.0
    R1 = pde_run_gv(N, dt, a, nu, 2*T + dt, (1.0, 2.0), blow_cert=1e9, decay_abs=-1)
    R2 = pde_run_gv(N, dt/2, a, 2*nu, T + dt/2, (0.5, 1.0), amp0=2.0, blow_cert=1e9, decay_abs=-1)
    for tA, tB in ((0.5, 1.0), (1.0, 2.0)):
        d = np.max(np.abs(R2["snaps"][tA] - 2.0*R1["snaps"][tB]))
        print("  |w_2(.,%.1f) - 2*w_1(.,%.1f)|_inf = %.3e   (exact symmetry; floor = solver error)"
              % (tA, tB, d))
    print("[%.1f s]" % (time.time() - t0))

def stage_va1():
    """a=1 exact anchor: on the steady family A*cos x the gCLM nonlinearity
    vanishes identically, and viscosity is diagonal on mode 1, so viscous
    De Gregorio from cos x solves EXACTLY omega(t) = e^{-nu t} cos x.
    (Hence nu*(1) = 0 for this data.) Machine-precision check."""
    print("stage va1: a=1 exact decay anchor  omega = e^{-nu t} cos x")
    N, dt, nu, T = 512, 5e-5, 0.1, 2.0
    ops = cv.make_ops(N)
    x = ops[0]
    w = np.cos(x)
    for n in range(1, int(round(T/dt)) + 1):
        w = rk4_v(w, dt, 1.0, nu, ops, rhs_gv)
        if n % int(round(1.0/dt)) == 0:
            t = n*dt
            print("  t=%.1f  |w - e^{-nu t} cos x|_inf = %.3e" % (t, np.max(np.abs(w - np.exp(-nu*t)*np.cos(x)))))

def stage_probe(a):
    """Coarse nu sweep to bracket nu*(a)."""
    print("stage probe: a=%+.3f coarse nu sweep (N=512, dt=5e-5)" % a); t0 = time.time()
    nus = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
    N, dt, Tmax = 512, 5e-5, 16.0
    for nu in nus:
        tA = time.time()
        R = pde_run_gv(N, dt, a, nu, Tmax)
        cls = classify(R)
        tr = R["track"]
        print("  nu=%.3f : %-16s t_stop=%.2f sup_end=%.3g peak=%.3g  [%.0f s]"
              % (nu, cls, R["t_stop"], tr[-1, 1] if len(tr) else float("nan"),
                 R["peak"], time.time() - tA))
    print("[total %.0f s]" % (time.time() - t0))

def stage_bis(a, lo, hi, iters=5):
    """Bisection: lo must be BLOWUP-certified, hi DECAY-certified (checked)."""
    print("stage bis: a=%+.3f bracket [%.4f, %.4f] (N=512, dt=5e-5)" % (a, lo, hi))
    t0 = time.time()
    N, dt = 512, 5e-5
    log = []
    def run(nu, Tmax):
        R = pde_run_gv(N, dt, a, nu, Tmax)
        return classify(R), R
    for it in range(iters):
        mid = 0.5*(lo + hi)
        Tmax = 16.0
        cls, R = run(mid, Tmax)
        if cls.startswith("UNDET"):
            cls, R = run(mid, 40.0)          # one extension near the boundary
        tr = R["track"]
        print("  it%d nu=%.5f : %-16s t_stop=%.2f peak=%.3g" % (it, mid, cls, R["t_stop"], R["peak"]))
        log.append((mid, cls, R["t_stop"], R["peak"]))
        if cls == "BLOWUP":
            lo = mid
        elif cls == "DECAY":
            hi = mid
        else:
            print("  UNDET persists at nu=%.5f even at Tmax=40 — recording half-open bracket" % mid)
            break
    print("  RESULT a=%+.3f : nu* in [%.5f, %.5f]  [%.0f s]" % (a, lo, hi, time.time() - t0))
    np.savez(os.path.join(DIR, "nustar_a%s.npz" % gs.atag(a)),
             a=a, lo=lo, hi=hi, log=np.array(log, dtype=object))

def stage_nu0x(lo=0.05, hi=0.06, iters=14, M=64, Tmax=300.0,
               blow_bar=100.0, decay_bar=0.05):
    """High-precision nu*(0) via the EXACT structure at a=0: the real cos-x
    problem maps exactly to the triangular q-hierarchy (S02): c_m' = -nu m^2 c_m
    + (1/2) sum c_j c_{m-j}, c_1(0) = -i; modes<=M are the true solution's
    modes. Bisect nu on the ODE system (fast) with long Tmax to beat critical
    slowing; monitor the tail mode for truncation honesty. This nu*(0) is the
    Fujita-type threshold constant of q_t = q^2/2 + nu q_xx on S^1 from -ie^{ix}."""
    from scipy.integrate import solve_ivp
    print("stage nu0x: exact-hierarchy bisection of nu*(0)  (M=%d, Tmax=%g)" % (M, Tmax))
    m2 = (np.arange(1, M+1)**2).astype(float)
    def f(t, y, nu):
        c = y[:M] + 1j*y[M:]
        conv = np.convolve(c, c)
        nl = np.zeros(M, complex); nl[1:] = 0.5*conv[:M-1]
        dc = -nu*m2*c + nl
        return np.concatenate([dc.real, dc.imag])
    def sup_event_factory(nu):
        # events on |c|_1 proxy: stop early on blowup or deep decay
        def ev_blow(t, y, *a):
            return np.sum(np.abs(y[:M] + 1j*y[M:])) - blow_bar
        ev_blow.terminal = True; ev_blow.direction = 1
        def ev_decay(t, y, *a):
            return np.sum(np.abs(y[:M] + 1j*y[M:])) - decay_bar
        ev_decay.terminal = True; ev_decay.direction = -1
        return ev_blow, ev_decay
    y0 = np.zeros(2*M); y0[M] = -1.0
    t0 = time.time()
    def run_one(nu):
        evb, evd = sup_event_factory(nu)
        sol = solve_ivp(f, (0.0, Tmax), y0, args=(nu,), method="DOP853",
                        rtol=1e-11, atol=1e-13, events=(evb, evd), max_step=0.5)
        # running max of the tail mode and of the sum (transient-awareness)
        c = sol.y[:M] + 1j*sol.y[M:]
        tailmax = np.max(np.abs(c[M-1]))
        summax = np.max(np.sum(np.abs(c), axis=0))
        if len(sol.t_events[0]):
            return "BLOWUP", sol.t_events[0][0], tailmax, summax
        if len(sol.t_events[1]):
            return "DECAY", sol.t_events[1][0], tailmax, summax
        return "UNDET(Tmax)", Tmax, tailmax, summax
    for label, nu in (("lo", lo), ("hi", hi)):
        verdict, tev, tailmax, summax = run_one(nu)
        print("  endpoint %s nu=%.7f : %-12s t_event=%.1f  max|c_M|=%.1e  max_sum=%.2e"
              % (label, nu, verdict, tev, tailmax, summax))
        if label == "lo" and verdict != "BLOWUP":
            print("  !! lo endpoint not blowup-certified — bracket invalid, stopping.")
            return
        if label == "hi" and verdict != "DECAY":
            print("  !! hi endpoint not decay-certified — bracket invalid, stopping.")
            return
    for it in range(iters):
        nu = 0.5*(lo + hi)
        verdict, tev, tailmax, summax = run_one(nu)
        print("  it%02d nu=%.7f : %-12s t_event=%.1f  max|c_M|=%.1e  max_sum=%.2e" %
              (it, nu, verdict, tev, tailmax, summax))
        if verdict == "BLOWUP":
            lo = nu
        elif verdict == "DECAY":
            hi = nu
        else:
            break
    print("  RESULT: nu*(0) in [%.7f, %.7f]  width %.1e  [%.0f s]"
          % (lo, hi, hi - lo, time.time() - t0))
    print("  reference constants: 1/18 = %.7f" % (1.0/18.0))

def stage_spot(a, nu):
    """N=1024 spot-check of a boundary cell (resolution robustness)."""
    print("stage spot: a=%.3f nu=%.5f at N=1024, dt=2.5e-5, Tmax=40" % (a, nu))
    t0 = time.time()
    R = pde_run_gv(1024, 2.5e-5, a, nu, 40.0)
    print("  %s : t_stop=%.2f peak=%.3g stop=%s  [%.0f s]"
          % (classify(R), R["t_stop"], R["peak"], R["stop"], time.time() - t0))

def stage_vfig():
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(DIR, "nustar_a*.npz")))
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rows.append((float(d["a"]), float(d["lo"]), float(d["hi"])))
    rows.sort()
    # probe-only brackets (S04 probe stages; x100-growth bar) and endpoints
    probes = [(-0.5, 0.0, 0.01), (-0.25, 0.01, 0.02), (0.9, 0.02, 0.16), (0.95, 0.0, 0.08)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    A = [r[0] for r in rows]
    mid = [0.5*(r[1] + r[2]) for r in rows]
    err = [0.5*(r[2] - r[1]) for r in rows]
    ax.errorbar(A, mid, yerr=err, fmt="o", capsize=4, color="C0", label="bisection bracket")
    Ap = [p[0] for p in probes]
    midp = [0.5*(p[1] + p[2]) for p in probes]
    errp = [0.5*(p[2] - p[1]) for p in probes]
    ax.errorbar(Ap, midp, yerr=errp, fmt="s", capsize=3, mfc="none", color="gray",
                label="probe bracket (coarse)")
    ax.plot([1.0], [0.0], "r*", ms=14, mfc="none", label=r"exact: $\nu^*(1)=0$ ($e^{-\nu t}\cos x$)")
    ax.plot([0.0], [0.05057], "d", color="purple", mfc="none", ms=9,
            label=r"excursion-law determination: $\nu^*(0) = 0.05057(5)$ (N16, bar-free)")
    ax.set_xlabel("a"); ax.set_ylabel(r"$\nu^*$")
    ax.set_title(r"Viscous gCLM phase boundary, $\omega_0=\cos x$, $S^1$, $\sigma=2$"
                 "\nblowup below, decay above; certified at the x100-growth bar (N15 caveat);"
                 "\n" r"$\nu^*(A\,\mathrm{data}) = A\,\nu^*$ exact")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "gclm_visc_phase.png"), dpi=150)
    print("figure saved: gclm_visc_phase.png")

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "vu":     stage_vu()
    elif st == "va1":  stage_va1()
    elif st == "vc0":  stage_vc0()
    elif st == "vcc":  stage_vcc()
    elif st == "amp":  stage_amp()
    elif st.startswith("probe:"): stage_probe(float(st[6:]))
    elif st.startswith("bis:"):
        parts = st.split(":")
        stage_bis(float(parts[1]), float(parts[2]), float(parts[3]))
    elif st == "vfig": stage_vfig()
    elif st == "nu0x": stage_nu0x()
    elif st == "nu0x96": stage_nu0x(lo=0.05470, hi=0.05480, iters=8, M=96)
    elif st == "nu0x128": stage_nu0x(lo=0.054, hi=0.058, iters=14, M=128,
                                     Tmax=600.0, blow_bar=1e4, decay_bar=1e-3)
    elif st == "nu0x192": stage_nu0x(lo=0.054, hi=0.058, iters=14, M=192,
                                     Tmax=600.0, blow_bar=1e4, decay_bar=1e-3)
    elif st == "nu0strict": stage_nu0x(lo=0.040, hi=0.054, iters=12, M=256,
                                       Tmax=600.0, blow_bar=1e4, decay_bar=1e-3)
    elif st.startswith("spot:"):
        parts = st.split(":")
        stage_spot(float(parts[1]), float(parts[2]))
    elif st == "nu0final":
        # decisive M/bar continuation at the contested bracket endpoints:
        # M=512 contains the predicted front-death scale (~m=445 at these nu);
        # bar=1e6 separates accelerating cascades from bar-grazing excursions.
        # iters=0 => endpoint verification only.
        stage_nu0x(lo=0.0507837, hi=0.0507871, iters=0, M=512, Tmax=600.0,
                   blow_bar=1e6, decay_bar=1e-3)
    else: raise SystemExit("unknown stage: %s" % st)
