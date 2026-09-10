#!/usr/bin/env python3
# gclm_scan.py — Rung 3 (gCLM / De Gregorio a-family)
# Equation (S^1 = 2*pi torus, mean-zero):
#     w_t + a*u*w_x = u_x * w,     u_x = H w   (u_hat = -w_hat/|k|, u_hat(0) = 0)
# a = 0 : CLM (rung-1 exact solution; T* = 2 for w0 = cos x)
# a = 1 : De Gregorio; -sin x and every translate/scale (incl. cos x) is steady.
#
# CALIBRATION STRUCTURE (derived S03, verified in stages u/cc):
# For COMPLEX fields w = sum_{m>=1} b_m e^{imx} (positive modes only) any
# quadratic nonlinearity is lower-triangular. Here, from u_hat_m = -b_m/m,
# (w_x)_m = i m b_m, (u_x)_m = -i b_m:
#     b_m' = i * sum_{j=1}^{m-1} ( a*(m-j)/j - 1 ) * b_j * b_{m-j},   b_1' = 0.
# Modes 1..M of the truncated hierarchy EQUAL those of the true complex
# solution (no truncation feedback) => DOP853 on the hierarchy is an
# independent quasi-exact reference at EVERY a. This transplants the S02
# method-(ii)/(iii) discipline to a != 0, where no real-data exact solution
# is available in-session. (SLSA arXiv:2411.01891 give closed-form S^1
# blowup solutions at a = 1/2 with Lambda^{0,1} dissipation — queued as a
# future external anchor; deliberately not transcribed this session.)
#
# Methods for the real-data scan:
#   (i)  FFT pseudospectral RK4 on w (N=1024, 2/3-rule K=341, dt=2.5e-5)
#   (ii) DOP853 on the full +/- mode Galerkin ODEs assembled by DIRECT
#        CONVOLUTION (np.convolve; no FFT) — same Galerkin system,
#        independent integrator and nonlinearity assembly.
# Deterministic. Stages:  u | c0 | cc | s1 | a:<value> | fig
#     python gclm_scan.py <stage>
import os, sys, time
import numpy as np
import clm_viscous as cv          # reuse verified components: make_ops, strip_fit, fit_T, verdict

DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- gCLM operators ----------------
def rhs_g(w, a, ops):
    """Real-field gCLM rhs, dealiased. ops = (x,k,hmult,mask,K) from cv.make_ops.
    Identical structure to S02 run_stage.py rhs_g (R5, verified 3.25e-15)."""
    x, k, hmult, mask, K = ops
    wh = np.fft.fft(w)
    absk = np.abs(k); absk[0] = 1.0
    u  = np.real(np.fft.ifft(-wh/absk * (np.abs(k) > 0)))
    ux = np.real(np.fft.ifft(hmult*wh))
    wx = np.real(np.fft.ifft(1j*k*wh*mask))
    ph = np.fft.fft(-a*u*wx + w*ux); ph[~mask] = 0.0
    return np.real(np.fft.ifft(ph))

def rhs_gc(w, a, ops):
    """Complex-field variant (positive-modes class): same algebra, no real casts."""
    x, k, hmult, mask, K = ops
    wh = np.fft.fft(w)
    absk = np.abs(k); absk[0] = 1.0
    u  = np.fft.ifft(-wh/absk * (np.abs(k) > 0))
    ux = np.fft.ifft(hmult*wh)
    wx = np.fft.ifft(1j*k*wh*mask)
    ph = np.fft.fft(-a*u*wx + w*ux); ph[~mask] = 0.0
    return np.fft.ifft(ph)

def rk4_g(w, dt, a, ops, rhs):
    k1 = rhs(w, a, ops)
    k2 = rhs(w + 0.5*dt*k1, a, ops)
    k3 = rhs(w + 0.5*dt*k2, a, ops)
    k4 = rhs(w + dt*k3, a, ops)
    return w + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

# ---------------- complex triangular hierarchy (quasi-exact reference, any a) ----------------
def hier_rhs(b, a):
    """b[0]=b_1,...,b[M-1]=b_M; returns db/dt from the triangular formula."""
    M = len(b)
    db = np.zeros(M, complex)
    for m in range(2, M+1):
        j = np.arange(1, m)
        coef = a*(m - j)/j - 1.0
        db[m-1] = 1j*np.sum(coef*b[j-1]*b[m-j-1])
    return db

def dop853_hier(a, M, teval, rtol=1e-12, atol=1e-14):
    from scipy.integrate import solve_ivp
    def f(t, y):
        db = hier_rhs(y[:M] + 1j*y[M:], a)
        return np.concatenate([db.real, db.imag])
    y0 = np.zeros(2*M); y0[0] = 1.0            # b_1(0) = 1  (w0 = e^{ix})
    sol = solve_ivp(f, (0.0, max(teval)), y0, t_eval=teval,
                    method="DOP853", rtol=rtol, atol=atol)
    return [sol.y[:M, i] + 1j*sol.y[M:, i] for i in range(len(teval))]

# ---------------- method (ii): DOP853 + direct-convolution Galerkin (real data) ----------------
def galerkin_rhs_full(c, a, M):
    """Real field w = sum_{m=1..M} c_m e^{imx} + c.c.; returns dc_m/dt, m=1..M.
    Assembled by direct convolution over the full mode range [-M..M] — no FFT.
    full[j] holds mode (j-M) for j=0..2M."""
    full = np.zeros(2*M+1, complex)
    full[M+1:] = c
    full[:M] = np.conj(c)[::-1]
    md = np.arange(-M, M+1).astype(float)
    absm = np.abs(md); absm[M] = 1.0
    uh  = -full/absm * (md != 0)               # u_hat
    wxh = 1j*md*full                           # (w_x)_hat
    uxh = -1j*np.sign(md)*full                 # (u_x)_hat = (Hw)_hat
    t1 = np.convolve(uh, wxh)                  # (u*w_x)_hat over [-2M..2M]
    t2 = np.convolve(full, uxh)                # (w*u_x)_hat
    nl = -a*t1 + t2
    return nl[2*M+1:3*M+1]                     # modes 1..M  (mode m at index 2M + m)

def dop853_real(a, M, teval, rtol=1e-12, atol=1e-14):
    from scipy.integrate import solve_ivp
    def f(t, y):
        dc = galerkin_rhs_full(y[:M] + 1j*y[M:], a, M)
        return np.concatenate([dc.real, dc.imag])
    y0 = np.zeros(2*M); y0[0] = 0.5            # w0 = cos x: c_1 = 1/2
    sol = solve_ivp(f, (0.0, max(teval)), y0, t_eval=teval,
                    method="DOP853", rtol=rtol, atol=atol)
    return [sol.y[:M, i] + 1j*sol.y[M:, i] for i in range(len(teval))]

def coeffs_to_w_real(c, N):
    wh = np.zeros(N, complex)
    M = len(c)
    wh[1:M+1] = c*N
    wh[-M:] = np.conj(wh[1:M+1])[::-1]
    return np.real(np.fft.ifft(wh))

# ---------------- real-data scan driver ----------------
def pde_run_g(N, dt, a, Tmax, checkpoints, track_every=100, spec_every=400):
    ops = cv.make_ops(N)
    x, k, hmult, mask, K = ops
    w = np.cos(x)
    cp = {int(round(t/dt)): t for t in checkpoints}
    snaps, track, spec = {}, [], []
    strict_h = None
    nsteps = int(round(Tmax/dt))
    stop_reason = "Tmax"
    t = 0.0
    for n in range(1, nsteps+1):
        w = rk4_g(w, dt, a, ops, rhs_g)
        t = n*dt
        if n in cp:
            snaps[cp[n]] = w.copy()
        if n % track_every == 0:
            track.append((t, float(np.max(np.abs(w)))))
        if n % spec_every == 0:
            d = cv.strip_fit(w, K)
            if d is not None:
                spec.append((t, d))
                if strict_h is None and d*K < 32:
                    strict_h = t
                if d*K < 12:
                    stop_reason = "strip collapsed (delta*K<12)"
                    break
        if not np.isfinite(w).all() or np.max(np.abs(w)) > 1e6:
            stop_reason = "amplitude blowup / nonfinite"
            break
    return dict(x=x, snaps=snaps, track=np.array(track), spec=np.array(spec),
                t_stop=t, stop=stop_reason, strict_h=strict_h, K=K, w_final=w)

def atag(a):
    return ("m" if a < 0 else "") + ("%g" % abs(a)).replace(".", "p")

def tmax_for(a):
    if a < 0:      return 4.0
    if a < 0.6:    return 8.0
    if a < 0.85:   return 12.0
    if a < 1.0:    return 20.0
    return 10.0

# ---------------- exact CLM solution (a=0 anchor, rung 1) ----------------
def w_ex(x, t):  return 4*np.cos(x)/(4 - 4*t*np.sin(x) + t*t)

# ================= stages =================
def stage_u():
    print("stage u: unit tests"); t0 = time.time()
    N = 256
    ops = cv.make_ops(N)
    x, k, hmult, mask, K = ops
    rng = np.random.default_rng(3)
    # random real band-limited field, modes 1..40
    wh = np.zeros(N, complex)
    for m in range(1, 41):
        c = rng.normal() + 1j*rng.normal()
        wh[m], wh[-m] = c, np.conj(c)
    w = np.real(np.fft.ifft(wh))
    # U1: a=0 reduces exactly to CLM rhs
    p = w*np.real(np.fft.ifft(hmult*np.fft.fft(w)))
    ph = np.fft.fft(p); ph[~mask] = 0.0
    rhs_clm = np.real(np.fft.ifft(ph))
    print("U1  |rhs_g(w,a=0) - rhs_CLM(w)|_inf          : %.3e" % np.max(np.abs(rhs_g(w, 0.0, ops) - rhs_clm)))
    # U2: mean conservation at several a
    for a in (0.7, 1.0, -0.5):
        print("U2  |mean(rhs_g)|, a=%+.1f                    : %.3e" % (a, abs(np.mean(rhs_g(w, a, ops)))))
    # U3: De Gregorio steady state, pointwise rhs
    print("U3  sup|rhs_g(-sin x, a=1)|                  : %.3e" % np.max(np.abs(rhs_g(-np.sin(x), 1.0, ops))))
    # U4: triangular hierarchy formula vs complex FFT rhs (independent assemblies)
    for a in (0.0, 0.5, 1.0, -0.5):
        M = 20
        b = (rng.normal(size=M) + 1j*rng.normal(size=M)) * 0.5**np.arange(M)
        wc = np.fft.ifft(np.concatenate([[0], b, np.zeros(N-M-1)])*N)
        r_fft = np.fft.fft(rhs_gc(wc, a, ops))/N
        r_hier = hier_rhs(b, a)
        print("U4  hierarchy vs FFT rhs (modes<=20), a=%+.1f : %.3e"
              % (a, np.max(np.abs(r_fft[1:M+1] - r_hier))))
    # U5: direct-convolution Galerkin rhs vs FFT rhs (real field, modes<=40)
    for a in (0.6, 1.0):
        c40 = (np.fft.fft(w)/N)[1:41]
        r_conv = galerkin_rhs_full(c40, a, 40)
        r_fft = (np.fft.fft(rhs_g(w, a, ops))/N)[1:41]
        print("U5  convolution vs FFT Galerkin rhs, a=%+.1f  : %.3e"
              % (a, np.max(np.abs(r_conv - r_fft))))
    # U6: sympy — symbolic hierarchy vs symbolic Fourier algebra, modes 2..4.
    # Modes represented as powers of z (z^m ~ e^{imx}); the operators act
    # diagonally per mode, so polynomial bookkeeping in z IS the Fourier algebra.
    try:
        import sympy as sp
        z = sp.symbols("z")
        aa = sp.symbols("a", real=True)
        bs = sp.symbols("b1 b2 b3", complex=True)
        wS  = sum(bs[m-1]*z**m for m in (1, 2, 3))
        uS  = sum(-bs[m-1]/m*z**m for m in (1, 2, 3))          # u_hat_m = -b_m/m
        uxS = sum(-sp.I*bs[m-1]*z**m for m in (1, 2, 3))       # (u_x)_m = -i b_m
        wxS = sum(sp.I*m*bs[m-1]*z**m for m in (1, 2, 3))      # (w_x)_m = i m b_m
        nlS = sp.expand(-aa*uS*wxS + wS*uxS)
        ok = []
        for m in (2, 3, 4):
            got = nlS.coeff(z, m)
            want = sp.expand(sp.I*sum((aa*sp.Rational(m - jj)/jj - 1)*bs[jj-1]*bs[m-jj-1]
                                      for jj in range(1, m) if m - jj <= 3 and jj <= 3))
            ok.append(sp.simplify(got - want) == 0)
        print("U6  sympy hierarchy coefficients m=2..4      :", "all match" if all(ok) else "MISMATCH %s" % ok)
    except Exception as e:
        print("U6  sympy SKIPPED:", e)
    print("[%.1f s]" % (time.time() - t0))

def stage_c0():
    print("stage c0: a=0 real-data calibration vs rung-1 exact solution"); t0 = time.time()
    N, dt = 1024, 5e-5
    cps = (0.5, 1.0, 1.5, 1.8, 1.85, 1.9)
    ops = cv.make_ops(N)
    x = ops[0]; K = ops[4]
    w = np.cos(x)
    cp = {int(round(t/dt)): t for t in cps}
    print("    K=%d; columns: on-grid err | pred floor (t/2)^K" % K)
    for n in range(1, int(round(1.9/dt)) + 1):
        w = rk4_g(w, dt, 0.0, ops, rhs_g)
        if n in cp:
            t = cp[n]
            err = np.max(np.abs(w - w_ex(x, t)))
            print("    t=%.2f  %.3e  %.3e" % (t, err, (t/2.0)**K))
    print("[%.1f s]" % (time.time() - t0))

def stage_cc():
    print("stage cc: complex triangular hierarchy vs complex pseudospectral, per a"); t0 = time.time()
    N, dt, M = 256, 1e-4, 48
    teval = (0.2, 0.4, 0.6)
    ops = cv.make_ops(N)
    for a in (0.0, 0.25, 0.5, 0.75, 1.0, -0.5):
        tA = time.time()
        bh = dop853_hier(a, M, teval)
        w = np.exp(1j*ops[0])                   # w0 = e^{ix}
        cpn = {int(round(t/dt)): i for i, t in enumerate(teval)}
        res, tail = {}, {}
        bad = False
        for n in range(1, int(round(max(teval)/dt)) + 1):
            w = rk4_g(w, dt, a, ops, rhs_gc)
            if not np.isfinite(w).all() or np.max(np.abs(w)) > 1e4:
                bad = True; break
            if n in cpn:
                i = cpn[n]
                bm = (np.fft.fft(w)/N)[1:M+1]
                res[i] = np.max(np.abs(bm - bh[i]))
                tail[i] = abs(bh[i][-1])
        line = "  a=%+.2f: " % a
        for i, t in enumerate(teval):
            if i in res:
                line += " t=%.1f err=%.2e (|b_%d|=%.1e)" % (t, res[i], M, tail[i])
        if bad:
            line += "  [stopped: complex amplitude blowup]"
        print(line + "   [%.1f s]" % (time.time() - tA))
    print("[total %.1f s]" % (time.time() - t0))

def stage_s1():
    print("stage s1: De Gregorio (a=1) steady state + perturbation decay"); t0 = time.time()
    # s1a: reproduce S02 R5 drift number
    N, dtg = 256, 1e-3
    ops = cv.make_ops(N)
    x = ops[0]
    w = -np.sin(x)
    for n in range(int(5.0/dtg)):
        w = rk4_g(w, dtg, 1.0, ops, rhs_g)
    print("  s1a  drift from -sin x after t=5 (N=256)     : %.3e   (S02: 3.251e-15)"
          % np.max(np.abs(w + np.sin(x))))
    # s1b: perturbation decay toward the steady manifold {-A sin(x-phi)} = span{sin x, cos x}
    N, dt = 512, 2.5e-4
    ops = cv.make_ops(N)
    x = ops[0]
    for eps, pert, name in ((0.1, np.cos(2*x), "0.1*cos2x"), (0.3, np.sin(3*x), "0.3*sin3x")):
        w = -np.sin(x) + eps*pert
        rec = []
        for n in range(1, int(round(40.0/dt)) + 1):
            w = rk4_g(w, dt, 1.0, ops, rhs_g)
            if n % 2000 == 0:
                wh = np.fft.fft(w)/N
                d_perp = np.sqrt(2*np.sum(np.abs(wh[2:N//2])**2))   # L2 of modes >= 2
                a1 = 2*np.abs(wh[1])
                rec.append((n*dt, d_perp, a1))
        rec = np.array(rec)
        np.savez(os.path.join(DIR, "gs1_%s.npz" % name.replace("*", "").replace(".", "p")), rec=rec)
        half = rec[rec[:, 0] >= 20.0]
        rate = np.polyfit(half[:, 0], np.log(half[:, 1]), 1)[0] if np.all(half[:, 1] > 0) else np.nan
        print("  s1b  w0=-sin x + %-9s: |P_{m>=2}w|_2 %.3e -> %.3e (t=%.0f->40), mode-1 %.4f -> %.4f, late log-slope %.4f"
              % (name, rec[0, 1], rec[-1, 1], rec[0, 0], rec[0, 2], rec[-1, 2], rate))
    print("[%.1f s]" % (time.time() - t0))

def stage_s1c():
    print("stage s1c: a=1 long-time relaxation (t=200), per-mode tracking"); t0 = time.time()
    N, dt = 512, 2.5e-4
    ops = cv.make_ops(N)
    x = ops[0]
    for eps, pert, name in ((0.1, np.cos(2*x), "0p1cos2x"), (0.3, np.sin(3*x), "0p3sin3x")):
        tA = time.time()
        w = -np.sin(x) + eps*pert
        rec = []
        for n in range(1, int(round(200.0/dt)) + 1):
            w = rk4_g(w, dt, 1.0, ops, rhs_g)
            if n % 2000 == 0:
                wh = np.fft.fft(w)/N
                rec.append((n*dt,
                            np.sqrt(2*np.sum(np.abs(wh[2:N//2])**2)),
                            2*np.abs(wh[1]), 2*np.abs(wh[2]), 2*np.abs(wh[3]),
                            float(np.max(np.abs(w)))))
        rec = np.array(rec)
        np.savez(os.path.join(DIR, "gs1c_%s.npz" % name), rec=rec)
        r = rec[-1]
        print("  %s: at t=200  |P_(m>=2)|=%.4e  |m1|=%.4f |m2|=%.4f |m3|=%.4e  sup=%.4f  "
              "rel_perp=|P|/|m1|=%.3f   [%.0f s]"
              % (name, r[1], r[2], r[3], r[4], r[5], r[1]/max(r[2], 1e-30), time.time()-tA))
    print("[total %.1f s]" % (time.time() - t0))

def stage_ccr():
    print("stage ccr: attribute cc error at a=0.75 — dt refinement of the complex PDE side")
    N, M = 256, 48
    teval = (0.6,)
    ops = cv.make_ops(N)
    bh = dop853_hier(0.75, M, teval, rtol=1e-13, atol=1e-15)
    for dt in (1e-4, 5e-5, 2.5e-5):
        w = np.exp(1j*ops[0])
        for n in range(int(round(0.6/dt))):
            w = rk4_g(w, dt, 0.75, ops, rhs_gc)
        bm = (np.fft.fft(w)/N)[1:M+1]
        print("  dt=%.1e : max mode err vs hierarchy = %.3e" % (dt, np.max(np.abs(bm - bh[0]))))

def stage_a(a):
    Tmax = tmax_for(a)
    cps = (0.25, 0.5) if a < 0 else (0.5, 1.0)
    print("stage a=%+.3f  (Tmax=%.0f)" % (a, Tmax)); t0 = time.time()
    R = pde_run_g(1024, 2.5e-5, a, Tmax, cps)
    print("  t_stop=%.3f (%s)  strict_horizon=%s  sup(t_stop)=%.3e" % (
        R["t_stop"], R["stop"],
        ("%.3f" % R["strict_h"]) if R["strict_h"] else "n/a",
        R["track"][-1, 1] if len(R["track"]) else float("nan")))
    # method (ii) cross-check at the checkpoints reached
    reached = [t for t in cps if t in R["snaps"]]
    if reached:
        cs = dop853_real(a, R["K"], reached)
        for i, t in enumerate(reached):
            d = np.max(np.abs(R["snaps"][t] - coeffs_to_w_real(cs[i], 1024)))
            print("    t=%.2f  (i)vs(ii): %.3e" % (t, d))
    if a == 1.0:
        x = R["x"]
        print("    a=1 consistency: drift of cos x from steadiness at t_stop: %.3e"
              % np.max(np.abs(R["w_final"] - np.cos(x))))
    if len(R["track"]) > 20:
        T_lin, T_quad = cv.fit_T(R["track"])
        note = "  (exact T*=2)" if a == 0.0 else ""
        print("    verdict: %s | T_est lin %.4f quad %.4f%s" % (cv.verdict(R), T_lin, T_quad, note))
    np.savez(os.path.join(DIR, "gscan_a%s.npz" % atag(a)),
             track=R["track"], spec=R["spec"], t_stop=R["t_stop"], stop=R["stop"],
             strict_h=(R["strict_h"] if R["strict_h"] else -1.0), K=R["K"],
             w_final=R["w_final"], a=a)
    print("[%.1f s]" % (time.time() - t0))

def stage_refit():
    """Fit hygiene, uniform rule: fit T only on strip-verified dynamics.
    Track points after the last spectrum measurement with delta*K >= 12 are
    dropped (beyond it the run is under-resolved by our own criterion; note
    that for mid-range a the strip is UNMEASURABLY WIDE — modes below the
    noise guard — for most of the run, which counts as resolved)."""
    import glob
    for f in sorted(glob.glob(os.path.join(DIR, "gscan_a*.npz"))):
        d = np.load(f, allow_pickle=True)
        a = float(d["a"]); tr = d["track"]; sp = d["spec"]; K = float(d["K"])
        if len(tr) < 25 or not ("collapsed" in str(d["stop"]) or "blowup" in str(d["stop"])):
            continue
        T0 = cv.fit_T(tr)
        if len(sp):
            ok = sp[sp[:, 1]*K >= 12.0]
            t_res = ok[-1, 0] if len(ok) else 0.0
        else:
            t_res = tr[-1, 0]            # strip never measurable => never under-resolved
        trc = tr[tr[:, 0] <= t_res]
        line = "  a=%+.3f: full T lin/quad %.4f/%.4f" % (a, T0[0], T0[1])
        if len(trc) > 25 and len(trc) < len(tr):
            T1 = cv.fit_T(trc)
            line += "  | strip-verified (t<=%.3f, drop %d pts) %.4f/%.4f" % (
                t_res, len(tr) - len(trc), T1[0], T1[1])
        else:
            T1 = T0
            line += "  | strip-verified fit unchanged (t_res=%.3f)" % t_res
        # honest re-classification: cv.verdict labels every strip-collapse stop
        # BLOWUP, but at high a the strip collapses by filamentation with no
        # amplitude growth — that is horizon-limited, not blowup evidence.
        growth = tr[-1, 1]/tr[0, 1]
        agree = abs(T1[0] - T1[1])/max(T1[0], 1e-9) if np.isfinite(T1[0]) and np.isfinite(T1[1]) else np.inf
        if growth > 10.0 and agree < 0.02:
            cls = "BLOWUP-SUPPORTED"
        elif growth < 3.0:
            cls = "UNDETERMINED (fine scales w/o amplitude growth; T_est not meaningful)"
        else:
            cls = "INDICATIVE-ONLY (growing at stop; T_est extrapolates past horizon)"
        print(line)
        print("            sup growth x%.1f, lin/quad spread %.1f%% -> %s"
              % (growth, 100*agree if np.isfinite(agree) else float("nan"), cls))

def stage_v2():
    """Robustness spot-checks: dt halving at a=0.5; N=2048 at a=0.7."""
    print("v2a: a=0.5, dt=1.25e-5 (halved), N=1024")
    R = pde_run_g(1024, 1.25e-5, 0.5, 8.0, ())
    print("  t_stop=%.3f (%s) strict_h=%s" % (R["t_stop"], R["stop"],
          ("%.3f" % R["strict_h"]) if R["strict_h"] else "n/a"))
    tr = R["track"]; cap = 0.5/(1.25e-5*341*0.5)
    trc = tr[tr[:, 1] <= cap]
    print("  full fit lin/quad: %.4f/%.4f | capped(sup<=%.0f): %.4f/%.4f"
          % (*cv.fit_T(tr), cap, *cv.fit_T(trc)))
    print("v2b: a=0.7, N=2048, dt=2.5e-5")
    R = pde_run_g(2048, 2.5e-5, 0.7, 12.0, ())
    print("  t_stop=%.3f (%s) strict_h=%s K=%d" % (R["t_stop"], R["stop"],
          ("%.3f" % R["strict_h"]) if R["strict_h"] else "n/a", R["K"]))
    tr = R["track"]; cap = 0.5/(2.5e-5*682*0.7)
    trc = tr[tr[:, 1] <= cap]
    if len(trc) > 25:
        print("  full fit lin/quad: %.4f/%.4f | capped(sup<=%.0f): %.4f/%.4f"
              % (*cv.fit_T(tr), cap, *cv.fit_T(trc)))
    np.savez(os.path.join(DIR, "gscan_v2b_a0p7_N2048.npz"), track=R["track"], spec=R["spec"])

def stage_n2(a):
    """N=2048 horizon-extension probe at high a: does sup start growing before
    the finer grid's strip collapses?"""
    print("stage n2: a=%.2f, N=2048, dt=2.5e-5, Tmax=20" % a); t0 = time.time()
    R = pde_run_g(2048, 2.5e-5, a, 20.0, ())
    tr = R["track"]
    print("  t_stop=%.3f (%s) strict_h=%s K=%d sup(t_stop)=%.3e (vs N=1024 horizon)"
          % (R["t_stop"], R["stop"],
             ("%.3f" % R["strict_h"]) if R["strict_h"] else "n/a", R["K"],
             tr[-1, 1] if len(tr) else float("nan")))
    if len(tr) > 20:
        T = cv.fit_T(tr)
        print("  sup growth x%.2f | T lin/quad %.4f/%.4f (interpret per refit rules)"
              % (tr[-1, 1]/tr[0, 1], T[0], T[1]))
    np.savez(os.path.join(DIR, "gscan_n2_a%s.npz" % atag(a)),
             track=R["track"], spec=R["spec"], K=R["K"], a=a)
    print("[%.1f s]" % (time.time() - t0))

def stage_pi():
    """Precision test of the conjecture T*(a=1/2, w0=cos x, S^1) = pi.
    N=2048 (K=682), dt=1.25e-5, fine tracking; windowed extrapolation battery
    with the strip-verified truncation rule."""
    print("stage pi: a=0.5 precision blowup-time test vs pi"); t0 = time.time()
    N, dt, a = 2048, 1.25e-5, 0.5
    R = pde_run_g(N, dt, a, 3.16, (), track_every=40, spec_every=200)
    tr, sp, K = R["track"], R["spec"], R["K"]
    print("  t_stop=%.5f (%s)  K=%d  track pts=%d  spec pts=%d"
          % (R["t_stop"], R["stop"], K, len(tr), len(sp)))
    np.savez(os.path.join(DIR, "gscan_pi_a0p5_N2048.npz"), track=tr, spec=sp, K=K)
    if len(sp):
        ok = sp[sp[:, 1]*K >= 12.0]
        t_res = ok[-1, 0] if len(ok) else tr[-1, 0]
        print("  last strip-verified time: %.5f (deltaK=%.1f there)"
              % (t_res, ok[-1, 1]*K if len(ok) else float("nan")))
    else:
        t_res = tr[-1, 0]
        print("  strip unmeasurable throughout (below noise guard): treating all as resolved")
    m0 = tr[:, 0] <= t_res
    t, inv = tr[m0, 0], 1.0/tr[m0, 1]
    print("  window  T_lin      |T-pi|     T_quad     |T-pi|")
    qs = []
    for w in (0.2, 0.1, 0.05, 0.025):
        m = t >= t[-1] - w
        if m.sum() < 8:
            continue
        pl = np.polyfit(t[m], inv[m], 1)
        Tl = -pl[1]/pl[0]
        pq = np.polyfit(t[m], inv[m], 2)
        r = np.roots(pq); r = r[np.isreal(r)].real; r = r[r > t[-1] - 0.02]
        Tq = r.min() if len(r) else np.nan
        qs.append((w, Tq))
        print("  %.3f   %.6f  %.2e   %.6f  %.2e"
              % (w, Tl, abs(Tl - np.pi), Tq, abs(Tq - np.pi)))
    if len(qs) >= 2:
        # Richardson on the two smallest windows assuming error ~ w^2 for T_quad
        (w1, T1), (w2, T2) = qs[-2], qs[-1]
        Tr = T2 + (T2 - T1)/((w1/w2)**2 - 1)
        print("  Richardson (last two quad windows): T=%.7f  |T-pi|=%.2e" % (Tr, abs(Tr - np.pi)))
    print("[%.1f s]" % (time.time() - t0))

def stage_pade():
    """Spectral-structure probe: ratios c_{m+1}/c_m of the analytic-signal
    coefficients at t=1 from w0 = cos x. Exactly geometric (single pole pair)
    ratios = -it/2 at a=0; branch-point drift at a=0.5; oscillatory at a=0.7."""
    for a in (0.0, 0.5, 0.7):
        cs = dop853_real(a, 60, (1.0,))[0]
        r = cs[1:20]/cs[0:19]
        print("a=%.1f  ratios c_(m+1)/c_m, m=1..10:" % a)
        print("   ", " ".join("%+.6f%+.6fj" % (x.real, x.imag) for x in r[:10]))
        print("    |ratio| drift over m=1..18: %.2e" % (abs(r).max() - abs(r).min()))

def stage_osw():
    """OSW-style discrimination at high a: over the measurable window, is the
    strip decay delta(t) closer to linear (finite-time zero => blowup-like) or
    exponential (never reaches zero => the global-existence signature OSW saw
    at a=1)? Weak evidence either way if the window is < ~1 e-fold; report rms
    of both fits and the window size honestly."""
    files = ["gscan_a0p8.npz", "gscan_a0p9.npz", "gscan_a0p95.npz", "gscan_n2_a0p9.npz"]
    for fn in files:
        p = os.path.join(DIR, fn)
        if not os.path.exists(p):
            continue
        tag = fn.replace("gscan_", "").replace(".npz", "")
        d = np.load(p, allow_pickle=True)
        sp, K = d["spec"], float(d["K"])
        t, dl = sp[:, 0], sp[:, 1]
        m = t >= t[len(t)//2]
        pl = np.polyfit(t[m], dl[m], 1)
        rl = np.sqrt(np.mean((np.polyval(pl, t[m]) - dl[m])**2))
        pe = np.polyfit(t[m], np.log(dl[m]), 1)
        re = np.sqrt(np.mean((np.exp(np.polyval(pe, t[m])) - dl[m])**2))
        print("  a=%s: deltaK %.1f->%.1f over t=[%.2f,%.2f] | lin rms %.2e (zero at t=%.2f) | "
              "exp rms %.2e (rate %.3f) | exp/lin %.2f"
              % (tag, dl[0]*K, dl[-1]*K, t[0], t[-1], rl, -pl[1]/pl[0], re, pe[0], re/rl))

def stage_rate():
    """Blowup-rate fit at a=0.5 on the stage-pi track: sup ~ C/(pi-t)^gamma."""
    d = np.load(os.path.join(DIR, "gscan_pi_a0p5_N2048.npz"), allow_pickle=True)
    tr = d["track"]; t, s = tr[:, 0], tr[:, 1]
    for w in (0.5, 0.2, 0.1, 0.05, 0.02):
        mm = (t > np.pi - w) & (t < np.pi - 1e-3)
        p = np.polyfit(np.log(np.pi - t[mm]), np.log(s[mm]), 1)
        print("  window %.2f: gamma=%.5f  C=%.5f" % (w, -p[0], np.exp(p[1])))
    m = (t > np.pi - 0.5) & (t < np.pi - 1e-3)
    c = s[m]*(np.pi - t[m])
    print("  sup*(pi-t): %.5f (at pi-0.5) -> %.5f (at pi-0.001); sqrt(3)=%.5f [unconverged]"
          % (c[0], c[-1], np.sqrt(3.0)))

def stage_rate4(N=2048):
    """S05: sharpen the sqrt(3) rate test at a=1/2 by a two-phase ride:
    dt=1.25e-5 to pi-1e-3, then rate-adaptive continuation toward breakdown.
    The N argument discriminates the deep-dip attribution: a resolution
    artifact moves deeper with N; a T*-offset stays put."""
    print("stage rate4: two-phase precision ride at a=0.5, N=%d" % N); t0 = time.time()
    dt1 = 1.25e-5
    T1 = np.pi - 1e-3
    R = pde_run_g(N, dt1, 0.5, dt1*round(T1/dt1), ())
    w = R["w_final"]; t = R["t_stop"]
    print("  phase 1 done: t=%.8f sup=%.6g (%s)" % (t, np.max(np.abs(w)), R["stop"]))
    ops = cv.make_ops(N)
    def sup_refined(w):
        """Grid max + 3-point parabolic refinement: kills the O((dx/l)^2)
        grid-sampling error of an off-grid smooth peak (the N-dependent 'dip'
        was exactly this error at the collapsing core scale)."""
        aw = np.abs(w)
        i = int(np.argmax(aw))
        sm, sp = aw[(i-1) % N], aw[(i+1) % N]
        s0 = aw[i]
        denom = 2.0*s0 - sm - sp
        if denom <= 0:
            return float(s0)
        return float(s0 + (sp - sm)**2/(8.0*denom))
    rec = []
    s = sup_refined(w)
    for n in range(1, 200000):
        dt2 = min(1e-6, 0.05/s)         # rate-adaptive: dt*sup = 0.05 => per-step
        w = rk4_g(w, dt2, 0.5, ops, rhs_g)   # RK4 error ~ (0.05)^5, temporally clean
        t += dt2
        s = sup_refined(w)
        if not np.isfinite(s) or s > 2e6 or t >= np.pi - 2e-7:
            break
        rec.append((t, s))
    rec = np.array(rec)
    np.savez(os.path.join(DIR, "grate4_a0p5_N%d.npz" % N), rec=rec)
    tt, ss = rec[:, 0], rec[:, 1]
    prod = ss*(np.pi - tt)
    print("  phase 2: %d pts, reached pi-t = %.2e, sup = %.4g" % (len(rec), np.pi - tt[-1], ss[-1]))
    print("  product sup*(pi-t): %.6f at pi-1e-3 ... %.6f at closest  (sqrt3 = %.6f)"
          % (prod[0], prod[-1], np.sqrt(3)))
    from scipy.optimize import curve_fit
    x = np.pi - tt
    # C fixed = sqrt3, exponent 2/3 (family prediction): fit c1 only
    c1 = np.sum((prod - np.sqrt(3))*x**(2.0/3.0))/np.sum(x**(4.0/3.0))
    r_fix = np.sqrt(np.mean((prod - (np.sqrt(3) + c1*x**(2.0/3.0)))**2))
    print("  fit A [C=sqrt3, beta=2/3]: c1=%.4f  rms=%.2e" % (c1, r_fix))
    p, _ = curve_fit(lambda x, C, c, b: C + c*x**b, x, prod, p0=(1.732, 0.3, 0.67), maxfev=40000)
    r_free = np.sqrt(np.mean((prod - (p[0] + p[1]*x**p[2]))**2))
    print("  fit B [free]: C=%.7f (|C-sqrt3|=%.1e)  c1=%.4f beta=%.4f  rms=%.2e"
          % (p[0], abs(p[0] - np.sqrt(3)), p[1], p[2], r_free))
    print("[%.0f s]" % (time.time() - t0))

def stage_rate5():
    """Post-process the rate4 tracks on RESOLUTION-CLEAN windows only:
    keep points with core width l(t) = 2*(3(pi-t)/32)^(1/3) >= 10*dx.
    Fit sup*(pi-t) = C + c1*(pi-t)^(2/3) with C free and with C = sqrt3."""
    from scipy.optimize import curve_fit
    for N in (2048, 4096):
        p = os.path.join(DIR, "grate4_a0p5_N%d.npz" % N)
        if not os.path.exists(p):
            continue
        rec = np.load(p)["rec"]
        tt, ss = rec[:, 0], rec[:, 1]
        x = np.pi - tt
        dx = 2*np.pi/N
        clean = 2.0*(3.0*x/32.0)**(1.0/3.0) >= 10.0*dx
        xc, pc = x[clean], (ss*x)[clean]
        print("N=%d: %d/%d points resolution-clean (pi-t >= %.1e)"
              % (N, clean.sum(), len(x), xc.min()))
        c1 = np.sum((pc - np.sqrt(3))*xc**(2.0/3.0))/np.sum(xc**(4.0/3.0))
        rms = np.sqrt(np.mean((pc - (np.sqrt(3) + c1*xc**(2.0/3.0)))**2))
        print("  [C=sqrt3 fixed]  c1=%.5f  rms=%.2e" % (c1, rms))
        pp, cov = curve_fit(lambda x, C, c: C + c*x**(2.0/3.0), xc, pc,
                            p0=(1.732, 0.2), maxfev=40000)
        perr = np.sqrt(np.diag(cov))
        print("  [C free, beta=2/3] C=%.6f +- %.1e  (C-sqrt3 = %+.2e)  c1=%.5f"
              % (pp[0], perr[0], pp[0] - np.sqrt(3), pp[1]))
        pp3, cov3 = curve_fit(lambda x, C, c, b: C + c*x**b, xc, pc,
                              p0=(1.732, 0.2, 0.667), maxfev=80000)
        perr3 = np.sqrt(np.diag(cov3))
        print("  [C, beta free]     C=%.6f +- %.1e  beta=%.4f +- %.3f"
              % (pp3[0], perr3[0], pp3[2], perr3[2]))

def stage_fig():
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(DIR, "gscan_a*.npz")))
    data = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        data.append(d)
    data.sort(key=lambda d: float(d["a"]))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for d in data:
        a = float(d["a"]); tr = d["track"]
        ax[0, 0].plot(tr[:, 0], 1.0/tr[:, 1], label="a=%+.2f" % a)
        spc = d["spec"]
        if len(spc):
            ax[0, 1].plot(spc[:, 0], spc[:, 1]*float(d["K"]), label="a=%+.2f" % a)
    ax[0, 0].axvline(2.0, color="k", ls=":", lw=1)
    ax[0, 0].set_xlabel("t"); ax[0, 0].set_ylabel(r"$1/\|\omega\|_\infty$")
    ax[0, 0].set_title(r"Inverse sup-norm, $\omega_0=\cos x$ (root = blowup)")
    ax[0, 0].legend(fontsize=7); ax[0, 0].grid(alpha=0.3)
    ax[0, 1].axhline(32, color="gray", ls="--", lw=1); ax[0, 1].axhline(12, color="r", ls="--", lw=1)
    ax[0, 1].set_xlabel("t"); ax[0, 1].set_ylabel(r"$\delta_{est}\cdot K$"); ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("Analyticity-strip collapse"); ax[0, 1].legend(fontsize=7); ax[0, 1].grid(alpha=0.3)
    As, Tl, Tq, Ai, Ti = [], [], [], [], []
    for d in data:
        a = float(d["a"]); tr = d["track"]
        if len(tr) > 20 and ("collapsed" in str(d["stop"]) or "blowup" in str(d["stop"])):
            tlin, tquad = cv.fit_T(tr)
            growth = tr[-1, 1]/tr[0, 1]
            spread = abs(tlin - tquad)/max(tlin, 1e-9)
            if growth > 10.0 and spread < 0.02:
                As.append(a); Tl.append(tlin); Tq.append(tquad)
            elif growth > 3.0:
                Ai.append(a); Ti.append(tquad)
            # growth < 3x: fits are not meaningful; excluded from the panel
    ax[1, 0].plot(As, Tl, "o-", label="T_lin (blowup-supported)")
    ax[1, 0].plot(As, Tq, "s--", label="T_quad")
    ax[1, 0].plot(Ai, Ti, "^", mfc="none", color="gray", label="indicative only")
    ax[1, 0].plot([0.5], [np.pi], "r*", ms=12, mfc="none", label=r"conjecture $T(1/2)=\pi$")
    ax[1, 0].axhline(2.0, color="gray", ls=":", lw=1)
    ax[1, 0].axvline(0.689, color="purple", ls=":", lw=1, label=r"$a_c$ (LSS, $\mathbb{R}$)")
    ax[1, 0].set_xlabel("a"); ax[1, 0].set_ylabel(r"$T_{est}(a)$")
    ax[1, 0].set_title(r"Blowup-time estimates, $\omega_0=\cos x$, $S^1$")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)
    for base in ("0p1cos2x", "0p3sin3x"):
        p200 = os.path.join(DIR, "gs1c_%s.npz" % base)
        p40 = os.path.join(DIR, "gs1_%s.npz" % base)
        p = p200 if os.path.exists(p200) else p40
        if os.path.exists(p):
            rec = np.load(p)["rec"]
            ax[1, 1].semilogy(rec[:, 0], rec[:, 1], label=base)
    ax[1, 1].set_xlabel("t"); ax[1, 1].set_ylabel(r"$\|P_{m\geq 2}\,\omega\|_2$")
    ax[1, 1].set_title("a=1: perturbation decay toward steady manifold")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "gclm_scan.png"), dpi=150)
    print("figure saved: gclm_scan.png")

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "u":      stage_u()
    elif st == "c0":   stage_c0()
    elif st == "cc":   stage_cc()
    elif st == "s1":   stage_s1()
    elif st == "s1c":  stage_s1c()
    elif st == "ccr":  stage_ccr()
    elif st == "refit": stage_refit()
    elif st == "v2":   stage_v2()
    elif st == "pi":   stage_pi()
    elif st.startswith("n2:"): stage_n2(float(st[3:]))
    elif st == "pade": stage_pade()
    elif st == "rate": stage_rate()
    elif st == "rate4": stage_rate4()
    elif st.startswith("rate4:"): stage_rate4(int(st.split(":")[1]))
    elif st == "rate5": stage_rate5()
    elif st == "osw":  stage_osw()
    elif st.startswith("a:"): stage_a(float(st[2:]))
    elif st == "fig":  stage_fig()
    else: raise SystemExit("unknown stage: %s" % st)
