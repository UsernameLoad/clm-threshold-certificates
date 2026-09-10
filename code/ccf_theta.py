#!/usr/bin/env python3
# ccf_theta.py — rung 4 (CCF theta-form).
# Equation (our conventions, sign-checked against CCF Annals 162 (2005) 1377-1389,
# arXiv:0706.1969, full text held locally):  theta_t = (H theta) theta_x  on S^1,
# whose x-derivative with omega = theta_x is EXACTLY gCLM at a = -1
# (omega_t - u omega_x = u_x omega, u = H theta, u_x = H omega).
# Mode 0 of theta is a decoupled slave: d/dt theta0_hat = -sum_k |k||theta_k|^2
# (an exact balance law, used as a per-step calibration diagnostic).
# Triangular hierarchy (positive-mode complex data): b_m' = sum_{j<m} (m-j) b_j b_{m-j},
# equivalent to the a=-1 omega-hierarchy under c_m = i m b_m (sympy-checked, stage u).
# Anchor battery per ccf-rung4-plan.md: u | c0 | bal | a1 | T
import os, sys, time
import numpy as np
import clm_viscous as cv
import gclm_scan as gs

DIR = os.path.dirname(os.path.abspath(__file__))

def rhs_th(th, ops):
    """Real-field theta rhs, dealiased: P_K[(H theta) * P_K theta_x]."""
    x, k, hmult, mask, K = ops
    thh = np.fft.fft(th)
    Hth = np.real(np.fft.ifft(hmult*thh))
    thx = np.real(np.fft.ifft(1j*k*thh*mask))
    ph = np.fft.fft(Hth*thx); ph[~mask] = 0.0
    return np.real(np.fft.ifft(ph))

def rhs_thc(th, ops):
    """Complex-field variant (no real casts) for the hierarchy cross-check."""
    x, k, hmult, mask, K = ops
    thh = np.fft.fft(th)
    Hth = np.fft.ifft(hmult*thh)
    thx = np.fft.ifft(1j*k*thh*mask)
    ph = np.fft.fft(Hth*thx); ph[~mask] = 0.0
    return np.fft.ifft(ph)

def rk4(th, dt, ops, rhs):
    k1 = rhs(th, ops)
    k2 = rhs(th + 0.5*dt*k1, ops)
    k3 = rhs(th + 0.5*dt*k2, ops)
    k4 = rhs(th + dt*k3, ops)
    return th + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ---------------- stages ----------------
def stage_u():
    """Unit tests: sympy hierarchy + mapping; operator-level derivative identity;
    velocity identity; balance law. All floors predicted <= ~1e-13."""
    print("stage u: rung-4 unit tests"); t0 = time.time()
    # U1: sympy — theta-hierarchy coefficients and the c = i m b mapping to a=-1
    import sympy as sp
    t = sp.symbols("t")
    M = 5
    b = [None]*(M+1); b[1] = sp.Integer(1)
    for m in range(2, M+1):
        b[m] = sp.integrate(sp.expand(sum((m-j)*b[j]*b[m-j] for j in range(1, m))), t)
    c = [None]*(M+1)
    c[1] = sp.I*1*b[1]
    okall = True
    for m in range(2, M+1):
        cm = sp.I*m*b[m]
        rhs = sp.integrate(sp.expand(sp.I*sum((sp.Rational(-m, j))*(sp.I*j*b[j])*(sp.I*(m-j)*b[m-j])
                                              for j in range(1, m))), t)
        okall &= sp.simplify(cm - rhs) == 0
    print("U1  sympy: theta-hierarchy -> a=-1 omega-hierarchy under c=imb, m<=5: %s"
          % ("all match" if okall else "MISMATCH"))
    # U2: operator identity d/dx rhs_th(theta) == rhs_g(theta_x, a=-1), band-limited field
    N = 256
    ops = cv.make_ops(N)
    x, k, hmult, mask, K = ops
    th = np.sin(x) + 0.3*np.cos(2*x) + 0.1*np.sin(5*x)
    lhs = np.real(np.fft.ifft(1j*k*np.fft.fft(rhs_th(th, ops))))
    thx = np.real(np.fft.ifft(1j*k*np.fft.fft(th)))
    rhsg = gs.rhs_g(thx, -1.0, ops)
    print("U2  |d/dx rhs_th - rhs_g(theta_x, a=-1)|_inf     : %.3e" % np.max(np.abs(lhs - rhsg)))
    # U3: velocity identity u[theta_x] == H theta
    wh = np.fft.fft(thx)
    absk = np.abs(k); absk[0] = 1.0
    u = np.real(np.fft.ifft(-wh/absk*(np.abs(k) > 0)))
    Hth = np.real(np.fft.ifft(hmult*np.fft.fft(th)))
    print("U3  |u[theta_x] - H theta|_inf                   : %.3e" % np.max(np.abs(u - Hth)))
    # U4: balance law: mean(rhs_th) == -sum_k |k| |theta_k|^2 (DFT normalization)
    thh = np.fft.fft(th)/N
    bal = -np.sum(np.abs(k)*np.abs(thh)**2)
    print("U4  |mean(rhs_th) - (-sum|k||theta_k|^2)|        : %.3e" % abs(np.mean(rhs_th(th, ops)) - bal))
    # U5: reduction to gCLM at the hierarchy level, float: theta-hier vs FFT rhs
    Mh = 20
    bm = (0.5 + 0.1*np.arange(1, Mh+1))*np.exp(1j*0.3*np.arange(1, Mh+1))
    N2 = 128
    ops2 = cv.make_ops(N2)
    thf = np.zeros(N2, complex)
    for m in range(1, Mh+1):
        thf[m] = bm[m-1]*N2
    th2 = np.fft.ifft(thf)
    r = np.fft.fft(rhs_thc(th2, ops2))/N2
    # hierarchy rhs directly: b_m' = sum (m-j) b_j b_{m-j}
    hh = np.array([sum((m-j)*bm[j-1]*bm[m-j-1] for j in range(1, m)) if m >= 2 else 0.0
                   for m in range(1, Mh+1)], complex)
    print("U5  theta-hierarchy vs FFT rhs (modes<=%d)       : %.3e"
          % (Mh, np.max(np.abs(r[1:Mh+1] - hh))))
    print("[%.1f s]" % (time.time() - t0))

def stage_c0():
    """Complex single-mode run vs the triangular hierarchy (quasi-exact)."""
    from scipy.integrate import solve_ivp
    print("stage c0: complex e^{ix} vs triangular hierarchy"); t0 = time.time()
    N, M, dt = 256, 48, 1e-4
    ops = cv.make_ops(N)
    x = ops[0]
    th = np.exp(1j*x)
    def f(t, y):
        b = y[:M] + 1j*y[M:]
        db = np.zeros(M, complex)
        for m in range(2, M+1):
            db[m-1] = sum((m-j)*b[j-1]*b[m-j-1] for j in range(1, m))
        return np.concatenate([db.real, db.imag])
    y0 = np.zeros(2*M); y0[0] = 1.0
    for tchk in (0.1, 0.2, 0.3):
        nst = int(round(tchk/dt))
        thc = th.copy()
        for _ in range(nst):
            thc = rk4(thc, dt, ops, rhs_thc)
        sol = solve_ivp(f, (0, tchk), y0, method="DOP853", rtol=1e-12, atol=1e-14)
        bh = sol.y[:M, -1] + 1j*sol.y[M:, -1]
        thh = np.fft.fft(thc)/N
        d = np.max(np.abs(thh[1:M+1] - bh))
        print("  t=%.1f: max_m|theta_m - b_m| = %.3e   (|b_%d| = %.1e)" % (tchk, d, M, abs(bh[-1])))
    print("[%.0f s]" % (time.time() - t0))

def stage_bal():
    """Balance-law drift along a real evolution from sin x."""
    print("stage bal: balance-law residual along the flow"); t0 = time.time()
    N, dt = 1024, 2.5e-5
    ops = cv.make_ops(N)
    x, k = ops[0], ops[1]
    th = np.sin(x)
    worst = 0.0
    for n in range(int(0.8/dt)):
        if n % 400 == 0:
            thh = np.fft.fft(th)/N
            bal = -np.sum(np.abs(k)*np.abs(thh)**2)
            res = abs(np.mean(rhs_th(th, ops)) - bal)
            worst = max(worst, res)
        th = rk4(th, dt, ops, rhs_th)
    print("  max |mean(rhs) + ||theta||_{H^1/2}^2| over t<=0.8: %.3e" % worst)
    print("[%.0f s]" % (time.time() - t0))

def stage_a1():
    """Derivative consistency: theta-run (from sin x) vs verified omega-run
    (gclm_scan a=-1 from cos x). Discretization-limited (different dealiased
    Galerkin systems), so refinement attribution is included."""
    print("stage a1: theta_x vs omega cross-solver consistency"); t0 = time.time()
    for (N, dt) in ((1024, 2.5e-5), (1024, 1.25e-5), (2048, 1.25e-5)):
        ops = cv.make_ops(N)
        x, k = ops[0], ops[1]
        th = np.sin(x)
        w = np.cos(x)
        out = []
        for tchk in (0.25, 0.5, 0.75, 1.0):
            nst = int(round(0.25/dt))
            for _ in range(nst):
                th = rk4(th, dt, ops, rhs_th)
                w = gs.rk4_g(w, dt, -1.0, ops, gs.rhs_g)
            thx = np.real(np.fft.ifft(1j*k*np.fft.fft(th)))
            out.append("t=%.2f: %.3e" % (tchk, np.max(np.abs(thx - w))))
        print("  N=%d dt=%.2e:  %s" % (N, dt, " | ".join(out)))
    print("[%.0f s]" % (time.time() - t0))

def strip_fit_guarded(w, K):
    """cv.strip_fit with an N-safe noise cut and a dynamic-range guard:
    at large N the [K/2, K] band can sit entirely on the fft roundoff plateau
    (~1e-16*N*max), where the 1e-12*max cut of the rung-1 instrument admits
    flat noise points and returns a bogus tiny delta (S08/S09 artifact at
    N=4096). Requires >= 2 decades of genuine decay in the fitted points;
    otherwise returns None (no measurement)."""
    A = np.abs(np.fft.fft(w))[1:K+1]
    kk = np.arange(K//2, K+1)
    band = A[K//2 - 1: K]
    noise = 1e-10*A.max()
    good = band > noise
    if good.sum() < 10:
        return None
    lb = np.log(band[good])
    if lb.max() - lb.min() < 4.6:      # < 2 decades: flat/noise band
        return None
    slope = np.polyfit(kk[:len(band)][good], lb, 1)[0]
    return -slope

def stage_T(N=2048):
    """First physics: the a=-1 cell from single-mode data, theta-form.
    Track sup|theta_x|, strip delta_est(theta_x), 1/sup fits; guard-stopped.
    Saves ccfT_N<N>.npz for the fit/classification pass."""
    print("stage T: a=-1 ride from theta0 = sin x (omega0 = cos x), N=%d" % N); t0 = time.time()
    dt = 1.25e-5
    ops = cv.make_ops(N)
    x, k, hmult, mask, K = ops
    th = np.sin(x)
    t = 0.0
    rec = []
    nstep = 0
    while t < 3.0:
        th = rk4(th, dt, ops, rhs_th)
        t += dt
        nstep += 1
        if nstep % 200 == 0:
            thx = np.real(np.fft.ifft(1j*k*np.fft.fft(th)))
            s = np.max(np.abs(thx))
            d = strip_fit_guarded(thx, K)
            dK = d*K if d is not None else np.nan
            rec.append((t, s, dK))
            if not np.isfinite(s) or s > 1e8:
                print("  amp stop at t=%.5f (sup=%.3g)" % (t, s)); break
            if d is not None and dK < 6.0:
                print("  strip stop at t=%.5f (deltaK=%.1f, sup=%.4g)" % (t, dK, s)); break
    rec = np.array(rec)
    np.savez(os.path.join(DIR, "ccfT_N%d.npz" % N), rec=rec, N=N, dt=dt)
    # preliminary window fits on 1/sup (lin/quad), strip-verified stretch only
    tt, ss = rec[:, 0], rec[:, 1]
    m = ss >= 3.0
    if m.sum() >= 8:
        for w_ in (0.2, 0.1, 0.05):
            mm = m & (tt >= tt[m][-1] - w_)
            if mm.sum() < 5:
                continue
            c1 = np.polyfit(tt[mm], 1.0/ss[mm], 1)
            c2 = np.polyfit(tt[mm], 1.0/ss[mm], 2)
            r2 = np.roots(c2); r2 = r2[np.isreal(r2)].real
            r2 = r2[np.argmin(np.abs(r2 - (-c1[1]/c1[0])))] if len(r2) else np.nan
            print("  window %.2f: T_lin=%.5f T_quad=%.5f  (last t=%.5f, sup=%.4g, growth x%.1f)"
                  % (w_, -c1[1]/c1[0], r2, tt[mm][-1], ss[mm][-1], ss[mm][-1]/1.0))
    print("[%.0f s]" % (time.time() - t0))

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "u":     stage_u()
    elif st == "c0":  stage_c0()
    elif st == "bal": stage_bal()
    elif st == "a1":  stage_a1()
    elif st.startswith("T"):
        stage_T(int(st.split(":")[1]) if ":" in st else 2048)
    else: raise SystemExit("unknown stage: %s" % st)
