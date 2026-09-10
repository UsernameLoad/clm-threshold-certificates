#!/usr/bin/env python3
# clm_viscous.py — Rung 2 (viscous CLM), rev B
# Equation:  w_t = w*H(w) + nu*w_xx  on the 2*pi torus, mean-zero data w0 = cos x.
#
# Complexification: q = Hw - i*w has only positive Fourier modes, and
#     q_t = q^2/2 + nu*q_xx        (exactly; H commutes with d_xx)
# => lower-triangular mode hierarchy for q = sum_{m>=1} c_m(t) e^{imx}:
#     c_m' = -nu*m^2*c_m + (1/2) * sum_{j=1}^{m-1} c_j c_{m-j},  c_1(0)=-i, c_m(0)=0.
# TRIANGULARITY => modes 1..M of the TRUE solution are computed exactly (no
# truncation feedback). NO RESONANCES: source exponents p satisfy
# p <= j^2+(m-j)^2 < m^2, so each c_m(t) is a finite sum of exponentials
# e^{-p*nu*t}: an explicit exact solution in the spirit of Schochet (CPAM 1986),
# derived independently here.
#
# KNOWN NUMERICAL LIMIT (measured in rev A, documented in session log): the
# exponential-sum representation is exact algebra but ILL-CONDITIONED in float64
# for small nu*t at large m (catastrophic cancellation). Policy: float64 series
# used only at nu=0.25 (verified vs mpmath); at nu=0.05 the exact series is
# evaluated in 40-digit mpmath at M=24 and compared PER MODE against the PDE.
#
# Methods:
#   (i)   FFT pseudospectral RK4 on w (N=1024, dealias K=341)
#   (ii)  DOP853 on coefficient ODEs truncated at M=K (same Galerkin system,
#         independent integrator/implementation - no FFT)
#   (iii) closed-form exponential-sum coefficients (exact in time)
# Deterministic. Reproduce with: python3 clm_viscous.py
import os, sys, time
import numpy as np

# ---------------- method (i): FFT pseudospectral ----------------
def make_ops(N):
    x = 2*np.pi*np.arange(N)/N
    k = np.fft.fftfreq(N, 1.0/N)
    hmult = -1j*np.sign(k); hmult[N//2] = 0.0
    K = N//3
    mask = np.abs(k) <= K
    return x, k, hmult, mask, K

def rhs_w(w, nu, k, hmult, mask):
    wh = np.fft.fft(w)
    Hw = np.real(np.fft.ifft(hmult*wh))
    ph = np.fft.fft(w*Hw); ph[~mask] = 0.0
    return np.real(np.fft.ifft(ph - nu*(k**2)*wh*mask))

def rk4(w, dt, nu, k, hmult, mask):
    k1 = rhs_w(w, nu, k, hmult, mask)
    k2 = rhs_w(w + 0.5*dt*k1, nu, k, hmult, mask)
    k3 = rhs_w(w + 0.5*dt*k2, nu, k, hmult, mask)
    k4 = rhs_w(w + dt*k3, nu, k, hmult, mask)
    return w + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def strip_fit(w, K):
    """delta_est from log-linear fit of |w_hat_k| over k in [K/2,K], using only
    points >= 100x the fft noise floor (rev A underestimated delta by fitting
    noise-flattened tail)."""
    A = np.abs(np.fft.fft(w))[1:K+1]
    kk = np.arange(K//2, K+1)
    band = A[K//2 - 1: K]
    noise = 1e-12 * A.max()
    good = band > noise
    if good.sum() < 10:
        return None
    slope = np.polyfit(kk[:len(band)][good], np.log(band[good]), 1)[0]
    return -slope

def pde_run(N, dt, nu, Tmax, checkpoints, track_every=100, spec_every=400):
    x, k, hmult, mask, K = make_ops(N)
    w = np.cos(x)
    cp = {int(round(t/dt)): t for t in checkpoints}
    snaps, track, spec = {}, [], []
    strict_h = None
    nsteps = int(round(Tmax/dt))
    stop_reason = "Tmax"
    t = 0.0
    for n in range(1, nsteps+1):
        w = rk4(w, dt, nu, k, hmult, mask)
        t = n*dt
        if n in cp:
            snaps[cp[n]] = w.copy()
        if n % track_every == 0:
            track.append((t, float(np.max(np.abs(w)))))
        if n % spec_every == 0:
            d = strip_fit(w, K)
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
                t_stop=t, stop=stop_reason, strict_h=strict_h, K=K)

# ---------------- method (ii): DOP853 on coefficient ODEs ----------------
def dop853_coeffs(nu, M, teval):
    from scipy.integrate import solve_ivp
    m2 = (np.arange(1, M+1)**2).astype(float)
    def f(t, y):
        c = y[:M] + 1j*y[M:]
        conv = np.convolve(c, c)
        nl = np.zeros(M, complex); nl[1:] = 0.5*conv[:M-1]
        dc = -nu*m2*c + nl
        return np.concatenate([dc.real, dc.imag])
    y0 = np.zeros(2*M); y0[M] = -1.0     # c_1(0) = -i
    sol = solve_ivp(f, (0.0, max(teval)), y0, t_eval=teval,
                    method="DOP853", rtol=1e-12, atol=1e-14)
    return [sol.y[:M, i] + 1j*sol.y[M:, i] for i in range(len(teval))]

# ---------------- method (iii): exact exponential-sum coefficients ----------------
def exact_coeffs(nu, M):
    assert nu > 0
    B = [None]*(M+1)
    b1 = np.zeros(2, complex); b1[1] = -1j
    B[1] = b1
    for m in range(2, M+1):
        L = m*m + 1
        s = np.zeros(L, complex)
        for j in range(1, m//2 + 1):
            pj = np.convolve(B[j], B[m-j])
            fac = 0.5 if (2*j == m) else 1.0
            s[:len(pj)] += fac*pj
        p = np.arange(L, dtype=float)
        denom = nu*(m*m - p); denom[m*m] = 1.0
        c = s/denom
        c[m*m] = -c.sum()
        B[m] = c
    return B

def eval_exact(B, nu, t):
    M = len(B) - 1
    c = np.zeros(M+1, complex)
    for m in range(1, M+1):
        p = np.arange(len(B[m]))
        c[m] = np.sum(B[m]*np.exp(-nu*t*p))
    return c[1:]

def mp_exact_coeffs(nu, M, dps=40):
    import mpmath as mp
    mp.mp.dps = dps
    B = [None]*(M+1)
    B[1] = {1: -mp.mpc(0, 1)}
    for m in range(2, M+1):
        s = {}
        for j in range(1, m):
            for pa, va in B[j].items():
                for pb, vb in B[m-j].items():
                    s[pa+pb] = s.get(pa+pb, mp.mpc(0)) + mp.mpf(0.5)*va*vb
        cm = {p: v/(mp.mpf(nu)*(m*m - p)) for p, v in s.items()}
        cm[m*m] = -sum(cm.values())
        B[m] = cm
    return B

def mp_eval(B, nu, t):
    # exponent must be formed in mpmath: forming -nu*t*p in float64 injects
    # ~1e-16 relative exponent error -> ~|B|*|p*nu*t|*1e-16 absolute noise,
    # which dominated at small nu (measured: 8e-6 at nu=0.05, m<=24, dps-independent)
    import mpmath as mp
    nut = mp.mpf(nu)*mp.mpf(t)
    out = []
    for m in range(1, len(B)):
        out.append(complex(sum(v*mp.exp(-nut*p) for p, v in B[m].items())))
    return np.array(out)

def coeffs_to_w(c, N):
    wh = np.zeros(N, complex)
    M = len(c)
    wh[1:M+1] = 1j*c/2 * N
    wh[-M:] = np.conj(wh[1:M+1])[::-1]
    return np.real(np.fft.ifft(wh))

def pde_coeffs(w, M):
    """c_m = -2i * w_hat_m extracted from a PDE snapshot."""
    wh = np.fft.fft(w)/len(w)
    return -2j*wh[1:M+1]

# ---------------- blowup-time extrapolation ----------------
def fit_T(track, window=0.25):
    t, s = track[:, 0], track[:, 1]
    tw = t >= t[-1] - window
    tt, yy = t[tw], 1.0/s[tw]
    lin = np.polyfit(tt, yy, 1)
    T_lin = -lin[1]/lin[0] if lin[0] < 0 else np.nan
    quad = np.polyfit(tt, yy, 2)
    r = np.roots(quad); r = r[np.isreal(r)].real; r = r[r > tt[-1]]
    T_quad = r.min() if len(r) else np.nan
    return T_lin, T_quad

def verdict(R):
    tr = R["track"]
    tw = tr[:, 0] >= tr[-1, 0] - 0.5
    slope = np.polyfit(tr[tw, 0], tr[tw, 1], 1)[0]
    if "collapsed" in R["stop"] or "blowup" in R["stop"]:
        return "BLOWUP (strip collapse)"
    if slope < 0:
        return "DECAY (sup decreasing at Tmax)"
    return "UNDETERMINED (still growing at Tmax)"

# ---------------- main ----------------
def main():
    t0 = time.time()
    import scipy
    print("python %s | numpy %s | scipy %s" %
          (sys.version.split()[0], np.__version__, scipy.__version__))
    print("="*74)

    # V1: numeric check of the complexification identity incl. viscous term
    N = 256
    x, k, hmult, mask, K0 = make_ops(N)
    rng = np.random.default_rng(1)
    wh = np.zeros(N, complex)
    for m in range(1, 31):
        c = rng.normal() + 1j*rng.normal()
        wh[m], wh[-m] = c, np.conj(c)
    w = np.real(np.fft.ifft(wh))
    nu_t = 0.137
    Hw = np.real(np.fft.ifft(hmult*np.fft.fft(w)))
    wt = w*Hw + nu_t*np.real(np.fft.ifft(-(k**2)*np.fft.fft(w)))
    lhs = np.real(np.fft.ifft(hmult*np.fft.fft(wt))) - 1j*wt
    q = Hw - 1j*w
    qxx = np.fft.ifft(-(k**2)*np.fft.fft(q))
    v1 = np.max(np.abs(lhs - (q*q/2 + nu_t*qxx)))
    print("V1  |q_t - (q^2/2 + nu q_xx)|_inf (random band-limited w): %.3e" % v1)

    # V2: symbolic recursion consistency at order 5 (symbolic nu, t)
    try:
        import sympy as sp
        nu_s, t_s, W = sp.symbols("nu t W", positive=True)
        c = {1: -sp.I*sp.exp(-nu_s*t_s)}
        for m in range(2, 6):
            src = sp.expand(sp.Rational(1, 2)*sum(c[j]*c[m-j] for j in range(1, m)))
            part = sp.integrate(sp.exp(nu_s*m**2*t_s)*src, t_s)
            c[m] = sp.expand(sp.exp(-nu_s*m**2*t_s)*(part - part.subs(t_s, 0)))
        q_s = sum(c[m]*W**m for m in range(1, 6))
        R = sp.expand(sp.diff(q_s, t_s) - q_s**2/2
                      + nu_s*sum(m**2*c[m]*W**m for m in range(1, 6)))
        bad = [m for m in range(1, 6) if sp.simplify(R.coeff(W, m)) != 0]
        print("V2  symbolic recursion residual, modes 1..5 nonzero at:", bad or "none (all zero)")
    except Exception as e:
        print("V2  sympy SKIPPED:", e)

    # V3: exact series at nu=0.25 (float64, conditioning OK) + mpmath at nu=0.05
    tA = time.time()
    B25 = exact_coeffs(0.25, 90)
    Bmp = mp_exact_coeffs(0.05, 24)
    c25_chk = eval_exact(B25, 0.25, 1.0)[:24]
    cmp_chk = mp_eval(mp_exact_coeffs(0.25, 24), 0.25, 1.0)
    print("V3  float64 vs mpmath(40dps) exact series, nu=0.25, m<=24, t=1: %.3e   [built in %.1f s]"
          % (np.max(np.abs(c25_chk - cmp_chk)), time.time()-tA))

    # ---- per-nu study ----
    print("-"*74)
    plan = [(0.0, 2.5), (0.01, 4.0), (0.05, 8.0), (0.1, 6.0), (0.25, 5.0)]
    cps = (0.5, 1.0, 1.2)
    dt = 2.5e-5
    results = {}
    for nu, Tmax in plan:
        tB = time.time()
        R = pde_run(1024, dt, nu, Tmax, cps)
        results[nu] = R
        print("nu=%.3f  t_stop=%.3f (%s)  strict_horizon=%s  sup(t_stop)=%.3e  [%.0f s]" % (
            nu, R["t_stop"], R["stop"],
            ("%.3f" % R["strict_h"]) if R["strict_h"] else "n/a",
            R["track"][-1, 1] if len(R["track"]) else float("nan"), time.time()-tB))

        if nu > 0:
            cs2 = dop853_coeffs(nu, R["K"], list(cps))
            for i, t in enumerate(cps):
                w1 = R["snaps"][t]
                d12 = np.max(np.abs(w1 - coeffs_to_w(cs2[i], 1024)))
                extra = ""
                if nu == 0.25:
                    c3 = eval_exact(B25, 0.25, t)
                    d13 = np.max(np.abs(w1 - coeffs_to_w(c3, 1024)))
                    a = np.abs(c3); rr = (a[-1]/a[-10])**(1/9.0)
                    tail = a[-1]*rr/(1-rr) if rr < 0.95 else np.inf
                    extra = "   (i)vs(iii,f64): %.3e  tail<=%.1e" % (d13, tail)
                if nu == 0.05:
                    cm = mp_eval(Bmp, 0.05, t)
                    cp_ = pde_coeffs(w1, 24)
                    extra = "   (i)vs(iii,mp) per-mode m<=24: %.3e" % np.max(np.abs(cm - cp_))
                print("    t=%.1f  (i)vs(ii): %.3e%s" % (t, d12, extra))

        if len(R["track"]) > 20:
            T_lin, T_quad = fit_T(R["track"])
            note = "  (exact T*=2)" if nu == 0.0 else ""
            print("    verdict: %s | T_est lin %.4f quad %.4f%s"
                  % (verdict(R), T_lin, T_quad, note))

    sp0 = results[0.0]["spec"]
    if len(sp0):
        i = int(np.argmin(np.abs(sp0[:, 0] - 1.75)))
        print("    delta_est(nu=0, t=%.2f) = %.4f  | exact ln(2/t) = %.4f"
              % (sp0[i, 0], sp0[i, 1], np.log(2.0/sp0[i, 0])))

    # ---- resolution robustness at nu=0.05, N=2048 ----
    print("-"*74)
    tC = time.time()
    R2 = pde_run(2048, dt, 0.05, 6.0, ())
    Ta = fit_T(results[0.05]["track"]); Tb = fit_T(R2["track"])
    print("R4  nu=0.05: N=1024 stop %.3f (%s) T_lin=%.4f T_quad=%.4f" %
          (results[0.05]["t_stop"], results[0.05]["stop"], Ta[0], Ta[1]))
    print("             N=2048 stop %.3f (%s) T_lin=%.4f T_quad=%.4f  [%.0f s]" %
          (R2["t_stop"], R2["stop"], Tb[0], Tb[1], time.time()-tC))

    # ---- Rung-3 teaser: gCLM a=1 (De Gregorio) steady state -sin(x) ----
    print("-"*74)
    N3 = 256
    x3, k3, hm3, mk3, K3 = make_ops(N3)
    absk = np.abs(k3); absk[0] = 1.0
    def rhs_g(w, a):
        wh = np.fft.fft(w)
        u  = np.real(np.fft.ifft(-wh/absk * (np.abs(k3) > 0)))
        ux = np.real(np.fft.ifft(hm3*wh))
        wx = np.real(np.fft.ifft(1j*k3*wh*mk3))
        ph = np.fft.fft(-a*u*wx + w*ux); ph[~mk3] = 0.0
        return np.real(np.fft.ifft(ph))
    w3 = -np.sin(x3)
    dtg = 1e-3
    for n in range(int(5.0/dtg)):
        k1 = rhs_g(w3, 1.0); k2 = rhs_g(w3+0.5*dtg*k1, 1.0)
        k3_ = rhs_g(w3+0.5*dtg*k2, 1.0); k4 = rhs_g(w3+dtg*k3_, 1.0)
        w3 = w3 + dtg*(k1+2*k2+2*k3_+k4)/6.0
    print("R5  gCLM a=1: drift from steady state -sin(x) after t=5: %.3e"
          % np.max(np.abs(w3 + np.sin(x3))))

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for nu, _ in plan:
            tr = results[nu]["track"]
            ax[0].plot(tr[:, 0], 1.0/tr[:, 1], label="nu=%.2f" % nu)
        ax[0].axvline(2.0, color="k", ls=":", lw=1)
        ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$1/\|\omega\|_\infty$")
        ax[0].set_title("Inverse sup-norm (root = blowup)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        for nu, _ in plan:
            spc = results[nu]["spec"]
            if len(spc):
                ax[1].plot(spc[:, 0], spc[:, 1]*results[nu]["K"], label="nu=%.2f" % nu)
        ax[1].axhline(32, color="gray", ls="--", lw=1)
        ax[1].axhline(12, color="r", ls="--", lw=1)
        ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\delta_{est}\cdot K$")
        ax[1].set_title("Analyticity-strip collapse"); ax[1].legend(fontsize=8)
        ax[1].set_yscale("log"); ax[1].grid(alpha=0.3)
        c3 = eval_exact(B25, 0.25, 1.2)
        ax[2].semilogy(np.arange(1, 91), np.abs(c3), label="nu=0.25 exact series")
        cpde = np.abs(pde_coeffs(results[0.25]["snaps"][1.2], 90))
        ax[2].semilogy(np.arange(1, 91), cpde, "--", label="nu=0.25 PDE")
        ax[2].set_xlabel("mode m"); ax[2].set_ylabel(r"$|c_m(1.2)|$")
        ax[2].set_title("Spectrum: exact series vs PDE"); ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "clm_viscous.png"), dpi=150)  # S03 path fix
        print("figure saved: clm_viscous.png")
    except Exception as e:
        print("figure SKIPPED:", e)

    print("="*74)
    print("wall time: %.1f s" % (time.time() - t0))

if __name__ == "__main__":
    main()
