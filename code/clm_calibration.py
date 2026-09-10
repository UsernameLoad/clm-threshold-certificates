#!/usr/bin/env python3
# clm_calibration.py — Rung 1 (inviscid CLM)
# Equation: w_t = w * H(w) on the 2*pi torus; H = periodic Hilbert transform
#   (Fourier symbol -i*sgn(k), so H(cos kx) = sin kx).
# Exact solution for w0 = cos(x)  [derived via z = Hw + i*w, z_t = z^2/2]:
#   w(x,t)   = 4 cos(x) / (4 - 4 t sin(x) + t^2)
#   Hw(x,t)  = (4 sin(x) - 2 t) / (4 - 4 t sin(x) + t^2)
#   ||w(t)||_inf = 4 / (4 - t^2)          blowup T* = 2 at x = pi/2
# Complex singularity of w(.,t) at x = pi/2 + i*ln(t/2)
#   => analyticity strip delta(t) = ln(2/t)
#   => predicted spectral truncation floor ~ (t/2)^K, K = dealias cutoff N/3.
# Deterministic. Reproduce all numbers with:  python3 clm_calibration.py
import os, sys, time
import numpy as np

# ---------------- operators ----------------
def make_ops(N):
    x = 2 * np.pi * np.arange(N) / N
    k = np.fft.fftfreq(N, 1.0 / N)          # [0..N/2-1, -N/2..-1]
    hmult = -1j * np.sign(k)
    hmult[N // 2] = 0.0                     # zero Nyquist mode
    K = N // 3                              # 2/3-rule dealias cutoff
    mask = np.abs(k) <= K
    return x, k, hmult, mask, K

def H(w, hmult):
    return np.real(np.fft.ifft(hmult * np.fft.fft(w)))

def rhs(w, hmult, mask):
    p = w * H(w, hmult)
    ph = np.fft.fft(p)
    ph[~mask] = 0.0
    return np.real(np.fft.ifft(ph))

def rk4_step(w, dt, hmult, mask):
    k1 = rhs(w, hmult, mask)
    k2 = rhs(w + 0.5 * dt * k1, hmult, mask)
    k3 = rhs(w + 0.5 * dt * k2, hmult, mask)
    k4 = rhs(w + dt * k3, hmult, mask)
    return w + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

# ---------------- exact solution ----------------
def w_ex(x, t):   return 4 * np.cos(x) / (4 - 4 * t * np.sin(x) + t * t)
def Hw_ex(x, t):  return (4 * np.sin(x) - 2 * t) / (4 - 4 * t * np.sin(x) + t * t)
def sup_ex(t):    return 4.0 / (4.0 - t * t)

# ---------------- driver ----------------
def run(N, dt, T, checkpoints):
    x, k, hmult, mask, K = make_ops(N)
    w = np.cos(x)
    cp = {int(round(t / dt)): t for t in checkpoints}
    out = {}
    nsteps = int(round(T / dt))
    for n in range(1, nsteps + 1):
        w = rk4_step(w, dt, hmult, mask)
        if n in cp:
            t = cp[n]
            rec = {}
            rec["err_grid"] = float(np.max(np.abs(w - w_ex(x, t))))
            rec["mean_drift"] = float(abs(np.mean(w)))
            # off-grid check: evaluate trig interpolant on a 4x finer grid
            M = 4 * N
            wh = np.fft.fft(w)
            pad = np.zeros(M, dtype=complex)
            pad[: N // 2] = wh[: N // 2]
            pad[-(N // 2):] = wh[-(N // 2):]
            wf = np.real(np.fft.ifft(pad)) * (M / N)
            xf = 2 * np.pi * np.arange(M) / M
            rec["err_fine"] = float(np.max(np.abs(wf - w_ex(xf, t))))
            # sup-norm error measured on the fine grid (still O(dx^2/16)
            # grid-sampling of the true supremum; see session log)
            rec["sup_err"] = float(abs(np.max(np.abs(wf)) - sup_ex(t)))
            rec["pred_floor"] = float((t / 2.0) ** K)
            out[t] = rec
    return out, K

def main():
    t0 = time.time()
    print("python %s | numpy %s" % (sys.version.split()[0], np.__version__))
    print("=" * 72)

    # ---- U: unit tests of the Hilbert transform & torus product identity ----
    N = 256
    x, k, hmult, mask, K = make_ops(N)
    u1 = np.max(np.abs(H(np.cos(x), hmult) - np.sin(x)))
    u2 = np.max(np.abs(H(np.sin(x), hmult) + np.cos(x)))
    rng = np.random.default_rng(0)
    fh = np.zeros(N, dtype=complex)
    for m in range(1, 21):
        c = rng.normal() + 1j * rng.normal()
        fh[m], fh[-m] = c, np.conj(c)
    f = np.real(np.fft.ifft(fh))
    u3 = np.max(np.abs(H(f * H(f, hmult), hmult) - 0.5 * (H(f, hmult) ** 2 - f ** 2)))
    u5 = max(np.max(np.abs(H(w_ex(x, t), hmult) - Hw_ex(x, t))) for t in (0.5, 1.0, 1.5))
    print("U1  |H(cos)-sin|_inf                  : %.3e" % u1)
    print("U2  |H(sin)+cos|_inf                  : %.3e" % u2)
    print("U3  |H(fHf)-((Hf)^2-f^2)/2|_inf       : %.3e   (random band-limited f, seed 0)" % u3)
    print("U5  |H(w_ex)-Hw_ex|_inf, t<=1.5       : %.3e" % u5)

    # ---- U4: symbolic verification of the exact solution ----
    try:
        import sympy as sp
        X, T_ = sp.symbols("x t", real=True)
        D = 4 - 4 * T_ * sp.sin(X) + T_ ** 2
        W = 4 * sp.cos(X) / D
        HW = (4 * sp.sin(X) - 2 * T_) / D
        r1 = sp.trigsimp(sp.expand((sp.diff(W, T_) - W * HW) * D ** 2))
        Z = HW + sp.I * W
        r2 = sp.trigsimp(sp.expand((sp.diff(Z, T_) - Z * Z / 2) * D ** 2))
        print("U4  sympy (w_t - w*Hw)*D^2            : %s" % r1)
        print("U4  sympy (z_t - z^2/2)*D^2           : %s" % r2)
    except Exception as e:
        print("U4  sympy verification SKIPPED:", e)

    # ---- R1: spatial (spectral) convergence at t = 1.5, dt = 5e-5 ----
    print("-" * 72)
    print("R1  spatial convergence at t=1.5 (dt=5e-5); predicted floor (0.75)^K")
    errs, Ks = [], []
    for Ni in (64, 128, 256, 512):
        o, Ki = run(Ni, 5e-5, 1.5, (1.5,))
        e = o[1.5]["err_grid"]
        print("    N=%5d  K=%4d  err=%.3e  pred=%.3e" % (Ni, Ki, e, (0.75) ** Ki))
        errs.append(e); Ks.append(Ki)
    fit = np.polyfit(Ks[:3], np.log(errs[:3]), 1)[0]
    print("    fitted decay rate (N=64..256): %.4f   | predicted ln(4/3) = %.4f"
          % (-fit, np.log(4.0 / 3.0)))

    # ---- R2: temporal order at t = 1.8, N = 1024 ----
    print("-" * 72)
    print("R2  temporal order at t=1.8, N=1024")
    es = []
    for dt in (8e-4, 4e-4, 2e-4):
        o, _ = run(1024, dt, 1.8, (1.8,))
        es.append(o[1.8]["err_grid"])
        print("    dt=%.1e  err=%.3e" % (dt, es[-1]))
    for i in range(len(es) - 1):
        print("    observed order: %.2f" % np.log2(es[i] / es[i + 1]))

    # ---- R3: headline run N=1024, dt=5e-5, to t=1.9 ----
    print("-" * 72)
    print("R3  headline: N=1024, dt=5e-5, checkpoints to t=1.9 (T*=2)")
    cps = (0.5, 1.0, 1.5, 1.8, 1.85, 1.9)
    out, K = run(1024, 5e-5, 1.9, cps)
    print("    K=%d;  columns: on-grid err | 4x off-grid err | sup-norm err | pred floor (t/2)^K" % K)
    for t in cps:
        r = out[t]
        print("    t=%.2f  %.3e  %.3e  %.3e  %.3e"
              % (t, r["err_grid"], r["err_fine"], r["sup_err"], r["pred_floor"]))
    print("    max mean drift over run: %.3e (mean is an exact invariant)"
          % max(out[t]["mean_drift"] for t in cps))

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = np.array(cps)
        eg = np.array([out[t]["err_grid"] for t in cps])
        ef = np.array([out[t]["err_fine"] for t in cps])
        pf = np.array([out[t]["pred_floor"] for t in cps])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].semilogy(ts, eg, "o-", label="on-grid max error")
        ax[0].semilogy(ts, ef, "s--", label="off-grid (4x) max error")
        ax[0].semilogy(ts, pf, "k:", label=r"predicted floor $(t/2)^K$")
        ax[0].set_xlabel("t"); ax[0].set_ylabel("max abs error")
        ax[0].set_title("CLM solver vs exact solution (N=1024)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        tt = np.linspace(0, 1.95, 400)
        ax[1].semilogy(tt, sup_ex(tt), "-", label=r"exact $4/(4-t^2)$")
        sup_meas = [sup_ex(t) + out[t]["sup_err"] for t in cps]  # display only
        ax[1].semilogy(ts, sup_meas, "o", label="measured sup |w|")
        ax[1].axvline(2.0, color="r", ls=":", label=r"$T^*=2$")
        ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\|\omega\|_\infty$")
        ax[1].set_title("Sup-norm growth toward blowup")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "clm_calibration.png"), dpi=150)  # S03 path fix
        print("    figure saved: clm_calibration.png")
    except Exception as e:
        print("    figure SKIPPED:", e)

    print("=" * 72)
    print("wall time: %.1f s" % (time.time() - t0))

if __name__ == "__main__":
    main()
