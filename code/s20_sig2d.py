#!/usr/bin/env python
"""
s20_sig2d.py -- THE sigma-DIAL 2D LEVEL.
Design + registered U9 battery:.
U9 verdicts, including three honest test-redesign records: 5c.

WHAT THIS ADDS.  The certified kappa level (bsq_solver.py, S12) solves
    th_t + u.grad(th) = kappa*Lap(th),   no-flux walls,  omega inviscid.
This level replaces Lap by the FRACTIONAL operator -(-Lap)^{sigma/2}:
    th_t + u.grad(th) = -kappa*(-Lap)^{sigma/2} th,
so sigma = 2 is the certified kappa level, sigma = 1 is the 2D critical
exponent (Hmidi-Keraani-Rousset: global regularity proven at beta = 1 for
Euler-Boussinesq with fractional thermal diffusion, beta < 1 OPEN), and
sigma < 1 is the theorem-open supercritical region where plan v2.0's
selected Phase IV program lives.

HOW (three facts established by the design probe recorded in the log 5a):
  (1) With the two no-flux rows eliminated -- [th_0; th_Ny] = G th_int, G
      from D_xi[0,:] th = D_xi[Ny,:] th = 0 -- the reduced y-operator is
          A_red(kx) = A0 + kx^2 I,      A0 = [-4 D_xi^2]_ii + [-4 D_xi^2]_ib G,
      so the EIGENVECTORS ARE kx-INDEPENDENT: one eigendecomposition
      serves every kx, and the kx dependence is the analytic shift.
  (2) A0's spectrum is real and non-negative (max|Im lam| = 0.0 exactly)
      and matches the exact Neumann eigenvalues (n pi)^2 to ~1e-12 for the
      resolved modes; the spurious high modes are real, positive and
      O(N^4), hence strongly damped by any decaying symbol.
  (3) cond(W) = 2.7-4.2 -- the eigenbasis is essentially perfectly
      conditioned, which is what makes an EXACT-EXPONENTIAL step viable.

Hence the diffusive half-step is exact in time:
    th_int  <-  W diag(exp(-kappa (lam_n + kx^2)^{sigma/2} h)) W^{-1} th_int
with wall values reconstructed by G.  No Crank-Nicolson, no Helmholtz
solve, no order reduction from the diffusion step, unconditionally stable
at every sigma including 0.

CONVENTION, on the record: 0^{sigma/2} := 0 for EVERY sigma >= 0 (the
fractional operator annihilates constants, matching the multiplier
|xi|^sigma).  So the sigma = 0 operator is I - P0, not the bare identity.
NOTE (U9g): this makes the theta-mean conserved only up to the leakage of
the strip-integral functional onto the UNRESOLVED eigenmodes (measured
5.4e-4 at n = 42 against 0.126 on the lam = 0 mode at Ny = 64), i.e. the
mean drift is a resolution monitor, not an exact invariant.  Measured
1e-10 (sigma = 0) to 8e-8 (sigma = 2) on compatible data.

bsq_solver.py is NOT EDITED: this is a subclass, so the gate's u/u7/u8
references stay valid, and the physics ride reuses the certified stage_p2
loop VERBATIM by class substitution.

Stages:
  u9                       the registered certification battery
  p3:<sigma>[:N[:kappa[:cfl[:tag[:A]]]]]   the sigma-dial ride
"""
import os
import sys
import time

import numpy as np
from scipy.fft import rfft, irfft

import bsq_solver
from bsq_solver import BSQ


# ------------------------------------------------------------------ the level
class BSQS(BSQ):
    """The kappa level with fractional thermal dissipation.

    sigma is None    -> the inherited Crank-Nicolson Laplacian D-step
                        (used only by the U9-REPRO obligation).
    sigma is a float -> the exact-exponential fractional D-step.
    """

    def __init__(self, *a, sigma=None, **kw):
        super().__init__(*a, **kw)
        self.sigma = sigma
        self._frac = None
        if sigma is not None:
            self._build_frac()

    # -- the k-independent Neumann eigenbasis (built once per configuration)
    def _build_frac(self):
        Ny = self.Ny
        D = self.ch.Dxi
        M = -4.0 * (D @ D)                       # -d2/dy2 on the CGL points
        idx = np.arange(1, Ny)                   # interior unknowns
        bnd = np.array([0, Ny])
        Bm = np.array([[D[0, 0], D[0, Ny]], [D[Ny, 0], D[Ny, Ny]]])
        G = -np.linalg.solve(Bm, D[np.ix_(bnd, idx)])       # (2, Ny-1)
        A0 = M[np.ix_(idx, idx)] + M[np.ix_(idx, bnd)] @ G
        lam, W = np.linalg.eig(A0)
        imax = float(np.max(np.abs(lam.imag))) if np.iscomplexobj(lam) else 0.0
        o = np.argsort(lam.real)
        lam = np.asarray(lam[o].real)
        W = np.asarray(W[:, o].real)
        W = W / np.linalg.norm(W, axis=0)
        self._frac = dict(A0=A0, G=G, lam=np.maximum(lam, 0.0), W=W,
                          Wi=np.linalg.inv(W), imax=imax, lam_raw=lam,
                          condW=np.linalg.cond(W), idx=idx, bnd=bnd)

    def symbol(self, sigma=None, h=1.0, kappa=None):
        """exp(-kappa (lam_n + kx^2)^{sigma/2} h), shape (Ny-1, nk).
        0^{sigma/2} := 0 for every sigma >= 0 (constants annihilated)."""
        s = self.sigma if sigma is None else sigma
        kap = self.kappa if kappa is None else kappa
        lam = self._frac['lam'][:, None] + (self.kx ** 2)[None, :]
        p = np.where(lam > 0.0, np.power(np.maximum(lam, 0.0), 0.5 * s), 0.0)
        return np.exp(-kap * p * h)

    def dstep(self, th, h):
        """The fractional diffusive substep over time h (exact exponential)."""
        if self.sigma is None:
            return super().dstep(th, h)
        if self.kappa == 0.0 or h == 0.0:
            return th
        F = self._frac
        idx, bnd, G, W, Wi = F['idx'], F['bnd'], F['G'], F['W'], F['Wi']
        E = self.symbol(h=h)
        TH = rfft(th, axis=1, workers=self.wk)               # (Ny+1, nk)
        V = W @ (E * (Wi @ TH[idx, :]))                      # interior state
        OUT = np.empty_like(TH)
        OUT[idx, :] = V
        OUT[bnd, :] = G @ V                                  # no-flux walls
        return irfft(OUT, n=self.Nx, axis=1, workers=self.wk)


# ------------------------------------------------------------------ helpers
def _mode(S, n, j):
    """cos(n pi y) cos(k_j x): an exact joint eigenfunction of the continuous
    Neumann operator, eigenvalue (n pi)^2 + k_j^2."""
    X = np.tile(S.x, (S.Ny + 1, 1))
    Y = np.tile(S.ch.y[:, None], (1, S.Nx))
    return np.cos(n * np.pi * Y) * np.cos(S.kx[j] * X)


def _neumann_random(S, nmax, jmax, seed, jmin=0):
    """A random WALL-COMPATIBLE field: a random combination of the exact
    Neumann eigenfunctions.  The S12 U8f/U8h lesson, re-learned at U9e v1:
    any clause about the D-step's action needs data that already satisfies
    the no-flux condition, because the step discards the input's wall values
    and reconstructs them (exactly as the certified CN tier does)."""
    r = np.random.default_rng(seed)
    X = np.tile(S.x, (S.Ny + 1, 1))
    Y = np.tile(S.ch.y[:, None], (1, S.Nx))
    g = np.zeros((S.Ny + 1, S.Nx))
    for n in range(0, nmax + 1):
        for j in range(jmin, jmax + 1):
            g += r.standard_normal() * np.cos(n * np.pi * Y) * np.cos(S.kx[j] * X)
    return g


def _lam_disc(S, n, j):
    """the SEMI-DISCRETE eigenvalue the level uses for mode (n, j)."""
    return S._frac['lam'][n] + S.kx[j] ** 2


# ------------------------------------------------------------------ battery
def stage_u9():
    print("stage u9: the sigma-dial level certification battery")
    print("  (floors + scaling models registered in"
          " written before the first run; three clauses were REDESIGNED after"
          " diagnosed failures -- v1 records kept, see 5c and s20_u9_diag.txt)")
    Lx = 1.0 / 6.0
    T, H = 0.1, 10
    rng = np.random.default_rng(20)

    # ---------------- U9-REPRO: the subclass must not touch the certified path
    N = 64
    R = BSQ(N, N, Lx=Lx, kappa=1e-2)
    Q = BSQS(N, N, Lx=Lx, kappa=1e-2, sigma=None)
    X = np.tile(R.x, (N + 1, 1))
    Y = np.tile(R.ch.y[:, None], (1, N))
    th0 = 1.0e4 * np.exp(-60.0 * (Y * (2.0 - Y)) ** 4) * \
        np.sin(2.0 * np.pi * X / Lx) ** 2
    wa, ta = np.zeros_like(th0), th0.copy()
    wb, tb = np.zeros_like(th0), th0.copy()
    for _ in range(20):
        wa, ta, _, _ = R.step_p2(wa, ta, 1e-6)
        wb, tb, _, _ = Q.step_p2(wb, tb, 1e-6)
    d = max(np.abs(wa - wb).max(), np.abs(ta - tb).max())
    print("  U9-REPRO sigma=None 20-step p2 vs BSQ : %.1e   (predict 0.0 EXACT)"
          "  %s" % (d, "PASS" if d == 0.0 else "FAIL -- STOP"))

    # ---------------- U9a: the eigen-decomposition
    print("  U9a eigen-decomposition (predict |Im|=0.0, min Re >= -1e-10,"
          " cond(W) <= 10, rel err <= 1e-10 @Ny=64 / 1e-9 @Ny=128)")
    for Ny in (32, 64, 128):
        S = BSQS(32, Ny, Lx=Lx, kappa=1e-2, sigma=1.0)
        F = S._frac
        lam = F['lam_raw']
        ex = (np.pi * np.arange(0, Ny - 1)) ** 2
        nn = max(2, Ny // 4)
        err = np.max(np.abs(lam[1:nn] - ex[1:nn]) / ex[1:nn])
        print("    Ny=%3d  max|Im lam|=%.1e  min Re lam=%+.2e  cond(W)=%.3f"
              "  max rel |lam_n-(n pi)^2| (n<=%d) = %.2e  lam_max=%.3e"
              % (Ny, F['imax'], lam.min(), F['condW'], nn - 1, err, lam.max()))

    # ---------------- U9b: pure-eigenmode decay, machine floor at every sigma
    print("  U9b pure-eigenmode decay vs the exact semi-discrete answer"
          " (predict <= 1e-11 rel at every sigma)")
    S = BSQS(64, 64, Lx=Lx, kappa=1e-2, sigma=1.0)
    worst = 0.0
    for sg in (0.0, 0.5, 1.0, 1.5, 2.0):
        S.sigma = sg
        wsg = wcont = 0.0
        for n in (0, 1, 2, 5):
            for j in (0, 1, 2):
                th = _mode(S, n, j)
                f = th.copy()
                for _ in range(H):
                    f = S.dstep(f, T / H)
                lam = _lam_disc(S, n, j)
                pred = np.exp(-S.kappa * (lam ** (0.5 * sg) if lam > 0 else 0.0) * T)
                wsg = max(wsg, np.abs(f - pred * th).max() / np.abs(th).max())
                lamc = (n * np.pi) ** 2 + S.kx[j] ** 2
                predc = np.exp(-S.kappa * (lamc ** (0.5 * sg) if lamc > 0 else 0.0) * T)
                wcont = max(wcont, abs(pred - predc))
        worst = max(worst, wsg)
        print("    sigma=%.2f  max rel err vs semi-discrete = %.2e"
              "   [discretization vs continuum: %.2e, reported]" % (sg, wsg, wcont))
    print("    U9b worst over all sigma: %.2e   %s"
          % (worst, "PASS" if worst <= 1e-11 else "FAIL"))

    # ---------------- U9c: sigma = 2 against the certified CN kappa level
    print("  U9c sigma=2 exact exponential vs the certified CN D-step"
          " (predict dt-halving ratio in [3.5, 4.5])")
    S2 = BSQS(64, 64, Lx=Lx, kappa=1e-2, sigma=2.0)
    Rc = BSQ(64, 64, Lx=Lx, kappa=1e-2)
    base = Rc.dstep(Rc.dealias_f(rng.standard_normal((65, 64))), 1e-2)
    prev = None
    for dt in (1e-3, 5e-4, 2.5e-4):
        nst = int(round(0.02 / dt))
        fa = base.copy(); fb = base.copy()
        for _ in range(nst):
            fa = S2.dstep(fa, dt); fb = Rc.dstep(fb, dt)
        e = np.abs(fa - fb).max() / np.abs(base).max()
        r = "" if prev is None else "   ratio %.2f" % (prev / e)
        print("    v1 (one CN smoothing step only)     dt=%.1e %4d steps"
              "  |frac-CN| rel = %.4e%s" % (dt, nst, e, r))
        prev = e
    print("    v1 DIAGNOSIS: lam_max(Ny=64) = %.2e, so z = kappa*lam*dt = %.1f"
          " at dt=1e-3 -- CN is NOT asymptotic on the top modes, which is what"
          " the 6.28 first ratio measures."
          % (S2._frac['lam'].max(), 1e-2 * S2._frac['lam'].max() * 1e-3))
    g = _neumann_random(S2, 16, 10, 7)
    prev = None
    for dt in (1e-3, 5e-4, 2.5e-4, 1.25e-4):
        nst = int(round(0.02 / dt))
        fa = g.copy(); fb = g.copy()
        for _ in range(nst):
            fa = S2.dstep(fa, dt); fb = Rc.dstep(fb, dt)
        e = np.abs(fa - fb).max() / np.abs(g).max()
        r = "" if prev is None else "   ratio %.2f" % (prev / e)
        print("    v2 (wall-COMPATIBLE band-limited)   dt=%.1e %4d steps"
              "  |frac-CN| rel = %.4e%s" % (dt, nst, e, r))
        prev = e

    # ---------------- U9d: kappa = 0 bit-identity
    S0 = BSQS(48, 48, Lx=Lx, kappa=0.0, sigma=1.0)
    X = np.tile(S0.x, (49, 1)); Y = np.tile(S0.ch.y[:, None], (1, 48))
    th = np.exp(-30.0 * (Y - 0.5) ** 2) * np.sin(2.0 * np.pi * X / Lx) ** 2
    w = 0.1 * np.sin(2.0 * np.pi * X / Lx) * np.sin(np.pi * Y)
    wa, ta = w.copy(), th.copy(); wb, tb = w.copy(), th.copy()
    for _ in range(20):
        wa, ta, _, _ = S0.step_p2(wa, ta, 1e-4)
        wb, tb, _, _ = S0.step(wb, tb, 1e-4)[:4]
    print("  U9d kappa=0 20-step p2-vs-plain       : %.1e   (predict 0.0 EXACT)"
          % max(np.abs(wa - wb).max(), np.abs(ta - tb).max()))

    # ---------------- U9e: THE sigma = 0 DEGENERATE ANCHOR
    Sz = BSQS(64, 64, Lx=Lx, kappa=1e-2, sigma=0.0)
    print("  U9e sigma=0 degenerate anchor (I - P0).  v1 AS REGISTERED FAILED"
          " (6.1e-1) for TWO reasons, both in the CLAUSE:")
    print("     (a) wall-INCOMPATIBLE data -- v1 residual by row: walls"
          " 1.06e+00 / interior 3.16e-05 (the S12 U8f/U8h trap, third time);")
    print("     (b) a normalisation bug -- integral() returns the STRIP"
          " integral (Lx x average) and v1 subtracted it as an average,"
          " injecting c'(1-E)(1-Lx)/E ~ 1e-6, which matches v1's interior"
          " residual exactly.  Both mine; the level was never implicated.")
    for lab, f0 in (("wall-COMPATIBLE Neumann-mode data",
                     _neumann_random(Sz, 16, 10, 7)),
                    ("same, kx=0 column excluded",
                     _neumann_random(Sz, 16, 10, 7, jmin=1))):
        avg0 = Sz.integral(f0) / Sz.Lx
        f = f0.copy()
        for _ in range(H):
            f = Sz.dstep(f, T / H)
        avgT = Sz.integral(f) / Sz.Lx
        mf0 = f0 - avg0
        print("     v2 %-34s mean-free rel resid = %.2e"
              % (lab, np.abs((f - avgT) * np.exp(Sz.kappa * T) - mf0).max()
                 / np.abs(mf0).max()))
    Fz = Sz._frac
    v = _neumann_random(Sz, 16, 10, 7)[Fz['idx'], 0]
    P0 = np.outer(Fz['W'][:, 0], Fz['Wi'][0, :])
    E1 = np.exp(-Sz.kappa * (T / H))
    lhs = Fz['W'] @ (Sz.symbol(h=T / H)[:, 0] * (Fz['Wi'] @ v))
    print("     v2 operator identity |D(h)v - (P0 + e^{-kh}(I-P0))v| = %.2e"
          "   [spread of P0 v = %.1e, i.e. P0 v IS constant]"
          % (np.abs(lhs - (P0 @ v + E1 * (v - P0 @ v))).max(), np.ptp(P0 @ v)))

    # ---------------- U9f: no-flux enforcement from incompatible data
    Sf = BSQS(64, 64, Lx=Lx, kappa=1e-2, sigma=1.0)
    g0 = Sf.dealias_f(rng.standard_normal((65, 64)))     # wall-INCOMPATIBLE
    gy = Sf.ddy(Sf.dstep(g0, 1e-3))
    print("  U9f no-flux after one D-step (incompatible in): %.2e rel"
          "   (predict <= 1e-9)"
          % (max(np.abs(gy[0]).max(), np.abs(gy[-1]).max())
             / np.abs(Sf.ddy(g0)).max()))

    # ---------------- U9g: mean conservation -- v1 REFUTED, converted
    print("  U9g theta-mean conservation: v1 (<= 1e-13, registered WITHOUT a"
          " scaling model for this quantity) is REFUTED.  Mechanism, measured:")
    Sm0 = BSQS(64, 64, Lx=Lx, kappa=1e-2, sigma=2.0)
    wt = Sm0.ch.wq[Sm0._frac['idx']] + Sm0.ch.wq[Sm0._frac['bnd']] @ Sm0._frac['G']
    proj = wt @ Sm0._frac['W']
    nmax = 1 + int(np.argmax(np.abs(proj[1:])))
    print("     the strip-integral functional is NOT a left null vector of A0:"
          " |w~.W[:,0]| = %.4e, max_{n>=1} = %.2e at n = %d (lam = %.2e)"
          % (abs(proj[0]), np.abs(proj[1:]).max(), nmax, Sm0._frac['lam'][nmax]))
    print("     -> the mean is conserved to the field's content in the"
          " UNRESOLVED eigenmodes; the drift is a free resolution monitor.")
    line = []
    for sg in (0.0, 0.5, 1.0, 1.5, 2.0):
        Sm = BSQS(64, 64, Lx=Lx, kappa=1e-2, sigma=sg)
        a0 = _neumann_random(Sm, 16, 10, 11) + 3.0
        i0 = Sm.integral(a0)
        a = a0.copy()
        for _ in range(H):
            a = Sm.dstep(a, T / H)
        b0 = Sm.dealias_f(np.random.default_rng(11).standard_normal((65, 64))) + 3.0
        j0 = Sm.integral(b0)
        b = b0.copy()
        for _ in range(H):
            b = Sm.dstep(b, T / H)
        line.append("s=%.2f: %.1e/%.1e" % (sg, abs(Sm.integral(a) - i0) / abs(i0),
                                           abs(Sm.integral(b) - j0) / abs(j0)))
    print("     compatible/incompatible:  " + "   ".join(line))

    # ---------------- U9h: resolution doubling
    print("  U9h resolution doubling.  v1 (n = 2 fixed) is VOID: that mode is"
          " already at machine floor at Ny = 32, so the clause had no content.")
    print("     v2 -- mode index scaled with the grid, n = Ny/3, sigma = 1:")
    for Ny in (32, 64, 128, 192):
        Sh = BSQS(64, Ny, Lx=Lx, kappa=1e-2, sigma=1.0)
        n = max(2, Ny // 3)
        lam = Sh._frac['lam'][n]
        lamc = (n * np.pi) ** 2
        print("       Ny=%3d  n=Ny/3=%3d  |lam_disc - lam_cont|/lam = %.3e"
              % (Ny, n, abs(lam - lamc) / lamc))

    # ---------------- U9i: free rows
    Si = BSQS(64, 64, Lx=Lx, kappa=1e-2, sigma=2.0)
    lamk = Si._frac['lam'][:, None] + (Si.kx ** 2)[None, :]
    direct = np.where(lamk > 0, np.exp(-Si.kappa * lamk * 1e-3), 1.0)
    print("  U9i sigma=2 symbol vs direct exp(-kappa(lam+k^2)h): %.1e (free)"
          % np.abs(Si.symbol(h=1e-3) - direct).max())
    for sg in (0.99, 1.00, 1.01, 1.99, 2.00, 2.01):
        print("     symbol continuity: sigma=%.2f  sum|E| = %.10f"
              % (sg, Si.symbol(sigma=sg, h=1e-3).sum()))


# ------------------------------------------------------------------ physics
def stage_p3(sigma, N=256, kappa=1e-2, cfl=0.4, tag="", A=1.0e4):
    """The sigma-dial ride.  The RIDE LOOP IS bsq_solver.stage_p2 VERBATIM --
    only the class it constructs is substituted -- so the certified P2 ride
    (taxonomy, certified window, snapshots, records) is reused rather than
    re-implemented."""
    class _Sub(BSQ):
        def __new__(cls, *a, **kw):
            return BSQS(*a, sigma=sigma, **kw)
    old = bsq_solver.BSQ
    tagS = ("s%s" % ("%.3f" % sigma).replace(".", "p")) + (("_" + tag) if tag else "")
    try:
        bsq_solver.BSQ = _Sub
        print("stage p3[sigma=%g]: the sigma-dial level; ride loop = certified"
              " stage_p2 verbatim" % sigma, flush=True)
        bsq_solver.stage_p2(N=N, kappa=kappa, cfl=cfl, tag=tagS, A=A)
    finally:
        bsq_solver.BSQ = old
    src = "p2_k%s_N%d%s_%s.npz" % (("%.6f" % kappa).replace(".", "p"), N,
                                   "" if A == 1.0e4 else "_A%g" % A, tagS)
    dst = "p3_%s_k%s_N%d%s.npz" % (tagS, ("%.6f" % kappa).replace(".", "p"), N,
                                   "" if A == 1.0e4 else "_A%g" % A)
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
        print("  record renamed -> %s" % dst, flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "u9":
        stage_u9()
    elif st.startswith("p3"):
        p = st.split(":")
        stage_p3(float(p[1]),
                 N=int(p[2]) if len(p) > 2 else 256,
                 kappa=float(p[3]) if len(p) > 3 else 1e-2,
                 cfl=float(p[4]) if len(p) > 4 else 0.4,
                 tag=p[5] if len(p) > 5 else "",
                 A=float(p[6]) if len(p) > 6 else 1.0e4)
    else:
        print("stages: u9 | p3:<sigma>[:N[:kappa[:cfl[:tag[:A]]]]]")
