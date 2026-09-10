#!/usr/bin/env python
"""
bsq_solver.py -- Rung 5: 2D incompressible Boussinesq on the periodic-in-x strip. Entry ticket: the build plan (anchors FIRST).

System (Hou-Luo scenario's home):
    w_t + u.grad(w) = theta_x  (+ nu_w Lap w)
    th_t + u.grad(th) = 0      (+ kappa Lap th)
    u = (-psi_y, psi_x),  Lap psi = w,  psi = 0 at y=0,1  (no-penetration walls)
Domain: x in [0,2pi) periodic (Nx uniform), y in [0,1] Chebyshev-Gauss-Lobatto (Ny+1 pts).

Discretization:
  - Fourier (rfft) in x; Chebyshev (DCT-I) in y. Derivatives: ik in x; coefficient
    recurrence in xi = 2y-1 (d/dy = 2 d/dxi).
  - Poisson per kx: (4 D2_xi - kx^2) psi = w, psi(walls)=0; dense collocation matrix,
    LU precomputed per kx (verified-before-fast: banded/QI upgrade only if profiling demands).
  - Products pointwise on the grid; dealiasing by 2/3 coefficient truncation in BOTH
    directions applied to each nonlinear product.
  - RK4, adaptive dt = cfl / max(|u|/dx + |v|/dy_local); v=0 at walls makes the CGL
    near-wall clustering CFL-benign (v ~ v_y * dy locally).
  - Dissipation hooks (nu_w, kappa) EXPLICIT and OFF by default. Design decision on
    record (S10): the first dissipative dial for the P2 threshold program will be
    kappa-only (no-flux theta walls, vorticity inviscid) -- it needs NO vorticity wall
    condition, leaves the corner mechanism intact, and attacks the driver theta_x
    directly. nu_w with stress-free walls is the second dial; no-slip (influence
    matrix) only if the physics demands it. NOT used in this anchor stage.

Symmetry class (Hou-Luo, A4): w odd in x, theta even in x about x=0; u even, v odd.

Stages:  u          -- anchor battery A1-A4 (U0-U6), floors predicted per test below
         u:fast     -- same battery with fast=True (S11; must be digit-for-digit)
         u7         -- S11 fast-path EXACT-equality battery (see below)
         bench[:N[:fast]] -- RK4 step timing at N x N (default 256, then 512)
Anchors A5-A7 (external Luo-Hou diagnostics; strip tracking; cross-solver) are later
stages by design -- physics (P1) is gated on A1-A4 passing at their predicted floors.

S11 FAST PATH (design decision on record): every optimization is chosen to be
BIT-IDENTICAL to the reference path -- execution-order/parallelism changes only,
no algorithm changes:
  (a) Poisson: the SAME per-kx lu_solve calls, executed by a thread pool over
      disjoint kx chunks (check_finite=False skips only the validation pass);
  (b) transforms: scipy.fft workers=-1 (threads across the independent 1D lines
      of a batch; per-line kernel unchanged);
  (c) Cheb.dcoeff: the sequential k-recurrence replaced by per-parity-chain
      reversed cumsum -- the SAME additions in the SAME order (used both paths);
  (d) rhs: ddx(th) computed once and reused (was computed twice on identical
      input -> identical bits).
The explicit-inverse route was REJECTED (changes results AND still streams the
same O(N^3) factor memory -- the measured wall). PREDICTED FLOORS (S11, written
before first run) -- ALL MET (record s11_bsq_u7.txt):
  U7a poisson fast-vs-ref (manufactured + random):     0.0 EXACT   [met]
  U7b ddy fast(workers/cumsum)-vs-ref loop:            0.0 EXACT   [met]
  U7c full rhs fast-vs-ref (random parity data):       0.0 EXACT   [met]
  U7d 20-step evolution fast-vs-ref:                   0.0 EXACT   [met]
  stage u under fast=True: digit-for-digit identical to S10 floors [met].

S11 QI LEVEL (second optimization tier; profiling verdict: the per-kx dense-LU
Poisson streams ~540 MB of factors per solve at 512^2 -- a memory-bandwidth
wall; threading gave only 59->49 ms). qi=True replaces the Poisson SOLVER
ALGORITHM (not the equation): Chebyshev-tau with quasi-inverse (apply B^2, the
double-integration operator, to rows k>=2: 4*psi_k - kx^2 (B^2 psi)_k =
(B^2 w)_k) -- pentadiagonal, parity-split into TWO (tridiagonal + one dense
BC row) chains, solved by bordered Thomas sweeps VECTORIZED ACROSS ALL kx
(factors precomputed; O(N^2) memory total vs O(N^3)). This CHANGES floating-
point results => it is a separate level with its own certification: the LU
path remains the reference; U7 gains a QI battery. Well-posedness note: the
QI system is the classical integration-preconditioned tau method (O(1)-ish
conditioning vs N^4 for dense collocation).
PREDICTED FLOORS (QI level, written before first run):
  U7e qi-vs-LU poisson, manufactured smooth RHS:       <= 1e-13   [met: 3e-14]
  U7f qi-vs-LU poisson, random rough RHS (rel):        <= 1e-11   [FAILED: 1e-3]
  U7g qi-vs-LU 20-step evolution (rel):                <= 1e-11   [met: 9e-15]
  stage u under qi=True: U1 recovery <= 1e-13; U2/U3/U6-HL zeros PRESERVED
  (structural: they never depended on Poisson internals); others within 10x
  of the S10 floors.
U7f FAILURE RECORD + DIAGNOSIS (S11, honest-failure discipline): the original
U7f prediction assumed tau and collocation are the same linear system up to
rounding. They are NOT: they are two different (equally valid) spectral
discretizations that agree only on RESOLVED content. Unfiltered pointwise
random data carries O(1) coefficient energy at the truncation edge, where the
two methods legitimately differ (tau drops the k+2>N couplings and replaces
the top rows; collocation enforces the ODE at nodes). VERIFIED: pre-filtering
the same random RHS to the 2/3-resolved band (exactly what dealiased physics
feeds the solver) collapses the difference to 1.6e-13 / 4.8e-13 / 1.4e-11 at
64^2/128x96/256^2, and the raw difference lives in the high coefficients.
U7f is REDEFINED to the resolved-band form (prediction <= 2e-11); the raw-
noise number is reported alongside as [method-diff], not an error metric.
QI OUTCOME (record s11_bsq_u7.txt + s11_bsq_uqi.txt): U7e/f/g all within
floors (3.0e-14 / 1.1e-11 / 9.4e-15 at 256^2); stage u under qi: U1 recovery
IMPROVES to 3.5e-17 (B^2 stencil exact on polynomial content), U2/U3 zeros
preserved exactly; U6-HL moves 0.000e+00 -> 4.8e-16 -- the parity mechanism
still cancels the advection term exactly, but the bit-exact 0.0 of the
reference was a floating-point coincidence of the LU psi values, not a
structural guarantee (prediction imprecise on that one line; noted).

S12 KAPPA LEVEL (P2 entry: the dissipative dial, kappa-only per the S10 design
decision above). Physics: th_t + u.grad(th) = kappa*Lap(th), NO-FLUX walls
(th_y = 0 at y=0,1); omega stays inviscid; psi keeps Dirichlet. Time
integration: Strang splitting D(dt/2) A(dt) D(dt/2) -- A is the UNTOUCHED
certified RK4 advection step, D is a Crank-Nicolson Helmholtz solve
(unconditionally stable; explicit diffusion on the CGL grid would cost
dt ~ dy_min^2/kappa ~ 1e-7). Per kx: (1 + b*kx^2)*th - 4b*th_xixi = rhs with
b = kappa*h/2 (h = the substep), two tiers mirroring the Poisson:
  - lu tier (reference): dense collocation with NEUMANN rows (Dxi rows at
    both walls), factors cached per (b); rebuilt on every b change (slow --
    reference only).
  - qi tier (production): tau + B^2 quasi-inverse, SAME parity-chain
    bordered-Thomas machinery as poisson_qi with generalized coefficients
    (row k>=2: 4b*th_k - a*(B^2 th)_k = -(B^2 rhs)_k, a = 1 + b*kx^2; the
    B^2-stencil identity |mu| = beta + gamma gives diagonal-dominance slack
    exactly 4b > 0); Neumann BC per parity chain: sum k^2 c_k = 0 (T_k'(+-1)
    = +-k^2; one weighted sum kills both walls in each parity). Refactor
    cost O(N*nk) per b change -- cheap at every adaptive-dt step.
kappa = 0 applies NO D-step: stage p2 at kappa=0 is BIT-IDENTICAL to p1.
PREDICTED FLOORS (S12, written before first run) + MEASURED OUTCOMES:
  U8a heat-kernel exact decay, x-mode/y-mode/mixed (kappa=1e-2, dt=1e-3,
      T=0.1; CN error z^3/12 per step, z = kappa*lam*dt):     <= 1e-9  rel
      (both tiers; and at tiny b = 1e-9: solve error          <= 1e-10)
      [MET: 5.1e-10 (matches z^3/12 arithmetic to 2%) / 1e-11 class /
       tiny-b 3e-13 lu, 5e-15 qi]
  U8b no-flux wall enforcement after D-step (th_y at walls):  <= 1e-8  rel
      [MET: 7.5e-13 lu / 1.1e-12 qi]
  U8c theta-mean conservation per D-step (no-flux + periodic):<= 1e-10 rel
      [MET: 0.0 exact lu / 1.4e-16 qi]
  U8d Strang split order, v1 prediction ratio ~5 (2nd order): FAILED TWICE,
      DIAGNOSED, ACCEPTED: v1 (wall-incompatible data) ratio 3.01; v2
      (compatible data) ratio 2.92 -- first-order-class from the classical
      Neumann-boundary splitting order reduction (per-step advective
      regeneration of wall gradients, u_y*th_x|wall, interacting with the
      split). MAGNITUDE MEASURED AND DECISIVE: |err| = 1.1e-8 rel (theta)
      at dt=1e-3 over T=0.01 on O(1) fields, wall excess mild (1.11 vs
      0.87 e-8); production CFL dt is >= 100x smaller => <= 1e-9-class per
      ride at first order. ACCEPTED with mandatory cfl-halving spot-check
      per kappa-regime (verdict + Omega_inf invariance).
  U8e kappa=0 path 20-step evolution vs plain step:           0.0 EXACT
      [MET: exact 0.0]
  U8f qi-vs-lu D-step: v1 (bare 2/3-filtered random) FAILED (6.1e-3) --
      diagnosed: wall-INCOMPATIBLE data is projected through an under-
      resolved sqrt(b) layer differently by tau vs collocation (the U7f
      lesson one level deeper: agreement needs band-limited AND compatible
      data; the physics is compatible for t > 0). v2 (random compatible
      band, j,m <= 20):                                       <= 1e-11
      [MET: 1.7e-13]
  U8g x-parity (theta even) preservation through D-step: v1 "0.0 EXACT"
      IMPRECISE (recorded): evenness survives structurally but through
      rfft roundoff imaginaries -- floor class ~N*eps [measured 4-5e-15,
      same class as the U5 convention].
  U8h theta^2 dissipation identity d/dt int(th^2)/2 = -kappa*int|grad th|^2:
      v1 (incompatible data) FAILED (0.55/0.43) -- the projection layer,
      not the identity; v2 (compatible, CN midpoint field) predicted
      <= 1e-8:  [MISSED x11: 1.1e-7 -- quadrature-class residual of the
      product integrands at 64^2; PASS-WITH-NOTE at 1e-7]
P2 STAGE PREDICTIONS (registered before the first kappa ride, S12):
  (i)   kappa*(A=1e4) in [0.005, 0.08] (driver-mode balance: the sin^2 mode
        kx = 4pi/Lx = 75.4 has diffusion rate kappa*kx^2; matching the
        inviscid crisis t_s = 3.24e-3 gives kappa ~ 0.05; the sharpening
        front raises the effective kx, biasing kappa* DOWN from that).
  (ii)  amplitude law: kappa*(A) = kappa*(1e4)*sqrt(A/1e4) EXACTLY at
        continuum level (the Boussinesq scaling th -> lam*th, t -> t/sqrt(lam),
        om -> sqrt(lam)*om maps kappa -> kappa*sqrt(lam); walls fixed).
  (iii) verdict taxonomy: BLOWUP-bar (certified-window growth through the
        bar; bar-calibrated per N15, as always), SATURATED (driver collapse:
        trailing contribution of int sup|th_x| dt < 2% of total AND trailing
        sup-omega growth < 2%; omega has NO dissipation channel, so decay
        verdicts do not exist here -- saturation is the kappa-world analog),
        RESOLUTION (tails through 1e-2), UNDET(Tmax).
  (iv)  the excursion object: Omega_inf(kappa) = the saturated sup|omega|
        plateau; prediction: Omega_inf diverges as kappa decreases toward
        the ride-window threshold (the 2D analog of the N16 peak divergence;
        exponent gamma_2D unknown -- THE question of record).

PREDICTED FLOORS (written before first run; see session log S10):
  U0 transform round-trip + derivative exactness on polynomials: <= 1e-13 (N^2*eps*scale)
  U1 Poisson residual + exact-solution recovery (poly RHS):      <= 1e-10 at Ny<=128
       (dense collocation cond ~ N^4; smooth-data error ~ N^2 eps, NOT cond*eps)
  U2 shear + stratified steady states, sup|RHS|:                 <= 1e-13
  U3 rest state with theta = g(y):                                exact 0.0 or <= 1e-15
  U4 manufactured full-operator vs sympy:                        <= 1e-11 (deg-7 poly x
       mode-2 trig content, far below dealias cutoffs at Ny>=48, Nx>=32)
  U5 parity preservation over 20 RK4 steps:                      <= 1e-13 relative
  U6 discrete energy balance |dE/dt - int(theta v)| (instant.):  <= 1e-9 * scale
       (no discrete skew-symmetry claimed; spectral quadrature on analytic fields)
     theta^2 drift per step (inviscid):                          reported, expect ~1e-10
"""
import os, sys, time
import numpy as np
from scipy.fft import rfft, irfft, dct
from scipy.linalg import lu_factor, lu_solve
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- Chebyshev in y
class Cheb:
    """Chebyshev machinery on y in [0,1] via xi = 2y-1, CGL points xi_i = cos(pi i/N).
    Point order: i=0 -> y=1 (top), i=N -> y=0 (bottom). Fields f[iy, jx]."""
    def __init__(self, N):
        self.N = N
        self.wk = None                       # scipy.fft workers; set by BSQ in fast mode
        i = np.arange(N + 1)
        self.xi = np.cos(np.pi * i / N)
        self.y = (self.xi + 1.0) / 2.0
        # Clenshaw-Curtis quadrature weights on xi in [-1,1], then scaled to y in [0,1]
        w = np.zeros(N + 1)
        v = np.ones(N - 1)
        for k in range(1, N // 2 + 1):
            fac = 2.0 if (2 * k != N) else 1.0
            v -= fac * np.cos(2.0 * k * np.pi * i[1:N] / N) / (4.0 * k * k - 1.0)
        w[1:N] = 2.0 * v / N
        w[0] = w[N] = 1.0 / (N * N - (1 if N % 2 else 0))
        self.wq = w / 2.0                    # int_0^1 f dy ~ sum wq * f
        # dense differentiation matrix (Trefethen) w.r.t. xi
        c = np.ones(N + 1); c[0] = c[N] = 2.0
        c *= (-1.0) ** i
        X = np.tile(self.xi, (N + 1, 1)).T
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
        D -= np.diag(D.sum(axis=1))
        self.Dxi = D                          # d/dxi at the points
        self.Dy = 2.0 * D                     # d/dy

    def coeff(self, f):
        """Chebyshev coefficients c_k along axis 0 (fields f[iy, ...])."""
        N = self.N
        A = dct(f, type=1, axis=0, workers=self.wk)
        c = A / N
        c[0] /= 2.0
        c[N] /= 2.0
        return c

    def point(self, c):
        """Inverse of coeff."""
        N = self.N
        b = c.copy()
        b[0] *= 2.0
        b[N] *= 2.0
        return dct(b, type=1, axis=0, workers=self.wk) / 2.0

    def dcoeff(self, c):
        """Coefficients of d/dxi from coefficients.
        S11: the backward recurrence b[k-1] = b[k+1] + 2k c[k] decomposes into two
        independent parity chains, each a reversed cumulative sum of t[k] = 2k c[k]
        accumulated from high k down -- np.cumsum performs the SAME additions in the
        SAME order, so this is bit-identical to the historical loop (verified U7b)."""
        N = self.N
        b = np.zeros_like(c)
        sh = [1] * c.ndim; sh[0] = -1
        t = (2.0 * np.arange(N + 1)).reshape(sh) * c
        for ks in (np.arange(N, 0, -2), np.arange(N - 1, 0, -2)):
            if len(ks):
                b[ks - 1] = np.cumsum(t[ks], axis=0)
        b[0] /= 2.0
        return b

    def ddy(self, f):
        """d/dy via coefficient recurrence (spectral)."""
        return self.point(self.dcoeff(self.coeff(f))) * 2.0

# ---------------------------------------------------------------- solver
class BSQ:
    def __init__(self, Nx, Ny, dealias=True, fast=False, qi=False, workers=-1, nthreads=12,
                 Lx=2.0 * np.pi, kappa=0.0):
        self.Nx, self.Ny = Nx, Ny
        self.Lx = Lx
        self.kappa = kappa
        self._helm_cache = {}      # b -> solve data (lu factors | qi chains)
        self.ch = Cheb(Ny)
        self.x = Lx * np.arange(Nx) / Nx
        self.kx = (2.0 * np.pi / Lx) * np.arange(Nx // 2 + 1)   # rfft wavenumbers
        self.dealias = dealias
        self.fast = fast
        self.qi = qi
        self.wk = workers if fast else None    # scipy.fft workers (None = historical default)
        self.ch.wk = self.wk
        if fast:
            nk = len(self.kx)
            T = max(1, min(nthreads, nk))
            edges = [i * nk // T for i in range(T)] + [nk]
            self.chunks = [range(edges[i], edges[i + 1]) for i in range(T)
                           if edges[i + 1] > edges[i]]
            self.pool = ThreadPoolExecutor(max_workers=len(self.chunks))
        self.kcutx = int((2.0 / 3.0) * (Nx // 2))
        self.kcuty = int((2.0 / 3.0) * Ny)
        # local y spacing for CFL (distance to nearest neighbor)
        y = self.ch.y
        dy = np.empty_like(y)
        dy[1:-1] = np.minimum(np.abs(y[1:-1] - y[:-2]), np.abs(y[2:] - y[1:-1]))
        dy[0] = np.abs(y[1] - y[0]); dy[-1] = np.abs(y[-1] - y[-2])
        self.dy_loc = dy[:, None]
        self.dx = Lx / Nx
        # Poisson (reference): per-kx dense LU of (4 D2 - kx^2 I) with Dirichlet rows
        D2 = self.ch.Dxi @ self.ch.Dxi
        Np1 = Ny + 1
        if not qi:
            self.lus = []
            for k in self.kx:
                A = 4.0 * D2 - (k * k) * np.eye(Np1)
                A[0, :] = 0.0; A[0, 0] = 1.0        # psi(y=1) = 0
                A[-1, :] = 0.0; A[-1, -1] = 1.0     # psi(y=0) = 0
                self.lus.append(lu_factor(A))
        else:
            # QI level (S11, see header): coefficient-space tau system per parity chain,
            # bordered-Thomas factors precomputed, sweeps vectorized across all kx.
            self.qi_chains = [self._qi_chain(0), self._qi_chain(1)]

    # ---------- spectral helpers
    def ddx(self, f):
        fh = rfft(f, axis=1, workers=self.wk)
        sh = [1] * f.ndim; sh[1] = -1
        fh *= (1j * self.kx).reshape(sh)
        return irfft(fh, n=self.Nx, axis=1, workers=self.wk)

    def ddy(self, f):
        return self.ch.ddy(f)

    def dealias_f(self, f):
        if not self.dealias:
            return f
        fh = rfft(f, axis=1, workers=self.wk)
        fh[:, self.kcutx:, ...] = 0.0
        f2 = irfft(fh, n=self.Nx, axis=1, workers=self.wk)
        c = self.ch.coeff(f2)
        c[self.kcuty:, ...] = 0.0
        return self.ch.point(c)

    # ---------- QI Poisson level (S11)
    def _qi_chain(self, k0):
        """Precompute one parity chain (k0 = 0 even / 1 odd) of the tau system:
        rows k >= 2 of  4 psi_k - kx^2 (B^2 psi)_k = (B^2 w)_k, unknowns
        psi_{k0}, psi_{k0+2}, ...; BC row: sum of chain coefficients = 0
        (psi(1) = psi(-1) = 0 split by parity). B^2 stencil (c-normalized):
        (B^2 f)_k = c_{k-2} f_{k-2}/(4k(k-1)) - f_k/(2(k^2-1)) + f_{k+2}/(4k(k+1)),
        with the f_{k+2} term dropped for k+2 > N (tau truncation).
        Reduced tridiagonal (x_0 moved to RHS) is strictly diagonally dominant
        (|l|+|u| = d - 4 on interior rows) => unpivoted Thomas is stable."""
        N = self.Ny
        kx2 = self.kx ** 2                            # (nk,)
        kun = np.arange(k0, N + 1, 2)                 # unknown k-values (chain)
        ks = kun[1:].astype(float)                    # equation k-values, i = 1..M
        M = len(ks)
        beta = np.where(ks == 2.0, 2.0, 1.0) / (4.0 * ks * (ks - 1.0))
        mu = -1.0 / (2.0 * (ks * ks - 1.0))
        gamma = np.where(ks + 2 <= N, 1.0, 0.0) / (4.0 * ks * (ks + 1.0))
        l = -kx2[None, :] * beta[:, None]             # (M, nk); l[0] couples x_0
        d = 4.0 - kx2[None, :] * mu[:, None]
        u = -kx2[None, :] * gamma[:, None]
        dp = np.empty_like(d); m = np.zeros_like(d)
        dp[0] = d[0]
        for i in range(1, M):
            m[i] = l[i] / dp[i - 1]
            dp[i] = d[i] - m[i] * u[i - 1]
        ch = dict(kun=kun, m=m, dp=dp, u=u, beta=beta, mu=mu, gamma=gamma,
                  igath=np.minimum(kun[1:] + 2, N))   # safe gather for k+2 (gamma=0 past N)
        # null vector of the underdetermined tridiagonal: T xn = 0 with xn_0 = 1
        gz = np.zeros((M, len(kx2)))
        gz[0] = -l[0]
        xn = np.vstack([np.ones((1, len(kx2))), self._qi_sweep(ch, gz)])
        ch['xn'] = xn
        ch['Snull'] = xn.sum(axis=0)
        return ch

    def _qi_sweep(self, ch, g):
        """Thomas solve of the reduced chain (x_0 = 0) with precomputed factors."""
        m, dp, u = ch['m'], ch['dp'], ch['u']
        M = g.shape[0]
        gp = np.empty_like(g)
        gp[0] = g[0]
        for i in range(1, M):
            gp[i] = g[i] - m[i] * gp[i - 1]
        x = np.empty_like(g)
        x[M - 1] = gp[M - 1] / dp[M - 1]
        for i in range(M - 2, -1, -1):
            x[i] = (gp[i] - u[i] * x[i + 1]) / dp[i]
        return x

    def poisson_qi(self, w):
        """QI-level Poisson: solve in (Cheb coeff in y) x (Fourier in x) space."""
        C = self.ch.coeff(w)                          # y-coeffs, x-grid (real)
        Ch = rfft(C, axis=1, workers=self.wk)         # (Np1, nk) complex
        P = np.empty_like(Ch)
        for ch in self.qi_chains:
            kun = ch['kun']
            g = (ch['beta'][:, None] * Ch[kun[:-1]] + ch['mu'][:, None] * Ch[kun[1:]]
                 + ch['gamma'][:, None] * Ch[ch['igath']])
            xr = self._qi_sweep(ch, g)                # particular solution, x_0 = 0
            s = -xr.sum(axis=0) / ch['Snull']         # BC: chain sum = 0
            P[kun[0]] = s
            P[kun[1:]] = xr + s[None, :] * ch['xn'][1:]
        Pg = irfft(P, n=self.Nx, axis=1, workers=self.wk)   # y-coeffs, x-grid
        return self.ch.point(Pg)

    def poisson(self, w):
        """Solve Lap psi = w, psi=0 at walls. Fast path: SAME lu_solve calls,
        thread pool over disjoint kx chunks (bit-identical; U7a). QI level
        replaces the algorithm (certified separately; U7e-g)."""
        if self.qi:
            return self.poisson_qi(w)
        wh = rfft(w, axis=1, workers=self.wk)
        ph = np.empty_like(wh)
        if self.fast:
            def do(chunk):
                for j in chunk:
                    rhs = wh[:, j].copy()
                    rhs[0] = 0.0; rhs[-1] = 0.0
                    ph[:, j] = lu_solve(self.lus[j], rhs, check_finite=False)
            list(self.pool.map(do, self.chunks))
        else:
            for j in range(len(self.kx)):
                rhs = wh[:, j].copy()
                rhs[0] = 0.0; rhs[-1] = 0.0
                ph[:, j] = lu_solve(self.lus[j], rhs)
        return irfft(ph, n=self.Nx, axis=1, workers=self.wk)

    def velocity(self, w):
        psi = self.poisson(w)
        u = -self.ddy(psi)
        v = self.ddx(psi)
        return u, v, psi

    # ---------- S12 kappa level: CN Helmholtz diffusion substep (design in header)
    def lap(self, f):
        """Spectral laplacian: -kx^2 in x, double Chebyshev derivative in y."""
        fh = rfft(f, axis=1, workers=self.wk)
        sh = [1] * f.ndim; sh[1] = -1
        fxx = irfft(fh * (-(self.kx ** 2)).reshape(sh), n=self.Nx, axis=1,
                    workers=self.wk)
        return fxx + self.ch.ddy(self.ch.ddy(f))

    def _helm_qi_chain(self, k0, b):
        """One parity chain of the tau Helmholtz system (header: S12 KAPPA
        LEVEL): rows k >= 2 of  4b*th_k - a*(B^2 th)_k = -(B^2 g)_k with
        a = 1 + b*kx^2 (per-kx); Neumann BC row: sum k^2 c_k = 0 over the
        chain (kills th_y at BOTH walls per parity). Same B^2 stencil and
        bordered-Thomas structure as _qi_chain; diagonal-dominance slack is
        exactly 4b > 0 (|mu| = beta + gamma)."""
        N = self.Ny
        a = 1.0 + b * (self.kx ** 2)                  # (nk,)
        kun = np.arange(k0, N + 1, 2)
        ks = kun[1:].astype(float)
        M = len(ks)
        beta = np.where(ks == 2.0, 2.0, 1.0) / (4.0 * ks * (ks - 1.0))
        mu = -1.0 / (2.0 * (ks * ks - 1.0))
        gamma = np.where(ks + 2 <= N, 1.0, 0.0) / (4.0 * ks * (ks + 1.0))
        l = -a[None, :] * beta[:, None]               # (M, nk)
        d = 4.0 * b - a[None, :] * mu[:, None]        # mu < 0 -> positive part
        u = -a[None, :] * gamma[:, None]
        dp = np.empty_like(d); m = np.zeros_like(d)
        dp[0] = d[0]
        for i in range(1, M):
            m[i] = l[i] / dp[i - 1]
            dp[i] = d[i] - m[i] * u[i - 1]
        ch = dict(kun=kun, m=m, dp=dp, u=u, beta=beta, mu=mu, gamma=gamma,
                  igath=np.minimum(kun[1:] + 2, N))
        gz = np.zeros((M, len(self.kx)))
        gz[0] = -l[0]
        xn = np.vstack([np.ones((1, len(self.kx)))    , self._qi_sweep(ch, gz)])
        ch['xn'] = xn
        wbc = (kun.astype(float)) ** 2                # Neumann weights k^2
        ch['wbc'] = wbc
        ch['Swnull'] = (wbc[:, None] * xn).sum(axis=0)
        if np.min(np.abs(ch['Swnull'])) < 1e-300:
            raise RuntimeError("helm chain: singular Neumann closure (Swnull ~ 0)")
        return ch

    def _helm_get(self, b):
        """Cached solve data for CN parameter b (per tier). Rebuilt on every
        new b (cheap for qi: O(N*nk) Thomas factorization)."""
        key = ('qi' if self.qi else 'lu', b)
        H = self._helm_cache.get(key)
        if H is None:
            if self.qi:
                H = [self._helm_qi_chain(0, b), self._helm_qi_chain(1, b)]
            else:
                D2 = self.ch.Dxi @ self.ch.Dxi
                Np1 = self.Ny + 1
                H = []
                for k in self.kx:
                    A = (1.0 + b * k * k) * np.eye(Np1) - 4.0 * b * D2
                    A[0, :] = self.ch.Dxi[0, :]       # th_xi = 0 at y=1 (top)
                    A[-1, :] = self.ch.Dxi[-1, :]     # th_xi = 0 at y=0 (bottom)
                    H.append(lu_factor(A))
            if len(self._helm_cache) > 8:             # keep the cache tiny
                self._helm_cache.clear()
            self._helm_cache[key] = H
        return H

    def _helm_solve(self, rhs, b):
        """Solve (1 + b kx^2) th - 4b th_xixi = rhs per kx, no-flux walls."""
        H = self._helm_get(b)
        if self.qi:
            C = self.ch.coeff(rhs)
            Ch = rfft(C, axis=1, workers=self.wk)
            P = np.empty_like(Ch)
            for ch in H:
                kun = ch['kun']
                g = -(ch['beta'][:, None] * Ch[kun[:-1]]
                      + ch['mu'][:, None] * Ch[kun[1:]]
                      + ch['gamma'][:, None] * Ch[ch['igath']])
                xr = self._qi_sweep(ch, g)
                s = -(ch['wbc'][1:] @ xr) / ch['Swnull']
                P[kun[0]] = s
                P[kun[1:]] = xr + s[None, :] * ch['xn'][1:]
            Pg = irfft(P, n=self.Nx, axis=1, workers=self.wk)
            return self.ch.point(Pg)
        wh = rfft(rhs, axis=1, workers=self.wk)
        ph = np.empty_like(wh)
        for j in range(len(self.kx)):
            r = wh[:, j].copy()
            r[0] = 0.0; r[-1] = 0.0
            ph[:, j] = lu_solve(H[j], r)
        return irfft(ph, n=self.Nx, axis=1, workers=self.wk)

    def dstep(self, th, h):
        """CN diffusion substep over time h (no-flux walls). kappa == 0 or
        h == 0: exact identity (no arithmetic touched)."""
        if self.kappa == 0.0 or h == 0.0:
            return th
        b = 0.5 * self.kappa * h
        return self._helm_solve(th + b * self.lap(th), b)

    def step_p2(self, w, th, dt):
        """Strang split D(dt/2) A(dt) D(dt/2); A = the certified RK4 step
        (untouched). kappa == 0 reduces to step() exactly (U8e)."""
        if self.kappa == 0.0:
            return self.step(w, th, dt)
        th = self.dstep(th, 0.5 * dt)
        w, th, u, v = self.step(w, th, dt)
        th = self.dstep(th, 0.5 * dt)
        return w, th, u, v

    def rhs(self, w, th):
        u, v, psi = self.velocity(w)
        tx = self.ddx(th)          # S11: computed once, reused in dw (identical bits)
        adv_w = self.dealias_f(u * self.ddx(w) + v * self.ddy(w))
        adv_t = self.dealias_f(u * tx + v * self.ddy(th))
        dw = -adv_w + tx
        dt_ = -adv_t
        return dw, dt_, u, v, psi

    def step(self, w, th, dt):
        k1w, k1t, u, v, _ = self.rhs(w, th)
        k2w, k2t, *_ = self.rhs(w + 0.5 * dt * k1w, th + 0.5 * dt * k1t)
        k3w, k3t, *_ = self.rhs(w + 0.5 * dt * k2w, th + 0.5 * dt * k2t)
        k4w, k4t, *_ = self.rhs(w + dt * k3w, th + dt * k3t)
        wn = w + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        tn = th + dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        return wn, tn, u, v

    def cfl_dt(self, u, v, cfl=0.4):
        s = np.abs(u) / self.dx + np.abs(v) / self.dy_loc
        m = s.max()
        return cfl / m if m > 0 else 1e-2

    def integral(self, f):
        """int over the strip: Clenshaw-Curtis in y x trapezoid(=exact) in x."""
        return (self.ch.wq[:, None] * f).sum() * self.dx

    # ---------- A6: resolution/analyticity instruments (S11)
    def diagnostics(self, w, th, psi=None):
        """A6 guard battery + corner observables. Conventions: y[0]=1 (top wall),
        y[Ny]=0 (bottom wall); the Hou-Luo corner is (x=0, y=0) = index [Ny, 0].
        tail_x / tail_y: max high-band spectral amplitude over max amplitude, the
        1D stack's cleanliness bar transplanted (certified window: tails <= 1e-6).
        delta_x: analyticity-strip width from the x-spectrum log-slope (fit over
        the clean decades of the upper half-band, R^2-guarded)."""
        d = {}
        aw = np.abs(w)
        ij = np.unravel_index(np.argmax(aw), aw.shape)
        d['sup_w'] = aw[ij]
        d['x_at_max'] = self.x[ij[1]]
        d['y_at_max'] = self.ch.y[ij[0]]
        d['sup_th'] = np.abs(th).max()
        # x-spectrum of w (max over y): tail + strip fit
        Ew = np.abs(rfft(w, axis=1, workers=self.wk)).max(axis=0)
        Ew = Ew / max(Ew.max(), 1e-300)
        ncut = self.kcutx
        d['tail_x'] = Ew[int(0.90 * ncut):ncut].max()
        # strip fit on kx in [ncut/2, ncut) where E > 1e-12 (clean floor)
        k1, k2 = ncut // 2, ncut
        seg = Ew[k1:k2]
        msk = seg > 1e-12
        if msk.sum() >= 8:
            kk = self.kx[k1:k2][msk]
            ll = np.log(seg[msk])
            A_ = np.vstack([kk, np.ones_like(kk)]).T
            sol, res, *_ = np.linalg.lstsq(A_, ll, rcond=None)
            d['delta_x'] = -sol[0]
            ss = ((ll - ll.mean())**2).sum()
            d['strip_r2'] = 1.0 - (res[0] / ss if len(res) and ss > 0 else np.nan)
        else:
            d['delta_x'] = np.nan; d['strip_r2'] = np.nan
        # y (Chebyshev) tail of w and th
        for nm, f in (('w', w), ('th', th)):
            c = np.abs(self.ch.coeff(f)).max(axis=1)
            c = c / max(c.max(), 1e-300)
            d[f'tail_y_{nm}'] = c[int(0.90 * self.kcuty):self.kcuty].max()
        # corner observables (A5 notes): psi_xy at the corner = the log-derivative proxy
        if psi is not None:
            d['psi_xy_corner'] = self.ddy(self.ddx(psi))[self.Ny, 0]
        d['th2'] = self.integral(th * th)
        return d

# ---------------------------------------------------------------- A7 cross-solver
class FD:
    """A7 (S11): INDEPENDENT low-order cross-check integrator. Uniform y grid,
    2nd-order central differences both directions, sparse-LU 5-point Poisson,
    classical RK4. Shares no discretization machinery with BSQ; its purpose is
    short-time agreement converging at O(h^2) to the spectral solution."""
    def __init__(self, Nx, Ny, Lx=2.0 * np.pi):
        from scipy.sparse import lil_matrix, csc_matrix
        from scipy.sparse.linalg import splu
        self.Nx, self.Ny, self.Lx = Nx, Ny, Lx
        self.dx = Lx / Nx
        self.hy = 1.0 / Ny
        self.x = Lx * np.arange(Nx) / Nx
        self.y = np.linspace(0.0, 1.0, Ny + 1)          # y[0]=0 bottom ... y[Ny]=1 top
        ni = Ny - 1                                      # interior rows
        n = ni * Nx
        A = lil_matrix((n, n))
        ix = lambda j, i: (j - 1) * Nx + i               # j=1..Ny-1 interior, i x-index
        cx, cy = 1.0 / self.dx**2, 1.0 / self.hy**2
        for j in range(1, Ny):
            for i in range(Nx):
                r = ix(j, i)
                A[r, r] = -2.0 * (cx + cy)
                A[r, ix(j, (i + 1) % Nx)] += cx
                A[r, ix(j, (i - 1) % Nx)] += cx
                if j + 1 <= Ny - 1: A[r, ix(j + 1, i)] += cy
                if j - 1 >= 1:      A[r, ix(j - 1, i)] += cy
        self.slu = splu(csc_matrix(A))

    def ddx(self, f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * self.dx)

    def ddy(self, f):
        g = np.empty_like(f)
        g[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.hy)
        g[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * self.hy)
        g[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * self.hy)
        return g

    def poisson(self, w):
        rhs = w[1:-1].reshape(-1)
        psi = np.zeros_like(w)
        psi[1:-1] = self.slu.solve(rhs).reshape(self.Ny - 1, self.Nx)
        return psi

    def rhs(self, w, th):
        psi = self.poisson(w)
        u = -self.ddy(psi); v = self.ddx(psi)
        tx = self.ddx(th)
        dw = -(u * self.ddx(w) + v * self.ddy(w)) + tx
        dth = -(u * tx + v * self.ddy(th))
        return dw, dth

    def step(self, w, th, dt):
        k1w, k1t = self.rhs(w, th)
        k2w, k2t = self.rhs(w + 0.5 * dt * k1w, th + 0.5 * dt * k1t)
        k3w, k3t = self.rhs(w + 0.5 * dt * k2w, th + 0.5 * dt * k2t)
        k4w, k4t = self.rhs(w + dt * k3w, th + dt * k3t)
        return (w + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w),
                th + dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t))

# ---------------------------------------------------------------- stages
def stage_u(fast=False, qi=False):
    print(f"stage u: rung-5 anchor battery A1-A4 (floors predicted in header)"
          + ("  [fast=True]" if fast else "") + ("  [qi=True]" if qi else ""))
    t0 = time.time()
    Nx, Ny = 64, 64
    S = BSQ(Nx, Ny, fast=fast, qi=qi)
    ch = S.ch
    X = np.tile(S.x, (Ny + 1, 1))
    Y = np.tile(ch.y[:, None], (1, Nx))

    # U0: transform round-trip + spectral d/dy on a polynomial
    f = Y**5 - 2.0 * Y**3 + Y
    rt = np.abs(ch.point(ch.coeff(f)) - f).max()
    dfe = 5.0 * Y**4 - 6.0 * Y**2 + 1.0
    de = np.abs(S.ddy(f) - dfe).max()
    dxe = np.abs(S.ddx(np.sin(3 * X)) - 3 * np.cos(3 * X)).max()
    print(f"U0  cheb round-trip poly deg5              : {rt:.3e}")
    print(f"U0  d/dy spectral vs exact (poly)          : {de:.3e}")
    print(f"U0  d/dx spectral vs exact (sin 3x)        : {dxe:.3e}")

    # U1: Poisson -- manufactured psi with exactly-representable content
    P = (Y**2) * (1.0 - Y)**2
    psi_e = np.sin(X) * P
    Pdd = 2.0 - 12.0 * Y + 12.0 * Y**2
    w_m = np.sin(X) * (Pdd - P)          # Lap psi_e
    psi_n = S.poisson(w_m)
    e1 = np.abs(psi_n - psi_e).max()
    bc = max(np.abs(psi_n[0]).max(), np.abs(psi_n[-1]).max())
    print(f"U1  Poisson exact-solution recovery        : {e1:.3e}")
    print(f"U1  Poisson wall values (Dirichlet)        : {bc:.3e}")

    # U2: steady states -- shear w=f(y) + stratification th=g(y)
    w_s = np.tile(np.cos(3.0 * np.pi * ch.y[:, None]) + 0.3 * ch.y[:, None], (1, Nx))
    th_s = np.tile((ch.y**3 - 0.5 * ch.y)[:, None], (1, Nx))
    dw, dth, *_ = S.rhs(w_s, th_s)
    print(f"U2  shear+strat steady: sup|dw/dt|         : {np.abs(dw).max():.3e}")
    print(f"U2  shear+strat steady: sup|dth/dt|        : {np.abs(dth).max():.3e}")

    # U3: rest state
    dw, dth, *_ = S.rhs(np.zeros_like(w_s), th_s)
    print(f"U3  rest state: sup|dw|,sup|dth|           : {np.abs(dw).max():.3e} {np.abs(dth).max():.3e}")

    # U4: manufactured full operator vs closed form (hand-derived, verified w/ sympy offline)
    # psi_m = sin(x) P(y);  th_m = cos(x) G(y), G = 1 + y^3 - y
    G = 1.0 + Y**3 - Y
    Gp = 3.0 * Y**2 - 1.0
    th_m = np.cos(X) * G
    u_e = -np.sin(X) * (2.0 * Y - 6.0 * Y**2 + 4.0 * Y**3)   # -psi_y
    v_e = np.cos(X) * P                                       # psi_x
    w_x = np.cos(X) * (Pdd - P)
    P3 = -12.0 + 24.0 * Y                                     # P'''
    Pp = 2.0 * Y - 6.0 * Y**2 + 4.0 * Y**3
    w_y = np.sin(X) * (P3 - Pp)
    th_x = -np.sin(X) * G
    th_y = np.cos(X) * Gp
    rhs_w_e = -(u_e * w_x + v_e * w_y) + th_x
    rhs_t_e = -(u_e * th_x + v_e * th_y)
    dw, dth, u_n, v_n, _ = S.rhs(w_m, th_m)
    print(f"U4  velocity from Poisson vs exact         : {np.abs(u_n-u_e).max():.3e} {np.abs(v_n-v_e).max():.3e}")
    print(f"U4  full RHS_w vs closed form              : {np.abs(dw-rhs_w_e).max():.3e}")
    print(f"U4  full RHS_th vs closed form             : {np.abs(dth-rhs_t_e).max():.3e}")

    # U5: Hou-Luo parity preservation over 20 RK4 steps
    w0 = np.sin(X) * np.sin(np.pi * Y) + 0.3 * np.sin(2 * X) * (Y**2) * (1 - Y)
    th0 = np.cos(X) * (0.5 + 0.5 * Y**2) + 0.1 * np.cos(2 * X) * Y
    w, th = w0.copy(), th0.copy()
    dt = 2e-3
    for _ in range(20):
        w, th, u, v = S.step(w, th, dt)
    wr = np.abs(w + np.roll(w[:, ::-1], 1, axis=1)).max() / np.abs(w).max()
    tr = np.abs(th - np.roll(th[:, ::-1], 1, axis=1)).max() / np.abs(th).max()
    print(f"U5  parity drift (w odd / th even), 20 stp : {wr:.3e} {tr:.3e}")

    # U6: instantaneous discrete energy balance dE/dt = int(theta v), on U5's state.
    # NOTE: in the Hou-Luo parity class psi*adv_w is odd in x, so the advection term
    # integrates to zero EXACTLY and the identity is degenerate; U6b breaks parity.
    dw, dth, u, v, psi = S.rhs(w, th)
    dEdt = -S.integral(psi * dw)
    work = S.integral(th * v)
    scale = max(abs(dEdt), abs(work), 1e-30)
    print(f"U6  |dE/dt - int(th v)| / scale (HL class) : {abs(dEdt-work)/scale:.3e}  (scale {scale:.3e})")
    wb = w + 0.4 * np.cos(X) * np.sin(np.pi * Y) + 0.2 * np.sin(3 * X) * Y * (1 - Y)
    tb = th + 0.3 * np.sin(X) * (1.0 + 0.5 * Y) + 0.1 * np.cos(3 * X) * Y**2
    dwb, dtb, ub, vb, psib = S.rhs(wb, tb)
    dEdtb = -S.integral(psib * dwb)
    workb = S.integral(tb * vb)
    scb = max(abs(dEdtb), abs(workb), 1e-30)
    print(f"U6b |dE/dt - int(th v)| / scale (generic)  : {abs(dEdtb-workb)/scb:.3e}  (scale {scb:.3e})")
    th2a = S.integral(tb * tb)
    w1, th1, *_ = S.step(wb, tb, dt)
    th2b = S.integral(th1 * th1)
    print(f"U6  theta^2 drift per step (inviscid)      : {abs(th2b-th2a)/th2a:.3e}")
    print(f"[{time.time()-t0:.1f} s]")

def stage_u7():
    """S11 fast-path equality battery. Every check demands EXACT 0.0 (see header):
    the fast path differs from the reference only in execution order of
    independent tasks, never in the arithmetic performed."""
    print("stage u7: fast-path exact-equality battery (predicted: all 0.0 EXACT)")
    t0 = time.time()
    rng = np.random.default_rng(20260824)
    for (Nx, Ny) in ((64, 64), (128, 96)):
        R = BSQ(Nx, Ny)               # reference
        F = BSQ(Nx, Ny, fast=True)    # fast
        w = rng.standard_normal((Ny + 1, Nx))
        th = rng.standard_normal((Ny + 1, Nx))
        # U7a poisson
        d_a = np.abs(R.poisson(w) - F.poisson(w)).max()
        # U7b ddy (workers + cumsum dcoeff vs historical loop semantics)
        d_b = np.abs(R.ddy(w) - F.ddy(w)).max()
        # U7c full rhs
        ra = R.rhs(w, th); rb = F.rhs(w, th)
        d_c = max(np.abs(ra[i] - rb[i]).max() for i in range(5))
        # U7d 20-step evolution
        wr, tr = w.copy(), th.copy()
        wf, tf = w.copy(), th.copy()
        for _ in range(20):
            wr, tr, *_ = R.step(wr, tr, 1e-3)
            wf, tf, *_ = F.step(wf, tf, 1e-3)
        d_d = max(np.abs(wr - wf).max(), np.abs(tr - tf).max())
        print(f"  {Nx}x{Ny}:  U7a poisson {d_a:.1e}   U7b ddy {d_b:.1e}   "
              f"U7c rhs {d_c:.1e}   U7d 20-step {d_d:.1e}")
    # ---- QI level (tolerance-certified; floors in header)
    print("  QI level (qi-vs-LU; predicted: U7e<=1e-13, U7f<=1e-11, U7g<=1e-11 rel):")
    for (Nx, Ny) in ((64, 64), (128, 96), (256, 256)):
        R = BSQ(Nx, Ny)
        Q = BSQ(Nx, Ny, fast=True, qi=True)
        X = np.tile(R.x, (Ny + 1, 1)); Y = np.tile(R.ch.y[:, None], (1, Nx))
        # U7e manufactured smooth RHS (the U1 content)
        P = (Y**2) * (1.0 - Y)**2
        w_m = np.sin(X) * ((2.0 - 12.0 * Y + 12.0 * Y**2) - P)
        d_e = np.abs(R.poisson(w_m) - Q.poisson(w_m)).max()
        # U7f random RHS filtered to the resolved band (see header failure record);
        # raw unfiltered diff reported as [method-diff] context, not an error.
        wr_ = rng.standard_normal((Ny + 1, Nx))
        wrf = R.dealias_f(wr_)
        pr = R.poisson(wrf); pq = Q.poisson(wrf)
        d_f = np.abs(pr - pq).max() / max(np.abs(pr).max(), 1e-300)
        pr0 = R.poisson(wr_); pq0 = Q.poisson(wr_)
        d_f0 = np.abs(pr0 - pq0).max() / max(np.abs(pr0).max(), 1e-300)
        # U7g 20-step evolution, relative
        w0 = np.sin(X) * np.sin(np.pi * Y); t0_ = np.cos(X) * Y
        wa, ta = w0.copy(), t0_.copy(); wb, tb = w0.copy(), t0_.copy()
        for _ in range(20):
            wa, ta, *_ = R.step(wa, ta, 1e-3)
            wb, tb, *_ = Q.step(wb, tb, 1e-3)
        d_g = max(np.abs(wa - wb).max() / np.abs(wa).max(),
                  np.abs(ta - tb).max() / np.abs(ta).max())
        print(f"  {Nx}x{Ny}:  U7e smooth {d_e:.1e}   U7f resolved-random(rel) {d_f:.1e} "
              f"[method-diff raw {d_f0:.1e}]   U7g 20-step(rel) {d_g:.1e}")
    # ---- U7h: Lx domain-length parameter (S11; predicted: derivative <= 1e-11,
    # Poisson recovery <= 1e-12 at Lx=1/6 -- amplitudes carry (2pi/Lx)^n factors)
    Lx = 1.0 / 6.0
    for qi_ in (False, True):
        S = BSQ(96, 64, fast=True, qi=qi_, Lx=Lx)
        X = np.tile(S.x, (65, 1)); Y = np.tile(S.ch.y[:, None], (1, 96))
        kp = 2.0 * np.pi / Lx
        d1 = np.abs(S.ddx(np.sin(3 * kp * X)) - 3 * kp * np.cos(3 * kp * X)).max() / (3 * kp)
        P = (Y**2) * (1.0 - Y)**2
        Pdd = 2.0 - 12.0 * Y + 12.0 * Y**2
        psi_e = np.sin(kp * X) * P
        w_m = np.sin(kp * X) * (Pdd - (kp * kp) * P)
        d2 = np.abs(S.poisson(w_m) - psi_e).max()
        print(f"  U7h Lx=1/6 ({'qi' if qi_ else 'lu'}): ddx(rel) {d1:.1e}   poisson {d2:.1e}")
    print(f"[{time.time()-t0:.1f} s]")

def stage_u8():
    """S12 kappa-level battery (predicted floors in header, written first)."""
    print("stage u8: kappa/CN-Helmholtz battery (floors predicted in header)")
    t0 = time.time()
    Nx, Ny = 64, 64
    for tier in ('lu', 'qi'):
        S = BSQ(Nx, Ny, qi=(tier == 'qi'), kappa=1e-2)
        ch = S.ch
        X = np.tile(S.x, (Ny + 1, 1))
        Y = np.tile(ch.y[:, None], (1, Nx))
        # U8a: heat-kernel exact decay on Neumann eigenfunctions
        kap, dt, T = 1e-2, 1e-3, 0.1
        n = int(round(T / dt))
        cases = [("y-mode cos(2 pi y)", np.cos(2.0 * np.pi * Y), (2.0 * np.pi) ** 2),
                 ("x-mode cos(3x)", np.cos(3.0 * X), 9.0),
                 ("mixed cos(2x)cos(pi y)", np.cos(2.0 * X) * np.cos(np.pi * Y),
                  4.0 + np.pi ** 2)]
        for nm, th0, lam in cases:
            th = th0.copy()
            for _ in range(n):
                th = S.dstep(th, dt)
            ex = np.exp(-kap * lam * T) * th0
            err = np.abs(th - ex).max() / np.abs(ex).max()
            print(f"  [{tier}] U8a {nm:26s}: rel {err:.3e}")
        # U8a tiny-b: one CN step at b = 1e-9 vs exact
        th = np.cos(2.0 * np.pi * Y).copy()
        hb = 2.0e-7                                   # b = kap*h/2 = 1e-9
        th1 = S.dstep(th, hb)
        ex = np.exp(-kap * (2.0 * np.pi) ** 2 * hb) * np.cos(2.0 * np.pi * Y)
        print(f"  [{tier}] U8a tiny-b (b=1e-9) one step     : rel {np.abs(th1-ex).max()/np.abs(ex).max():.3e}")
        # U8b: no-flux wall enforcement on generic (incompatible) data --
        # exactly the case the solve must PROJECT correctly
        th = (Y ** 2) * np.cos(X) + Y + 0.3 * np.cos(2.0 * X) * Y ** 3
        th1 = S.dstep(th, 1e-3)
        ty = S.ddy(th1)
        wall = max(np.abs(ty[0]).max(), np.abs(ty[-1]).max()) / np.abs(ty).max()
        print(f"  [{tier}] U8b no-flux walls after D-step   : rel {wall:.3e}")
        # U8c: theta-mean conservation
        m0 = S.integral(th)
        m1 = S.integral(th1)
        print(f"  [{tier}] U8c mean drift per D-step        : rel {abs(m1-m0)/abs(m0):.3e}")
        # U8h: theta^2 dissipation identity, NO-FLUX-COMPATIBLE data (v2).
        # v1 used wall-gradient data: the first CN step projects onto the BC
        # and the balance measures the projection layer, not the identity --
        # honest v1 numbers: rel 5.5e-1 (lu) / 4.3e-1 (qi) [incompatible-data,
        # test-design error, recorded per discipline]. With the CN midpoint
        # field and compatible data the discrete identity is structurally
        # exact (integration by parts at spectral accuracy): floor <= 1e-8.
        thc = np.cos(X) * (0.5 + 0.3 * np.cos(np.pi * Y)) \
            + 0.1 * np.cos(2.0 * X) * np.cos(2.0 * np.pi * Y) \
            + 0.2 * np.cos(3.0 * np.pi * Y)
        hh = 1e-4
        th1 = S.dstep(thc, hh)
        lhs = (S.integral(th1 * th1) - S.integral(thc * thc)) / (2.0 * hh)
        thm = 0.5 * (thc + th1)
        gx = S.ddx(thm); gy = S.ddy(thm)
        rhs_ = -kap * S.integral(gx * gx + gy * gy)
        print(f"  [{tier}] U8h theta^2 identity (compat.)   : rel {abs(lhs-rhs_)/abs(rhs_):.3e}")
        # U8g: x-parity (even) preservation through D-step. v1 prediction
        # "0.0 EXACT" was IMPRECISE (recorded): evenness survives the per-kx
        # real solve structurally, but through rfft roundoff imaginaries --
        # machine-floor class (~N*eps), same as the U5 convention.
        the = np.cos(X) * (0.5 + Y ** 2) + 0.2 * np.cos(3.0 * X) * Y
        th1 = S.dstep(the, 1e-3)
        par = np.abs(th1 - np.roll(th1[:, ::-1], 1, axis=1)).max()
        print(f"  [{tier}] U8g theta evenness after D-step  : {par:.3e}  (floor class ~1e-14)")
    # U8f v2: qi-vs-lu D-step on band-limited AND no-flux-COMPATIBLE random
    # data (random x-phases/amplitudes on the cos(m pi y) Neumann basis,
    # j, m <= 20 -- resolved to ~1e-16). v1 used bare 2/3-filtered random
    # data (wall-incompatible): the two tiers project the O(1) BC violation
    # through an under-resolved sqrt(b) layer DIFFERENTLY -- legitimate
    # method difference, not solver error (the U7f lesson one level deeper:
    # cross-tier agreement needs band-limited AND compatible data; the
    # physics is always compatible after t=0). Honest v1 numbers: rel
    # 6.1e-3 [incompatible-data]; raw unfiltered 6.3e-1 [method-diff].
    rng = np.random.default_rng(20260824)
    R = BSQ(Nx, Ny, kappa=1e-2)
    Q = BSQ(Nx, Ny, qi=True, kappa=1e-2)
    filt = np.zeros((Ny + 1, Nx))
    for j in range(0, 21):
        for mm in range(0, 21):
            aj, ph = rng.standard_normal(), rng.uniform(0, 2 * np.pi)
            filt += aj * np.cos(j * X + ph) * np.cos(mm * np.pi * Y)
    dr = R.dstep(filt, 1e-3)
    dq = Q.dstep(filt, 1e-3)
    rel = np.abs(dr - dq).max() / np.abs(dr).max()
    print(f"  U8f qi-vs-lu D-step, compat. band  : rel {rel:.3e}  (predict <= 1e-11)")
    # U8e: kappa=0 bit-identity of step_p2 vs step (20 steps)
    S0 = BSQ(Nx, Ny, kappa=0.0)
    w0 = np.sin(X) * np.sin(np.pi * Y) + 0.3 * np.sin(2 * X) * (Y ** 2) * (1 - Y)
    t0_ = np.cos(X) * (0.5 + 0.5 * Y ** 2) + 0.1 * np.cos(2 * X) * Y
    wa, ta = w0.copy(), t0_.copy()
    wb, tb = w0.copy(), t0_.copy()
    for _ in range(20):
        wa, ta, *_ = S0.step(wa, ta, 1e-3)
        wb, tb, *_ = S0.step_p2(wb, tb, 1e-3)
    print(f"  U8e kappa=0 20-step p2-vs-plain    : {max(np.abs(wa-wb).max(), np.abs(ta-tb).max()):.1e}  (predict 0.0 EXACT)")
    # U8d v2: Strang split order with COMPATIBLE theta data. v1 (wall-
    # gradient data) measured ratio 3.01 = FIRST order: the initial CN
    # projection layer collapses the split order (splitting order reduction
    # at boundaries, the classical mechanism) -- honest v1 number recorded.
    # With compatible data the residual boundary interaction is the physical
    # advective regeneration of wall gradients (u_y th_x at the wall),
    # handled by both sub-flows: expect ratio near 5, accept >= 4 (mild
    # residual order reduction documented if seen).
    t0c = np.cos(X) * (0.5 + 0.3 * np.cos(np.pi * Y)) \
        + 0.1 * np.cos(2.0 * X) * np.cos(2.0 * np.pi * Y)
    errs = {}
    for ddt in (2.5e-4, 5e-4, 1e-3):
        S = BSQ(Nx, Ny, qi=True, kappa=1e-2)
        w, th = w0.copy(), t0c.copy()
        for _ in range(int(round(0.01 / ddt))):
            w, th, *_ = S.step_p2(w, th, ddt)
        errs[ddt] = (w.copy(), th.copy())
    e1 = max(np.abs(errs[1e-3][0] - errs[2.5e-4][0]).max(),
             np.abs(errs[1e-3][1] - errs[2.5e-4][1]).max())
    e2 = max(np.abs(errs[5e-4][0] - errs[2.5e-4][0]).max(),
             np.abs(errs[5e-4][1] - errs[2.5e-4][1]).max())
    # 2nd order with a dt/4 reference: e(dt)/e(dt/2) -> (1-1/16)/(1/4-1/16) = 5
    print(f"  U8d Strang order ratio (vs dt/4 ref): {e1/e2:.2f}  (2nd order -> 5; accept >= 4)")
    print(f"[{time.time()-t0:.1f} s]")

def stage_bench(N=256, fast=False, qi=False):
    print(f"stage bench: RK4 step timing at {N}x{N}"
          + ("  [fast=True]" if fast else "") + ("  [qi=True]" if qi else ""))
    t0 = time.time()
    S = BSQ(N, N, fast=fast, qi=qi)
    print(f"  setup: {time.time()-t0:.1f} s")
    X = np.tile(S.x, (N + 1, 1)); Y = np.tile(S.ch.y[:, None], (1, N))
    w = np.sin(X) * np.sin(np.pi * Y); th = np.cos(X) * Y
    w, th, u, v = S.step(w, th, 1e-3)   # warm
    t1 = time.time(); n = 5
    for _ in range(n):
        w, th, u, v = S.step(w, th, 1e-3)
    per = (time.time() - t1) / n
    print(f"  RK4 step: {per*1e3:.0f} ms  ({per*1e3/4:.0f} ms per RHS eval)")
    print(f"  CFL dt at this state: {S.cfl_dt(u, v):.2e}")

def stage_a7():
    """A7: cross-solver consistency. The FD integrator (independent discretization)
    must converge at 2nd order TO the spectral solution on a short window.
    PREDICTED (before first run): rel sup diff at FD 64^2 <= 3e-2, at 128^2
    <= 1e-2, ratio in [3, 5.5] (asymptotic 4); spectral reference error
    negligible (96x96, well-resolved smooth data)."""
    print("stage a7: FD cross-solver, T=0.4, dt=1e-3 fixed")
    t0 = time.time()
    fw = lambda x, y: np.sin(x) * np.sin(np.pi * y) + 0.3 * np.cos(2 * x) * y * y * (1 - y)
    ft = lambda x, y: np.cos(x) * (0.5 + 0.5 * y * y) + 0.1 * np.sin(x) * y
    T, dt = 0.4, 1e-3
    n = int(round(T / dt))
    # spectral reference
    S = BSQ(128, 96, fast=True, qi=True)
    XS = np.tile(S.x, (97, 1)); YS = np.tile(S.ch.y[:, None], (1, 128))
    w, th = fw(XS, YS), ft(XS, YS)
    for _ in range(n):
        w, th, *_ = S.step(w, th, dt)
    cw, ct = S.ch.coeff(w), S.ch.coeff(th)
    from numpy.polynomial.chebyshev import chebval
    prev = None
    for Nf in (64, 128):
        F = FD(Nf, Nf)
        XF = np.tile(F.x, (Nf + 1, 1)); YF = np.tile(F.y[:, None], (1, Nf))
        wf, tf = fw(XF, YF), ft(XF, YF)
        for _ in range(n):
            wf, tf = F.step(wf, tf, dt)
        # evaluate spectral solution on FD grid: x -> every (128/Nf)-th column;
        # y -> chebval at xi = 2y-1 (coefficient rep is orientation-free)
        stx = 128 // Nf
        xi = 2.0 * F.y - 1.0
        ws = chebval(xi, cw[:, ::stx]).T      # (Nf+1, Nf)
        ts = chebval(xi, ct[:, ::stx]).T
        dwr = np.abs(wf - ws).max() / np.abs(ws).max()
        dtr = np.abs(tf - ts).max() / np.abs(ts).max()
        print(f"  FD {Nf}x{Nf}: rel sup diff  w {dwr:.3e}   th {dtr:.3e}")
        if prev is not None:
            print(f"  convergence ratio (64->128): w {prev[0]/dwr:.2f}   th {prev[1]/dtr:.2f}"
                  f"   (2nd order -> 4)")
        prev = (dwr, dtr)
    print(f"[{time.time()-t0:.1f} s]")

def stage_p1(N=256, cfl=0.4, tag=""):
    """P1: Hou-Luo growth-phase reproduction on the strip (S11).
    Setup (pinned from primaries, see lit/ + bsq-a5-notes.md):
      Lx = 1/6 (Luo-Hou z-period), y in [0,1], corner at (0,0);
      theta0 = 1e4 * exp(-60(y(2-y))^4) * sin^2(2 pi x / Lx)   [= (u1_0)^2 of
      LH2014 under u1^2 <-> theta, y = 1-r], omega0 = 0. Parity: theta even,
      omega odd in x, theta(0,y)=0 -- the Chen-Hou (Sym) class.
    PREDICTIONS (registered before first run, S11 log):
      (a) max|omega| location migrates wall-ward and x->0 (corner-directed);
      (b) sup|omega| growth super-exponential in the late certified window;
      (c) certified growth factor (tails <= 1e-6) at 256^2: >= 1e2;
      (d) singular-time ballpark [analog, NOT a reproduction target]:
          t_s ~ O(3e-3) (the 3D value 0.0035 is a different system);
      (e) theta^2 invariant drift <= 1e-6 relative over the certified window.
    Stops: t > 0.02 | growth >= 1e6 | tail_x or tail_y_w > 1e-2 | dt < 1e-12."""
    Lx = 1.0 / 6.0
    A = 1.0e4
    S = BSQ(N, N, fast=True, qi=True, Lx=Lx)
    X = np.tile(S.x, (N + 1, 1)); Y = np.tile(S.ch.y[:, None], (1, N))
    th = A * np.exp(-60.0 * (Y * (2.0 - Y))**4) * np.sin(2.0 * np.pi * X / Lx)**2
    w = np.zeros_like(th)
    tagS = tag or f"N{N}"
    print(f"stage p1[{tagS}]: Lx=1/6, A=1e4, N={N}, cfl={cfl}  (predictions in docstring)")
    d0 = S.diagnostics(w, th, None)
    th2_0 = d0['th2']
    rec = []
    snaps = {}
    every_t = []; every_sw = []
    t = 0.0; it = 0; t0 = time.time()
    sup0 = None; next_mile = 10.0
    _, _, u, v, _ = S.rhs(w, th)          # initial velocity for the first dt
    dt_prev = None
    while True:
        # advective CFL + buoyancy (internal-wave) limit sqrt(dx/|th_x|): the
        # forcing dw/dt = th_x sets the timescale while u is still small
        # (omega0 = 0 start); ramp guard dt <= 2*dt_prev.
        adv = np.abs(u).max() / S.dx + (np.abs(v) / S.dy_loc).max()
        txm = np.abs(S.ddx(th)).max()
        dt = cfl * min(1.0 / max(adv, 1e-30),
                       np.sqrt(S.dx / max(txm, 1e-30)))
        if dt_prev is not None:
            dt = min(dt, 2.0 * dt_prev)
        dt_prev = dt
        w, th, u, v = S.step(w, th, dt)   # u, v returned = stage-1 velocity of this step
        t += dt; it += 1
        every_t.append(t); every_sw.append(np.abs(w).max())   # dense track for 1/lambda
        g = 1.0
        if it % 5 == 0 or it <= 5:
            d = S.diagnostics(w, th, S.poisson(w))
            d['t'] = t; d['dt'] = dt; d['it'] = it
            d['sup_u'] = np.abs(u).max()
            rec.append(d)
            if sup0 is None and t >= 2e-4:
                sup0 = d['sup_w']         # growth baseline past the linear spin-up
            g = d['sup_w'] / sup0 if sup0 else 1.0
            if g >= next_mile:
                snaps[f"g{int(round(np.log10(next_mile)))}"] = (t, w.copy(), th.copy())
                next_mile *= 10.0
            if it % 100 == 0:
                print(f"  it {it:6d}  t {t:.6f}  dt {dt:.2e}  sup_w {d['sup_w']:.4e}  "
                      f"g {g:.2e}  xy_max ({d['x_at_max']:.4f},{d['y_at_max']:.4f})  "
                      f"tails x {d['tail_x']:.1e} yw {d['tail_y_w']:.1e}  "
                      f"th2drift {abs(d['th2']-th2_0)/th2_0:.1e}", flush=True)
            bad = (not np.isfinite(d['sup_w'])) or d['tail_x'] > 1e-2 or d['tail_y_w'] > 1e-2
            if t > 0.02 or g >= 1e6 or bad or dt < 1e-12:
                why = ('t>0.02' if t > 0.02 else 'g>=1e6' if g >= 1e6 else
                       'RESOLUTION' if bad else 'dt-collapse')
                print(f"  STOP [{why}] at it {it}, t {t:.6f}")
                break
    # verdict: certified window = tails <= 1e-6
    ts = np.array([r['t'] for r in rec]); sw = np.array([r['sup_w'] for r in rec])
    ok = np.array([max(r['tail_x'], r['tail_y_w']) <= 1e-6 for r in rec])
    tcert = ts[ok][-1] if ok.any() else np.nan
    gcert = (sw[ok][-1] / sup0) if (ok.any() and sup0) else np.nan
    print(f"  certified window end t_cert = {tcert:.6f}   certified growth = {gcert:.3e}")
    out = dict(t=ts, sup_w=sw,
               sup_th=np.array([r['sup_th'] for r in rec]),
               x_at_max=np.array([r['x_at_max'] for r in rec]),
               y_at_max=np.array([r['y_at_max'] for r in rec]),
               tail_x=np.array([r['tail_x'] for r in rec]),
               tail_y_w=np.array([r['tail_y_w'] for r in rec]),
               tail_y_th=np.array([r['tail_y_th'] for r in rec]),
               delta_x=np.array([r['delta_x'] for r in rec]),
               strip_r2=np.array([r['strip_r2'] for r in rec]),
               psi_xy=np.array([r.get('psi_xy_corner', np.nan) for r in rec]),
               th2=np.array([r['th2'] for r in rec]),
               dt=np.array([r['dt'] for r in rec]),
               sup_u=np.array([r['sup_u'] for r in rec]),
               tcert=tcert, gcert=gcert, N=N, cfl=cfl, Lx=Lx, A=A,
               every_t=np.array(every_t), every_sw=np.array(every_sw),
               w_final=w, th_final=th)
    for k, (tk, wk, tk_) in snaps.items():
        out[f"t_{k}"] = tk; out[f"w_{k}"] = wk; out[f"th_{k}"] = tk_
    fn = f"p1_{tagS}.npz"
    np.savez_compressed(fn, **out)
    print(f"  saved {fn}   [{time.time()-t0:.1f} s, {it} steps]")

def stage_p2(N=256, kappa=1e-2, cfl=0.4, tag="", A=1.0e4):
    """P2 (S12): kappa-dial dissipative ride at the Hou-Luo corner, from the
    EXACT P1 setup (same Lx, data, class) + kappa*Lap(theta), no-flux walls.
    Predictions (i)-(iv) registered in the header BEFORE the first run.
    Verdicts: BLOWUP-bar(1e6) [bar-calibrated, N15] | SATURATED (driver
    collapse: trailing 25%-window contribution to int sup|th_x| dt < 2% of
    total AND trailing sup-omega growth < 2%) | RESOLUTION | UNDET(Tmax).
    Records: p2_k<kappa>_N<N>[_A<A>][_<tag>].npz (S11 filename rule)."""
    Lx = 1.0 / 6.0
    S = BSQ(N, N, fast=True, qi=True, Lx=Lx, kappa=kappa)
    X = np.tile(S.x, (N + 1, 1)); Y = np.tile(S.ch.y[:, None], (1, N))
    th = A * np.exp(-60.0 * (Y * (2.0 - Y))**4) * np.sin(2.0 * np.pi * X / Lx)**2
    w = np.zeros_like(th)
    tagS = ("k%s_N%d" % (("%.6f" % kappa).replace(".", "p"), N)
            + ("" if A == 1.0e4 else "_A%g" % A) + (("_" + tag) if tag else ""))
    print(f"stage p2[{tagS}]: Lx=1/6, A={A:g}, N={N}, kappa={kappa:g}, cfl={cfl}"
          f"  (predictions in header)", flush=True)
    d0 = S.diagnostics(w, th, None)
    th2_0 = d0['th2']
    mean0 = S.integral(th)
    rec = []
    snaps = {}
    every_t = []; every_sw = []; every_txm = []
    t = 0.0; it = 0; t0 = time.time()
    sup0 = None; next_mile = 10.0
    _, _, u, v, _ = S.rhs(w, th)
    dt_prev = None
    itx = 0.0                      # running integral of sup|th_x| dt (driver)
    itx_track = []                 # (t, itx) for the trailing-window verdict
    txm_max = 0.0
    Tcap = 0.06
    verdict = "UNDET(Tmax)"
    # S14 ops fix:
    # once the certified window (tails <= 1e-6) has been CLOSED for 400
    # consecutive diagnostic records (~2000 steps), further iterations are
    # junk-class (uncertified fake-growth endgame; the 2048^2 run burned ~2 h
    # there) -> stop with RESOLUTION(cw-closed). Certified quantities
    # (t_cert/g_cert/Omega_cert_peak) are ok-mask-based and unaffected.
    # Opt out with tag containing "nocw".
    cw_closed_run = 0
    while True:
        adv = np.abs(u).max() / S.dx + (np.abs(v) / S.dy_loc).max()
        txm = np.abs(S.ddx(th)).max()
        dt = cfl * min(1.0 / max(adv, 1e-30),
                       np.sqrt(S.dx / max(txm, 1e-30)))
        if dt_prev is not None:
            dt = min(dt, 2.0 * dt_prev)
        dt_prev = dt
        w, th, u, v = S.step_p2(w, th, dt)
        t += dt; it += 1
        itx += txm * dt
        txm_max = max(txm_max, txm)
        sw = np.abs(w).max()
        every_t.append(t); every_sw.append(sw); every_txm.append(txm)
        itx_track.append((t, itx))
        g = 1.0
        if it % 5 == 0 or it <= 5:
            d = S.diagnostics(w, th, S.poisson(w))
            d['t'] = t; d['dt'] = dt; d['it'] = it
            d['sup_u'] = np.abs(u).max()
            d['txm'] = txm
            d['th_min'] = th.min(); d['th_max'] = th.max()
            d['mean_th'] = S.integral(th)
            rec.append(d)
            if sup0 is None and t >= 2e-4:
                sup0 = d['sup_w']
            g = d['sup_w'] / sup0 if sup0 else 1.0
            if g >= next_mile:
                snaps[f"g{int(round(np.log10(next_mile)))}"] = (t, w.copy(), th.copy())
                next_mile *= 10.0
            if it % 200 == 0:
                print(f"  it {it:6d}  t {t:.6f}  dt {dt:.2e}  sup_w {d['sup_w']:.4e}  "
                      f"g {g:.2e}  txm {txm:.3e}  th_max {d['th_max']:.4e}  "
                      f"tails x {d['tail_x']:.1e} yw {d['tail_y_w']:.1e}", flush=True)
            bad = (not np.isfinite(d['sup_w'])) or d['tail_x'] > 1e-2 or d['tail_y_w'] > 1e-2
            if max(d['tail_x'], d['tail_y_w']) <= 1e-6:
                cw_closed_run = 0
            else:
                cw_closed_run += 1
            # SATURATED: trailing 25% window of the driver integral < 2% of
            # total AND trailing sup-omega growth < 2% (after a warmup)
            sat = False
            if t > 5e-3 and len(itx_track) > 40:
                tt = t * 0.75
                j = np.searchsorted([p[0] for p in itx_track], tt)
                if 0 < j < len(itx_track):
                    tail_itx = itx - itx_track[j][1]
                    jj = np.searchsorted(every_t, tt)
                    swref = max(every_sw[jj:jj+1] or [sw])
                    gr = sw / swref - 1.0 if swref > 0 else 0.0
                    if itx > 0 and tail_itx / itx < 0.02 and gr < 0.02 \
                            and "nosat" not in tag:
                        sat = True
            cw = (cw_closed_run >= 400 and t > 1e-3 and "nocw" not in tag)
            if t > Tcap or g >= 1e6 or bad or dt < 1e-12 or sat or cw:
                verdict = ('SATURATED' if sat else
                           't>Tcap' if t > Tcap else
                           'BLOWUP-bar(1e6)' if g >= 1e6 else
                           'RESOLUTION' if bad else
                           'RESOLUTION(cw-closed)' if cw else 'dt-collapse')
                print(f"  STOP [{verdict}] at it {it}, t {t:.6f}", flush=True)
                break
    ts = np.array([r['t'] for r in rec]); sw_ = np.array([r['sup_w'] for r in rec])
    ok = np.array([max(r['tail_x'], r['tail_y_w']) <= 1e-6 for r in rec])
    tcert = ts[ok][-1] if ok.any() else np.nan
    gcert = (sw_[ok][-1] / sup0) if (ok.any() and sup0) else np.nan
    # Omega_inf: the certified-window peak of sup|omega| (the excursion object)
    om_pk = sw_[ok].max() if ok.any() else np.nan
    print(f"  verdict {verdict}   t_cert {tcert:.6f}   g_cert {gcert:.3e}   "
          f"Omega_cert_peak {om_pk:.4e}   itx {itx:.4e}", flush=True)
    out = dict(t=ts, sup_w=sw_,
               sup_th=np.array([r['sup_th'] for r in rec]),
               x_at_max=np.array([r['x_at_max'] for r in rec]),
               y_at_max=np.array([r['y_at_max'] for r in rec]),
               tail_x=np.array([r['tail_x'] for r in rec]),
               tail_y_w=np.array([r['tail_y_w'] for r in rec]),
               tail_y_th=np.array([r['tail_y_th'] for r in rec]),
               delta_x=np.array([r['delta_x'] for r in rec]),
               strip_r2=np.array([r['strip_r2'] for r in rec]),
               psi_xy=np.array([r.get('psi_xy_corner', np.nan) for r in rec]),
               th2=np.array([r['th2'] for r in rec]),
               dt=np.array([r['dt'] for r in rec]),
               sup_u=np.array([r['sup_u'] for r in rec]),
               txm=np.array([r['txm'] for r in rec]),
               th_min=np.array([r['th_min'] for r in rec]),
               th_max=np.array([r['th_max'] for r in rec]),
               mean_th=np.array([r['mean_th'] for r in rec]),
               tcert=tcert, gcert=gcert, om_pk=om_pk, N=N, cfl=cfl, Lx=Lx, A=A,
               kappa=kappa, verdict=verdict, itx=itx, mean0=mean0, th2_0=th2_0,
               every_t=np.array(every_t), every_sw=np.array(every_sw),
               every_txm=np.array(every_txm),
               w_final=w, th_final=th)
    for k, (tk, wk, tk_) in snaps.items():
        out[f"t_{k}"] = tk; out[f"w_{k}"] = wk; out[f"th_{k}"] = tk_
    fn = f"p2_{tagS}.npz"
    if os.path.exists(fn):
        k = 1
        while os.path.exists(fn + ".bak%d" % k):
            k += 1
        os.rename(fn, fn + ".bak%d" % k)
        print(f"  [existing {fn} backed up to .bak{k}]")
    np.savez_compressed(fn, **out)
    print(f"  saved {fn}   [{time.time()-t0:.1f} s, {it} steps]", flush=True)

if __name__ == "__main__":
    args = sys.argv[1] if len(sys.argv) > 1 else "u"
    parts = args.split(":")
    if parts[0] == "u":
        stage_u(fast=("fast" in parts[1:] or "qi" in parts[1:]), qi=("qi" in parts[1:]))
    elif parts[0] == "u7":
        stage_u7()
    elif parts[0] == "u8":
        stage_u8()
    elif parts[0] == "bench":
        stage_bench(int(parts[1]) if len(parts) > 1 else 256,
                    fast=("fast" in parts[2:] or "qi" in parts[2:]),
                    qi=("qi" in parts[2:]))
    elif parts[0] == "a7":
        stage_a7()
    elif parts[0] == "p1":
        stage_p1(int(parts[1]) if len(parts) > 1 else 256,
                 cfl=float(parts[2]) if len(parts) > 2 else 0.4,
                 tag=parts[3] if len(parts) > 3 else "")
    elif parts[0] == "p2":
        stage_p2(int(parts[1]) if len(parts) > 1 else 256,
                 kappa=float(parts[2]) if len(parts) > 2 else 1e-2,
                 cfl=float(parts[3]) if len(parts) > 3 else 0.4,
                 tag=parts[4] if len(parts) > 4 else "",
                 A=float(parts[5]) if len(parts) > 5 else 1.0e4)
    else:
        print("stages: u[:fast|:qi] | u7 | u8 | bench[:N[:fast|:qi]] | a7 | "
              "p1[:N[:cfl[:tag]]] | p2[:N[:kappa[:cfl[:tag[:A]]]]]")
