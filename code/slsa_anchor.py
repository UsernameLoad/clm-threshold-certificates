#!/usr/bin/env python3
# slsa_anchor.py — transcription + verification of the SLSA a=1/2
# exact pole solutions (Silantyev–Lushnikov–Siegel–Ambrose, arXiv:2411.01891,
# Section 3.1, sigma=0), and the v0 -> 1- boundary-limit analysis that appears
# to explain our conjecture T*(a=1/2, cos x) = pi  (ledger N1).
#
# TRANSCRIBED OBJECTS (equation numbers = SLSA's; transcription verified by the
# stages below against their own norm formulas and against our PDE solver):
#   Real reduction (56): omega_-2 = i*w2 (w2 real), v_c real in (0,1)u(1,inf).
#   Field (41)+(42), real form (X = tan(x/2)):
#     omega_-(x) = -(2 v w2/(1-v^2)) * [ 1/(X-iv) - i/(1+v) ]
#                  + i*w2 * [ 1/(X-iv)^2 + 1/(1+v)^2 ]
#     omega(x)   = 2 Re[omega_-(x)]        (omega_av = 0 in this reduction)
#   ODEs (57), nu=0:
#     dw2/dt = w2^2 (1-2v^2+5v^4) / (4 v^2 (1-v^2)^2)
#     dv/dt  = -w2 (1+v^2) / (4 v (1-v^2))
#   Implicit solution (60), nu=0, with F(v) = v(v^2-1)/(v^2+1)^2 + arctan(v):
#     F(v(t)) = F(v0) + R0*t,   R0 = 2 w2(0) v0 / ((v0^2+1)^2 (v0^2-1))
#   Collapse: (A) v->0+ (F->0) or (B) v->inf (F->pi/2); |w2|->inf either way.
#   Case-A collapse time (w2(0)>0, 0<v0<1):
#     t_c(v0, w2(0)) = F(v0) (v0^2+1)^2 (1-v0^2) / (2 w2(0) v0)
#   L2 norm (44): ||omega||_L2^2 = 2 pi w2^2 (1+v^2) / (v^3 (1-v^2)^2)
#
# OUR limit observation (ledger N10, this session): matching cos x's L2 norm
# (int cos^2 = pi) forces w2(0) = v0^(3/2)(1-v0^2)/sqrt(2(1+v0^2)) -> (1-v0^2)/2
# as v0->1-, giving R0 -> -1/4, F(v0) -> pi/4, hence t_c -> pi.
# Stages: field | ode | anchor | limit
import os, sys, time
import numpy as np
import clm_viscous as cv
import gclm_scan as gs

DIR = os.path.dirname(os.path.abspath(__file__))

def F_impl(v):
    return v*(v*v - 1)/(v*v + 1)**2 + np.arctan(v)

def slsa_field(x, v, w2):
    X = np.tan(x/2.0)
    om1 = -(2.0*v*w2/(1.0 - v*v))
    wm = om1*(1.0/(X - 1j*v) - 1j/(1.0 + v)) \
         + 1j*w2*(1.0/(X - 1j*v)**2 + 1.0/(1.0 + v)**2)
    return 2.0*np.real(wm)

def slsa_L2sq(v, w2):
    return 2.0*np.pi*w2*w2*(1.0 + v*v)/(v**3*(1.0 - v*v)**2)

def rate0(v0, w20):
    return 2.0*w20*v0/((v0*v0 + 1.0)**2*(v0*v0 - 1.0))

def tc_caseA(v0, w20):
    return F_impl(v0)*(v0*v0 + 1.0)**2*(1.0 - v0*v0)/(2.0*w20*v0)

def w2_L2match(v0):
    """w2(0) > 0 with ||omega||_L2^2 = pi (the L2 norm of cos x)."""
    return np.sqrt(v0**3*(1.0 - v0*v0)**2/(2.0*(1.0 + v0*v0)))

def ode_rhs(t, y):
    w2, v = y
    dw2 = w2*w2*(1.0 - 2.0*v*v + 5.0*v**4)/(4.0*v*v*(1.0 - v*v)**2)
    dv = -w2*(1.0 + v*v)/(4.0*v*(1.0 - v*v))
    return [dw2, dv]

# ---------------- stages ----------------
def stage_field():
    """Transcription check 1: the constructed field must be mean-zero and
    reproduce SLSA's own L2 formula (44); independent trapezoid integration."""
    print("stage field: transcription checks of (41)+(42) real reduction")
    N = 4096
    x = 2*np.pi*np.arange(N)/N
    for v0, w20 in ((0.5, 0.1), (0.8, 0.05), (0.95, w2_L2match(0.95))):
        w = slsa_field(x, v0, w20)
        mean = np.abs(np.mean(w))
        L2sq = np.sum(w*w)*(2*np.pi/N)
        pred = slsa_L2sq(v0, w20)
        print("  v0=%.2f w2=%.4f : |mean|=%.2e   L2^2 num %.10f vs formula %.10f  (rel %.1e)"
              % (v0, w20, mean, L2sq, pred, abs(L2sq - pred)/pred))

def stage_ode():
    """Transcription check 2: ODEs (57) vs implicit solution (60), nu=0."""
    from scipy.integrate import solve_ivp
    print("stage ode: ODE (57) vs implicit solution (60)")
    for v0, w20 in ((0.6, 0.08), (0.9, w2_L2match(0.9))):
        tc = tc_caseA(v0, w20)
        te = np.linspace(0, 0.95*tc, 8)
        sol = solve_ivp(ode_rhs, (0, te[-1]), [w20, v0], t_eval=te,
                        method="DOP853", rtol=1e-12, atol=1e-14)
        R0 = rate0(v0, w20)
        err = np.max(np.abs(F_impl(sol.y[1]) - (F_impl(v0) + R0*sol.t)))
        print("  v0=%.2f w2=%.4f : tc(formula)=%.6f  max|F(v(t)) - F(v0) - R0 t| = %.2e"
              % (v0, w20, tc, err))
        # collapse-time cross-check: integrate close to tc, confirm v -> 0
        sol2 = solve_ivp(ode_rhs, (0, 0.999*tc), [w20, v0],
                         method="DOP853", rtol=1e-12, atol=1e-14)
        print("      at t=0.999 tc: v=%.4f  w2=%.4f  (case-A: v->0+, w2->inf)"
              % (sol2.y[1, -1], sol2.y[0, -1]))

def stage_anchor():
    """THE EXTERNAL ANCHOR: evolve SLSA exact pole data with OUR pseudospectral
    inviscid gCLM solver (gs.rhs_g, a=1/2) and compare the field at checkpoints
    against the pole-dynamics reconstruction; compare observed blowup vs tc."""
    from scipy.integrate import solve_ivp
    print("stage anchor: our PDE solver vs SLSA exact pole solution (a=1/2, nu=0)")
    t0 = time.time()
    v0 = 0.6
    w20 = 0.08
    tc = tc_caseA(v0, w20)
    print("  v0=%.2f w2=%.4f: tc(formula) = %.6f" % (v0, w20, tc))
    N, dt = 1024, 2.5e-5
    ops = cv.make_ops(N)
    x = ops[0]
    w = slsa_field(x, v0, w20)
    # checkpoints snapped to exact dt multiples so the ODE reference is
    # evaluated at the PDE's own times (else the comparison is dominated by
    # |omega_t| * (time-rounding) ~ 1e-6, as seen in the first run)
    cps = [dt*round(f*tc/dt) for f in (0.25, 0.5, 0.75, 0.9)]
    cpn = {int(round(t/dt)): t for t in cps}
    nmax = max(cpn) + 1
    sol = solve_ivp(ode_rhs, (0, 0.95*tc), [w20, v0], t_eval=cps,
                    method="DOP853", rtol=1e-12, atol=1e-14)
    errs = {}
    n = 0
    while n < nmax:
        w = gs.rk4_g(w, dt, 0.5, ops, gs.rhs_g)
        n += 1
        if n in cpn:
            i = cps.index(cpn[n])
            wex = slsa_field(x, sol.y[1, i], sol.y[0, i])
            errs[cpn[n]] = np.max(np.abs(w - wex))
    for t in cps:
        print("  t=%.4f (=%.2f tc): |PDE - pole solution|_inf = %.3e" % (t, t/tc, errs[t]))
    # ride to blowup from the SLSA data and estimate T, compare to tc
    w = slsa_field(x, v0, w20)
    track = []
    stop = "Tmax"
    for n in range(1, int(round(1.02*tc/dt)) + 1):
        w = gs.rk4_g(w, dt, 0.5, ops, gs.rhs_g)
        if n % 100 == 0:
            s = float(np.max(np.abs(w)))
            track.append((n*dt, s))
            if s > 1e6 or not np.isfinite(s):
                stop = "blowup"
                break
    tr = np.array(track)
    tr = tr[np.isfinite(tr[:, 1])]
    Tl, Tq = cv.fit_T(tr)
    print("  ride-to-blowup: stop=%s at t=%.4f | T_lin=%.6f T_quad=%.6f vs tc=%.6f (diff %.1e/%.1e)"
          % (stop, tr[-1, 0], Tl, Tq, tc, abs(Tl - tc), abs(Tq - tc)))
    print("[%.0f s]" % (time.time() - t0))

def stage_limit():
    """The v0->1- boundary limit (ledger N10): with L2-matched amplitude,
    (a) does the family data converge to a translate of cos x?
    (b) tc(v0) -> pi?"""
    print("stage limit: v0->1- boundary limit of the SLSA family")
    N = 4096
    x = 2*np.pi*np.arange(N)/N
    print("  v0        w2(0)       tc(v0)      tc-pi        min_shift ||field - cos(x-s)||_inf")
    for v0 in (0.9, 0.99, 0.999, 0.9999):
        w20 = w2_L2match(v0)
        tc = tc_caseA(v0, w20)
        w = slsa_field(x, v0, w20)
        # best translate match against cos: use FFT phase of mode 1
        wh = np.fft.fft(w)/N
        s = np.angle(wh[1])            # cos(x-s) has mode-1 coefficient e^{-is}/2... sign check numerically
        best = None
        for sh in (s, -s, np.pi - s, s - np.pi):
            d = np.max(np.abs(w - np.cos(x - sh)))
            best = d if best is None else min(best, d)
        print("  %.4f   %.6f   %.8f  %+.2e   %.4e" % (v0, w20, tc, tc - np.pi, best))
    print("  (formula limits: w2 -> (1-v0^2)/2, R0 -> -1/4, F -> pi/4, tc -> pi)")

def stage_sym():
    """Verification of the boundary-limit identity (hand derivation + numeric
    confirmation; the earlier sympy sp.limit route hangs on this expression).
    HAND DERIVATION (recorded here as the proof of the identity):
      With w2 = (1-v^2)/2 the log-cancellation constraint gives
        omega_-1 = -2 v w2/(1-v^2) = -v  exactly.
      At v = 1 the double-pole term carries the factor w2 = 0, so
        omega_-|_{v=1} = -[ 1/(X - i) - i/2 ],   X = tan(x/2).
      With X = -i(Z-1)/(Z+1), Z = e^{ix}:  X - i = -2iZ/(Z+1), hence
        1/(X - i) = (i/2)(1 + e^{-ix})   =>   omega_-|_{v=1} = -(i/2) e^{-ix}
        =>  omega = 2 Re[omega_-] = -sin x   exactly.
      So the L2-matched family data converge to -sin x = cos(x + pi/2); the
      family's case-A collapse point x=0 sits at distance pi/2 from the crest,
      matching the cos-x run's singularity above x = pi/2."""
    print("stage sym: boundary-limit identity (hand-derived; numeric check)")
    N = 1 << 14
    x = 2*np.pi*np.arange(N)/N
    for eps in (1e-3, 1e-5, 1e-7):
        v = 1.0 - eps
        w = slsa_field(x, v, (1.0 - v*v)/2.0)
        d = np.max(np.abs(w - (-np.sin(x))))
        print("  1-v0 = %.0e : ||field - (-sin x)||_inf = %.3e   (expected O(1-v0))" % (eps, d))

def stage_rate3():
    """(a) Verify sup|field| ~ (3*sqrt(3)/4) w2/v^2 near collapse on family
    fields. (b) Family-limit prediction sup*(tc-t) -> sqrt(3); test the cos-x
    stage-pi track against sqrt(3)/(pi-t) with a correction fit."""
    print("stage rate3: sqrt(3) rate — family formula check + cos-x comparison")
    N = 1 << 16
    x = 2*np.pi*np.arange(N)/N
    for v, w2 in ((0.05, 1.0), (0.02, 1.0), (0.01, 1.0)):
        s = np.max(np.abs(slsa_field(x, v, w2)))
        pred = (3*np.sqrt(3)/4)*w2/v**2
        print("  family sup at v=%.3f: numeric %.6g vs (3sqrt3/4)w2/v^2 = %.6g  (rel %.1e)"
              % (v, s, pred, abs(s - pred)/pred))
    d = np.load(os.path.join(DIR, "gscan_pi_a0p5_N2048.npz"), allow_pickle=True)
    tr = d["track"]; t, s = tr[:, 0], tr[:, 1]
    m = (t > np.pi - 0.3) & (t < np.pi - 1e-3)
    prod = s[m]*(np.pi - t[m])
    # correction fit: prod = C + c1*(pi-t)^beta ; test C = sqrt(3)
    from scipy.optimize import curve_fit
    def f(tt, C, c1, beta):
        return C + c1*tt**beta
    p, _ = curve_fit(f, np.pi - t[m], prod, p0=(1.73, 0.5, 0.5), maxfev=20000)
    print("  cos-x run: fit sup*(pi-t) = C + c1*(pi-t)^beta over (pi-0.3, pi-0.001):")
    print("    C = %.6f   (sqrt(3) = %.6f, |C - sqrt3| = %.1e)   c1=%.4f beta=%.3f"
          % (p[0], np.sqrt(3), abs(p[0] - np.sqrt(3)), p[1], p[2]))

def stage_cd():
    """Continuity/shadowing demonstration for the N10 proof route: evolve the
    PDE from -sin x (the family's boundary-limit data) and measure its sup
    distance D(t) to the EXACT family solution at v0 = 0.99, 0.999. Collapse of
    D(t)/(1-v0) onto one curve = the linear-response/continuous-dependence
    regime, i.e. the -sin x solution is family-shadowed up to near collapse."""
    from scipy.integrate import solve_ivp
    print("stage cd: shadowing of the -sin x solution by the SLSA family")
    N, dt = 1024, 2.5e-5
    ops = cv.make_ops(N)
    x = ops[0]
    for v0 in (0.99, 0.999):
        w20 = (1.0 - v0*v0)/2.0
        tc = tc_caseA(v0, w20)
        fr = (0.25, 0.5, 0.75, 0.9, 0.95)
        cps = [dt*round(f*tc/dt) for f in fr]
        sol = solve_ivp(ode_rhs, (0, max(cps)), [w20, v0], t_eval=cps,
                        method="DOP853", rtol=1e-12, atol=1e-14)
        w = -np.sin(x)
        cpn = {int(round(t/dt)): i for i, t in enumerate(cps)}
        line = "  v0=%.3f (tc=%.4f, 1-v0=%.0e):" % (v0, tc, 1 - v0)
        n = 0
        while n < max(cpn):
            w = gs.rk4_g(w, dt, 0.5, ops, gs.rhs_g)
            n += 1
            if n in cpn:
                i = cpn[n]
                wex = slsa_field(x, sol.y[1, i], sol.y[0, i])
                D = np.max(np.abs(w - wex))
                line += "  D(%.2ftc)=%.2e [D/(1-v0)=%.2f]" % (fr[i], D, D/(1 - v0))
        print(line)

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "field":    stage_field()
    elif st == "ode":    stage_ode()
    elif st == "anchor": stage_anchor()
    elif st == "limit":  stage_limit()
    elif st == "sym":    stage_sym()
    elif st == "rate3":  stage_rate3()
    elif st == "cd":     stage_cd()
    else: raise SystemExit("unknown stage: %s" % st)
