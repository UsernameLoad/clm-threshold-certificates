#!/usr/bin/env python3
# cap_hier.py — computer-assisted-proof track
# Target theorem: finite-time breakdown of sigma=2
# viscous CLM on S^1 from cos x at explicit nu (candidate nu = 0.04), via the
# q-hierarchy  q_m' = -nu m^2 q_m + (1/2) sum_{j<m} q_j q_{m-j},  q_1 = e^{-nu t},
# all q_m >= 0, blowup <=> Sigma q_m = infinity at finite time.
#
# Proof architecture (dyadic block ladder, S08 derivation — recorded in
# cap-blowup-note.md):
#   S_k(t) := sum_{m in [2^k, 2^{k+1})} q_m(t)   (block sums)
#   S_{k+1}(t) >= (1/2) int e^{-nu 4^{k+2}(t-s)} S_k(s)^2 ds        (exact minorant)
#   With windows h_k = 1/(nu 4^{k+2}) (sum of windows finite and tiny):
#   A_{k+1} = (1-1/e) A_k^2/(2 nu 4^{k+2}),  L := lim A_k^{1/2^k},
#   L > 1  <=>  A_{k0} > A*(k0) := (2 nu/(1-1/e)) 4^{k0+3}
#   and L > 1  ==>  Sigma_m q_m = infinity on an explicit window ==> no smooth
#   PDE solution beyond t1 (contradiction with rapid Fourier decay).
# Stages: proto[:nu[:M]] | iv0 | enc:<...> (interval stages added as built)
import os, sys, time
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

def NL(c):
    M = len(c)
    conv = np.convolve(c, c)
    out = np.zeros(M)
    out[1:] = 0.5*conv[:M-1]
    return out

def q_lawson(nu, M, Tmax, alpha=0.2, dt_max=0.01, stop=None, record_every=25):
    """Lawson-IFRK4 on the REAL q-hierarchy (exact stiff diagonal; mode 1 is
    exact e^{-nu t} automatically since it has no nonlinear feedback). Mirrors
    hier_if.ifrk4_run. stop(t, q) -> True ends the run. Scouting accuracy
    (alpha=0.2) unless stated."""
    m2 = (np.arange(1, M+1)**2).astype(float)
    q = np.zeros(M); q[0] = 1.0
    t = 0.0
    dt_prev = -1.0; E1 = E2 = None
    out = []
    nstep = 0
    while t < Tmax:
        S = np.sum(q)
        if nstep % record_every == 0:
            out.append((t, q.copy()))
            if stop is not None and stop(t, q):
                break
        if not np.isfinite(S) or S > 1e14:
            break
        dt = min(dt_max, alpha/max(S, 1e-12), Tmax - t)
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = NL(q)
        U2 = E2*(q + 0.5*dt*k1)
        k2 = NL(U2)
        U3 = E2*q + 0.5*dt*k2
        k3 = NL(U3)
        U4 = E1*q + dt*E2*k3
        k4 = NL(U4)
        q = E1*q + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt
        nstep += 1
    return out

def stage_proto(nu=0.04, M=512):
    """Float prototype: when do the dyadic block sums cross the ladder
    thresholds A*(k0)? Locates the cheapest certification point (k0, t1)."""
    print("stage proto: nu=%.4f M=%d — block sums vs ladder thresholds" % (nu, M))
    t0 = time.time()
    thresh = {k: (2*nu/(1 - np.e**-1))*4.0**(k + 3) for k in range(3, 9)}
    for k in sorted(thresh):
        print("  A*(k0=%d) [seed block m in [%d,%d)] = %.4g" % (k, 2**k, 2**(k+1), thresh[k]))
    ks = [k for k in sorted(thresh) if 2**(k+1) <= M]
    crossings = {}
    state = dict(nrec=0, smax=0.0, decaying=0)
    def stop(t, q):
        S = np.sum(q)
        for k in ks:
            if k not in crossings:
                Sk = np.sum(q[2**k - 1: 2**(k+1) - 1])
                if Sk >= thresh[k]:
                    crossings[k] = (t, Sk)
                    print("  CROSS k0=%d at t=%.4f (S_k=%.4g, Sigma_M=%.4g)" % (k, t, Sk, S))
        state["nrec"] += 1
        if state["nrec"] % 400 == 0:
            bl = " ".join("S%d=%.3g" % (k, np.sum(q[2**k - 1: 2**(k+1) - 1])) for k in ks)
            print("    [t=%.4f Sigma_M=%.4g | %s]" % (t, S, bl))
        # end when the truncated observable is clearly past its peak and shrinking
        if S > state["smax"]:
            state["smax"], state["decaying"] = S, 0
        elif S < 0.5*state["smax"]:
            state["decaying"] += 1
            if state["decaying"] > 40:
                print("  [truncated peak passed: Sigma_M max %.4g, now %.4g — ending]" % (state["smax"], S))
                return True
        return len(crossings) == len(ks)
    out = q_lawson(nu, M, 12.0, stop=stop)
    ts = np.array([o[0] for o in out])
    Sig = np.array([np.sum(o[1]) for o in out])
    print("  Sigma milestones: " + " | ".join("S=%.3g at t=%.3f" % (s, ts[np.argmax(Sig >= s)])
          for s in (10, 100, 1e3, 1e4, 1e5, 1e7) if Sig.max() >= s))
    print("  end: t=%.4f Sigma_M=%.4g   (block sums for m<=M are exact by triangularity)"
          % (ts[-1], Sig[-1]))
    for k in sorted(crossings):
        t, Sk = crossings[k]
        print("  SEED CANDIDATE k0=%d: S_%d = %.4g >= %.4g at t = %.4f" % (k, k, Sk, thresh[k], t))
    if not crossings:
        print("  no threshold crossed by t=%.1f at M=%d — deepen M or t" % (12.0, M))
    # mode magnitudes at the earliest crossing (for interval-arithmetic sizing)
    if crossings:
        k0 = min(crossings)
        t1 = crossings[k0][0]
        i = int(np.argmin(np.abs(ts - t1)))
        q = out[i][1]
        print("  at t1=%.3f: max q_m (m<%d) = %.3g, q_%d = %.3g, q_%d = %.3g"
              % (t1, 2**(k0+1), q[:2**(k0+1)].max(), 2**k0, q[2**k0 - 1],
                 2**(k0+1) - 1, q[2**(k0+1) - 2]))
    print("[%.0f s]" % (time.time() - t0))


# ------------------------- certified sweep (stage enc) -------------------------
# Rigor model: numpy float64 (lo, hi) pairs on NONNEGATIVE data with
#  - global outward factors CI/CD covering all composite-op rounding
#    (Higham gamma_n bounds; dot lengths <= M=512 << 1200), plus absolute ETA
#    against underflow;
#  - certified exponentials/kernels via arb (python-flint) at EXACT dyadic
#    inputs (nu = 41/1024, h in {2^-8-j}), converted outward by one ulp;
#  - per-step step-hulls: lower = decay-only  H_lo = E_lo*qa_lo;
#    upper = a-posteriori super-hull  H_hi = C_g*qa_hi verified via
#    Phi(H) := qa_hi + h*F_hi(H) <= 0.999*H  (monotone invariant region);
#  - endpoint Duhamel bracket  qb in E*[qa] + [F_lo,F_hi]*G(h),
#    G = (1-e^{-nu m^2 h})/(nu m^2) certified.
# Soundness is inductive in m (triangularity): no Picard iteration anywhere.
U53 = 2.0**-53
CI = 1.0 + 1200*U53
CD = 1.0 - 1200*U53
ETA = 1e-280

def _tables(nu_num, nu_den, j, M):
    """Certified E, G tables for h = 2^(-8-j): outward float64 bounds via arb."""
    from flint import arb, ctx
    ctx.prec = 80
    E_lo = np.empty(M); E_hi = np.empty(M); G_lo = np.empty(M); G_hi = np.empty(M)
    h = 2.0**(-8 - j)
    for m in range(1, M + 1):
        x = float(nu_num*m*m)*h/nu_den          # exact: 41*m^2*2^-k, k <= 60
        b = (-arb(x)).exp()
        E_lo[m-1] = np.nextafter(float(b.lower()), -np.inf)
        E_hi[m-1] = np.nextafter(float(b.upper()), np.inf)
        g = (1 - b)/arb(float(nu_num*m*m)/nu_den)
        G_lo[m-1] = np.nextafter(float(g.lower()), -np.inf)
        G_hi[m-1] = np.nextafter(float(g.upper()), np.inf)
    return h, np.maximum(E_lo, 0.0), E_hi, np.maximum(G_lo, 0.0), G_hi

def _conv_upper(H):
    F = 0.5*np.convolve(H, H)[:len(H)]
    out = np.zeros(len(H))
    out[1:] = F[:-1]*CI + ETA
    return out

def _conv_lower(H):
    F = 0.5*np.convolve(H, H)[:len(H)]
    out = np.zeros(len(H))
    out[1:] = np.maximum(F[:-1]*CD, 0.0)
    return out

def stage_enc(M=512, k0=8, alpha=0.01, target_margin=4.0, maxsteps=800000):
    """Certified enclosure sweep for the blowup certificate at nu = 41/1024.
    Runs until the sliding window of length W = (4/(3 nu)) 4^-(k0+2) has
    certified min S_k0_lo >= target_margin * A*(k0)_hi, then prints the
    certificate. Self-terminating on success; honest failure report if
    widths blow up first."""
    from flint import arb, ctx
    ctx.prec = 80
    nu_num, nu_den = 41, 1024
    nu = nu_num/nu_den
    print("stage enc: certified sweep  nu = %d/%d, M = %d, k0 = %d, alpha = %g" % (nu_num, nu_den, M, k0, alpha))
    t0 = time.time()
    # certified threshold A* (upper bound) and window W (upper bound)
    nu_a = arb(nu_num)/arb(nu_den)
    Astar = 2*nu_a/(1 - (-arb(1)).exp())*arb(4)**(k0 + 3)
    Astar_hi = np.nextafter(float(Astar.upper()), np.inf)
    W = float((4/(3*nu_a)*arb(4)**(-(k0 + 2))).upper())*1.02
    print("  A*(k0=%d) <= %.6g ; window W = %.3e ; target S_lo >= %.6g"
          % (k0, Astar_hi, W, target_margin*Astar_hi))
    lo_blk, hi_blk = 2**k0 - 1, 2**(k0 + 1) - 1
    qa_lo = np.zeros(M); qa_hi = np.zeros(M)
    qa_lo[0] = qa_hi[0] = 1.0
    t = 0.0
    tab = {}
    win = []                                # (t_step_start, h, S_lo_hull)
    nstep = 0
    best = (0.0, -1.0)
    while nstep < maxsteps:
        S_hi = np.sum(qa_hi)*CI
        # h-level from the MIDPOINT size (any h is sound — hull closure is
        # verified per step — so we must not key h on the inflated upper
        # bound: that causes a width/step-count death spiral)
        S_mid = 0.5*(np.sum(qa_lo) + np.sum(qa_hi))
        j = max(0, int(np.ceil(np.log2(max(S_mid/(alpha*2.0**8), 1.0)))))
        j = min(j, 44)
        if j not in tab:
            tab[j] = _tables(nu_num, nu_den, j, M)
        h, E_lo, E_hi, G_lo, G_hi = tab[j]
        # step-hulls: lower = decay-only; upper = monotone-iterated super-hull
        # (support doubles per iteration, so ~log2(M)+O(1) iterations suffice;
        #  invariant-region lemma: Phi(H) <= 0.999 H with outward rounding
        #  implies sup_step q <= H).
        ok = False
        for jtry in range(j, min(j + 13, 45)):
            if jtry > 44:
                break
            if jtry not in tab:
                tab[jtry] = _tables(nu_num, nu_den, jtry, M)
            h, E_lo, E_hi, G_lo, G_hi = tab[jtry]
            H_lo = np.maximum(E_lo*qa_lo*CD, 0.0)
            H_hi = qa_hi*(1.0 + 3.0*alpha) + ETA
            for _ in range(40):
                F_hi = _conv_upper(H_hi)
                Phi = (qa_hi + F_hi*h*CI)*CI + ETA
                if np.all(Phi <= 0.999*H_hi + 1e-300):
                    ok = True
                    break
                H_hi = np.maximum(H_hi, Phi*1.02)
            if ok:
                break
        if not ok:
            bad = int(np.argmax(Phi/np.maximum(0.999*H_hi, 1e-300)))
            print("  FAIL: hull not closing at t=%.6f (S_hi=%.3g); worst mode %d: Phi/H=%.3g (Phi=%.3g H=%.3g)"
                  % (t, S_hi, bad + 1, Phi[bad]/(0.999*H_hi[bad]), Phi[bad], H_hi[bad])); break
        F_lo = _conv_lower(H_lo)
        # endpoint Duhamel bracket
        qb_lo = np.maximum((E_lo*qa_lo + F_lo*G_lo)*CD, 0.0)
        qb_hi = (E_hi*qa_hi + F_hi*G_hi)*CI + ETA
        # window bookkeeping: hull-certified block lower bound over this step
        S_blk_lo = np.sum(H_lo[lo_blk:hi_blk])*CD
        win.append((t, h, S_blk_lo))
        tw = t + h - W
        while win and win[0][0] + win[0][1] <= tw:
            win.pop(0)
        if win and win[0][0] <= tw:
            wmin = min(w[2] for w in win)
            if wmin > best[1]:
                best = (win[0][0], wmin)
            if wmin >= target_margin*Astar_hi:
                t1 = win[0][0]
                print("  CERTIFIED at t1 = %.8f: min over [t1, t1+W] of S_%d >= %.6g  (margin %.2f x A*)"
                      % (t1, k0, wmin, wmin/Astar_hi))
                print("  ==> Sigma q_m(t) = INFINITY for t in [t1 + %.3e, t1 + %.3e]" % (W*0.75/1.02, W))
                print("  ==> no smooth solution of viscous CLM (nu = 41/1024) from cos x exists beyond t = %.6f"
                      % (t1 + W))
                live = qa_hi > 1e-6*qa_hi.max()
                print("  [%d steps, %.0f s; max live rel width: %.3f]"
                      % (nstep, time.time() - t0,
                         np.max((qa_hi[live] - qa_lo[live])/np.maximum(qa_lo[live], 1e-300))))
                np.savez(os.path.join(DIR, "cap_cert_k%d.npz" % k0), t1=t1, W=W,
                         S_lo=wmin, Astar_hi=Astar_hi, qa_lo=qa_lo, qa_hi=qa_hi, t=t,
                         nu_num=nu_num, nu_den=nu_den, M=M, k0=k0, alpha=alpha)
                return
        qa_lo, qa_hi = qb_lo, qb_hi
        t += h
        nstep += 1
        if nstep % 25000 == 0:
            Sbh = np.sum(qa_hi[lo_blk:hi_blk])
            print("    [t=%.5f S_lo/S_hi=%.4g/%.4g blk_lo=%.3g blk_hi/lo=%.2f  %.0fs]"
                  % (t, np.sum(qa_lo), S_hi, S_blk_lo,
                     Sbh/max(np.sum(qa_lo[lo_blk:hi_blk]), 1e-300), time.time() - t0))
    print("  NOT certified (best window bound %.4g = %.2f x A* at t=%.6f); %d steps [%.0f s]"
          % (best[1], best[1]/Astar_hi, best[0], nstep, time.time() - t0))

# ------------------- decay-side certificate (stage dec0) -------------------
# Integration-free rigorous certificate of GLOBAL DECAY at a given nu, hence
# nu*(0) <= nu by the S08 nu-comparison lemma (the positive q-system is
# cooperative; larger nu has pointwise-smaller RHS, so Kamke comparison gives
# q^{nu2} <= q^{nu1} componentwise for nu2 >= nu1 — decay propagates upward).
# Structure of the certificate (all quantities as arb balls; proofs to the note):
#   q1 = e^{-nu t} exactly.
#   q2, q3, q4: EXACT exponential sums, from the closed Duhamel chain
#     (each mode is a linear scalar ODE forced by lower modes; integrals of
#      exponentials stay exponential sums; coefficients are rational in nu).
#   Tail Y = sum_{m>=5} q_m:  Y' <= -25 nu Y + (1/2)(q3+q4+Y)(2T+Y),
#     T = q1+q2+q3+q4  (derived by subtracting the full F_2, F_3, F_4 mass
#     from the square; valid for every truncation, hence for the sum).
#   Barrier: if there is B > 0 with  (1/2)(G+B)(2Tbar+B) < 25 nu B  where
#     G >= sup_t (q3+q4), Tbar >= sup_t T, then Y(t) < B for all t (Y(0)=0),
#     and since T -> 0 explicitly, Y -> 0: GLOBAL DECAY.
#   Sufficient explicit B-existence check: the quadratic
#     B^2/2 + B(Tbar + G/2 - 25 nu) + G*Tbar < 0 for some B > 0
#     <=> Tbar + G/2 < 25 nu  and  (25 nu - Tbar - G/2)^2 > 2 G Tbar.
# Sup bounds for exponential sums: certified grid + derivative-Lipschitz tail
# (crude V1: sum of |coeff| * sup of each decaying term on [t_i, t_{i+1}]).

def _expsum_mul(a, b):
    """Terms are (coef: arb, k: int) meaning sum coef*e^{-k nu t}; exponents are
    exact integer multiples of nu throughout the chain."""
    out = {}
    for (c1, k1) in a:
        for (c2, k2) in b:
            key = k1 + k2
            out[key] = out.get(key) + c1*c2 if key in out else c1*c2
    return [(c, k) for (k, c) in out.items()]

def _duhamel(terms, m2, nu):
    """Solve x' = -m^2 nu x + f, x(0)=0, f = sum c e^{-k nu t} (all k != m^2):
    x = sum c/((m^2-k) nu) (e^{-k nu t} - e^{-m^2 nu t})."""
    out = []
    tail = None
    for (c, k) in terms:
        assert k != m2
        w = c/((m2 - k)*nu)
        out.append((w, k))
        tail = w if tail is None else tail + w
    out.append((-tail, m2))
    return out

def _supbound(terms, arb, nu, tmax, n=2000):
    """Rigorous upper bound for sup_{t>=0} of V(t) = sum c e^{-mu t}
    (signed c, mu > 0). On each grid cell [t_k, t_{k+1}]:
       sup V <= V(t_k) + (t_{k+1}-t_k) * L(t_k),
    where L(t_k) = sum |c| mu e^{-mu t_k} bounds |V'| on the cell (each
    |c mu e^{-mu t}| is decreasing). The [tmax, inf) tail is bounded by
    sum |c| e^{-mu tmax}. All evaluation in arb; returns an arb upper bound."""
    ts = [tmax*k*k/(n*n) for k in range(n + 1)]     # quadratic clustering near 0
    best = None
    for j in range(n):
        tk = arb(ts[j]); dt = arb(ts[j+1] - ts[j])
        V = None; L = None
        for (c, k) in terms:
            mu = k*nu
            e = (-(mu*tk)).exp()
            V = c*e if V is None else V + c*e
            L = abs(c)*mu*e if L is None else L + abs(c)*mu*e
        cell = V + dt*L
        if best is None or float(cell.upper()) > float(best.upper()):
            best = cell
    tail = None
    for (c, k) in terms:
        e = abs(c)*(-(k*nu*arb(tmax))).exp()
        tail = e if tail is None else tail + e
    if float(tail.upper()) > float(best.upper()):
        best = tail
    return best

def stage_dec0():
    """Scan nu downward; report the smallest nu at which the decay certificate
    closes. Result: THEOREM nu*(0) <= that nu (with the comparison lemma)."""
    from flint import arb, ctx
    ctx.prec = 120
    print("stage dec0: integration-free decay certificates (exact modes 1-4 + tail barrier)")
    for nu_txt in ("0.30", "0.25", "0.22", "0.20", "0.18", "0.16", "0.15", "0.14", "0.13", "0.12", "0.11", "0.10", "0.09", "0.08"):
        nu = arb(nu_txt)
        one = arb(1)
        q1 = [(one, 1)]                                    # e^{-nu t}
        F2 = [(c/2, k) for (c, k) in _expsum_mul(q1, q1)]
        q2 = _duhamel(F2, 4, nu)
        F3 = _expsum_mul(q1, q2)                           # (1/2)*2*q1q2
        q3 = _duhamel(F3, 9, nu)
        q2sq = _expsum_mul(q2, q2)
        F4 = _expsum_mul(q1, q3) + [(c/2, k) for (c, k) in q2sq]
        q4 = _duhamel(F4, 16, nu)
        # sup bounds via certified grid + Lipschitz cells
        tmax = 60.0/float(nu_txt)
        s1 = one                                           # sup q1 = 1
        s2 = _supbound(q2, arb, nu, tmax); s3 = _supbound(q3, arb, nu, tmax); s4 = _supbound(q4, arb, nu, tmax)
        Tbar = s1 + s2 + s3 + s4
        G = s3 + s4
        lhs = Tbar + G/2
        ok1 = lhs < 25*nu
        disc = (25*nu - lhs)**2 - 2*G*Tbar if ok1 else None
        ok = bool(ok1) and bool(disc > 0)
        print("  nu=%s: sup q2..q4 = %.4f %.4f %.4f  Tbar=%.4f  G=%.4f  need Tbar+G/2 < %.3f: %s%s"
              % (nu_txt, float(s2.upper()), float(s3.upper()), float(s4.upper()),
                 float(Tbar.upper()), float(G.upper()), float((25*nu).lower()),
                 "PASS" if ok1 else "fail",
                 ("; disc>0: %s -> %s" % (bool(disc > 0), "CERTIFIED" if ok else "fail")) if ok1 else ""))
        if not ok1:
            print("  [certificate stops closing here; smallest certified nu is the last CERTIFIED above]")
            break

def stage_dec1(mP=8):
    """Generalized integration-free decay certificate with modes 1..mP explicit.
    P := q_1+...+q_mP (exact exponential sums; Duhamel chain — never resonant
    since forcing exponents <= 1+(m-1)^2 < m^2).  Y := sum_{m>mP} q_m obeys
       Y' <= -(mP+1)^2 nu Y + (1/2)(R + 2PY + Y^2),
    R := sum_{j+l>mP, j,l<=mP} q_j q_l  (explicit exponential sum; sup-bounded
    directly so in-t correlations are kept).  Barrier B > 0 exists iff
       Pbar < (mP+1)^2 nu   and   ((mP+1)^2 nu - Pbar)^2 > Rbar.
    Certifies global decay at nu, hence nu*(0) <= nu (comparison lemma)."""
    from flint import arb, ctx
    ctx.prec = 160
    print("stage dec1: explicit modes 1..%d + tail barrier at -%d nu" % (mP, (mP+1)**2))
    for nu_txt in ("0.14", "0.13", "0.12", "0.11", "0.10", "0.095", "0.09", "0.085",
                   "0.08", "0.075", "0.07", "0.065", "0.06"):
        nu = arb(nu_txt)
        t0 = time.time()
        q = {1: [(arb(1), 1)]}
        for m in range(2, mP + 1):
            F = {}
            for j in range(1, m):
                for (c1, k1) in q[j]:
                    for (c2, k2) in q[m - j]:
                        key = k1 + k2
                        F[key] = F.get(key) + c1*c2/2 if key in F else c1*c2/2
            q[m] = _duhamel([(c, k) for (k, c) in F.items()], m*m, nu)
        # P(t) and R(t) as explicit sums
        Pterms = {}
        for m in range(1, mP + 1):
            for (c, k) in q[m]:
                Pterms[k] = Pterms.get(k) + c if k in Pterms else c
        Rterms = {}
        for j in range(1, mP + 1):
            for l in range(1, mP + 1):
                if j + l > mP:
                    for (c1, k1) in q[j]:
                        for (c2, k2) in q[l]:
                            key = k1 + k2
                            Rterms[key] = Rterms.get(key) + c1*c2 if key in Rterms else c1*c2
        # sup bounds by per-mode signed evaluation on a cubic-clustered grid:
        # sup_cell f <= f(t_k) + dt*|f'(t_k)| + (dt^2/2)*B2(t_k), with f, f'
        # evaluated as SIGNED sums (arb cancels; representations may have huge
        # cancelling coefficients) and only the second-derivative envelope
        # B2 = sum |c| mu^2 e^{-mu t_k} taken crudely (damped by dt^2).
        tmax = 80.0/float(nu_txt)
        n = 3000
        ts = [tmax*k**3/float(n)**3 for k in range(n + 1)]
        def ev(terms, tk, order):
            v = None
            for (c, k) in terms:
                mu = k*nu
                w = c*((-(mu*arb(tk))).exp())
                if order == 1:
                    w = -mu*w
                elif order == 2:
                    w = abs(c)*mu*mu*((-(mu*arb(tk))).exp())
                v = w if v is None else v + w
            return v
        Plist = [(c, k) for (k, c) in Pterms.items()]
        Rlist = [(c, k) for (k, c) in Rterms.items()]
        def supb(terms):
            best = None
            for j in range(n):
                dt = arb(ts[j+1] - ts[j])
                cell = ev(terms, ts[j], 0) + dt*abs(ev(terms, ts[j], 1)) + dt*dt/2*ev(terms, ts[j], 2)
                if best is None or float(cell.upper()) > float(best.upper()):
                    best = cell
            tail = None
            for (c, k) in terms:
                e = abs(c)*((-(k*nu*arb(tmax))).exp())
                tail = e if tail is None else tail + e
            return best if float(best.upper()) >= float(tail.upper()) else tail
        Pbar = supb(Plist)
        Rbar = supb(Rlist)
        thr = ((mP + 1)**2)*nu
        ok1 = Pbar < thr
        disc = (thr - Pbar)**2 - Rbar if ok1 else None
        ok = bool(ok1) and bool(disc > 0)
        print("  nu=%-5s: Pbar=%.4f (<%.3f: %s)  Rbar=%.4f  disc>0: %s -> %s  [%d P-terms, %d R-terms, %.0fs]"
              % (nu_txt, float(Pbar.upper()), float(thr.lower()), "ok" if ok1 else "FAIL",
                 float(Rbar.upper()), bool(disc > 0) if ok1 else "-",
                 "CERTIFIED" if ok else "fail", len(Pterms), len(Rterms), time.time() - t0))
        if not ok:
            print("  [stops closing; the smallest CERTIFIED nu above is the theorem bound]")
            break


def stage_dec2(mP=8):
    """V3 decay certificate: time-dependent piecewise-linear supersolution of
    the scalar comparison  Y' <= f(t,Y) = -Thr Y + (1/2)(R(t) + 2P(t)Y + Y^2),
    Thr = (mP+1)^2 nu, Y(0) = 0, followed by a static-basin landing check with
    tail sups over [T, inf).  Uses R(t)'s full transient profile instead of its
    global sup (the binding constraint of dec1).  All bounds in arb."""
    from flint import arb, ctx
    ctx.prec = 160
    print("stage dec2: time-dependent barrier, explicit modes 1..%d, tail at -%d nu" % (mP, (mP+1)**2))
    for nu_txt in ("0.08", "0.075", "0.072", "0.07", "0.068", "0.066", "0.064", "0.062", "0.06"):
        nu = arb(nu_txt)
        t0 = time.time()
        q = {1: [(arb(1), 1)]}
        for m in range(2, mP + 1):
            F = {}
            for j in range(1, m):
                for (c1, k1) in q[j]:
                    for (c2, k2) in q[m - j]:
                        key = k1 + k2
                        F[key] = F.get(key) + c1*c2/2 if key in F else c1*c2/2
            q[m] = _duhamel([(c, k) for (k, c) in F.items()], m*m, nu)
        Pterms = {}
        for m in range(1, mP + 1):
            for (c, k) in q[m]:
                Pterms[k] = Pterms.get(k) + c if k in Pterms else c
        Rterms = {}
        for j in range(1, mP + 1):
            for l in range(1, mP + 1):
                if j + l > mP:
                    for (c1, k1) in q[j]:
                        for (c2, k2) in q[l]:
                            key = k1 + k2
                            Rterms[key] = Rterms.get(key) + c1*c2 if key in Rterms else c1*c2
        Pl = [(c, k) for (k, c) in Pterms.items()]
        Rl = [(c, k) for (k, c) in Rterms.items()]
        Thr = ((mP + 1)**2)*nu
        def ev(terms, tk, order):
            # order 0,1,2: SIGNED sums of the 0th/1st/2nd derivative; order 3:
            # crude absolute bound sum |c| mu^3 e^{-mu t} (decreasing in t).
            v = None
            for (c, k) in terms:
                mu = k*nu
                e = (-(mu*arb(tk))).exp()
                if order == 0: w = c*e
                elif order == 1: w = -mu*c*e
                elif order == 2: w = mu*mu*c*e
                else: w = abs(c)*mu*mu*mu*e
                v = w if v is None else v + w
            return v
        def cellsup(terms, ta, h):
            H = arb(h)
            return (ev(terms, ta, 0) + H*abs(ev(terms, ta, 1))
                    + H*H/2*abs(ev(terms, ta, 2)) + H*H*H/6*ev(terms, ta, 3))
        def tailsup(terms, T, ngrid=800):
            span = 80.0/float(nu_txt)
            tsg = [T + span*k*k/(ngrid*ngrid) for k in range(ngrid + 1)]
            best = None
            for j in range(ngrid):
                cell = cellsup(terms, tsg[j], tsg[j+1] - tsg[j])
                if best is None or float(cell.upper()) > float(best.upper()):
                    best = cell
            tl = None
            for (c, k) in terms:
                e = abs(c)*((-(k*nu*arb(tsg[-1]))).exp())
                tl = e if tl is None else tl + e
            return best if float(best.upper()) >= float(tl.upper()) else tl
        # march the barrier from t_s = 1 with the inviscid-comparison start:
        # q_m(t) <= (t/2)^{m-1} e^{-m nu t}  (5-line induction lemma), so
        # Y(t_s) <= sum_{m>mP} (t_s/2)^{m-1} = (t_s/2)^{mP}/(1 - t_s/2).
        ts_ = 1.0
        h = 0.0004/float(nu_txt)
        Tmax = 240.0/float(nu_txt)
        ceil = 60.0*float(nu_txt)
        B = (arb(ts_)/2)**mP/(1 - arb(ts_)/2)
        t = ts_
        ok = None
        ncheck = int(3.0/float(nu_txt)/h)      # try landing every ~3/nu
        k = 0
        while t < Tmax:
            Rc = cellsup(Rl, t, h)
            Pc = cellsup(Pl, t, h)
            sk = arb(0)
            for _ in range(3):
                Bhi = B + arb(h)*(sk if float(sk.lower()) > 0 else arb(0))
                fb = -Thr*B + (Rc + 2*Pc*Bhi + Bhi*Bhi)/2
                sk = fb*arb("1.001") if float(fb.upper()) > 0 else fb
            B = B + arb(h)*sk
            if float(B.lower()) < 0:
                B = arb(0)
            t += h
            k += 1
            if float(B.upper()) > ceil:
                ok = False
                print("  nu=%-5s: barrier exceeded ceiling %.3f at t=%.1f -> fail  [%.0fs]"
                      % (nu_txt, ceil, t, time.time() - t0))
                break
            if k % ncheck == 0:
                Pt = tailsup(Pl, t); Rt = tailsup(Rl, t)
                if Pt < Thr:
                    disc = (Thr - Pt)**2 - Rt
                    if disc > 0:
                        Bp = (Thr - Pt) + disc.sqrt()
                        if B < Bp:
                            ok = True
                            print("  nu=%-5s: CERTIFIED — landed at T=%.1f (B=%.4f < B+=%.4f; Ptail=%.4f Rtail=%.4f)  [%.0fs]"
                                  % (nu_txt, t, float(B.upper()), float(Bp.lower()),
                                     float(Pt.upper()), float(Rt.upper()), time.time() - t0))
                            break
        if ok is None:
            print("  nu=%-5s: no landing by Tmax=%.0f (B=%.4f) -> fail  [%.0fs]"
                  % (nu_txt, Tmax, float(B.upper()), time.time() - t0))
        if ok is False or ok is None:
            print("  [stops closing; smallest CERTIFIED nu above is the bound]")
            break

def stage_dec3(mP=10):
    """V4 decay certificate (S10): TWO-BLOCK tail barrier. Blocks: B1 = modes
    mP+1..m2 with m2 = 2mP+1, B2 = modes > m2. Structural facts (derivation in
    the S10 log): with this split (i) no B1xB1 / B1xB2 / B2xB2 product lands in
    B1 and (ii) no exact-pair product lands past m2 (j,l <= mP => j+l <= 2mP <
    m2+1), so R2 == 0. Hence the comparison system is
        Y1' <= -T1 Y1 + (1/2) R(t) + P(t) Y1                      [LINEAR]
        Y2' <= -T2 Y2 + P(t)(Y1+Y2) + (1/2)(Y1+Y2)^2
    with T1 = (mP+1)^2 nu, T2 = (m2+1)^2 nu, R = dec2's R. Quasimonotone =>
    componentwise vector comparison. The dec2 failure mode (transient R driving
    the scalar barrier over the quadratic cliff) is structurally removed: the
    transient hits the linear block only. PREDICTED (before first run, S10 log):
    certifies to nu ~ 0.058-0.065; fails when sup_t P(t) approaches T1 = 121nu.
    All bounds in arb; march + landing generalize dec2 componentwise."""
    from flint import arb, ctx
    ctx.prec = 160
    m2 = 2*mP + 1
    print("stage dec3: two-block barrier, exact modes 1..%d, B1 at -%d nu (LINEAR), B2 at -%d nu"
          % (mP, (mP+1)**2, (m2+1)**2))
    for nu_txt in ("0.075", "0.072", "0.07", "0.068", "0.066", "0.064", "0.062",
                   "0.06", "0.058", "0.056", "0.054"):
        nu = arb(nu_txt)
        t0 = time.time()
        q = {1: [(arb(1), 1)]}
        for m in range(2, mP + 1):
            F = {}
            for j in range(1, m):
                for (c1, k1) in q[j]:
                    for (c2, k2) in q[m - j]:
                        key = k1 + k2
                        F[key] = F.get(key) + c1*c2/2 if key in F else c1*c2/2
            q[m] = _duhamel([(c, k) for (k, c) in F.items()], m*m, nu)
        Pterms = {}
        for m in range(1, mP + 1):
            for (c, k) in q[m]:
                Pterms[k] = Pterms.get(k) + c if k in Pterms else c
        Rterms = {}
        for j in range(1, mP + 1):
            for l in range(1, mP + 1):
                if j + l > mP:          # all these sums are <= 2mP < m2+1: R2 = 0
                    for (c1, k1) in q[j]:
                        for (c2, k2) in q[l]:
                            key = k1 + k2
                            Rterms[key] = Rterms.get(key) + c1*c2 if key in Rterms else c1*c2
        Pl = [(c, k) for (k, c) in Pterms.items()]
        Rl = [(c, k) for (k, c) in Rterms.items()]
        T1 = ((mP + 1)**2)*nu
        T2 = ((m2 + 1)**2)*nu
        def ev(terms, tk, order):
            v = None
            for (c, k) in terms:
                mu = k*nu
                e = (-(mu*arb(tk))).exp()
                if order == 0: w = c*e
                elif order == 1: w = -mu*c*e
                elif order == 2: w = mu*mu*c*e
                else: w = abs(c)*mu*mu*mu*e
                v = w if v is None else v + w
            return v
        def cellsup(terms, ta, h):
            H = arb(h)
            return (ev(terms, ta, 0) + H*abs(ev(terms, ta, 1))
                    + H*H/2*abs(ev(terms, ta, 2)) + H*H*H/6*ev(terms, ta, 3))
        def tailsup(terms, T, ngrid=800):
            span = 80.0/float(nu_txt)
            tsg = [T + span*k*k/(ngrid*ngrid) for k in range(ngrid + 1)]
            best = None
            for j in range(ngrid):
                cell = cellsup(terms, tsg[j], tsg[j+1] - tsg[j])
                if best is None or float(cell.upper()) > float(best.upper()):
                    best = cell
            tl = None
            for (c, k) in terms:
                e = abs(c)*((-(k*nu*arb(tsg[-1]))).exp())
                tl = e if tl is None else tl + e
            return best if float(best.upper()) >= float(tl.upper()) else tl
        # warm start at t_s = 1 (Sakajo comparison bound, dropping e^{-m nu}):
        ts_ = 1.0
        h = 0.0004/float(nu_txt)
        Tmax = 240.0/float(nu_txt)
        # ceilings are march-abort heuristics, not certificate conditions. B1 is
        # LINEAR (cannot blow up; transient peaks are legitimate), so its abort
        # level is generous; B2's quadratic cliff sits at ~2*T2 = 968 nu, so 30
        # is still far below the cliff at every nu scanned.
        ceil1 = 60.0
        ceil2 = 60.0
        B1 = None
        for m in range(mP + 1, m2 + 1):
            w = (arb(ts_)/2)**(m - 1)
            B1 = w if B1 is None else B1 + w
        B2 = (arb(ts_)/2)**m2/(1 - arb(ts_)/2)
        t = ts_
        ok = None
        ncheck = int(3.0/float(nu_txt)/h)
        k = 0
        while t < Tmax:
            Rc = cellsup(Rl, t, h)
            Pc = cellsup(Pl, t, h)
            s1 = arb(0); s2 = arb(0)
            for _ in range(3):
                # endpoint values with the SIGNED slope (backward-Euler style for
                # descending cells -- evaluating f at the start of a descent
                # under-counts f since dissipation weakens along it; S10 fix)
                B1h = B1 + arb(h)*s1
                B2h = B2 + arb(h)*s2
                if float(B1h.lower()) < 0: B1h = arb(0)
                if float(B2h.lower()) < 0: B2h = arb(0)
                f1 = -T1*B1h + Rc/2 + Pc*B1h
                f2 = -T2*B2h + Pc*(B1h + B2h) + (B1h + B2h)*(B1h + B2h)/2
                s1 = f1*arb("1.001") if float(f1.upper()) > 0 else f1
                s2 = f2*arb("1.001") if float(f2.upper()) > 0 else f2
            # PROOF OBLIGATION (final check; the construction above is untrusted):
            # verify s_i.lower >= f_i.upper with f_i evaluated at the cell-end
            # barrier values. On failure promote s_i to a THIN ball just above
            # f_i.upper (a certified upper slope; also handles f straddling 0).
            # Terminates: raising s raises f by only ~h*Lip*(ds) << ds.
            for _ in range(40):
                B1h = B1 + arb(h)*s1
                B2h = B2 + arb(h)*s2
                if float(B1h.lower()) < 0: B1h = arb(0)
                if float(B2h.lower()) < 0: B2h = arb(0)
                # f evaluated on the WHOLE cell range of B: dissipation at the
                # endpoint hull [min(B,Bh), max(B,Bh)] upper bound = -T*min;
                # growth at max. Conservative: dissipation at Bh if Bh<B else B,
                # growth at max(B,Bh). Implemented via interval hull:
                B1r = B1.union(B1h); B2r = B2.union(B2h)
                f1 = -T1*B1r + Rc/2 + Pc*B1r
                f2 = -T2*B2r + Pc*(B1r + B2r) + (B1r + B2r)*(B1r + B2r)/2
                ok1 = float(s1.lower()) >= float(f1.upper())
                ok2 = float(s2.lower()) >= float(f2.upper())
                if ok1 and ok2:
                    break
                if not ok1:
                    s1 = arb(np.nextafter(float(f1.upper()), np.inf))*arb("1.0001") \
                         if float(f1.upper()) > 0 else arb(np.nextafter(float(f1.upper()), np.inf))
                if not ok2:
                    s2 = arb(np.nextafter(float(f2.upper()), np.inf))*arb("1.0001") \
                         if float(f2.upper()) > 0 else arb(np.nextafter(float(f2.upper()), np.inf))
            else:
                raise RuntimeError("dec3: slope verification failed to close at t=%.4f" % t)
            B1 = B1 + arb(h)*s1
            B2 = B2 + arb(h)*s2
            if float(B1.lower()) < 0: B1 = arb(0)
            if float(B2.lower()) < 0: B2 = arb(0)
            t += h
            k += 1
            if float(B1.upper()) > ceil1 or float(B2.upper()) > ceil2:
                ok = False
                print("  nu=%-5s: barrier exceeded ceiling (B1=%.3f B2=%.3f) at t=%.1f -> fail  [%.0fs]"
                      % (nu_txt, float(B1.upper()), float(B2.upper()), t, time.time() - t0))
                break
            if k % ncheck == 0:
                Pt = tailsup(Pl, t); Rt = tailsup(Rl, t)
                if Pt < T1:
                    B1s = Rt/2/(T1 - Pt)
                    # forward bound for Y1 on [T,inf): linear stable dynamics under
                    # frozen sup-coefficients => Y1 <= max(B1(T), B1*) forever.
                    B1chk = B1 if float(B1.upper()) > float(B1s.upper()) else B1s
                    if T2 > Pt + B1chk:
                        disc = (T2 - Pt - B1chk)**2 - (2*Pt*B1chk + B1chk*B1chk)
                        if disc > 0:
                            B2p = (T2 - Pt - B1chk) + disc.sqrt()
                            if B2 < B2p:
                                ok = True
                                print("  nu=%-5s: CERTIFIED — landed at T=%.1f (B1=%.4g<=B1*=%.4g; B2=%.4g<B2+=%.4g; Ptail=%.4f)  [%.0fs]"
                                      % (nu_txt, t, float(B1.upper()), float(B1chk.upper()),
                                         float(B2.upper()), float(B2p.lower()),
                                         float(Pt.upper()), time.time() - t0))
                                break
        if ok is None:
            print("  nu=%-5s: no landing by Tmax=%.0f (B1=%.4g B2=%.4g) -> fail  [%.0fs]"
                  % (nu_txt, Tmax, float(B1.upper()), float(B2.upper()), time.time() - t0))
        if ok is False or ok is None:
            print("  [stops closing; smallest CERTIFIED nu above is the bound]")
            break

def stage_dec4(mP=10, NB=4, hdiv=1, prec=160):
    """V5 decay certificate (S10): N-BLOCK CASCADE barrier. Blocks G_k =
    (m_{k-1}, m_k], k = 1..NB (m_0 = mP, m_k = 2 m_{k-1} + 1, G_NB = tail to
    infinity with T_NB from its lower edge). With this spacing every block's
    self-products land BEYOND it, so blocks 1..NB-1 are LINEAR in themselves;
    only the last block carries the quadratic, against (m_{NB-1}+1)^2 nu
    dissipation. Generous-assignment soundness: each block's inequality uses
    the crude upper forcing (full cross/lower products), sound per-block:
        f_k = -T_k Y_k + (1/2)R(t)[k=1] + P(Y_k + S_{<k}) + (1/2)S_{<k}^2   (k < NB)
        f_NB = -T_NB Y_NB + P*S_all + (1/2)S_all^2,   S_<k = sum_{j<k} Y_j.
    Quasimonotone => componentwise vector comparison; march + proof
    obligations exactly as dec3, vectorized over blocks. Landing: freeze
    tail sups; solve the static system blockwise upward (linear blocks give
    B_k* = (R_k/2 + P S_<k* + S_<k*^2/2)/(T_k - P); last block: quadratic
    basin as dec3 with B1chk -> S_<NB bound). PREDICTED (before first run):
    certifies 0.066-0.068 at NB=4; the edge is set by block 1's exponential
    transient gain, so gains beyond ~0.064 need exact-mode extension (mP up)
    rather than more blocks."""
    from flint import arb, ctx
    ctx.prec = prec
    edges = [mP]
    for _ in range(NB - 1):
        edges.append(2*edges[-1] + 1)
    Ts = [((edges[k] + 1)**2) for k in range(NB)]   # T_k / nu, k=1..NB (edges[k-1]+1)^2... careful below
    print("stage dec4: %d-block cascade, exact modes 1..%d, block edges %s, dissipations %s x nu"
          % (NB, mP, edges, [ (edges[k]+1)**2 for k in range(NB) ]))
    for nu_txt in ("0.07", "0.068", "0.066", "0.064", "0.062", "0.06"):
        nu = arb(nu_txt)
        t0 = time.time()
        q = {1: [(arb(1), 1)]}
        for m in range(2, mP + 1):
            F = {}
            for j in range(1, m):
                for (c1, k1) in q[j]:
                    for (c2, k2) in q[m - j]:
                        key = k1 + k2
                        F[key] = F.get(key) + c1*c2/2 if key in F else c1*c2/2
            q[m] = _duhamel([(c, k) for (k, c) in F.items()], m*m, nu)
        Pterms = {}
        for m in range(1, mP + 1):
            for (c, k) in q[m]:
                Pterms[k] = Pterms.get(k) + c if k in Pterms else c
        Rterms = {}
        for j in range(1, mP + 1):
            for l in range(1, mP + 1):
                if j + l > mP:
                    for (c1, k1) in q[j]:
                        for (c2, k2) in q[l]:
                            key = k1 + k2
                            Rterms[key] = Rterms.get(key) + c1*c2 if key in Rterms else c1*c2
        Pl = [(c, k) for (k, c) in Pterms.items()]
        Rl = [(c, k) for (k, c) in Rterms.items()]
        # T_k: block k spans (edges[k-1], edges[k]] (k=1..NB-1 with edges[0]=mP),
        # block NB spans (edges[NB-1], inf). Dissipation floor of block k is
        # (lower edge + 1)^2 nu.
        lo_edges = [mP] + edges[1:NB]           # lower edges of blocks 1..NB: mP, m1, ..., m_{NB-1}
        Tk = [((lo_edges[k] + 1)**2)*nu for k in range(NB)]
        def ev(terms, tk, order):
            v = None
            for (c, k) in terms:
                mu = k*nu
                e = (-(mu*arb(tk))).exp()
                if order == 0: w = c*e
                elif order == 1: w = -mu*c*e
                elif order == 2: w = mu*mu*c*e
                else: w = abs(c)*mu*mu*mu*e
                v = w if v is None else v + w
            return v
        def cellsup(terms, ta, h):
            H = arb(h)
            return (ev(terms, ta, 0) + H*abs(ev(terms, ta, 1))
                    + H*H/2*abs(ev(terms, ta, 2)) + H*H*H/6*ev(terms, ta, 3))
        def tailsup(terms, T, ngrid=800):
            span = 80.0/float(nu_txt)
            tsg = [T + span*k*k/(ngrid*ngrid) for k in range(ngrid + 1)]
            best = None
            for j in range(ngrid):
                cell = cellsup(terms, tsg[j], tsg[j+1] - tsg[j])
                if best is None or float(cell.upper()) > float(best.upper()):
                    best = cell
            tl = None
            for (c, k) in terms:
                e = abs(c)*((-(k*nu*arb(tsg[-1]))).exp())
                tl = e if tl is None else tl + e
            return best if float(best.upper()) >= float(tl.upper()) else tl
        # warm starts at t_s = 1 (Sakajo bound per block)
        ts_ = 1.0
        h = 0.0004/float(nu_txt)/hdiv
        Tmax = 240.0/float(nu_txt)
        B = []
        for k in range(NB):
            lo = lo_edges[k]
            hi_ = edges[k+1] if k + 1 < len(edges) and k < NB - 1 else None
            v = None
            if k < NB - 1:
                for m in range(lo + 1, edges[k+1] + 1) if k+1 < len(edges) else []:
                    w = (arb(ts_)/2)**(m - 1)
                    v = w if v is None else v + w
                if v is None: v = arb(0)
            else:
                v = (arb(ts_)/2)**lo_edges[k]/(1 - arb(ts_)/2)
            B.append(v)
        t = ts_
        ok = None
        ncheck = int(3.0/float(nu_txt)/h)
        kstep = 0
        CEIL = 200.0
        def fvec(Bv, Rc, Pc):
            out = []
            Sacc = arb(0)
            for k in range(NB):
                if k < NB - 1:
                    fk = -Tk[k]*Bv[k] + (Rc/2 if k == 0 else arb(0)) + Pc*(Bv[k] + Sacc) + Sacc*Sacc/2
                else:
                    Sall = Sacc + Bv[k]
                    fk = -Tk[k]*Bv[k] + Pc*Sall + Sall*Sall/2
                out.append(fk)
                Sacc = Sacc + Bv[k]
            return out
        # certified per-cell DUHAMEL march (Lawson-style; unconditionally
        # stable in h -- the stiff last block has T*h > 1, where explicit
        # slope marching cannot close; S10 stiffness fix). On each cell:
        # Y_k' <= -T_k Y_k + G_k(t) with G_k <= Gbar_k (cell sup, using the
        # other blocks' cell hulls) implies, by exact linear comparison,
        # Y_k(end) <= E_k Y_k(a) + Gbar_k (1-E_k)/T_k. Monotone in all
        # inputs; a 3-pass hull iteration + final verification closes it.
        Ek = [(-Tk[k]*arb(h)).exp() for k in range(NB)]
        Gk = [(1 - Ek[k])/Tk[k] for k in range(NB)]
        def gvec(Bv, Rc, Pc):
            out = []
            Sacc = arb(0)
            for k in range(NB):
                if k < NB - 1:
                    gk = (Rc/2 if k == 0 else arb(0)) + Pc*(Bv[k] + Sacc) + Sacc*Sacc/2
                else:
                    Sall = Sacc + Bv[k]
                    gk = Pc*Sall + Sall*Sall/2
                out.append(gk)
                Sacc = Sacc + Bv[k]
            return out
        while t < Tmax:
            Rc = cellsup(Rl, t, h)
            Pc = cellsup(Pl, t, h)
            H = list(B)
            Bend = None
            for _ in range(3):
                G = gvec(H, Rc, Pc)
                Bend = [Ek[k]*B[k] + G[k]*Gk[k] for k in range(NB)]
                H = [B[k].union(Bend[k]) for k in range(NB)]
            # verification: with the final hulls H, the forcing bound must
            # reproduce (or exceed) the Bend actually used; iterate if not.
            closed = False
            for _ in range(40):
                G2 = gvec(H, Rc, Pc)
                Bend2 = [Ek[k]*B[k] + G2[k]*Gk[k] for k in range(NB)]
                oks = [float(Bend2[k].upper()) <= float(Bend[k].upper()) for k in range(NB)]
                if all(oks):
                    closed = True
                    break
                Bend = [Bend[k].union(Bend2[k]) for k in range(NB)]
                H = [B[k].union(Bend[k]) for k in range(NB)]
            if not closed:
                ok = False
                print("  nu=%-5s: Duhamel hull failed to close at t=%.4f (h=%.2e) -> fail"
                      % (nu_txt, t, h))
                break
            B = Bend
            for k in range(NB):
                if float(B[k].lower()) < 0: B[k] = arb(0)
            t += h
            kstep += 1
            if max(float(B[k].upper()) for k in range(NB)) > CEIL:
                ok = False
                print("  nu=%-5s: barrier exceeded ceiling (%s) at t=%.1f -> fail  [%.0fs]"
                      % (nu_txt, " ".join("B%d=%.3g" % (k+1, float(B[k].upper())) for k in range(NB)),
                         t, time.time() - t0))
                break
            if kstep % ncheck == 0:
                Pt = tailsup(Pl, t); Rt = tailsup(Rl, t)
                if not (Pt < Tk[0]):
                    continue
                # static system blockwise: linear blocks upward, then the last basin
                Ss = arb(0)
                Bstat = []
                feasible = True
                for k in range(NB - 1):
                    if not (Pt < Tk[k]):
                        feasible = False; break
                    Bk = ((Rt/2 if k == 0 else arb(0)) + Pt*Ss + Ss*Ss/2)/(Tk[k] - Pt)
                    Bk = Bk if float(Bk.upper()) > float(B[k].upper()) else B[k]
                    Bstat.append(Bk)
                    Ss = Ss + Bk
                if not feasible:
                    continue
                if Tk[NB-1] > Pt + Ss:
                    disc = (Tk[NB-1] - Pt - Ss)**2 - (2*Pt*Ss + Ss*Ss)
                    if disc > 0:
                        Bp = (Tk[NB-1] - Pt - Ss) + disc.sqrt()
                        if B[NB-1] < Bp:
                            ok = True
                            print("  nu=%-5s: CERTIFIED — landed at T=%.1f (%s; Blast=%.3g<B+=%.3g; Ptail=%.4f)  [%.0fs]"
                                  % (nu_txt, t,
                                     " ".join("B%d<=%.3g" % (k+1, float(Bstat[k].upper())) for k in range(NB-1)),
                                     float(B[NB-1].upper()), float(Bp.lower()),
                                     float(Pt.upper()), time.time() - t0))
                            break
        if ok is None:
            print("  nu=%-5s: no landing by Tmax=%.0f (%s) -> fail  [%.0fs]"
                  % (nu_txt, Tmax, " ".join("B%d=%.3g" % (k+1, float(B[k].upper())) for k in range(NB)),
                     time.time() - t0))
        if ok is False or ok is None:
            print("  [stops closing; smallest CERTIFIED nu above is the bound]")
            break

def stage_dec5(mP=14, NB=4, hdiv=4, prec=160, NT=9,
               nus=("0.064", "0.062", "0.06", "0.058", "0.056"), ceil=200.0):
    """V6 decay certificate (S11) = dec4's cascade + march with the DEEP-TAYLOR
    cell-sup evaluator. S11 DIAGNOSIS (scratch dec4_diag.py, on the record in
    the session log): dec4's mP=14 nu=0.062 'fail at t~1.1' was a PESSIMISM
    ARTIFACT of the order-3 cell-sup remainder -- the H^3/6 * Sigma|c| mu^3
    term evaluates to 5.9e3 against a TRUE forcing R(t~1.05) = 2.6e-4 (the
    exponential-sum coefficients at low exponents k~37..46 carry |c| ~ 8e11
    with ~15 orders of pointwise cancellation, and e^{-mu t} ~ 0.05 there
    gives no suppression). The phantom B1 equilibrium (Rc/2)/(T1-P) crosses
    the CEIL=200 ceiling at nu=0.062 (~275) but not at 0.064 (~107) -- hence
    the deterministic, precision-independent nu-cliff (prec-320 value-identical,
    S10). FIX: Taylor the cell-sup to order NT with SIGNED evaluations at all
    orders below the remainder; each order multiplies the |c|-weighted term by
    ~h*mu_eff ~ 5e-3. Single-pass per term (one exp, rolling powers) -- also
    faster than the old 4-pass evaluator. Soundness: sup_{s in [0,h]} |f(t+s)|
    <= sum_{n<=NT} h^n/n! |f^(n)(t)| + h^{NT+1}/(NT+1)! sup_cell |f^(NT+1)|,
    with sup_cell |f^(NT+1)| <= Sigma |c| mu^{NT+1} e^{-mu t} (mu > 0).
    PREDICTED (before first run): Rc(mP=14, 0.062, t~1.05) drops 5.9e3 ->
    <= 1e-2; mP=14 h/4 CERTIFIES 0.062 (the genuine mP=12 transient defeat at
    0.062 is relieved by the smaller true R at mP=14: 1.5e-4 vs 8.8e-4, and
    T1 = 225nu vs 169nu); 0.060 likely still fails on the GENUINE transient
    (expect all-blocks ceiling breach near t~5-6); if 0.060 certifies, run
    mP=16. Everything else is dec4 verbatim (same march, same obligations,
    same landing logic)."""
    from flint import arb, ctx
    ctx.prec = prec
    edges = [mP]
    for _ in range(NB - 1):
        edges.append(2*edges[-1] + 1)
    print("stage dec5: %d-block cascade, exact modes 1..%d, block edges %s, dissipations %s x nu, NT=%d"
          % (NB, mP, edges, [(edges[k]+1)**2 for k in range(NB)], NT))
    for nu_txt in nus:
        nu = arb(nu_txt)
        t0 = time.time()
        q = {1: [(arb(1), 1)]}
        for m in range(2, mP + 1):
            F = {}
            for j in range(1, m):
                for (c1, k1) in q[j]:
                    for (c2, k2) in q[m - j]:
                        key = k1 + k2
                        F[key] = F.get(key) + c1*c2/2 if key in F else c1*c2/2
            q[m] = _duhamel([(c, k) for (k, c) in F.items()], m*m, nu)
        Pterms = {}
        for m in range(1, mP + 1):
            for (c, k) in q[m]:
                Pterms[k] = Pterms.get(k) + c if k in Pterms else c
        Rterms = {}
        for j in range(1, mP + 1):
            for l in range(1, mP + 1):
                if j + l > mP:
                    for (c1, k1) in q[j]:
                        for (c2, k2) in q[l]:
                            key = k1 + k2
                            Rterms[key] = Rterms.get(key) + c1*c2 if key in Rterms else c1*c2
        Pl = [(c, k) for (k, c) in Pterms.items()]
        Rl = [(c, k) for (k, c) in Rterms.items()]
        lo_edges = [mP] + edges[1:NB]
        Tk = [((lo_edges[k] + 1)**2)*nu for k in range(NB)]
        fact = [arb(1)]
        for n in range(1, NT + 2):
            fact.append(fact[-1]*n)
        def cellsup(terms, ta, h):
            """Deep-Taylor cell sup (see docstring). Single pass over terms."""
            H = arb(h)
            Hp = [arb(1)]
            for n in range(1, NT + 2):
                Hp.append(Hp[-1]*H)
            S = [arb(0) for _ in range(NT + 1)]   # signed f^(n)(ta)
            rem = arb(0)                           # Sigma |c| mu^{NT+1} e^{-mu ta}
            for (c, k) in terms:
                mu = k*nu
                e = (-(mu*arb(ta))).exp()
                ce = c*e
                p = arb(1)
                for n in range(NT + 1):
                    S[n] = S[n] + ce*p
                    p = p*(-mu)
                rem = rem + abs(c)*e*(mu**(NT + 1))
            v = arb(0)
            for n in range(NT + 1):
                v = v + abs(S[n])*Hp[n]/fact[n]
            return v + rem*Hp[NT + 1]/fact[NT + 1]
        def tailsup(terms, T, ngrid=800):
            span = 80.0/float(nu_txt)
            tsg = [T + span*k*k/(ngrid*ngrid) for k in range(ngrid + 1)]
            best = None
            for j in range(ngrid):
                cell = cellsup(terms, tsg[j], tsg[j+1] - tsg[j])
                if best is None or float(cell.upper()) > float(best.upper()):
                    best = cell
            tl = None
            for (c, k) in terms:
                e = abs(c)*((-(k*nu*arb(tsg[-1]))).exp())
                tl = e if tl is None else tl + e
            return best if float(best.upper()) >= float(tl.upper()) else tl
        ts_ = 1.0
        h = 0.0004/float(nu_txt)/hdiv
        Tmax = 240.0/float(nu_txt)
        B = []
        for k in range(NB):
            v = None
            if k < NB - 1:
                for m in range(lo_edges[k] + 1, edges[k+1] + 1):
                    w = (arb(ts_)/2)**(m - 1)
                    v = w if v is None else v + w
                if v is None: v = arb(0)
            else:
                v = (arb(ts_)/2)**lo_edges[k]/(1 - arb(ts_)/2)
            B.append(v)
        t = ts_
        ok = None
        ncheck = int(3.0/float(nu_txt)/h)
        kstep = 0
        CEIL = ceil
        Ek = [(-Tk[k]*arb(h)).exp() for k in range(NB)]
        Gk = [(1 - Ek[k])/Tk[k] for k in range(NB)]
        def gvec(Bv, Rc, Pc):
            out = []
            Sacc = arb(0)
            for k in range(NB):
                if k < NB - 1:
                    gk = (Rc/2 if k == 0 else arb(0)) + Pc*(Bv[k] + Sacc) + Sacc*Sacc/2
                else:
                    Sall = Sacc + Bv[k]
                    gk = Pc*Sall + Sall*Sall/2
                out.append(gk)
                Sacc = Sacc + Bv[k]
            return out
        while t < Tmax:
            Rc = cellsup(Rl, t, h)
            Pc = cellsup(Pl, t, h)
            H = list(B)
            Bend = None
            for _ in range(3):
                G = gvec(H, Rc, Pc)
                Bend = [Ek[k]*B[k] + G[k]*Gk[k] for k in range(NB)]
                H = [B[k].union(Bend[k]) for k in range(NB)]
            closed = False
            for _ in range(40):
                G2 = gvec(H, Rc, Pc)
                Bend2 = [Ek[k]*B[k] + G2[k]*Gk[k] for k in range(NB)]
                oks = [float(Bend2[k].upper()) <= float(Bend[k].upper()) for k in range(NB)]
                if all(oks):
                    closed = True
                    break
                Bend = [Bend[k].union(Bend2[k]) for k in range(NB)]
                H = [B[k].union(Bend[k]) for k in range(NB)]
            if not closed:
                ok = False
                print("  nu=%-5s: Duhamel hull failed to close at t=%.4f (h=%.2e) -> fail"
                      % (nu_txt, t, h))
                break
            B = Bend
            for k in range(NB):
                if float(B[k].lower()) < 0: B[k] = arb(0)
            t += h
            kstep += 1
            if max(float(B[k].upper()) for k in range(NB)) > CEIL:
                ok = False
                print("  nu=%-5s: barrier exceeded ceiling (%s) at t=%.1f -> fail  [%.0fs]"
                      % (nu_txt, " ".join("B%d=%.3g" % (k+1, float(B[k].upper())) for k in range(NB)),
                         t, time.time() - t0))
                break
            if kstep % ncheck == 0:
                Pt = tailsup(Pl, t); Rt = tailsup(Rl, t)
                if not (Pt < Tk[0]):
                    continue
                Ss = arb(0)
                Bstat = []
                feasible = True
                for k in range(NB - 1):
                    if not (Pt < Tk[k]):
                        feasible = False; break
                    Bk = ((Rt/2 if k == 0 else arb(0)) + Pt*Ss + Ss*Ss/2)/(Tk[k] - Pt)
                    Bk = Bk if float(Bk.upper()) > float(B[k].upper()) else B[k]
                    Bstat.append(Bk)
                    Ss = Ss + Bk
                if not feasible:
                    continue
                if Tk[NB-1] > Pt + Ss:
                    disc = (Tk[NB-1] - Pt - Ss)**2 - (2*Pt*Ss + Ss*Ss)
                    if disc > 0:
                        Bp = (Tk[NB-1] - Pt - Ss) + disc.sqrt()
                        if B[NB-1] < Bp:
                            ok = True
                            print("  nu=%-5s: CERTIFIED — landed at T=%.1f (%s; Blast=%.3g<B+=%.3g; Ptail=%.4f)  [%.0fs]"
                                  % (nu_txt, t,
                                     " ".join("B%d<=%.3g" % (k+1, float(Bstat[k].upper())) for k in range(NB-1)),
                                     float(B[NB-1].upper()), float(Bp.lower()),
                                     float(Pt.upper()), time.time() - t0))
                            break
        if ok is None:
            print("  nu=%-5s: no landing by Tmax=%.0f (%s) -> fail  [%.0fs]"
                  % (nu_txt, Tmax, " ".join("B%d=%.3g" % (k+1, float(B[k].upper())) for k in range(NB)),
                     time.time() - t0))
        if ok is False or ok is None:
            print("  [stops closing; smallest CERTIFIED nu above is the bound]")
            break

if __name__ == "__main__":
    st = sys.argv[1]
    if st.startswith("proto"):
        parts = st.split(":")
        stage_proto(float(parts[1]) if len(parts) > 1 else 0.04,
                    int(parts[2]) if len(parts) > 2 else 512)
    elif st.startswith("enc"):
        parts = st.split(":")
        stage_enc(k0=int(parts[1]) if len(parts) > 1 else 8,
                  alpha=float(parts[2]) if len(parts) > 2 else 0.01)
    elif st == "dec0":
        stage_dec0()
    elif st.startswith("dec1"):
        parts = st.split(":")
        stage_dec1(int(parts[1]) if len(parts) > 1 else 8)
    elif st.startswith("dec2"):
        parts = st.split(":")
        stage_dec2(int(parts[1]) if len(parts) > 1 else 8)
    elif st.startswith("dec3"):
        parts = st.split(":")
        stage_dec3(int(parts[1]) if len(parts) > 1 else 10)
    elif st.startswith("dec5"):
        parts = st.split(":")
        stage_dec5(int(parts[1]) if len(parts) > 1 else 14,
                   int(parts[2]) if len(parts) > 2 else 4,
                   int(parts[3]) if len(parts) > 3 else 4,
                   int(parts[4]) if len(parts) > 4 else 160,
                   int(parts[5]) if len(parts) > 5 else 9,
                   nus=tuple(parts[6].split(",")) if len(parts) > 6 else ("0.064", "0.062", "0.06", "0.058", "0.056"),
                   ceil=float(parts[7]) if len(parts) > 7 else 200.0)
    elif st.startswith("dec4"):
        parts = st.split(":")
        stage_dec4(int(parts[1]) if len(parts) > 1 else 10,
                   int(parts[2]) if len(parts) > 2 else 4,
                   int(parts[3]) if len(parts) > 3 else 1,
                   int(parts[4]) if len(parts) > 4 else 160)
    else: raise SystemExit("unknown stage: %s" % st)
