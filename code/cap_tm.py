#!/usr/bin/env python3
# cap_tm.py — Taylor-model certified integrator
# for the q-hierarchy (the blowup side of the two-sided threshold bracket).
#
#   q_m' = -nu m^2 q_m + F_m,  F_m = (1/2) sum_{j<m} q_j q_{m-j},  q_1(0)=1,
#   q_m(0)=0 (m>=2), all q_m >= 0.
#
# Per step [a, a+h]: each mode is a degree-p Taylor polynomial in tau = t-a
# with midpoint-radius interval coefficients, plus a remainder bound rho_m.
# Coefficient recursion (exact, includes the stiff linear term):
#   a_{i+1} = (-nu m^2 a_i + F_i)/(i+1),  F_i = (1/2) sum_{i1+i2=i} conv(A_i1, A_i2)
# Remainder (rigorous, forward-substitution — no iteration, by triangularity):
#   defect D_m = sum of tau^p..tau^{2p} coefficients of (T' + nu m^2 T - F(T)),
#   bounded by |coef| h^i sums; then
#   rho >= h * ( conv(S, rho) + (1/2) conv(rho, rho) + D ) * CI
#   with S_m = sup-step |T_m| <= sum_i |a_i| h^i, solved exactly in one forward
#   pass (rho_m depends only on rho_{j<m}; the rho^2 term uses lower modes only),
#   then verified a posteriori with outward rounding.
# Soundness inductive in m; stiffness cap h <= (p+1)/(2 nu M^2); high order
# makes the per-step relative injection ~ (lambda h)^{p+1} — this is what beats
# the S08 conditioning wall (amplification ~ S(t1)/S(s) ~ 1e7).
# FP rigor: mid ops in float64 round-to-nearest; after every composite op,
# rad += RND*|mid| with RND = 1e-13 (covers Higham gamma_n, n <= 512, margin x1.7).
# Stages:  tmv | tmrun[:margin] | excv1[:M] | exc:<nu_num>:<nu_den>[:M[:mS[:tlmin[:budget[:jbump]]]]]
# S13: excursion-tracking certified decay barrier added (design frozen S12,
# excursion-barrier-design.md): stage exc marches THROUGH the near-threshold
# excursion with a certified tail block and lands on a static absorbing set.
# Additive only — tm_step's returned per-step sup now includes rho (consumed
# solely by the new stages; tmv re-verified bit-identical post-change).
import os, sys, time
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
RND = 1e-13
ETA = 1e-290

def _convmr(am, ar, bm, br):
    """Midpoint-radius interval convolution (vector over modes, index = m-2):
    result of sum_{j} A_j B_{m-j}. Returns (mid, rad)."""
    mm = np.convolve(am, bm)
    rr = (np.convolve(np.abs(am), br) + np.convolve(ar, np.abs(bm))
          + np.convolve(ar, br) + RND*np.convolve(np.abs(am), np.abs(bm)) + ETA)
    return mm, rr

_TABS = {}

def _ktables(nu_num, nu_den, j, M, pmax):
    """Certified tables for h = 2^-j: E = e^{-nu m^2 h} and kernels
    K_i = int_0^h e^{-nu m^2 (h-s)} s^i ds, i = 0..pmax, via the recursion
    K_i = (h^i - i K_{i-1})/(nu m^2), K_0 = (1-E)/(nu m^2), evaluated in arb
    (handles the small-x cancellation), outward float64. All K_i > 0."""
    from flint import arb, ctx
    ctx.prec = 120
    h = 2.0**(-j)
    E_lo = np.empty(M); E_hi = np.empty(M)
    K_lo = np.empty((pmax + 1, M)); K_hi = np.empty((pmax + 1, M))
    for m in range(1, M + 1):
        lam = arb(nu_num*m*m)/arb(nu_den)
        x = lam*arb(h)
        b = (-x).exp()
        E_lo[m-1] = max(np.nextafter(float(b.lower()), -np.inf), 0.0)
        E_hi[m-1] = np.nextafter(float(b.upper()), np.inf)
        K = (1 - b)/lam
        K_lo[0, m-1] = max(np.nextafter(float(K.lower()), -np.inf), 0.0)
        K_hi[0, m-1] = np.nextafter(float(K.upper()), np.inf)
        hp = arb(h)
        for i in range(1, pmax + 1):
            K = (hp - i*K)/lam
            hp = hp*arb(h)
            K_lo[i, m-1] = max(np.nextafter(float(K.lower()), -np.inf), 0.0)
            K_hi[i, m-1] = np.nextafter(float(K.upper()), np.inf)
    return E_lo, E_hi, K_lo, K_hi

def tm_step(qm, qr, nu_num, nu_den, M, j, p):
    """One certified Lawson-Taylor step, h = 2^-j.
    Levels b_i (scaled a_i h^i) from the FULL recursion (stiff term included:
    within-step radii inflation is a bounded constant, ~e^{x_max}, applied only
    to the small nonlinear increment). Endpoint via the kernel form:
      q(h) in E*[q(a)] + sum_i (F_i h^-i)*K_i  +/- (radii + rho),
    with K_i > 0 certified; the state radius contracts by E and gains only
    O((lambda h)) small terms — no e^{+x} inflation of the state.
    Lower-hull fact used downstream: N-term >= 0 for the true solution."""
    nu = nu_num/nu_den
    h = 2.0**(-j)
    key = (j, M, p)
    if key not in _TABS:
        _TABS[key] = _ktables(nu_num, nu_den, j, M, 2*p + 1)
    E_lo, E_hi, K_lo, K_hi = _TABS[key]
    m2h = nu*h*(np.arange(1, M + 1)**2).astype(float)
    Bm = [qm.copy()]; Br = [qr.copy()]
    Fms = []; Frs = []
    for i in range(2*p + 1):
        fm = np.zeros(2*M - 1); fr = np.zeros(2*M - 1)
        for i1 in range(max(0, i - p), min(i, p) + 1):
            cm, cr = _convmr(Bm[i1], Br[i1], Bm[i - i1], Br[i - i1])
            fm += 0.5*cm; fr += 0.5*cr + RND*np.abs(0.5*cm)
        Fm = np.zeros(M); Fr = np.zeros(M)
        Fm[1:] = fm[:M - 1]; Fr[1:] = fr[:M - 1]
        Fms.append(Fm); Frs.append(Fr)
        if i <= p:
            nm = (-m2h*Bm[i] + h*Fm)/(i + 1)
            nr = (m2h*Br[i] + h*Fr)/(i + 1) + RND*np.abs(nm) + ETA
            if not (np.isfinite(nm).all() and np.isfinite(nr).all()):
                return None
            Bm.append(nm); Br.append(nr)
    # step sup S: sup L <= qm+qr; sup N <= sum_i (F_i^+ + Fr_i) K_i(h)/h^i
    # (kernels increasing in tau; scaled: F-levels carry h^i via b-scaling, and
    # K_i(h)/h^i <= h/(i+1) <= h; use K_hi/h^i exactly).
    S = qm + qr
    for i in range(p + 1):
        S = S + (np.maximum(Fms[i], 0.0) + Frs[i])*(K_hi[i]/h**i)
    # rho: fixed point  rho >= K0_hi*(conv(S,rho) + conv(rho,rho)/2) + Dker
    # with Dker = kernel-weighted defect: levels p+1..2p of F (never matched)
    # plus the level cut (p+1)|b_{p+1}| entering through K_p-scale <= h.
    Dker = (p + 1)*(np.abs(Bm[p + 1]) + Br[p + 1])
    for i in range(p + 1, 2*p + 1):
        Dker = Dker + (np.abs(Fms[i]) + Frs[i])*(K_hi[min(i, 2*p + 1)]/h**i if False else h)
    Dker = Dker*(1 + RND) + ETA
    K0h = K_hi[0]
    rho = np.zeros(M)
    for mI in range(1, M):
        acc = Dker[mI]
        acc += np.dot(S[:mI], rho[mI - 1::-1][:mI])*K0h[mI]
        acc += 0.5*np.dot(rho[:mI], rho[mI - 1::-1][:mI])*K0h[mI]
        rho[mI] = acc*(1 + 3e-12) + ETA
    convS = np.zeros(M); convR = np.zeros(M)
    cs = np.convolve(S, rho); cr2 = np.convolve(rho, rho)
    convS[1:] = cs[:M - 1]; convR[1:] = cr2[:M - 1]
    need = K0h*(convS + 0.5*convR)*(1 + 5e-13) + Dker*(1 + 5e-13) + ETA
    if not (np.isfinite(rho).all() and np.all(rho[1:] >= need[1:])):
        return None
    # endpoint: E*[qa] + sum_i F_i K_i(h)/h^i  +/- (F radii * K + rho)
    Nm = np.zeros(M); Nr = np.zeros(M)
    for i in range(p + 1):
        w_lo = K_lo[i]/h**i; w_hi = K_hi[i]/h**i
        Nm = Nm + Fms[i]*0.5*(w_lo + w_hi)
        Nr = Nr + np.abs(Fms[i])*0.5*(w_hi - w_lo) + Frs[i]*w_hi + RND*np.abs(Nm)
    qm2 = E_lo*qm*0.5 + E_hi*qm*0.5 + Nm
    qr2 = (E_hi - E_lo)*0.5*np.abs(qm) + E_hi*qr + Nr + rho + RND*np.abs(qm2) + ETA
    if not (np.isfinite(qm2).all() and np.isfinite(qr2).all()):
        return None
    # S13 (excursion barrier): the returned per-step sup must bound the TRUE
    # solution q = T + e over the whole step, so add the remainder rho (valid
    # on all of [0,h] by the K0-monotone Picard bound above). Additive-safe:
    # no existing stage consumes the returned S (tmrun's seed_cb ignores it);
    # the rho fixed-point above still uses the T-only S internally.
    return np.maximum(qm2, 0.0), qr2, S + rho

def sweep(nu_num, nu_den, M, p, t_end, alphaT=0.35, progress=None, seed_cb=None, xmax=2.0):
    """Certified Lawson-TM sweep; dyadic h = 2^-j chosen from
    j >= log2(max(nu M^2/xmax, Smid/alphaT)); tables cached per j."""
    nu = nu_num/nu_den
    jstiff = int(np.ceil(np.log2(max(nu*M*M/xmax, 1.0))))
    qm = np.zeros(M); qm[0] = 1.0
    qr = np.zeros(M)
    t = 0.0
    nstep = 0
    t0 = time.time()
    while t < t_end - 2.0**-46:
        Smid = np.sum(qm)
        j = max(jstiff, int(np.ceil(np.log2(max(Smid/alphaT, 1.0)))))
        rem = t_end - t
        if 2.0**-j > rem:
            j = int(np.ceil(-np.log2(max(rem, 2.0**-49))))
        j = min(j, 50)
        r = tm_step(qm, qr, nu_num, nu_den, M, j, p)
        tries = 0
        while r is None and tries < 8 and j < 50:
            j += 1; r = tm_step(qm, qr, nu_num, nu_den, M, j, p); tries += 1
        if r is None:
            print("  TM FAIL at t=%.6f" % t); return None
        qm, qr, S_hi = r
        h = 2.0**(-j)
        t += h
        nstep += 1
        if progress and nstep % progress == 0:
            live = qm > 1e-8*max(qm.max(), 1e-300)
            print("    [t=%.5f  Sig~%.4g  relw(live)=%.2e  h=2^-%d  %.0fs]"
                  % (t, np.sum(qm), np.max(qr[live]/np.maximum(qm[live], 1e-300)), j,
                     time.time() - t0))
        if seed_cb is not None:
            st = seed_cb(t, h, qm, qr, S_hi)
            if st:
                return st
    return (t, qm, qr)

# ---------------- S13: excursion-tracking certified decay barrier ----------
# Design: excursion-barrier-design.md (frozen S12). March the true solution's
# enclosure THROUGH the near-threshold excursion (modes 1..M are the true
# modes by triangularity — no truncation error), carry a certified scalar
# bound Ybar(t) >= sum_{m>M} q_m (the tail block, a PROOF component), then
# verify a certified LANDING: from t_land on, a static absorbing-set argument
# finishes global decay. Proof obligations printed per the S10 standing rule.
#
# Tail-block lemma (per step [t, t+h], all q_m >= 0):
#   Y := sum_{m>M} q_m obeys Y' <= -lamT*Y + P(s)*Y + Y^2/2 + R(s)/2,
#   lamT = nu (M+1)^2, P(s) = sum_{m<=M} q_m <= P_step := sum_m S_m,
#   R(s) = sum_{j+l>M; j,l<=M} q_j q_l <= R_step := sum_{j+l>M} S_j S_l,
#   with S the per-step certified sup (tm_step's returned S, includes rho).
#   Hull H verified via (lamT - P - H/2) H >= g := R_step/2  ==> [0,H]
#   invariant; then Y(t+h) <= Y(t)/(1 + a2 h) + g h with
#   a2 = lamT - P - H/2 >= 0  (e^{-x} <= 1/(1+x), (1-e^{-x})/x <= 1).
#   Y(0) = 0 exactly (data has no modes above M).
#
# Landing lemma (split mS; certified in arb at t_land): with
#   Pm[1] = qhi_1(tl)  (q_1(s) = q_1(tl) e^{-nu(s-tl)}, decreasing),
#   Pm[m] = qhi_m(tl) + (1/2) sum_{j<m} Pm[j] Pm[m-j] / (nu m^2)   (m <= mS)
# (triangular Duhamel sup chain, induction in m on the closed cooperative
# subsystem), the low block obeys sup_{s>=tl} sum_{m<=mS} q_m <= Pbar and
# the high block Yhi = sum_{m>mS} q_m obeys the Riccati inequality with
# lamS = nu (mS+1)^2, forcing Rbar = sum_{j+l>mS; j,l<=mS} Pm[j] Pm[l]:
#   OBLIGATIONS: (O2) Pbar < lamS; (O3) exists H: (lamS - Pbar - H/2) H
#   > Rbar/2; (O4) Yhi(tl) <= sum_{mS<m<=M}(qm+qr) + Ybar(tl) < H.
#   ==> [0,H] invariant for Yhi, linear rate a = lamS - Pbar - H/2 > 0 gives
#   limsup Yhi <= Rbar/(2a); restarting at later tl (q_1 -> 0 exactly) sends
#   Pbar, Rbar -> 0, hence Yhi -> 0 and every mode -> 0: GLOBAL DECAY.
#   By the nu-comparison lemma (N18a) the certificate covers all nu' >= nu.
# Float-nu note: tm_step's level recursion uses float nu (1-ulp, 1.1e-17 rel)
# inside mid-rad arithmetic whose per-op RND = 1e-13 inflation dominates it by
# 4 orders; kernels/E use exact-rational arb nu. Same argument as the Higham
# coverage in the header; landing checks use exact-rational arb throughout.

def _tail_update(Y, S_step, h, lamT, M):
    """One certified tail-block step. Returns (Ynew, P_step, R_step) or
    (None, P, R) if the hull cannot close (F3: front pressing the truncation).
    Outward float on positive quantities (nudges cover round-nearest sums)."""
    eps = 1e-12
    P = float(np.sum(S_step))*(1 + eps)
    cs = np.convolve(S_step, S_step)            # index k <-> mode m = k+2
    R = float(np.sum(cs[M - 1:]))*(1 + eps)     # m > M  <=>  k >= M-1
    g = 0.5*R
    H = max(2.0*Y, 1e-280)
    ok = False
    for _ in range(240):
        a2 = lamT*(1 - eps) - P - 0.5*H
        if a2 > 0.0 and a2*H >= g*(1 + eps):
            ok = True
            break
        H *= 4.0
    if not ok:
        return None, P, R
    Ynew = (Y/(1.0 + a2*h))*(1 + 3e-13) + g*h*(1 + 3e-13) + 1e-290
    return Ynew, P, R

def _land_check(nu_num, nu_den, mS, M, qm, qr, Ytail):
    """Certified landing check (arb ball arithmetic; strict comparisons are
    certified — True only when provable). Returns dict with obligations."""
    from flint import arb, ctx
    ctx.prec = 120
    nu_a = arb(nu_num)/arb(nu_den)
    qhi = [arb(float(qm[i])) + arb(float(qr[i])) for i in range(mS)]
    Pm = [None]*(mS + 1)
    Pm[1] = qhi[0]
    for m in range(2, mS + 1):
        F = arb(0)
        for j in range(1, m):
            F += Pm[j]*Pm[m - j]
        Pm[m] = qhi[m - 1] + (F/2)/(nu_a*(m*m))
    Pbar = arb(0)
    for m in range(1, mS + 1):
        Pbar += Pm[m]
    lamS = nu_a*((mS + 1)**2)
    o2 = bool(Pbar < lamS)
    Rbar = arb(0)
    for j in range(1, mS + 1):
        for l in range(1, mS + 1):
            if j + l > mS:
                Rbar += Pm[j]*Pm[l]
    Yt_f = float(np.sum(qm[mS:] + qr[mS:]))*(1 + 1e-12) + Ytail
    Yt = arb(Yt_f)
    best = None
    if o2:
        cands = [Yt_f*4 + 1e-9, 0.125, 0.25, 0.5, float((lamS - Pbar).mid()), 1.0, 1.5]
        for Hf in cands:
            if Hf <= 0:
                continue
            H = arb(Hf)
            lhs = (lamS - Pbar - H/2)*H
            rhs = Rbar/2
            if bool(Yt < H) and bool(lhs > rhs):
                mg = float((lhs/rhs).lower()) if float(rhs.upper()) > 0 else float("inf")
                if best is None or mg > best[1]:
                    best = (Hf, mg)
    return dict(ok=(best is not None), o2=o2,
                Pbar=float(Pbar.upper()), lamS=float(lamS.lower()),
                Rbar=float(Rbar.upper()), Yt=Yt_f,
                H=(best[0] if best else 0.0), margin=(best[1] if best else 0.0),
                Pm=[float(Pm[m].upper()) for m in range(1, mS + 1)])

def stage_exc(nu_num, nu_den, M=384, mS=4, tl_min=20.0, t_budget=40.0,
              jbump=0, p=6, land_every=64):
    """The excursion-tracking certified decay run: march + tail block +
    landing. On success prints the theorem line and saves the certificate."""
    nu = nu_num/nu_den
    lamT = nu*((M + 1)**2)*(1 - 1e-13)
    print("stage exc: nu=%d/%d=%.6f M=%d mS=%d tl_min=%.1f budget=%.1f jbump=%d"
          % (nu_num, nu_den, nu, M, mS, tl_min, t_budget, jbump))
    st = dict(Y=0.0, Ymax=0.0, relwmax=0.0, Sigpk=0.0, tpk=0.0, nstep=0,
              nland=0, fail=None, land=None, Pmax=0.0)
    t0 = time.time()
    def cb(t, h, qm, qr, S_hi):
        st["nstep"] += 1
        Yn, P, R = _tail_update(st["Y"], S_hi, h, lamT, M)
        if Yn is None:
            st["fail"] = ("F3-tail-hull", t)
            print("  [F3] tail hull cannot close at t=%.4f (P=%.3g R=%.3g) — raise M"
                  % (t, P, R))
            return ("FAIL", t)
        st["Y"] = Yn
        st["Ymax"] = max(st["Ymax"], Yn)
        st["Pmax"] = max(st["Pmax"], P)
        Sig = float(np.sum(qm))
        if Sig > st["Sigpk"]:
            st["Sigpk"], st["tpk"] = Sig, t
        live = qm > 1e-8*max(qm.max(), 1e-300)
        relw = float(np.max(qr[live]/np.maximum(qm[live], 1e-300))) if live.any() else 0.0
        st["relwmax"] = max(st["relwmax"], relw)
        if relw > 1e-4:
            st["fail"] = ("F1-width", t)
            print("  [F1] enclosure width blowup at t=%.4f (relw=%.2e)" % (t, relw))
            return ("FAIL", t)
        if t >= tl_min and st["nstep"] % land_every == 0:
            st["nland"] += 1
            L = _land_check(nu_num, nu_den, mS, M, qm, qr, st["Y"])
            # land at the first attempt with COMFORTABLE margin (>= 3), not
            # the thinnest passing one — margins only grow with t (q_1 decays)
            if L["ok"] and L["margin"] >= 3.0:
                st["land"] = (t, L)
                return ("LANDED", t)
        if t > t_budget:
            st["fail"] = ("F2-budget", t)
            print("  [F2] no landing by t=%.2f — budget exhausted" % t)
            return ("FAIL", t)
        return None
    r = sweep(nu_num, nu_den, M, p, t_budget + 1.0, alphaT=0.35,
              progress=5000, seed_cb=cb, xmax=2.0/2**jbump)
    el = time.time() - t0
    print("  march: %d steps, %.0f s; Sigma_peak(march) = %.4f at t=%.4f; "
          "max relw(live) = %.2e; tail block Ybar_max = %.3e, P_max = %.4f"
          % (st["nstep"], el, st["Sigpk"], st["tpk"], st["relwmax"],
             st["Ymax"], st["Pmax"]))
    if r is None:
        print("  RESULT: TM FAIL (integrator width/step failure) — no certificate")
        return st
    if isinstance(r, tuple) and r[0] == "LANDED":
        tl, L = st["land"]
        print("  LANDING at t = %.5f (attempt %d): obligations" % (tl, st["nland"]))
        print("    (O2) Pbar = %.6f < lamS = %.6f : %s   [Pm: %s]"
              % (L["Pbar"], L["lamS"], L["o2"],
                 " ".join("%.4f" % x for x in L["Pm"])))
        print("    (O3) hull H = %.4f with margin (lhs/rhs) = %.2f : True" % (L["H"], L["margin"]))
        print("    (O4) Y_tot(tl) = %.3e < H : True   (tail-block share %.1e)"
              % (L["Yt"], st["Y"]))
        print("  *** DECAY CERTIFIED (march-through-excursion + certified landing) ***")
        print("  ==> THEOREM: sigma=2 dissipative CLM on S^1 from cos x decays")
        print("      globally for nu = %d/%d = %.6f (and every nu' >= nu by the" % (nu_num, nu_den, nu))
        print("      nu-comparison lemma)  ==> nu*(0) <= %.6f" % nu)
        fn = os.path.join(DIR, "exc_cert_%d_%d_M%d.npz" % (nu_num, nu_den, M))
        np.savez(fn, nu_num=nu_num, nu_den=nu_den, M=M, mS=mS, p=p,
                 t_land=tl, Pbar=L["Pbar"], lamS=L["lamS"], Rbar=L["Rbar"],
                 Ytot=L["Yt"], H=L["H"], margin=L["margin"], Pm=L["Pm"],
                 Ybar_max=st["Ymax"], relw_max=st["relwmax"],
                 Sig_peak=st["Sigpk"], t_peak=st["tpk"], nstep=st["nstep"],
                 jbump=jbump)
        print("  certificate saved: %s" % os.path.basename(fn))
        return st
    print("  RESULT: %s at t=%.4f — honest failure, no certificate"
          % (st["fail"][0] if st["fail"] else "unknown", st["fail"][1] if st["fail"] else -1))
    return st

def _lawson_ref(nu, M, T, alpha, dt_max):
    """Float Lawson-IFRK4 reference landing on T EXACTLY (q_lawson's kernel;
    its every-25-step recording misses the endpoint by up to ~25*dt, which
    would swamp the tight enclosure radii — the S13 V1 diagnosis)."""
    import cap_hier as ch
    m2 = (np.arange(1, M + 1)**2).astype(float)
    q = np.zeros(M); q[0] = 1.0
    t = 0.0
    dt_prev = -1.0; E1 = E2 = None
    while T - t > 1e-12:
        S = np.sum(q)
        dt = min(dt_max, alpha/max(S, 1e-12), T - t)
        if dt != dt_prev:
            E1 = np.exp(-nu*m2*dt); E2 = np.exp(-nu*m2*dt/2.0)
            dt_prev = dt
        k1 = ch.NL(q)
        U2 = E2*(q + 0.5*dt*k1)
        k2 = ch.NL(U2)
        U3 = E2*q + 0.5*dt*k2
        k3 = ch.NL(U3)
        U4 = E1*q + dt*E2*k3
        k4 = ch.NL(U4)
        q = E1*q + (dt/6.0)*(E1*k1 + 2.0*E2*(k2 + k3) + k4)
        t += dt
    return q

def stage_excv1(M=256):
    """Excursion-barrier validation V1: containment of a KNOWN decay
    trajectory (nu = 7/100 through its excursion) + tail-block exercise.
    Floors registered in"""
    print("stage excv1: containment at nu=0.07, M=%d, to t=12 exactly" % M); t0 = time.time()
    import cap_hier as ch
    tf = 12.0
    qf = _lawson_ref(0.07, M, tf, 0.01, 0.002)
    nu = 0.07
    lamT = nu*((M + 1)**2)*(1 - 1e-13)
    st = dict(Y=0.0, Ymax=0.0, Sigpk=0.0, tpk=0.0)
    def cb(t, h, qm, qr, S_hi):
        Yn, P, R = _tail_update(st["Y"], S_hi, h, lamT, M)
        if Yn is None:
            return ("FAIL", t)
        st["Y"] = Yn
        st["Ymax"] = max(st["Ymax"], Yn)
        Sig = float(np.sum(qm))
        if Sig > st["Sigpk"]:
            st["Sigpk"], st["tpk"] = Sig, t
        return None
    r = sweep(7, 100, M, 6, tf, alphaT=0.35, progress=10000, seed_cb=cb)
    if r is None or isinstance(r, tuple) and r and r[0] == "FAIL":
        print("  V1 FAIL: march did not complete")
        return
    t, qm, qr = r
    np.savez(os.path.join(DIR, "s13_excv1_state.npz"), t=t, qm=qm, qr=qr,
             tf=tf, qf=qf, Ybar=st["Y"], Ymax=st["Ymax"])
    live = qf > 1e-8*max(qf.max(), 1e-300)
    def compare(qref, tag):
        tol = qr + 1e-9*np.abs(qref) + 1e-20
        inn = np.abs(qref - qm) <= tol
        rat = (np.abs(qref - qm)/(qr + 1e-9*np.abs(qref) + 1e-300))
        print("  V1 [%s] contained on live modes: %s (worst ratio %.4f)"
              % (tag, bool(np.all(inn[live])), np.max(rat[live])))
        bad = np.argsort(rat*live)[::-1][:5]
        for i in bad:
            if rat[i]*live[i] > 0.5:
                print("      m=%-4d ref=%.6e  mid=%.6e  rad=%.2e  |d|/tol=%.2f"
                      % (i + 1, qref[i], qm[i], qr[i], rat[i]))
    compare(qf, "lawson a=0.01")
    # tighter second reference at the SAME exact endpoint — discriminates
    # reference error vs enclosure unsoundness
    qf2 = _lawson_ref(0.07, M, tf, 0.002, 5e-4)
    compare(qf2, "lawson a=0.002")
    dr = np.max((np.abs(qf2 - qf)/(qr + 1e-9*np.abs(qf2) + 1e-300))[live])
    print("  V1 ref-vs-ref |loose-tight|/rad worst on live: %.4f "
          "(>~1 ==> the loose reference's own error exceeds enclosure width)" % dr)
    relw = float(np.max(qr[live]/np.maximum(qm[live], 1e-300)))
    print("  V1 march Sigma_peak = %.4f at t = %.4f; relw(live,end) = %.2e; Ybar(end) = %.3e (max %.3e)"
          % (st["Sigpk"], st["tpk"], relw, st["Y"], st["Ymax"]))
    print("[%.0f s]" % (time.time() - t0))

def stage_tmv():
    """Validation battery:
    V1: TM enclosures vs EXACT exponential-sum modes (dec1 machinery) at
        nu = 0.075, t = 2.0 — external machine-precision anchor;
    V2: containment under step-halving (alphaT/2) at nu = 41/1024, t = 3.0;
    V3: TM vs float Lawson at nu = 41/1024, t = 4.0 (inside enclosure?)."""
    print("stage tmv: Taylor-model validation"); t0 = time.time()
    from flint import arb, ctx
    import cap_hier as ch
    ctx.prec = 160
    # V1
    M = 32; p = 6
    r = sweep(3, 40, M, p, 2.0)   # nu = 3/40 = 0.075 exactly
    t, qm, qr = r
    nu_a = arb("0.075")
    q = {1: [(arb(1), 1)]}
    for m in range(2, 9):
        F = {}
        for j in range(1, m):
            for (c1, k1) in q[j]:
                for (c2, k2) in q[m - j]:
                    key = k1 + k2
                    F[key] = F.get(key) + c1*c2/2 if key in F else c1*c2/2
        q[m] = ch._duhamel([(c, k) for (k, c) in F.items()], m*m, nu_a)
    worst = 0.0; okall = True
    for m in range(1, 9):
        ex = None
        for (c, k) in q[m]:
            w = c*((-(k*nu_a*arb(2))).exp())
            ex = w if ex is None else ex + w
        exf = float(ex.mid()) if hasattr(ex, 'mid') else float(ex)
        inn = abs(exf - qm[m-1]) <= qr[m-1] + 1e-30
        okall &= bool(inn)
        worst = max(worst, abs(exf - qm[m-1])/max(qr[m-1], 1e-300))
        if m <= 3:
            print("  V1 m=%d: exact=%.12e  TM mid=%.12e  rad=%.1e  contained=%s"
                  % (m, exf, qm[m-1], qr[m-1], inn))
    print("  V1 modes 1..8 all contained: %s  (worst |err|/rad = %.3f)" % (okall, worst))
    # V2 containment under refinement
    M2 = 64; p2 = 6
    rA = sweep(41, 1024, M2, p2, 3.0, alphaT=0.35)
    rB = sweep(41, 1024, M2, p2, 3.0, alphaT=0.175)
    okc = np.all(np.abs(rA[1] - rB[1]) <= rA[2] + rB[2] + 1e-25)
    print("  V2 halved-step enclosures mutually consistent: %s" % okc)
    # V3 float Lawson inside TM enclosure
    out = ch.q_lawson(41.0/1024.0, M2, 3.90, alpha=0.02, dt_max=0.005)
    tf, qf = out[-1]
    rC = sweep(41, 1024, M2, p2, tf)
    live = qf > 1e-13*qf.max()
    inn = np.abs(qf - rC[1]) <= rC[2] + 1e-9*np.abs(qf) + 1e-20
    print("  V3 float traj (t=%.3f) inside TM enclosure on live modes: %s (max ratio %.3f)"
          % (tf, np.all(inn[live]), np.max((np.abs(qf - rC[1])/(rC[2] + 1e-9*np.abs(qf) + 1e-300))[live])))
    print("[%.0f s]" % (time.time() - t0))

def stage_tmrun(margin_goal=2.0, nu_num=41, nu_den=1024, t_max=4.60):
    """THE BLOWUP CERTIFICATE: nu = 41/1024, M = 512, p = 6. March to the seed
    zone; certify min over a window of length W of S_8(t) >= margin * A*(8).
    On success: Sigma q_m = infinity at an explicit time — periodic sigma=2
    blowup from cos x. Constants A*, W certified in arb (reuse cap_hier)."""
    from flint import arb, ctx
    ctx.prec = 120
    nu = float(nu_num)/nu_den
    M, p = 512, 6
    nu_a = arb(nu_num)/arb(nu_den)
    Astar = 2*nu_a/(1 - (-arb(1)).exp())*arb(4)**11
    Astar_hi = np.nextafter(float(Astar.upper()), np.inf)
    W = float((4/(3*nu_a)*arb(4)**(-10)).upper())*1.02
    print("stage tmrun: nu=%d/%d=%.6f M=%d p=%d;  A*(8) <= %.6g, W = %.3e, goal margin %.1f"
          % (nu_num, nu_den, nu, M, p, Astar_hi, W, margin_goal))
    lo_blk, hi_blk = 255, 511
    win = []
    best = [0.0, -1.0]
    def seed_cb(t, h, qm, qr, S_hi):
        # window bookkeeping with the certified step-hull LOWER bound:
        # q >= max(0, endpoint-mid - rad) is NOT a step bound; use the certified
        # decay-only lower hull: over the step q_m >= (qm-qr at start)*e^{-nu m^2 h}
        # >= (qm-qr)*(1 - nu m^2 h). Cheap and sound.
        m2h = nu*(np.arange(1, M + 1)**2).astype(float)*h
        low_hull = np.maximum(qm - qr, 0.0)*np.maximum(1.0 - m2h, 0.0)
        Sblk_lo = np.sum(low_hull[lo_blk:hi_blk])*(1 - 1e-12)
        win.append((t - h, h, Sblk_lo))
        tw = t - W
        while win and win[0][0] + win[0][1] <= tw:
            win.pop(0)
        if win and win[0][0] <= tw:
            wmin = min(w[2] for w in win)
            if wmin > best[1]:
                best[0], best[1] = win[0][0], wmin
            if wmin >= margin_goal*Astar_hi:
                t1 = win[0][0]
                print("  *** CERTIFIED: min over [%.6f, %.6f] of S_8 >= %.6g = %.2f x A*(8) ***"
                      % (t1, t1 + W, wmin, wmin/Astar_hi))
                print("  ==> Sigma q_m(t) = INFINITY for explicit t <= %.6f" % (t1 + W))
                print("  ==> THEOREM: viscous CLM (sigma=2, nu=%d/%d) from cos x on S^1" % (nu_num, nu_den))
                print("      admits NO smooth solution beyond t = %.6f." % (t1 + W))
                np.savez(os.path.join(DIR, "cap_tm_cert_%d_%d.npz" % (nu_num, nu_den)),
                         t1=t1, W=W, S_lo=wmin, Astar_hi=Astar_hi, M=M, p=p,
                         margin=wmin/Astar_hi, nu_num=nu_num, nu_den=nu_den)
                return ("CERT", t1, wmin)
        return None
    r = sweep(nu_num, nu_den, M, p, t_max, alphaT=0.35, progress=2000, seed_cb=seed_cb)
    if r is None or (isinstance(r, tuple) and r[0] != "CERT"):
        print("  not certified; best window bound %.4g (= %.2f x A*) at t=%.6f"
              % (best[1], best[1]/Astar_hi, best[0]))

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "tmv":
        stage_tmv()
    elif st.startswith("excv1"):
        parts = st.split(":")
        stage_excv1(int(parts[1]) if len(parts) > 1 else 256)
    elif st.startswith("exc:"):
        parts = st.split(":")
        stage_exc(int(parts[1]), int(parts[2]),
                  M=int(parts[3]) if len(parts) > 3 else 384,
                  mS=int(parts[4]) if len(parts) > 4 else 4,
                  tl_min=float(parts[5]) if len(parts) > 5 else 20.0,
                  t_budget=float(parts[6]) if len(parts) > 6 else 40.0,
                  jbump=int(parts[7]) if len(parts) > 7 else 0)
    elif st.startswith("tmrun"):
        parts = st.split(":")
        stage_tmrun(float(parts[1]) if len(parts) > 1 else 2.0,
                    int(parts[2]) if len(parts) > 2 else 41,
                    int(parts[3]) if len(parts) > 3 else 1024,
                    float(parts[4]) if len(parts) > 4 else 4.60)
    else:
        raise SystemExit("unknown stage: %s" % st)
