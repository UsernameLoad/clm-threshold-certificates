#!/usr/bin/env python3
# s22_cap_sig.py — S22: THE SIGMA-PARAMETERIZED CAP TIER (B-1 build).
# Entry ticket:.
#
# cap_hier.py are NOT edited — the gate's references stand; this file
# re-implements ONLY the lambda-touching pieces with sigma threaded through,
# importing everything else from cap_tm (the S20 s20_sig2d.py pattern).
#
# The hierarchy at general sigma:
#   q_m' = -nu m^sigma q_m + (1/2) sum_{j<m} q_j q_{m-j},  q_1(0)=1, q_m(0)=0,
#   all q_m >= 0; modes 1..M are the TRUE solution's modes by triangularity.
#
# SIGMA DISPATCH (audit A-2): at sigma == 2 every lambda-touching line takes
# cap_tm's ORIGINAL expression (exact dyadic-rational lambda), so the sigma=2
# path is arithmetically the banked one (battery B-A1 tests bit-identity).
# The generic path builds lambda = (nu exact rational) * m^sigma as an arb
# BALL; positivity and non-resonance are certified where used.
#
# FLOAT-LAMBDA SOUNDNESS (extends the S13 float-nu note): the level
# recursion's float m^sigma (np.power, <= 2-ulp ~ 2e-16 rel) sits three
# orders below the per-op RND = 1e-13 mid-rad inflation that already covers
# float nu; kernels/E use arb balls of the exact lambda throughout.
#
# Stages:
#   tmv2                       — B-A1: the tmv battery through the new sweep at sigma=2
#   sig1                       — B-B1: TM tier vs Sakajo Lemma 3 closed form at sigma=1
#   decs2                      — B-A2: multiset dec chain at sigma=2 vs banked dec1 row
#   decs0                      — B-B2: multiset dec chain at sigma=0 vs ALSS closed form
#   widths                     — B-C : K-table ball-width report, sigma=0.5 vs 2
#   dec1s:<sig>:<mP>:<nu,...>  — static dec1-form barrier at general sigma (mP<=10)
#   excs:<sig>:<num>:<den>[:M[:mS[:tlmin[:budget[:p]]]]] — THE CERTIFICATE:
#       excursion-tracking certified decay march at general sigma
import os, sys, time
import numpy as np
import cap_tm as ct
import cap_hier as ch

DIR = os.path.dirname(os.path.abspath(__file__))
RND = ct.RND
ETA = ct.ETA

# ---------------------------------------------------------------- sigma plumbing
_MSIG = {}
def _msig(M, sigma_txt):
    """float m^sigma, m = 1..M; sigma=2 takes the EXACT original expression."""
    key = (M, sigma_txt)
    if key not in _MSIG:
        if float(sigma_txt) == 2.0:
            _MSIG[key] = (np.arange(1, M + 1)**2).astype(float)
        else:
            _MSIG[key] = np.power(np.arange(1, M + 1, dtype=float), float(sigma_txt))
    return _MSIG[key]

_TABS = {}
def _ktables_sig(nu_num, nu_den, sigma_txt, j, M, pmax):
    """cap_tm._ktables with lam = nu*m^sigma; sigma=2 dispatches to the exact
    original rational line. K_i = int_0^h e^{-lam(h-s)} s^i ds > 0 for every
    lam > 0 (the recursion computes the same integral; positivity is generic),
    so the >=0 clipping remains sound at every sigma."""
    from flint import arb, ctx
    ctx.prec = 120
    h = 2.0**(-j)
    E_lo = np.empty(M); E_hi = np.empty(M)
    K_lo = np.empty((pmax + 1, M)); K_hi = np.empty((pmax + 1, M))
    sig_is2 = (float(sigma_txt) == 2.0)
    sig_a = None if sig_is2 else arb(sigma_txt)
    nu_a = None if sig_is2 else arb(nu_num)/arb(nu_den)
    for m in range(1, M + 1):
        if sig_is2:
            lam = arb(nu_num*m*m)/arb(nu_den)          # cap_tm line 58, verbatim
        else:
            lam = nu_a*(arb(m)**sig_a)                 # ball of the exact lambda
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

def tm_step_sig(qm, qr, nu_num, nu_den, sigma_txt, M, j, p):
    """cap_tm.tm_step, verbatim, with (a) the table call sigma-keyed and
    (b) m2h = nu*h*m^sigma. Nothing else differs."""
    nu = nu_num/nu_den
    h = 2.0**(-j)
    key = (sigma_txt, j, M, p)
    if key not in _TABS:
        _TABS[key] = _ktables_sig(nu_num, nu_den, sigma_txt, j, M, 2*p + 1)
    E_lo, E_hi, K_lo, K_hi = _TABS[key]
    m2h = nu*h*_msig(M, sigma_txt)
    Bm = [qm.copy()]; Br = [qr.copy()]
    Fms = []; Frs = []
    for i in range(2*p + 1):
        fm = np.zeros(2*M - 1); fr = np.zeros(2*M - 1)
        for i1 in range(max(0, i - p), min(i, p) + 1):
            cm, cr = ct._convmr(Bm[i1], Br[i1], Bm[i - i1], Br[i - i1])
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
    S = qm + qr
    for i in range(p + 1):
        S = S + (np.maximum(Fms[i], 0.0) + Frs[i])*(K_hi[i]/h**i)
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
    Nm = np.zeros(M); Nr = np.zeros(M)
    for i in range(p + 1):
        w_lo = K_lo[i]/h**i; w_hi = K_hi[i]/h**i
        Nm = Nm + Fms[i]*0.5*(w_lo + w_hi)
        Nr = Nr + np.abs(Fms[i])*0.5*(w_hi - w_lo) + Frs[i]*w_hi + RND*np.abs(Nm)
    qm2 = E_lo*qm*0.5 + E_hi*qm*0.5 + Nm
    qr2 = (E_hi - E_lo)*0.5*np.abs(qm) + E_hi*qr + Nr + rho + RND*np.abs(qm2) + ETA
    if not (np.isfinite(qm2).all() and np.isfinite(qr2).all()):
        return None
    return np.maximum(qm2, 0.0), qr2, S + rho

def sweep_sig(nu_num, nu_den, sigma_txt, M, p, t_end, alphaT=0.35, progress=None,
              seed_cb=None, xmax=2.0):
    """cap_tm.sweep with jstiff from nu*M^sigma (sigma=2 dispatches to nu*M*M)."""
    nu = nu_num/nu_den
    if float(sigma_txt) == 2.0:
        nMs = nu*M*M
    else:
        nMs = nu*(float(M)**float(sigma_txt))
    jstiff = int(np.ceil(np.log2(max(nMs/xmax, 1.0))))
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
        r = tm_step_sig(qm, qr, nu_num, nu_den, sigma_txt, M, j, p)
        tries = 0
        while r is None and tries < 8 and j < 50:
            j += 1; r = tm_step_sig(qm, qr, nu_num, nu_den, sigma_txt, M, j, p); tries += 1
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

# ------------------------------------------------ sigma tail block (S13 lemma, lambda-generic)
def _tail_update_sig(Y, S_step, h, lamT, M):
    """cap_tm._tail_update with the hull seed at the FIXED-POINT scale.
    The imported original seeds H at max(2Y, 1e-280) and quadruples at most
    240 times; at sub-unit sigma's LARGE steps (h = 2^-2-class) the
    first-step forcing g ~ 1e-121 with Y = 0 is unreachable from 1e-280 in
    240 quadruplings (4^240 ~ 1e144) — an implementation cap in the hull
    SEARCH, not in the lemma (at sigma = 2's jstiff-forced tiny steps it
    never fired). The seed now also starts at 1.5*g/(lamT - P); the
    a-posteriori verification (a2 > 0 and a2*H >= g(1+eps)) is UNCHANGED and
    remains the proof obligation. Recorded as S22 test-redesign #2."""
    eps = 1e-12
    P = float(np.sum(S_step))*(1 + eps)
    cs = np.convolve(S_step, S_step)            # index k <-> mode m = k+2
    R = float(np.sum(cs[M - 1:]))*(1 + eps)     # m > M  <=>  k >= M-1
    g = 0.5*R
    slack0 = lamT*(1 - eps) - P
    H = max(2.0*Y, (1.5*g/slack0 if slack0 > 0 else 0.0), 1e-280)
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

# ------------------------------------------------ sigma landing (S13 lemma, lambda-generic)
def _land_check_sig(nu_num, nu_den, sigma_txt, mS, M, qm, qr, Ytail):
    """cap_tm._land_check with lamS = nu (mS+1)^sigma and per-mode nu m^sigma.
    The lemma structure (triangular Duhamel sup chain + Riccati trap) is
    lambda-generic; only the rates change. sigma=2 dispatches to integer forms."""
    from flint import arb, ctx
    ctx.prec = 120
    nu_a = arb(nu_num)/arb(nu_den)
    sig_is2 = (float(sigma_txt) == 2.0)
    sig_a = None if sig_is2 else arb(sigma_txt)
    def lam_of(mm):
        return nu_a*(mm*mm) if sig_is2 else nu_a*(arb(mm)**sig_a)
    qhi = [arb(float(qm[i])) + arb(float(qr[i])) for i in range(mS)]
    Pm = [None]*(mS + 1)
    Pm[1] = qhi[0]
    for m in range(2, mS + 1):
        F = arb(0)
        for j in range(1, m):
            F += Pm[j]*Pm[m - j]
        Pm[m] = qhi[m - 1] + (F/2)/lam_of(m)
    Pbar = arb(0)
    for m in range(1, mS + 1):
        Pbar += Pm[m]
    lamS = lam_of(mS + 1)
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

def stage_excs(sigma_txt, nu_num, nu_den, M=256, mS=8, tl_min=15.0, t_budget=80.0,
               p=6, land_every=16, jbump=4):
    # jbump default 4 (h = 2^-6-class): at sub-unit sigma the stiffness cap is
    # so mild (nu M^sigma ~ 7) that the generic h = 2^-2 exceeds the COLD-START
    # resolution — modes born at the front's birth (m ~ 9, value ~ 1e-8·max)
    # lie beyond the p = 6 Taylor reach and carry O(10) relative width in rho
    # (S22 test-redesign #3; at sigma = 2 jstiff hid this). Finer steps put
    # every live mode inside the resolved range; cost ~ seconds.
    """THE FRACTIONAL-SIGMA DECAY CERTIFICATE: cap_tm.stage_exc's architecture
    (march + certified tail block + certified landing; margin-3 acceptance)
    at general sigma. Tail block = _tail_update_sig (the S13 lemma with the
    hull-seed redesign, verification unchanged); lamT = nu (M+1)^sigma."""
    nu = nu_num/nu_den
    sigf = float(sigma_txt)
    if sigf == 2.0:
        lamT = nu*((M + 1)**2)*(1 - 1e-13)
    else:
        lamT = nu*(float(M + 1)**sigf)*(1 - 1e-13)
    print("stage excs: sigma=%s nu=%d/%d=%.6f M=%d mS=%d tl_min=%.1f budget=%.1f jbump=%d"
          % (sigma_txt, nu_num, nu_den, nu, M, mS, tl_min, t_budget, jbump))
    st = dict(Y=0.0, Ymax=0.0, relwmax=0.0, Sigpk=0.0, tpk=0.0, nstep=0,
              nland=0, fail=None, land=None, Pmax=0.0)
    t0 = time.time()
    def cb(t, h, qm, qr, S_hi):
        st["nstep"] += 1
        Yn, P, R = _tail_update_sig(st["Y"], S_hi, h, lamT, M)
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
            L = _land_check_sig(nu_num, nu_den, sigma_txt, mS, M, qm, qr, st["Y"])
            if L["ok"] and L["margin"] >= 3.0:
                st["land"] = (t, L)
                return ("LANDED", t)
        if t > t_budget:
            st["fail"] = ("F2-budget", t)
            print("  [F2] no landing by t=%.2f — budget exhausted" % t)
            return ("FAIL", t)
        return None
    r = sweep_sig(nu_num, nu_den, sigma_txt, M, p, t_budget + 1.0, alphaT=0.35,
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
        print("  *** DECAY CERTIFIED at sigma=%s (march-through-excursion + certified landing) ***" % sigma_txt)
        print("  ==> THEOREM: the sigma=%s dissipative CLM hierarchy on S^1 from cos x" % sigma_txt)
        print("      decays globally for nu = %d/%d = %.6f, and for every nu' >= nu by" % (nu_num, nu_den, nu))
        print("      the nu-comparison lemma (N18a; its Kamke proof is sigma-generic)")
        print("      ==> nu*(sigma=%s) <= %.6f" % (sigma_txt, nu))
        print("  ==> TRANSPORT (Sakajo Kokyuroku 1326-5 Prop 1, alpha-monotonicity —")
        print("      his Proposition, cited; Nonlinearity-primary read condition inherited):")
        print("      global decay at EVERY sigma' >= %s for nu' >= %.6f — the certified" % (sigma_txt, nu))
        print("      decay QUADRANT {sigma' >= %s} x {nu' >= %.6f}." % (sigma_txt, nu))
        if float(sigma_txt) >= 1.0:
            print("      Bar beaten: nu > 1/(2e) = 0.183940 on every sigma >= 1 — the sigma=1")
            print("      Lemma-3 geometric closed form (Sakajo JMSUT Lemma 3) transported by")
            print("      Kokyuroku Prop 1 (the S24 §0 A-1 composition; sharp at sigma=1) — x%.2f." % ((0.5/np.e)/nu))
        else:
            print("      Bar beaten: the elementary comparison bound nu <= 1/2 (S21 free bracket).")
        tag = sigma_txt.replace(".", "p")
        fn = os.path.join(DIR, "exc_cert_s%s_%d_%d_M%d.npz" % (tag, nu_num, nu_den, M))
        np.savez(fn, sigma=sigma_txt, nu_num=nu_num, nu_den=nu_den, M=M, mS=mS, p=p,
                 t_land=tl, Pbar=L["Pbar"], lamS=L["lamS"], Rbar=L["Rbar"],
                 Ytot=L["Yt"], H=L["H"], margin=L["margin"], Pm=L["Pm"],
                 Ybar_max=st["Ymax"], relw_max=st["relwmax"],
                 Sig_peak=st["Sigpk"], t_peak=st["tpk"], nstep=st["nstep"])
        print("  certificate saved: %s" % os.path.basename(fn))
        return st
    print("  RESULT: %s at t=%.4f — honest failure, no certificate"
          % (st["fail"][0] if st["fail"] else "unknown", st["fail"][1] if st["fail"] else -1))
    return st

# ------------------------------------------------ the multiset-keyed static chain
def _chain_sig(nu_a, sig_a, mP, int_sigma=None):
    """Exact exponential-sum chain. Exponent keys: at INTEGER sigma the key is
    the exact integer sum_i part^sigma (the cap_hier collapse — coefficient
    cancellation happens INSIDE each key before any |.| envelope, which is
    what keeps the sup-bounds sharp: the B-A2 v1 lesson); at fractional sigma
    keys are partition multisets (no two provably-equal exponents collapse,
    but coefficients are mild there — the 31x non-resonance margin). Every
    Duhamel denominator is certified nonzero in ball arithmetic (provable at
    sigma<1 by strict superadditivity; at sigma=1 the chain is TOTALLY
    resonant and this function raises — audit A-2)."""
    from flint import arb
    pow_cache = {}
    if int_sigma is not None:
        def mkkey(parts):
            return sum(p**int_sigma for p in parts)
        def merge(k1, k2):
            return k1 + k2
        def powsum(key):
            if key not in pow_cache:
                pow_cache[key] = arb(key)
            return pow_cache[key]
        key1 = 1
    else:
        def mkkey(parts):
            return tuple(sorted(parts))
        def merge(k1, k2):
            return tuple(sorted(k1 + k2))
        def powsum(key):
            if key not in pow_cache:
                v = arb(0)
                for part in key:
                    v += arb(part)**sig_a
                pow_cache[key] = v
            return pow_cache[key]
        key1 = (1,)
    q = {1: {key1: arb(1)}}
    for m in range(2, mP + 1):
        F = {}
        for j in range(1, m):
            for k1, c1 in q[j].items():
                for k2, c2 in q[m - j].items():
                    key = merge(k1, k2)
                    F[key] = F.get(key) + c1*c2/2 if key in F else c1*c2/2
        mkey = mkkey((m,))
        lam_ps = powsum(mkey)
        out = {}
        acc = arb(0)
        for key, c in F.items():
            d = (powsum(key) - lam_ps)*nu_a          # mu - lam
            if not bool(d != 0):
                raise RuntimeError("resonant Duhamel denominator: m=%d key=%s"
                                   % (m, key))
            w = c/d
            out[key] = out.get(key) - w if key in out else -w
            acc += w
        out[mkey] = out.get(mkey) + acc if mkey in out else acc
        q[m] = out
    return q, powsum, merge

def _supb_sig(terms, powsum, nu_a, tmax, n=3000):
    """cap_hier.stage_dec1's certified sup bound (cubic-clustered grid,
    signed f and f', |c|mu^2 envelope for f''), with mu = nu*powsum(key)."""
    from flint import arb
    ts = [tmax*k**3/float(n)**3 for k in range(n + 1)]
    # pre-collapse to (mu, c) pairs once (mu as arb)
    tl = [(powsum(k)*nu_a, c) for k, c in terms.items()]
    best = None
    for jj in range(n):
        tk = arb(ts[jj]); dt = arb(ts[jj + 1] - ts[jj])
        v0 = None; v1 = None; v2 = None
        for mu, c in tl:
            e = (-(mu*tk)).exp()
            w = c*e
            v0 = w if v0 is None else v0 + w
            w1 = -mu*w
            v1 = w1 if v1 is None else v1 + w1
            w2 = abs(c)*mu*mu*e
            v2 = w2 if v2 is None else v2 + w2
        cell = v0 + dt*abs(v1) + dt*dt/2*v2
        if best is None or float(cell.upper()) > float(best.upper()):
            best = cell
    tail = None
    for mu, c in tl:
        e = abs(c)*((-(mu*arb(tmax))).exp())
        tail = e if tail is None else tail + e
    return best if float(best.upper()) >= float(tail.upper()) else tail

def _int_sigma_of(sigma_txt):
    f = float(sigma_txt)
    return int(f) if f == int(f) else None

def stage_dec1s(sigma_txt, mP, nu_txts):
    """dec1-form static barrier at general sigma (cross-check tier; mP <= 10 —
    the partition-count wall,. Threshold nu(mP+1)^sigma."""
    from flint import arb, ctx
    ctx.prec = 160
    sig_a = arb(sigma_txt)
    isig = _int_sigma_of(sigma_txt)
    print("stage dec1s: sigma=%s, explicit modes 1..%d + tail barrier at -(%d)^%s nu"
          % (sigma_txt, mP, mP + 1, sigma_txt))
    for nu_txt in nu_txts:
        nu = arb(nu_txt)
        t0 = time.time()
        q, powsum, merge = _chain_sig(nu, sig_a, mP, int_sigma=isig)
        Pterms = {}
        for m in range(1, mP + 1):
            for k, c in q[m].items():
                Pterms[k] = Pterms.get(k) + c if k in Pterms else c
        Rterms = {}
        for j in range(1, mP + 1):
            for l in range(1, mP + 1):
                if j + l > mP:
                    for k1, c1 in q[j].items():
                        for k2, c2 in q[l].items():
                            key = merge(k1, k2)
                            Rterms[key] = Rterms.get(key) + c1*c2 if key in Rterms else c1*c2
        tmax = 80.0/float(nu_txt)
        Pbar = _supb_sig(Pterms, powsum, nu, tmax)
        Rbar = _supb_sig(Rterms, powsum, nu, tmax)
        thr = nu*(arb(mP + 1)**sig_a)
        ok1 = bool(Pbar < thr)
        disc = (thr - Pbar)**2 - Rbar if ok1 else None
        ok = ok1 and bool(disc > 0)
        print("  nu=%-5s: Pbar=%.4f (<%.3f: %s)  Rbar=%.4f  disc>0: %s -> %s  [%d P-terms, %d R-terms, %.0fs]"
              % (nu_txt, float(Pbar.upper()), float(thr.lower()), "ok" if ok1 else "FAIL",
                 float(Rbar.upper()), bool(disc > 0) if ok1 else "-",
                 "CERTIFIED" if ok else "fail", len(Pterms), len(Rterms), time.time() - t0))
        if ok:
            print("    ==> nu*(sigma=%s) <= %s (nu-comparison lemma, sigma-generic) + the" % (sigma_txt, nu_txt))
            print("        Prop-1 decay quadrant {sigma' >= %s} x {nu' >= %s}" % (sigma_txt, nu_txt))

# ------------------------------------------------------------------ battery stages
def stage_tmv2():
    """B-A1: cap_tm.stage_tmv verbatim, but every sweep goes through the NEW
    sigma-parameterized path at sigma='2'. Must reproduce s22_gate_tmv.txt
    digit-for-digit mod wall-clock."""
    print("stage tmv2: Taylor-model validation (sigma-tier, sigma=2 dispatch)"); t0 = time.time()
    from flint import arb, ctx
    ctx.prec = 160
    M = 32; p = 6
    r = sweep_sig(3, 40, "2", M, p, 2.0)
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
    M2 = 64; p2 = 6
    rA = sweep_sig(41, 1024, "2", M2, p2, 3.0, alphaT=0.35)
    rB = sweep_sig(41, 1024, "2", M2, p2, 3.0, alphaT=0.175)
    okc = np.all(np.abs(rA[1] - rB[1]) <= rA[2] + rB[2] + 1e-25)
    print("  V2 halved-step enclosures mutually consistent: %s" % okc)
    out = ch.q_lawson(41.0/1024.0, M2, 3.90, alpha=0.02, dt_max=0.005)
    tf, qf = out[-1]
    rC = sweep_sig(41, 1024, "2", M2, p2, tf)
    live = qf > 1e-13*qf.max()
    inn = np.abs(qf - rC[1]) <= rC[2] + 1e-9*np.abs(qf) + 1e-20
    print("  V3 float traj (t=%.3f) inside TM enclosure on live modes: %s (max ratio %.3f)"
          % (tf, np.all(inn[live]), np.max((np.abs(qf - rC[1])/(rC[2] + 1e-9*np.abs(qf) + 1e-300))[live])))
    print("[%.0f s]" % (time.time() - t0))

def stage_sig1():
    """B-B1: the sigma=1 machine-floor anchor for the TM tier. Sakajo JMSUT
    Lemma 3 (A1=1 normalization, S19-verified): q_m(t) = (t/2)^{m-1} e^{-m nu t}.
    nu = 1/4, M = 32, p = 6, t = 2.0. Containment at every m <= 16."""
    print("stage sig1: TM tier vs Sakajo Lemma 3 exact closed form (sigma=1, nu=1/4)")
    t0 = time.time()
    from flint import arb, ctx
    ctx.prec = 160
    M, p = 32, 6
    r = sweep_sig(1, 4, "1", M, p, 2.0)
    if r is None:
        print("  FAIL: march did not complete"); return
    t, qm, qr = r
    nu_a = arb(1)/arb(4)
    t_a = arb(2)
    okall = True; worst = 0.0
    for m in range(1, 17):
        ex = ((t_a/2)**(m - 1))*((-(m*nu_a*t_a)).exp())
        exf = float(ex.mid())
        inn = abs(exf - qm[m-1]) <= qr[m-1] + 1e-30
        okall &= bool(inn)
        rat = abs(exf - qm[m-1])/max(qr[m-1], 1e-300)
        worst = max(worst, rat)
        if m <= 3 or m in (8, 16):
            print("  m=%-2d: exact=%.12e  TM mid=%.12e  rad=%.1e  contained=%s  |err|/rad=%.4f"
                  % (m, exf, qm[m-1], qr[m-1], inn, rat))
    print("  B-B1 modes 1..16 all contained: %s  (worst |err|/rad = %.4f; registered bar 0.5)"
          % (okall, worst))
    print("[%.0f s]" % (time.time() - t0))

def stage_decs2():
    """B-A2: the multiset chain at sigma=2 vs the banked dec1 mP=8 row
    (s08_cap.txt second ladder: nu=0.09 -> Pbar=3.1011 Rbar=13.1841 thr 7.290
    CERTIFIED). Printed-digit grade (accumulation order differs; registered)."""
    print("stage decs2: multiset chain at sigma=2 vs banked dec1 mP=8 rows")
    print("  banked (s08_cap.txt): nu=0.09 : Pbar=3.1011 (<7.290)  Rbar=13.1841  CERTIFIED")
    stage_dec1s("2", 8, ["0.09"])

def stage_decs0():
    """B-B2: the sigma=0 ALSS-closed-form arithmetic anchor for the chain.
    q_m(t) = e^{-nu t} * tau^{m-1}/2^{m-1}, tau = (1-e^{-nu t})/nu (ALSS 2022
    5.4/6 — theirs, cited). Containment of chain values at t in {1, 5, 20};
    Pbar sup-bound vs the exact sup = 1 (Sigma monotone decreasing at
    nu > 1/2, N16 S19 block). The 2a-6 closure degeneration is REPORTED."""
    print("stage decs0: multiset chain at sigma=0 vs the ALSS closed form (nu=0.51, mP=8)")
    t0 = time.time()
    from flint import arb, ctx
    ctx.prec = 160
    nu_a = arb("0.51")
    sig_a = arb("0")
    q, powsum, merge = _chain_sig(nu_a, sig_a, 8, int_sigma=0)
    okall = True; worstmag = 0.0
    for tt in (1.0, 5.0, 20.0):
        t_a = arb(tt)
        u = (-(nu_a*t_a)).exp()
        tau = (1 - u)/nu_a
        for m in (2, 5, 8):
            ex = u*(tau/2)**(m - 1)
            val = None
            for key, c in q[m].items():
                w = c*((-(powsum(key)*nu_a*t_a)).exp())
                val = w if val is None else val + w
            d = val - ex
            ok = not bool(d != 0)          # ball difference must not be provably nonzero
            okall &= ok
            mag = abs(float(d.mid())) + float(d.rad())
            worstmag = max(worstmag, mag)
            if tt == 5.0:
                print("  t=%.0f m=%d: chain=%.12e  exact=%.12e  |diff| <= %.1e  consistent=%s"
                      % (tt, m, float(val.mid()), float(ex.mid()), mag, ok))
    print("  B-B2 containment at t in {1,5,20}, m in {2,5,8}: %s  (worst |diff| %.1e)"
          % (okall, worstmag))
    Pterms = {}
    for m in range(1, 9):
        for k, c in q[m].items():
            Pterms[k] = Pterms.get(k) + c if k in Pterms else c
    Pbar = _supb_sig(Pterms, powsum, nu_a, 80.0/0.51)
    pb = float(Pbar.upper())
    print("  B-B2 Pbar sup-bound = %.6f vs exact sup Sigma_{m<=8}(0) = 1 (must be >= 1, <= 1.05): %s"
          % (pb, (pb >= 1.0 and pb <= 1.05)))
    thr = nu_a*(arb(9)**sig_a)
    print("  REPORTED (2a-6, not scored): T1 = nu*(mP+1)^0 = %.4f < Pbar — the dec1-form"
          % float(thr.mid()))
    print("  barrier structurally cannot close at sigma=0 at any nu < 1 (the sigma->0")
    print("  degeneration derived in advance; the ALSS anchor is arithmetic, not closure).")
    print("[%.0f s]" % (time.time() - t0))

def stage_widths():
    """B-C: K-table ball-width report. Registered: fractional-exponent widths
    <= 1e-15 relative at every (i, m) — i.e. the one-ulp outward float64
    conversion dominates, and irrational exponents cost nothing."""
    print("stage widths: K-table relative ball widths (arb, prec 120)")
    from flint import arb, ctx
    ctx.prec = 120
    for sigma_txt, nu_pair, jj in (("0.5", (17, 50), 2), ("0.5", (17, 50), 8),
                                   ("2", (41, 1024), 8)):
        nu_num, nu_den = nu_pair
        h = 2.0**(-jj)
        sig_is2 = (float(sigma_txt) == 2.0)
        sig_a = None if sig_is2 else arb(sigma_txt)
        nu_a = arb(nu_num)/arb(nu_den)
        wmax = 0.0; wmax_f = 0.0
        for m in (1, 2, 16, 128, 512):
            lam = arb(nu_num*m*m)/arb(nu_den) if sig_is2 else nu_a*(arb(m)**sig_a)
            K = (1 - (-(lam*arb(h))).exp())/lam
            hp = arb(h)
            for i in range(1, 14):
                K = (hp - i*K)/lam
                hp = hp*arb(h)
                relw = float(K.rad())/max(abs(float(K.mid())), 1e-300)
                wmax = max(wmax, relw)
                lo = float(K.lower()); hi = float(K.upper())
                wf = (np.nextafter(hi, np.inf) - np.nextafter(lo, -np.inf))/max(abs(hi), 1e-300)
                wmax_f = max(wmax_f, wf)
        print("  sigma=%-3s j=%d (h=2^-%d): max arb rel width = %.2e; after outward float64: %.2e"
              % (sigma_txt, jj, jj, wmax, wmax_f))
    print("  registered bar: fractional arb widths <= 1e-15 relative (float64 conversion dominates)")

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "tmv2":
        stage_tmv2()
    elif st == "sig1":
        stage_sig1()
    elif st == "decs2":
        stage_decs2()
    elif st == "decs0":
        stage_decs0()
    elif st == "widths":
        stage_widths()
    elif st.startswith("dec1s:"):
        parts = st.split(":")
        stage_dec1s(parts[1], int(parts[2]), parts[3].split(","))
    elif st.startswith("excs:"):
        parts = st.split(":")
        stage_excs(parts[1], int(parts[2]), int(parts[3]),
                   M=int(parts[4]) if len(parts) > 4 else 256,
                   mS=int(parts[5]) if len(parts) > 5 else 8,
                   tl_min=float(parts[6]) if len(parts) > 6 else 15.0,
                   t_budget=float(parts[7]) if len(parts) > 7 else 80.0,
                   p=int(parts[8]) if len(parts) > 8 else 6,
                   jbump=int(parts[9]) if len(parts) > 9 else 4)
    else:
        raise SystemExit("unknown stage: %s" % st)
