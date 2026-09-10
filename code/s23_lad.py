#!/usr/bin/env python3
# s23_lad.py — S23: THE SIGMA-GENERIC DYADIC BLOCK-LADDER + BLOWUP MARCH.
#
#   A*(k0) = (2 nu/(1-1/e)) * 2^{sigma (k0+3)},
#   H(k0)  = 2^{-sigma (k0+2)} / (nu (1 - 2^{-sigma})),
# reducing VERBATIM to the banked S08/S09 constants at sigma = 2).
# cap_tm.py / cap_hier.py / s22_cap_sig.py are NOT edited: the march reuses
# s22_cap_sig.sweep_sig (gate-bound via tmv2) through its seed_cb hook.
#
# Stages:
#   const:<sig>:<num>:<den>:<k0>       — print certified A*(k0), H(k0)
#   repro                              — REPRODUCTION OBLIGATION: the new
#       constant arithmetic at sigma=2 dispatch vs the banked S09 certificate
#       npz tuples (Astar_hi, margin, W-vs-H), digit-for-digit
#   scout:<sig>:<nu>:<M>[:tmax[:dt0]]  — float IFRK4 map of block sums S_k(t)
#       vs 2.1*A*(k)/1.05*H(k) for every k0 candidate (a PROBE, not a clause)
#   blow:<sig>:<num>:<den>:<M>:<k0>[:margin[:p[:jbump[:tmax]]]] — THE
#       CERTIFIED BLOWUP MARCH: sweep_sig + start-of-step decay hulls
#       (sound per-interval lower bounds — note: cap_tm's stage_tmrun labels
#       end-of-step hulls with the step interval; under the strict reading
#       its certified window shifts by one step h ~ 2.4e-7 << W with the
#       theorem intact —, window
#       deque covering W = 1.02*H, obligations O-B1..O-B5 in-code.
#
# Soundness of the in-step hull (O-B1): each q_m is a supersolution of pure
# decay, so for s in [t_n, t_n + h]: q_m(s) >= q_m(t_n) e^{-nu m^sigma h}
# >= q_m(t_n) (1 - nu m^sigma h); block-worst rate over I_{k0} is
# nu (2^{k0+1})^sigma. All comparisons use certified lower endpoints
# (qm - qr clipped at 0) and float-conservative factors.
import os, sys, time
import numpy as np
import s22_cap_sig as cs

DIR = os.path.dirname(os.path.abspath(__file__))


def _consts_arb(sigma_txt, nu_num, nu_den, k0):
    """Certified A*(k0) upper bound and H(k0) upper bound. sigma = 2
    dispatches to cap_tm.stage_tmrun's EXACT expressions (bit-identity
    with the banked certificates is the repro obligation)."""
    from flint import arb, ctx
    ctx.prec = 120
    nu_a = arb(nu_num)/arb(nu_den)
    if float(sigma_txt) == 2.0:
        Astar = 2*nu_a/(1 - (-arb(1)).exp())*arb(4)**(k0 + 3)          # cap_tm line 514 form
        H = 4/(3*nu_a)*arb(4)**(-(k0 + 2))                             # cap_tm line 516 form
    else:
        sig_a = arb(sigma_txt)
        Astar = 2*nu_a/(1 - (-arb(1)).exp())*(arb(2)**(sig_a*(k0 + 3)))
        H = (arb(2)**(-sig_a*(k0 + 2)))/(nu_a*(1 - arb(2)**(-sig_a)))
    Astar_hi = np.nextafter(float(Astar.upper()), np.inf)
    H_hi = float(H.upper())
    return Astar_hi, H_hi


def stage_const(sigma_txt, nu_num, nu_den, k0):
    A, H = _consts_arb(sigma_txt, nu_num, nu_den, k0)
    nu = nu_num/nu_den
    print("const: sigma=%s nu=%d/%d=%.6f k0=%d" % (sigma_txt, nu_num, nu_den, nu, k0))
    print("  A*(k0) <= %.10g   (trigger bar; goal >= 2x)" % A)
    print("  H(k0)  <= %.6e   (required window; W = 1.02*H)" % H)


def stage_repro():
    """The new constant arithmetic must reproduce the banked S09 certificate
    tuples through the sigma=2 dispatch, digit-for-digit."""
    print("stage repro: sigma-generic ladder constants vs banked S09 certificates")
    ok_all = True
    for fn, k0 in [("cap_tm_cert.npz", 8), ("cap_tm_cert_46_1024.npz", 8)]:
        d = np.load(os.path.join(DIR, fn))
        nn = int(d["nu_num"]) if "nu_num" in d.files else 41
        nd = int(d["nu_den"]) if "nu_den" in d.files else 1024
        A, H = _consts_arb("2", nn, nd, k0)
        W_req = H*1.02
        A_bank = float(d["Astar_hi"]); W_bank = float(d["W"])
        S_lo = float(d["S_lo"]); mg_bank = float(d["margin"])
        mg_new = S_lo/A
        bitA = (A == A_bank)
        bitW = (W_req == W_bank)
        bitM = ("%.13f" % mg_new) == ("%.13f" % mg_bank)
        okW = (W_bank >= H)
        ok = bitA and bitW and bitM and okW and (S_lo >= 2.0*A)
        ok_all = ok_all and ok
        print("  %s (nu=%d/%d, k0=%d):" % (fn, nn, nd, k0))
        print("    A* new %.10f  banked %.10f  bit-identical: %s" % (A, A_bank, bitA))
        print("    W  new %.10e banked %.10e bit-identical: %s (and W >= H: %s)"
              % (W_req, W_bank, bitW, okW))
        print("    margin new %.13f banked %.13f  digit-identical: %s" % (mg_new, mg_bank, bitM))
        print("    S_lo %.6f >= 2*A*: %s" % (S_lo, S_lo >= 2.0*A))
    print("  REPRO %s" % ("PASS — the sigma-generic constants reduce to the banked"
                          " S09 certificate arithmetic verbatim" if ok_all else "FAIL"))


# ---------------------------------------------------------------- float scout
def _nl(q):
    c = np.convolve(q, q)
    F = np.zeros_like(q)
    F[1:] = 0.5*c[:q.size - 1]
    return F


def stage_scout(sigma_txt, nu, M, tmax=60.0, dt0=2e-3):
    """Float IFRK4 of the exact-triangular hierarchy (modes 1..M are the true
    solution's modes); maps S_k(t) against the ladder bars. PROBE ONLY."""
    sig = float(sigma_txt)
    lam = nu*np.power(np.arange(1, M + 1, dtype=float), sig)
    kmax = int(np.log2(M + 1)) - 1
    ks = list(range(2, kmax + 1))
    bars = {}
    nu_frac = nu.as_integer_ratio() if isinstance(nu, float) else (nu, 1)
    for k in ks:
        A, H = _consts_arb(sigma_txt, nu_frac[0], nu_frac[1], k)
        bars[k] = (A, H)
    q = np.zeros(M); q[0] = 1.0
    t = 0.0
    rec = []
    t0 = time.time()
    nstep = 0
    ktop = ks[-1]
    bar_top = 2.1*bars[ktop][0]
    hold_top = 1.6*bars[ktop][1]
    t_cross = None          # first time the TOP block crosses its bar
    max_steps = 1500000
    while t < tmax and nstep < max_steps:
        Sig = q.sum()
        dt = min(dt0, 0.4/max(Sig, 1.0))
        E = np.exp(-lam*dt/2.0)
        k1 = _nl(q)
        u2 = E*(q + (dt/2)*k1); k2 = _nl(u2)
        u3 = E*q + (dt/2)*k2;   k3 = _nl(u3)
        u4 = E*E*q + dt*E*k3;   k4 = _nl(u4)
        q = E*E*q + (dt/6.0)*(E*E*k1 + 2*E*(k2 + k3) + k4)
        t += dt
        nstep += 1
        if nstep % 25 == 0 or dt < dt0:
            row = [t, q.sum()]
            for k in ks:
                row.append(q[2**k - 1: 2**(k + 1) - 1].sum())
            rec.append(row)
        if t_cross is None and q[2**ktop - 1: 2**(ktop + 1) - 1].sum() >= bar_top:
            t_cross = t
        if t_cross is not None and t >= t_cross + hold_top:
            print("  [scout stops: top-block hold window observed (t_cross=%.4f + %.3f)]"
                  % (t_cross, hold_top))
            break
        if not np.isfinite(q.sum()):
            print("  [scout stops: overflow at t = %.4f]" % t)
            break
    rec = np.array(rec)
    print("scout: sigma=%s nu=%.6f M=%d  (%.1f s, %d steps, t_end=%.3f, Sig_end=%.4g)"
          % (sigma_txt, nu, M, time.time() - t0, nstep, t, q.sum()))
    print("  exact q1 check: |q1 - e^{-nu t}|/e^{-nu t} = %.2e"
          % (abs(q[0] - np.exp(-nu*t))/np.exp(-nu*t)))
    print("  %6s %10s" % ("k0", "2.1*A*") + "  earliest [t1, t1+1.05H] with min S_k >= 2.1*A*  (maxS_k)")
    for i, k in enumerate(ks):
        A, H = bars[k]
        bar = 2.1*A
        Wn = 1.05*H
        Sk = rec[:, 2 + i]
        tt = rec[:, 0]
        hit = None
        jlo = 0
        for jhi in range(len(tt)):
            while tt[jhi] - tt[jlo] > Wn*1.2:
                jlo += 1
            if tt[jhi] - tt[jlo] >= Wn and Sk[jlo:jhi + 1].min() >= bar:
                hit = (tt[jlo], tt[jhi])
                break
        print("  %6d %10.4g   %s   (max %.4g%s)"
              % (k, bar,
                 ("HIT [%.3f, %.3f]" % hit) if hit else "no window in scout range",
                 Sk.max(), "" if Sk.max() < bar else " > bar"))


# ---------------------------------------------------------------- certified march
def stage_blow(sigma_txt, nu_num, nu_den, M, k0, margin_goal=2.0, p=6, jbump=4,
               t_max=60.0):
    """THE CERTIFIED SIGMA-GENERIC BLOWUP MARCH (Lemma L trigger).
    Modes <= M are exact by triangularity, so the trigger needs NO tail block
    and NO landing: certify min over a window of length W = 1.02*H of the
    START-OF-STEP decay hulls of S_{k0} >= margin * A*(k0)."""
    from flint import arb, ctx
    ctx.prec = 120
    nu = nu_num/nu_den
    if M < 2**(k0 + 1) - 1:
        print("O-B4 FAIL: M=%d < 2^{k0+1}-1 = %d" % (M, 2**(k0 + 1) - 1)); return
    Astar_hi, H_hi = _consts_arb(sigma_txt, nu_num, nu_den, k0)
    W = H_hi*1.02
    # block-worst decay rate of the SEED block I_{k0} (O-B1), certified upper:
    if float(sigma_txt) == 2.0:
        lam_blk = float((arb(nu_num)/arb(nu_den)*arb(2**(k0 + 1))**2).upper())
    else:
        lam_blk = float((arb(nu_num)/arb(nu_den)*(arb(2**(k0 + 1))**arb(sigma_txt))).upper())
    lo_i, hi_i = 2**k0 - 1, 2**(k0 + 1) - 1     # numpy indices for modes in I_{k0}
    print("stage blow: sigma=%s nu=%d/%d=%.6f M=%d p=%d k0=%d jbump=%d"
          % (sigma_txt, nu_num, nu_den, nu, M, p, k0, jbump))
    print("  A*(k0) <= %.10g;  H <= %.6e;  W = 1.02*H = %.6e;  goal margin %.1f"
          % (Astar_hi, H_hi, W, margin_goal))
    st = dict(prev=None, win=[], best=(0.0, -1.0), relwmax=0.0, Sigpk=0.0,
              nstep=0, fail=None)
    t0 = time.time()

    def seed_cb(t, h, qm, qr, S_hi):
        st["nstep"] += 1
        Sig = float(np.sum(qm))
        st["Sigpk"] = max(st["Sigpk"], Sig)
        live = qm > 1e-8*max(qm.max(), 1e-300)
        relw = float(np.max(qr[live]/np.maximum(qm[live], 1e-300))) if live.any() else 0.0
        st["relwmax"] = max(st["relwmax"], relw)
        if relw > 1e-4:
            st["fail"] = ("F1-width", t)
            print("  [F1] enclosure width blowup at t=%.4f (relw=%.2e)" % (t, relw))
            return ("FAIL", t)
        if st["prev"] is not None:
            pqm, pqr, pt = st["prev"]
            # START-of-step hull: q_m(s) >= (pqm - pqr)*(1 - lam_m h) on [pt, pt+h]
            decf = max(0.0, 1.0 - lam_blk*h)*(1 - 1e-12)
            Sblk_lo = float(np.sum(np.maximum(pqm[lo_i:hi_i] - pqr[lo_i:hi_i], 0.0)))*decf
            st["win"].append((pt, h, Sblk_lo))
            tw = t - W
            win = st["win"]
            while win and win[0][0] + win[0][1] <= tw:
                win.pop(0)
            if win and win[0][0] <= tw:
                wmin = min(w[2] for w in win)
                if wmin > st["best"][1]:
                    st["best"] = (win[0][0], wmin)
                if wmin >= margin_goal*Astar_hi:
                    t1 = win[0][0]
                    print("  *** O-B1/O-B2/O-B3 CERTIFIED: min over [%.6f, %.6f] of"
                          % (t1, t1 + W))
                    print("      S_%d >= %.6g = %.4f x A*(%d)   (W = %.4e >= H = %.4e)"
                          % (k0, wmin, wmin/Astar_hi, k0, W, H_hi))
                    return ("CERT", t1, wmin)
        st["prev"] = (qm.copy(), qr.copy(), t)
        if t > t_max:
            st["fail"] = ("F2-budget", t)
            return ("FAIL", t)
        return None

    r = cs.sweep_sig(nu_num, nu_den, sigma_txt, M, p, t_max + 1.0, alphaT=0.35,
                     progress=5000, seed_cb=seed_cb, xmax=2.0/2**jbump)
    el = time.time() - t0
    print("  march: %d steps, %.0f s; Sigma_peak(march) = %.4f; max relw(live) = %.2e"
          % (st["nstep"], el, st["Sigpk"], st["relwmax"]))
    if isinstance(r, tuple) and len(r) == 3 and r[0] == "CERT":
        _, t1, wmin = r
        tbar = t1 + W
        print("  *** BLOWUP CERTIFIED at sigma=%s (dyadic block-ladder, sigma-generic) ***" % sigma_txt)
        print("  ==> HIERARCHY THEOREM: for the sigma=%s dissipative hierarchy from" % sigma_txt)
        print("      q_1 = e^{-nu t} at nu = %d/%d = %.6f: Sigma q_m(t) = INFINITY for" % (nu_num, nu_den, nu))
        print("      every t >= %.6f (Lemma L,." % (tbar, wmin/Astar_hi))
        print("  ==> PDE COROLLARY: the sigma=%s dissipative CLM equation on S^1 from" % sigma_txt)
        print("      cos x admits NO smooth solution beyond t = %.6f." % tbar)
        print("  ==> nu-comparison (N18a, sigma-generic Kamke proof): blowup for EVERY")
        print("      nu' <= %.6f at sigma=%s  ==>  nu*(%s) >= %.6f." % (nu, sigma_txt, sigma_txt, nu))
        print("  ==> TRANSPORT (Sakajo Kokyuroku 1326-5 Prop 1, alpha-monotonicity — his")
        print("      Proposition, cited; Nonlinearity-primary read condition inherited):")
        print("      blowup at EVERY sigma' <= %s for nu' <= %.6f — the certified" % (sigma_txt, nu))
        print("      blowup QUADRANT {sigma' <= %s} x {nu' <= %.6f}." % (sigma_txt, nu))
        if float(sigma_txt) < 1.0:
            print("      Bar beaten: Sakajo JMSUT Thm 9 (blowup at nu <= 1/(2e) = 0.183940")
            print("      for every alpha < 1) — sharpened by %.0f%% at sigma=%s." % (100*(nu*2*np.e - 1), sigma_txt))
        else:
            thm5 = 0.5*np.exp(-3.0*float(sigma_txt) - 1.0)
            print("      Bars beaten above the corner: (i) Sakajo JMSUT Thm 5 (blowup at")
            print("      nu <= (1/2)e^{-3 alpha - 1} = %.4e at alpha=%s) — sharpened x%.0f;" % (thm5, sigma_txt, nu/thm5))
            print("      (ii) the transported S09 floor 46/1024 = 0.044922 on (1, 2] (S22 §0")
            print("      A-1, Prop-1 quadrant of the banked sigma=2 certificate) — lifted x%.2f." % (nu/0.044921875))
        tag = str(sigma_txt).replace(".", "p")
        fn = os.path.join(DIR, "lad_cert_s%s_%d_%d_M%d.npz" % (tag, nu_num, nu_den, M))
        np.savez(fn, sigma=sigma_txt, nu_num=nu_num, nu_den=nu_den, M=M, p=p, k0=k0,
                 t1=t1, W=W, H_hi=H_hi, S_lo=wmin, Astar_hi=Astar_hi,
                 margin=wmin/Astar_hi, t_bar=tbar, relw_max=st["relwmax"],
                 Sig_peak=st["Sigpk"], nstep=st["nstep"], lam_blk=lam_blk)
        print("  certificate saved: %s" % os.path.basename(fn))
        return
    print("  RESULT: %s — honest failure, no certificate (best window bound %.4g = %.2f x A* at t=%.4f)"
          % (st["fail"][0] if st["fail"] else "no-trigger", st["best"][1],
             st["best"][1]/Astar_hi if st["best"][1] > 0 else 0.0, st["best"][0]))


if __name__ == "__main__":
    a = sys.argv[1]
    if a.startswith("const:"):
        _, s, nn, nd, k0 = a.split(":")
        stage_const(s, int(nn), int(nd), int(k0))
    elif a == "repro":
        stage_repro()
    elif a.startswith("scout:"):
        pp = a.split(":")
        stage_scout(pp[1], float(pp[2]), int(pp[3]),
                    tmax=float(pp[4]) if len(pp) > 4 else 60.0,
                    dt0=float(pp[5]) if len(pp) > 5 else 2e-3)
    elif a.startswith("blow:"):
        pp = a.split(":")
        stage_blow(pp[1], int(pp[2]), int(pp[3]), int(pp[4]), int(pp[5]),
                   margin_goal=float(pp[6]) if len(pp) > 6 else 2.0,
                   p=int(pp[7]) if len(pp) > 7 else 6,
                   jbump=int(pp[8]) if len(pp) > 8 else 4,
                   t_max=float(pp[9]) if len(pp) > 9 else 60.0)
    else:
        print("unknown stage", a)
