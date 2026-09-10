#!/usr/bin/env python
# S31 — THE OUTER-COUPLING DERIVATION PASS (.txt; zero rides).
# Stages (deterministic; every consumed instrument reproduced bit-identically
# this session — s31_repro_{verify,proj,grun,g2c}.txt, B-0a):
#   verify — B-1: the Taylor structure of the outer x front cross term
#            exact-sum-verified on each F1 vector (Delta_j, j = 1..7, the G-3
#            object; the exact forcing vs sum_k (-1)^k S_k/k! F^{(k)} on
#            m in [64, mhi]); the row-structure labels (m^0: S0*D ;
#            m^(s-2): -S1*A(s-1)) checked by the G-3 row-basis LSQ of the
#            EXACT forcing (the G-3 R_out column is the S30 reference).
#   c0     — B-2: the peak-dead c0 clause: |S0*D-hat(t_pk)| vs |R|_abs.
#   hout   — B-3 (+ B-5): the outer-forced h memory integral along the four
#            banked snap columns (s28_memint arithmetic verbatim with the
#            forcing F_h -> F_h(S28 FULL) + F_h^out(t)); dev_pred vs the
#            frozen dev; variants; the h sign history + the B-6 cap.
#   eleg   — B-4: the opened-slot e-hat re-measurement (s30_g2c.fit5
#            verbatim) on the ladders + the single-vector report rows.
import os, sys
import numpy as np
from math import gamma, pi, sin, log
from s25_threeterm import fit_slot
from s30_dclose import FROZEN, F1VEC, KWIN, KWIN_REPORT, row_basis, vec_fit, shape_vec, zeta, B
from s30_g2c import fit5, slope_lsq
import s28_memint as sm
import s27_theory as st

sys.stdout.reconfigure(encoding="utf-8")
DIR = os.path.dirname(os.path.abspath(__file__))
JOUT = 7   # the outer field's support (j = 1..7, the G-3 definition)

# frozen projection / requirement constants (s31_regcalc.txt, cross-checked)
DEH = {1.25: +2.3498, 1.375: +3.4768, 1.5: +5.5026, 1.75: +6.6721, 1.9: +3.4524}
XREQ = {1.25: +9.13801e-04, 1.375: +7.63213e-04, 1.5: +3.23078e-04, 1.75: -5.68642e-04, 1.9: -2.65483e-03}
DEV = {s: FROZEN[s][1] for s in FROZEN}

def ff(p, k):
    """falling factorial p(p-1)...(p-k+1)"""
    out = 1.0
    for i in range(k):
        out *= (p - i)
    return out

def outer_field(q, r, s, mhi):
    """the G-3 object in the e^{+delta j} frame: Dl_j = q_j e^{delta j} - F(j), j <= JOUT"""
    jj = np.arange(1, mhi + 1, dtype=float)
    Av, Dv, Ev, dv = r["A"], r["D"], r["E"], r["d"]
    Fj = Av*jj**(s - 1.0) + Dv + Ev*jj**(-s)
    qq = q[:mhi]*np.exp(dv*jj)
    Dl = qq[:JOUT] - Fj[:JOUT]
    return jj, Fj, Dl

def exact_forcing(Dl, Fj, wm):
    """Phi-tilde(m) = sum_j Dl_j F(m-j) + (1/2) sum Dl_j Dl_l  (G-3 verbatim)"""
    dout = np.empty(len(wm))
    for i, m in enumerate(wm):
        tot = 0.0
        for j in range(1, JOUT + 1):
            if m - j >= 1:
                tot += Dl[j - 1]*Fj[m - j - 1]
        if m <= 2*JOUT:
            for j in range(1, JOUT + 1):
                l = m - j
                if 1 <= l <= JOUT:
                    tot += 0.5*Dl[j - 1]*Dl[l - 1]
        dout[i] = tot
    return dout

def taylor_forcing(Dl, r, s, wm, K=3):
    """T_K(m) = sum_{k<=K} (-1)^k (S_k/k!) F^{(k)}(m); F = A m^(s-1) + D + E m^(-s)"""
    j = np.arange(1, JOUT + 1, dtype=float)
    S = [float(np.sum(j**k*Dl)) for k in range(K + 1)]
    Av, Dv, Ev = r["A"], r["D"], r["E"]
    m = wm.astype(float)
    T = np.zeros(len(m))
    fact = 1.0
    for k in range(K + 1):
        if k > 0:
            fact *= k
        Fk = Av*ff(s - 1.0, k)*m**(s - 1.0 - k) + Ev*ff(-s, k)*m**(-s - k) + (Dv if k == 0 else 0.0)
        T += (-1)**k*(S[k]/fact)*Fk
    return T, S

# ---------------------------------------------------------------- verify (B-1)
def mode_verify():
    print("=== B-1: the outer x front Taylor structure, exact-sum-verified on the F1 vectors ===")
    print("Delta_j := q_j e^{delta j} - F(j) (j = 1..7, the G-3 object); Phi(m) = sum_j Delta_j F(m-j) + Delta*Delta/2 (exact);")
    print("T_3(m) = sum_{k<=3} (-1)^k (S_k/k!) F^{(k)}(m), S_k = sum_j j^k Delta_j; labels: m^0 <- S_0 D ; m^(s-2) <- -S_1 A (s-1)")
    allok = True
    for fn, s, cls in F1VEC:
        nu, dev, k, dkreq, Rabs, breq = FROZEN[s]
        z, q, r, mm, mhi, sel = vec_fit(fn, s)
        Av, Dv, Ev, dv = r["A"], r["D"], r["E"], r["d"]
        jj, Fj, Dl = outer_field(q, r, s, mhi)
        wm = mm[sel].astype(int)
        Phi = exact_forcing(Dl, Fj, wm)
        T3, S = taylor_forcing(Dl, r, s, wm, K=3)
        hi = wm >= 64
        lo = wm < 64
        rel_hi = float(np.sqrt(np.mean((Phi[hi] - T3[hi])**2))/np.sqrt(np.mean(Phi[hi]**2))) if hi.any() else float("nan")
        rel_lo = float(np.sqrt(np.mean((Phi[lo] - T3[lo])**2))/np.sqrt(np.mean(Phi[lo]**2))) if lo.any() else float("nan")
        # the G-3 row-basis LSQ of the EXACT forcing (the S30 arithmetic verbatim)
        Jr, names = row_basis(s, wm.astype(float))
        cf = np.linalg.lstsq(Jr, Phi, rcond=None)[0]
        c0 = cf[names.index("m^0")]
        hname = "m^(s-2)" if "m^(s-2)" in names else "m^(1-s)"   # merged column at the 1.5 collision
        ch = cf[names.index(hname)]
        lab_h = -S[1]*Av*(s - 1.0)
        lab_0 = S[0]*Dv
        dev_h = abs(ch/lab_h - 1.0) if lab_h != 0 else float("inf")
        leak0 = abs(c0 - lab_0)
        ok_T = rel_hi <= 1e-3
        ok_h = dev_h <= 0.20
        ok_0 = leak0 <= Rabs/10.0
        allok &= (ok_h and ok_0)
        print("--- %s sigma=%.4f [%s] window [8, %d]  Delta(1..7) = %s" % (fn, s, cls, mhi, ", ".join("%+.2e" % x for x in Dl)))
        print("    S_0=%+.4e S_1=%+.4e S_2=%+.4e S_3=%+.4e   (S_0 = the G-3 S0; sign(S_1) = %+d)" % (S[0], S[1], S[2], S[3], int(np.sign(S[1]))))
        print("    Taylor T_3 vs exact: rel rms = %.2e on m in [64, %d] (tol 1e-3: %s) ; %.2e on [8, 64) (report)"
              % (rel_hi, mhi, "PASS" if ok_T else "FAIL — exact Phi used", rel_lo))
        print("    row-basis LSQ of the exact forcing: c[m^0] = %+.4e vs label S_0 D = %+.4e (|diff| = %.2e vs |R|/10 = %.2e: %s)"
              % (c0, lab_0, leak0, Rabs/10.0, "PASS" if ok_0 else "FAIL"))
        print("    c[%s] = %+.4e vs label -S_1 A (s-1) = %+.4e (rel dev %.3f, tol 0.20: %s)"
              % (hname, ch, lab_h, dev_h, "PASS" if ok_h else "FAIL"))
        print("    G-3 cross-check: c[m^0] reproduces the S30 R_out column (s30_grun.txt) at this row: %+.3e" % c0)
    print(">> B-1 VERDICT: %s" % ("the row-structure labels are VERIFIED at every F1 row (B-3 may consume -S_1 A (s-1))"
                                  if allok else "a label FAILED at some row — see rows; B-3 STOPS at the failed rows"))

# ---------------------------------------------------------------- c0 (B-2)
def mode_c0():
    print("=== B-2: THE c0-CLAUSE — the derived m^0-row outer term S_0 D-hat at the peak vector's own D-hat(t_pk) ===")
    verd = {}
    for fn, s, cls in F1VEC:
        nu, dev, k, dkreq, Rabs, breq = FROZEN[s]
        z, q, r, mm, mhi, sel = vec_fit(fn, s)
        jj, Fj, Dl = outer_field(q, r, s, mhi)
        S0 = float(np.sum(Dl))
        Dv = r["D"]
        term = S0*Dv
        rat = abs(term)/Rabs
        sgn_match = (np.sign(term) == np.sign(dev))
        if rat < 0.1:
            v = "EXCLUDED"
        elif rat >= 0.2 and sgn_match:
            v = "LIVE (price it)"
        else:
            v = "UNRESOLVED-at-provenance"
        verd[s] = v
        print("--- %s sigma=%.4f [%s]: S_0 = %+.4e ; D-hat(t_pk) = %+.4e ; S_0 D-hat = %+.3e ; |R|_abs = %.3e ; ratio = %.2e ; sign(R)=%+d ; %s"
              % (fn, s, cls, S0, Dv, term, Rabs, rat, int(np.sign(dev)), v))
    exc_all = all(v == "EXCLUDED" for v in verd.values())
    print(">> B-2 CLAUSE: %s" % ("the c0-channel is EXCLUDED at EVERY row (|S_0 D-hat| < |R|/10) — the fifth boundary's first leg"
                                  if exc_all else "NOT excluded at some row — see rows"))

# ---------------------------------------------------------------- hout (B-3 + B-5)
def col_fits_idx(z, sig):
    ts = z["snaps_t"]; cs = z["snaps_c"]
    rows = []
    for i in range(len(ts)):
        q = np.abs(cs[i])
        mmax_good = int(np.max(np.arange(1, len(q) + 1)[q > q.max()*1e-12]))
        if int(0.4*mmax_good) < 12:
            continue
        r = fit_slot(q, sig, -sig)
        rows.append((i, float(ts[i]), r, q, int(0.4*mmax_good)))
    return rows

def mode_hout():
    print("=== B-3: THE OUTER-FORCED h ACCOUNT (the s28_memint arithmetic verbatim + F_h^out(t) = -S_1(t) A(t) (s-1)) ===")
    print("(S28 FULL model, IC = 0 at the first snapshot, exponential midpoint NSUB = %d, 2x echo; projection de from s28_memint.proj_rows)" % sm.NSUB)
    devpred = {}
    signs = {}
    for (fn, sig) in sm.COLS:
        nu, tpk, t, A, D, E, rows = sm.column_arrays(fn, sig)
        z = np.load(os.path.join(DIR, fn))
        ridx = col_fits_idx(z, sig)
        assert len(ridx) == len(t) and np.allclose([x[1] for x in ridx], t)
        S0t = np.empty(len(t)); S1t = np.empty(len(t))
        for n, (i, ti, r, q, mhi) in enumerate(ridx):
            jj, Fj, Dl = outer_field(q, r, sig, max(mhi, JOUT + 1))
            j = np.arange(1, JOUT + 1, dtype=float)
            S0t[n] = float(np.sum(Dl)); S1t[n] = float(np.sum(j*Dl))
        Fout = -S1t*A*(sig - 1.0)
        # the S28 primary pieces (verbatim call)
        h28, f28, w28, FL, ihdt28 = sm.slot_integrals(sig, nu, tpk, t, A, D, E, full=True, ic_qs=False)
        Fh, Lh, Ff, Lf, Fw, Lw = FL
        h_out = sm.integrate_linear(t, Fh + Fout, Lh, tpk, 0.0, sm.NSUB)
        h_out2 = sm.integrate_linear(t, Fh + Fout, Lh, tpk, 0.0, 2*sm.NSUB)
        h_oo = sm.integrate_linear(t, Fout, Lh, tpk, 0.0, sm.NSUB)
        # projection responses
        powers = sm.uniq_powers(sig)
        out, orth, rpk, mhi = sm.proj_rows(fn, sig, powers)
        de_h = out[min(powers, key=lambda p: abs(p - (sig - 2.0)))][2]
        de_f = out[min(powers, key=lambda p: abs(p - (1.0 - 2.0*sig)))][2]
        de_w = out[min(powers, key=lambda p: abs(p - (1.0 - sig)))][2]
        dp_primary = de_h*h_out + de_f*f28 + de_w*w28
        dp_honly = de_h*h_out
        dp_outer = de_h*h_oo
        dp_s28 = de_h*h28 + de_f*f28 + de_w*w28
        devpred[sig] = dp_primary
        # sign history of h along the path (total forcing) + int h dt
        hpath = [0.0]; y = 0.0
        for i in range(len(t) - 1):
            y = sm.integrate_linear(t[i:i+2], (Fh + Fout)[i:i+2], Lh[i:i+2], t[i+1], y, sm.NSUB)
            hpath.append(y)
        hpath = np.array(hpath)
        tcap = np.minimum(t, tpk)
        ihdt = float(np.trapezoid(np.where(t <= tpk, hpath, hpath[-1]), tcap)) if t[-1] >= tpk else float(np.trapezoid(hpath, t))
        nsign = int(np.sum(np.diff(np.sign(hpath[np.abs(hpath) > 0])) != 0))
        Bg = sm.Bcont(sig, sig - 1.0)
        gcoup = float(np.mean(A))*Bg - nu
        Xg_max = rpk["rms"]/orth[min(powers, key=lambda p: abs(p - (2.0*sig - 2.0)))]
        cap = 2.0*Xg_max/abs(gcoup)
        print("--- %s sigma=%.3f nu=%.4f t0=%.4f t_pk=%.4f rows=%d" % (fn, sig, nu, t[0], tpk, len(t)))
        print("    S_0(t): [%+.3e .. %+.3e]  S_1(t): [%+.3e .. %+.3e]  sign(S_1) %s ; F_h^out in [%+.3e, %+.3e] ; F_h(S28) in [%+.3e, %+.3e]"
              % (S0t.min(), S0t.max(), S1t.min(), S1t.max(),
                 "SINGLE-SIGNED" if S1t.min()*S1t.max() > 0 else "SIGN-CHANGES", Fout.min(), Fout.max(), Fh.min(), Fh.max()))
        print("    S28 repro: h(t_pk) = %+.4e (s28_run.txt FULL IC=0 reference) ; f = %+.4e ; w = %+.4e" % (h28, f28, w28))
        print("    h_out(t_pk) = %+.4e (echo 2x: |dh| = %.1e) ; outer-only h = %+.4e ; de_h = %+.4f de_f = %+.4f de_w = %+.4f"
              % (h_out, abs(h_out2 - h_out), h_oo, de_h, de_f, de_w))
        print("    dev_pred PRIMARY (3-slot, outer-forced h) = %+.4e ; h-only = %+.4e ; outer-only = %+.4e ; S28-only = %+.4e ; FROZEN dev = %+.4e ; X_req(s-2) = %+.4e"
              % (dp_primary, dp_honly, dp_outer, dp_s28, DEV[sig], XREQ[sig]))
        print("    B-5: h(t) sign changes along the path = %d ; int h dt = %+.3e vs the B-6 cap %.3e (%s) ; g-row coeff = %+.4f"
              % (nsign, ihdt, cap, "inside" if abs(ihdt) <= cap else "OUTSIDE (single-signed-h cap)", 2.0*B(sig, 2.0*sig - 1.0)/B(sig, sig) - 1.0))
        signs[sig] = (np.sign(dp_primary) == np.sign(DEV[sig]))
    print("=== SCORING vs the FROZEN dev (PRIMARY variant) ===")
    for sig in (1.25, 1.375, 1.5, 1.75):
        print("  sigma=%.3f  dev_pred=%+.4e  dev=%+.4e  ratio=%+.3f  sign %s"
              % (sig, devpred[sig], DEV[sig], devpred[sig]/DEV[sig], "MATCH" if signs[sig] else "MISS"))
    nhit = sum(1 for s in signs if signs[s])
    m175 = signs[1.75]
    rat175 = devpred[1.75]/DEV[1.75]
    prices = m175 and nhit >= 3 and (0.2 <= rat175 <= 5.0)
    print("  B-3: %d/4 sign matches; mandatory sigma=1.75 row: %s; ratio at 1.75 = %+.3f  ==> %s"
          % (nhit, "MATCH" if m175 else "MISS", rat175,
             "PRICES — POSITIVE BRANCH ()" if prices
             else ("sign-pass, LIVE-unpriced (ratio outside [1/5, 5])" if (m175 and nhit >= 3)
                   else "FAILED (the mandatory sign row or the 3/4 clause) -> the registered letters bind")))
    if 1.5 in devpred and 1.75 in devpred:
        dpi = devpred[1.5] + (devpred[1.75] - devpred[1.5])*(1.59375 - 1.5)/0.25
        print("  interpolated dev_pred(1.59375) = %+.4e (sign %+d) ["
              % (dpi, int(np.sign(dpi))))
    dp19 = devpred[1.5] + (devpred[1.75] - devpred[1.5])*(1.9 - 1.5)/0.25
    print("  B-4-class [MODELED-EXTRAPOLATION, report]: linear-in-sigma dev_pred(1.9) = %+.4e vs frozen %+.4e" % (dp19, DEV[1.9]))

# ---------------------------------------------------------------- eleg (B-4)
def letters(shift_e, devpp, gain, cgstab, dev):
    if abs(shift_e) <= abs(dev)/5.0:
        return "CONTENT-ROBUST"
    if abs(devpp) <= abs(dev)/5.0 and gain >= 0.05 and cgstab <= 0.5:
        return "PRICES"
    if gain < 0.05 or cgstab > 0.5:
        return "DEGENERATE-UNRESOLVED"
    return "SHIFTED-NOT-PRICING"

def mode_eleg():
    print("=== B-4: THE OPENED-SLOT e-hat RE-MEASUREMENT (G-2c's twin on the e-hat leg; s30_g2c.fit5 verbatim) ===")
    print("(shift_e = e-hat'(iz) - e-hat_4(iz) at the D-zero-adjacent snapshot; gain = 1 - rms'/rms_4; cg-stability = std/|mean| over the n=4 window;")
    print(" dev'' = dev + shift_e - e_pred (k'/k - 1); letters per s31_reg.txt B-4)")
    verdicts = {}
    ladders = [(s, KWIN[s][3], KWIN[s][0], KWIN[s][1], "scored") for s in (1.25, 1.50, 1.75)] + \
              [(1.375, KWIN_REPORT[1.375], None, None, "REPORT")]
    for s, fn, h4, h2, cls in ladders:
        nu, dev, k, dkreq, Rabs, breq = FROZEN[s]
        ehat_frozen = -dev/breq
        e_pred = ehat_frozen - dev
        z = np.load(os.path.join(DIR, fn))
        snaps = np.abs(z["snaps_c"]); ts = np.array(z["snaps_t"], dtype=float)
        n = snaps.shape[0]
        base = [fit_slot(snaps[i], s, -s) for i in range(n)]
        print("--- sigma=%.4f ladder %s [%s] (frozen dev = %+.4e, e-hat_frozen = %+.4e, e_pred = %+.4e, k/nu2 = %.4f)"
              % (s, fn, cls, dev, ehat_frozen, e_pred, k))
        shapes = ["h", "w", "f", "U"] if abs(s - 1.5) > 1e-9 else ["w", "f", "U"]
        row = {}
        for nm in shapes:
            fits = []
            okall = True
            for i in range(n):
                f5 = fit5(snaps[i], s, nm, base[i])
                if f5 is None:
                    okall = False
                    break
                fits.append(f5)
            if not okall:
                print("    shape %-2s: 5-param fit FAILED to converge — not scoreable" % nm)
                row[nm] = "unscoreable"
                continue
            Dcol = np.array([f["D"] for f in fits])
            cgcol = np.array([f["cg"] for f in fits])
            E5 = np.array([f["E"] for f in fits]); E4 = np.array([b["E"] for b in base])
            r5 = np.array([f["rms"] for f in fits]); r4 = np.array([b["rms"] for b in base])
            zc = np.where(np.diff(np.sign(Dcol)) != 0)[0]
            if len(zc) == 0:
                print("    shape %-2s: no D-zero in the ladder — not scoreable" % nm)
                row[nm] = "unscoreable"
                continue
            iz = int(zc[0])
            t0 = float(ts[iz] + (ts[iz + 1] - ts[iz])*Dcol[iz]/(Dcol[iz] - Dcol[iz + 1]))
            if h4 is not None:
                in4 = [i for i in range(n) if abs(ts[i] - t0) <= h4 + 1e-12]
                in2 = [i for i in range(n) if abs(ts[i] - t0) <= h2 + 1e-12]
            else:
                in4, in2 = [], []
            if len(in2) < 2 or len(in4) < 3:
                in2 = [iz, iz + 1]
                in4 = list(range(max(0, iz - 1), min(n, iz + 3)))
            k4 = abs(slope_lsq(ts[in4], Dcol[in4])); k2 = abs(slope_lsq(ts[in2], Dcol[in2]))
            kR = (4.0*k2 - k4)/3.0
            stable = abs(k2/k4 - 1.0) <= 0.05
            kp = kR/(nu*nu)
            shift_e = E5[iz] - E4[iz]
            shift_e_b = E5[iz + 1] - E4[iz + 1]
            gain = 1.0 - r5[iz]/r4[iz]
            cgstab = float(np.std(cgcol[in4])/abs(np.mean(cgcol[in4]))) if abs(np.mean(cgcol[in4])) > 0 else float("inf")
            devpp = dev + shift_e - e_pred*(kp/k - 1.0)
            let = letters(shift_e, devpp, gain, cgstab, dev)
            if not stable:
                let = let + " [k' UNSTABLE>5% — reported]"
            row[nm] = let
            print("    shape %-2s: iz=%d t0=%.4f  e-hat_4=%+.4e e-hat'=%+.4e shift_e=%+.4e (echo iz+1: %+.4e)  cg=%+.3e (std/|mean| over n4 = %.2f)  rms_4=%.2e rms'=%.2e gain=%+.1f%%  k'/nu2=%.4f  dev''=%+.4e (dev %+.4e; |dev|/5 = %.1e)  -> %s"
                  % (nm, iz, t0, E4[iz], E5[iz], shift_e, shift_e_b, cgcol[iz], cgstab, r4[iz], r5[iz], 100*gain, kp, devpp, dev, abs(dev)/5.0, let))
        verdicts[s] = row
    def robust_all(s):
        return all(v.startswith("CONTENT-ROBUST") for v in verdicts.get(s, {}).values()) and len(verdicts.get(s, {})) > 0
    r175 = robust_all(1.75)
    nint = sum(1 for s in (1.25, 1.50) if robust_all(s))
    clause = r175 and nint >= 1
    deg175 = any(v.startswith("DEGENERATE") for v in verdicts.get(1.75, {}).values())
    pr175 = any(v.startswith("PRICES") for v in verdicts.get(1.75, {}).values())
    print(">> B-4 CLAUSE: mandatory 1.75 all-shapes ROBUST = %s ; {1.25, 1.5} all-shapes ROBUST count = %d/2 ; DEGENERATE at 1.75 = %s ; PRICES at 1.75 = %s"
          % (r175, nint, deg175, pr175))
    print(">> B-4 VERDICT: %s" % ("the e-hat leg is CONTENT-ROBUST (clause)" if clause else
                                  ("the e-hat leg is NOT content-robust — letters per row (DEGENERATE present at the mandatory row)" if deg175 else
                                   "the e-hat leg is NOT content-robust — letters per row")))
    print()
    print("=== single-vector REPORT rows (no ladder: letters by gain only; 'would-price' = |dev + shift_e| <= |dev|/5) ===")
    REP = [("sig1p56_nu0p092000_M8192.npz", 1.5625, +8.405e-04),
           ("sig1p62_nu0p084500_M8192.npz", 1.625, -5.6551e-04),
           ("sig1p90_nu0p060500_M2048.npz", 1.9, -9.1656e-03),
           ("snap_sig1p25_nu0p136000_M4096.npz", 1.25, +2.1472e-03),
           ("snapt_sig1p38_nu0p115500_M4096.npz", 1.375, +2.6535e-03),
           ("snap_sig1p50_nu0p098200_M8192.npz", 1.50, +1.7778e-03),
           ("snap_sig1p75_nu0p071500_M4096.npz", 1.75, -3.7940e-03)]
    for fn, s, dev in REP:
        z = np.load(os.path.join(DIR, fn))
        q = np.abs(z["c_pk"])
        base = fit_slot(q, s, -s)
        print("--- %s sigma=%.5f (dev = %+.4e ; e-hat_4 = %+.4e ; rms_4 = %.2e)" % (fn, s, dev, base["E"], base["rms"]))
        shapes = ["h", "w", "f", "U"] if abs(s - 1.5) > 1e-9 else ["w", "f", "U"]
        for nm in shapes:
            try:
                f5 = fit5(q, s, nm, base)
            except Exception as ex:
                print("    shape %-2s: n/a (%s)" % (nm, type(ex).__name__))
                continue
            if f5 is None:
                print("    shape %-2s: 5-param fit failed" % nm)
                continue
            shift_e = f5["E"] - base["E"]
            gain = 1.0 - f5["rms"]/base["rms"]
            if abs(shift_e) <= abs(dev)/5.0:
                let = "ROBUST-class"
            elif gain < 0.05:
                let = "DEGENERATE-class"
            else:
                let = "SHIFT-class"
            wp = abs(dev + shift_e) <= abs(dev)/5.0
            print("    shape %-2s: e-hat'=%+.4e shift_e=%+.4e cg=%+.3e rms'=%.2e gain=%+.1f%%  dev+shift=%+.4e (|dev|/5 = %.1e)  -> %s%s"
                  % (nm, f5["E"], shift_e, f5["cg"], f5["rms"], 100*gain, dev + shift_e, abs(dev)/5.0, let, " [would-price]" if wp else ""))

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if m == "verify": mode_verify()
    elif m == "c0": mode_c0()
    elif m == "hout": mode_hout()
    elif m == "eleg": mode_eleg()
    else: print("modes: verify | c0 | hout | eleg")
