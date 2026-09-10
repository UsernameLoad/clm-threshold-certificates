#!/usr/bin/env python
"""
s39_p3f.py -- THE PROFILE-STASH ADDITIVE DIAGNOSTIC for the sigma-dial ride.  Same design as
s35_p3d.py (certified S35: 22/22 bit-identical) and s36_p3e.py (S36: 4/4):
BSQSF(BSQSE) overrides ONLY diagnostics(), calls the parent FIRST (which itself
calls BSQSD.diagnostics -> the certified BSQS.diagnostics and stashes the S35
and S36 observers), then stashes ADDITIONAL pure observers -- the y-PROFILE of
the B* x-modes, so that no interior row has to be placed in advance (the S36
lesson: an interior-row discriminator for a spectral question is placed by the
spectral content's y-profile):
    pth      : |rfft_x theta| on EVERY CGL row at the columns [k_d] + B*
               (its max over y and its argmax row are the p3e keys Eth / iy_th
               restricted to those columns -- bit-identical by construction)     [(N+1) x 27]
    gth, gw  : the COMPLEX x-modes [k_d] + B* of theta and omega on the FIXED
               physical y-grid YG (0, log-spaced 1e-6..1e-1, the hint rows
               1e-3/2e-3/3e-3/5e-3 and the S36 rows 0.01/0.05) by EXACT Chebyshev
               evaluation of the spectral representation, f(y0, x) = sum_k c_k
               T_k(xi0), c = Cheb.coeff(f) -- the p3e formula; the same grid at
               every N                                                          [NG x 27] each
    th_g001, th_g005 : the theta rows at y = 0.01 / 0.05 from the grid evaluation
               (the identity with the p3e keys th_y001 / th_y005)                [Nx] each
    th_bot   : the driver-wall row again (the alignment key)
-- nothing returned to the ride changes.  The stash goes to a SEPARATE side
file p3f_<tagS>_k<kappa>_N<N>.npz; the S35 side file p3d_* and the S36 side
file p3e_* are ALSO written, exactly as s36_p3e.stage_p3e writes them (so the
S35 and S36 keys are reproduced too); the p3 record is renamed exactly as
stage_p3 renames it.  The stage REFUSES to run if any target filename exists.

bsq_solver.py, s20_sig2d.py, s35_p3d.py and s36_p3e.py are NOT edited.

Stage:  p3f:<sigma>:<N>:<kappa>:<cfl>:<tag>[:A]
"""
import os
import sys
import time

import numpy as np
from scipy.fft import rfft

import bsq_solver
from bsq_solver import BSQ
import s35_p3d
import s36_p3e
from s36_p3e import BSQSE
from s35_strip import BSTAR, kd_index

LAST = []
Y4 = [1e-3, 2e-3, 3e-3, 5e-3]
YG = np.unique(np.concatenate([[0.0], np.logspace(-6, -1, 161), Y4, [0.01, 0.05]]))
COLS = [kd_index] + list(BSTAR)


class BSQSF(BSQSE):
    """BSQSE + the y-profile observers of the B* modes, taken AFTER the parent."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.stash_f = []
        N = self.Ny
        kk = np.arange(N + 1)
        xi0 = 2.0 * YG - 1.0
        self._TG = np.cos(np.outer(np.arccos(xi0), kk))          # [NG x (N+1)]: T_k(xi0) for every grid row
        self._ig001 = int(np.argmin(np.abs(YG - 0.01)))
        self._ig005 = int(np.argmin(np.abs(YG - 0.05)))
        assert abs(YG[self._ig001] - 0.01) < 1e-15 and abs(YG[self._ig005] - 0.05) < 1e-15
        LAST.append(self)

    def diagnostics(self, w, th, psi=None):
        d = super().diagnostics(w, th, psi)           # BSQSE (-> BSQSD -> BSQS first): unchanged
        A_th = np.abs(rfft(th, axis=1, workers=self.wk))
        c_th = self.ch.coeff(th)
        c_w = self.ch.coeff(w)
        rows_th = self._TG @ c_th                      # [NG x Nx]: theta on the grid rows
        rows_w = self._TG @ c_w
        G_th = rfft(rows_th, axis=1, workers=self.wk)
        G_w = rfft(rows_w, axis=1, workers=self.wk)
        self.stash_f.append(dict(pth=A_th[:, COLS].copy(), gth=G_th[:, COLS].copy(), gw=G_w[:, COLS].copy(),
                                 th_g001=rows_th[self._ig001].copy(), th_g005=rows_th[self._ig005].copy(),
                                 th_bot=th[self.Ny].copy()))
        return d


def stage_p3f(sigma, N=256, kappa=1e-2, cfl=0.4, tag="", A=1.0e4):
    class _Sub(BSQ):
        def __new__(cls, *a, **kw):
            return BSQSF(*a, sigma=sigma, **kw)
    tagS = ("s%s" % ("%.3f" % sigma).replace(".", "p")) + (("_" + tag) if tag else "")
    ktag = ("%.6f" % kappa).replace(".", "p")
    asuf = "" if A == 1.0e4 else "_A%g" % A
    src = "p2_k%s_N%d%s_%s.npz" % (ktag, N, asuf, tagS)
    dst = "p3_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    side_d = "p3d_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    side_e = "p3e_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    side_f = "p3f_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    for fn in (src, dst, side_d, side_e, side_f):
        if os.path.exists(fn):
            print("REFUSED: %s already exists (never overwrite a banked record; choose a fresh tag)" % fn, flush=True)
            sys.exit(2)
    old = bsq_solver.BSQ
    n0 = len(LAST); n0e = len(s36_p3e.LAST); n0d = len(s35_p3d.LAST)
    try:
        bsq_solver.BSQ = _Sub
        print("stage p3f[sigma=%g]: the sigma-dial level + the S35 stash + the S36 extended observers + the PROFILE stash (YG %d rows, %d columns); "
              "ride loop = certified stage_p2 verbatim (class substitution as stage_p3)" % (sigma, len(YG), len(COLS)), flush=True)
        bsq_solver.stage_p2(N=N, kappa=kappa, cfl=cfl, tag=tagS, A=A)
    finally:
        bsq_solver.BSQ = old
    if not os.path.exists(src):
        print("ERROR: expected %s not written" % src, flush=True)
        sys.exit(3)
    os.rename(src, dst)
    print("  record renamed -> %s" % dst, flush=True)
    S = LAST[-1]
    assert len(LAST) == n0 + 1 and len(s36_p3e.LAST) == n0e + 1 and len(s35_p3d.LAST) == n0d + 1, "exactly one solver instance expected per ride"
    st = S.stash; se = S.stash_e; sf = S.stash_f
    z = np.load(dst)
    t = np.asarray(z["t"])
    if len(st) != len(t) + 1 or len(se) != len(t) + 1 or len(sf) != len(t) + 1:
        print("ALIGNMENT FAULT: %d / %d / %d stash entries vs %d saved records + 1" % (len(st), len(se), len(sf), len(t)), flush=True)
        sys.exit(4)
    sup_ok = all(st[i + 1]["sup_w"] == float(z["sup_w"][i]) for i in range(len(t)))
    dx_ok = all((st[i + 1]["delta_x"] == float(z["delta_x"][i])) or (np.isnan(st[i + 1]["delta_x"]) and np.isnan(z["delta_x"][i])) for i in range(len(t)))
    last_ok = np.array_equal(st[-1]["th_bot"], z["th_final"][N])
    last_ok_e = np.array_equal(se[-1]["th_bot"], z["th_final"][N])
    last_ok_f = np.array_equal(sf[-1]["th_bot"], z["th_final"][N])
    # the in-ride identities of the profile stash against the S36 stash (every record)
    ic = [COLS.index(k) for k in COLS]
    eth_ok = all(np.array_equal(sf[i]["pth"].max(axis=0), np.asarray(se[i]["Eth"])[COLS]) for i in range(len(sf)))
    iy_ok = all(np.array_equal(sf[i]["pth"].argmax(axis=0).astype(np.int32), np.asarray(se[i]["iy_th"])[COLS]) for i in range(len(sf)))
    def relmax(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    r001 = max(relmax(sf[i]["th_g001"], se[i]["th_y001"]) for i in range(len(sf)))
    r005 = max(relmax(sf[i]["th_g005"], se[i]["th_y005"]) for i in range(len(sf)))
    print("  stash alignment: sup_w bit-identical %s ; delta_x bit-identical %s ; last stash th_bot == th_final[Ny] %s (S35) / %s (S36) / %s (profile)"
          % (sup_ok, dx_ok, last_ok, last_ok_e, last_ok_f), flush=True)
    print("  profile identities: max_y pth == Eth[cols] bit-identical at every record %s ; argmax_y pth == iy_th[cols] %s ; grid rows y=0.01 / 0.05 vs the p3e Chebyshev rows: max rel diff %.2e / %.2e"
          % (eth_ok, iy_ok, r001, r005), flush=True)
    # the S35 side file, EXACTLY as s35_p3d / s36_p3e write it
    out = dict(t=t, t0_included=True,
               th_bot=np.array([s["th_bot"] for s in st[1:]]), th_top=np.array([s["th_top"] for s in st[1:]]),
               Ew=np.array([s["Ew"] for s in st[1:]]), Ew_max=np.array([s["Ew_max"] for s in st[1:]]),
               Ew_wall=np.array([s["Ew_wall"] for s in st[1:]]),
               th_bot_t0=st[0]["th_bot"], th_top_t0=st[0]["th_top"], Ew_t0=st[0]["Ew"], Ew_max_t0=st[0]["Ew_max"],
               sup_w_chk=np.array([s["sup_w"] for s in st[1:]]), tail_x_chk=np.array([s["tail_x"] for s in st[1:]]),
               delta_x_chk=np.array([s["delta_x"] for s in st[1:]]), strip_r2_chk=np.array([s["strip_r2"] for s in st[1:]]),
               tcert=float(z["tcert"]), gcert=float(z["gcert"]), om_pk=float(z["om_pk"]), itx=float(z["itx"]),
               verdict=str(z["verdict"]), N=N, kappa=kappa, sigma=sigma, cfl=cfl, A=A, Lx=float(z["Lx"]),
               source_p3=dst, align_ok=bool(sup_ok and dx_ok and last_ok))
    np.savez_compressed(side_d, **out)
    # the S36 side file, EXACTLY as s36_p3e writes it
    oute = dict(t=t, t0_included=True,
                Eth=np.array([s["Eth"] for s in se[1:]]), iy_th=np.array([s["iy_th"] for s in se[1:]]), iy_w=np.array([s["iy_w"] for s in se[1:]]),
                th_y005=np.array([s["th_y005"] for s in se[1:]]), th_y001=np.array([s["th_y001"] for s in se[1:]]),
                w_y005=np.array([s["w_y005"] for s in se[1:]]), w_y001=np.array([s["w_y001"] for s in se[1:]]),
                th_bot=np.array([s["th_bot"] for s in se[1:]]),
                Eth_t0=se[0]["Eth"], th_y005_t0=se[0]["th_y005"], th_y001_t0=se[0]["th_y001"],
                y_rows=np.array([0.05, 0.01]), node_check_index=S._inode,
                tcert=float(z["tcert"]), verdict=str(z["verdict"]), N=N, kappa=kappa, sigma=sigma, cfl=cfl, A=A, Lx=float(z["Lx"]),
                source_p3=dst, source_p3d=side_d, align_ok=bool(sup_ok and dx_ok and last_ok and last_ok_e))
    np.savez_compressed(side_e, **oute)
    # the NEW profile side file
    outf = dict(t=t, t0_included=True, yg=YG, cols=np.array(COLS, dtype=np.int32),
                pth=np.array([s["pth"] for s in sf[1:]]), gth=np.array([s["gth"] for s in sf[1:]]), gw=np.array([s["gw"] for s in sf[1:]]),
                th_g001=np.array([s["th_g001"] for s in sf[1:]]), th_g005=np.array([s["th_g005"] for s in sf[1:]]),
                th_bot=np.array([s["th_bot"] for s in sf[1:]]),
                pth_t0=sf[0]["pth"], gth_t0=sf[0]["gth"], gw_t0=sf[0]["gw"],
                tcert=float(z["tcert"]), verdict=str(z["verdict"]), N=N, kappa=kappa, sigma=sigma, cfl=cfl, A=A, Lx=float(z["Lx"]),
                source_p3=dst, source_p3d=side_d, source_p3e=side_e,
                eth_identity=bool(eth_ok), iy_identity=bool(iy_ok), row001_reldiff=r001, row005_reldiff=r005,
                align_ok=bool(sup_ok and dx_ok and last_ok and last_ok_e and last_ok_f and eth_ok and iy_ok))
    np.savez_compressed(side_f, **outf)
    print("  side files saved -> %s, %s and %s  [%d records x N=%d ; align_ok %s]" % (side_d, side_e, side_f, len(t), N, outf["align_ok"]), flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st.startswith("p3f"):
        p = st.split(":")
        stage_p3f(float(p[1]),
                  N=int(p[2]) if len(p) > 2 else 256,
                  kappa=float(p[3]) if len(p) > 3 else 1e-2,
                  cfl=float(p[4]) if len(p) > 4 else 0.4,
                  tag=p[5] if len(p) > 5 else "",
                  A=float(p[6]) if len(p) > 6 else 1.0e4)
    else:
        print("stages: p3f:<sigma>[:N[:kappa[:cfl[:tag[:A]]]]]")
