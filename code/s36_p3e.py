#!/usr/bin/env python
"""
s36_p3e.py -- THE EXTENDED ADDITIVE SAVED-FIELD DIAGNOSTIC for the sigma-dial
ride.  Same design as
s35_p3d.py (certified S35: bit-identical on 22/22 rows): BSQSE(BSQSD) overrides
ONLY diagnostics(), calls the parent FIRST (which itself calls the certified
BSQS.diagnostics first and stashes the S35 observers), then stashes ADDITIONAL
pure observers of theta OFF the driver wall:
    Eth      : |rfft_x theta|, max over y (unnormalised)                 [Nx/2+1]
    iy_th    : the y index carrying that max, per k_x                    [Nx/2+1]
    iy_w     : the y index carrying the omega x-spectrum's max, per k_x  [Nx/2+1]
    th_y005, th_y001 : theta at y = 0.05 and y = 0.01 (EXACT Chebyshev
               evaluation of the spectral representation at xi0 = 2 y0 - 1:
               f(y0, x) = sum_k c_k T_k(xi0), c = Cheb.coeff(f) -- the same
               physical y at every N)                                     [Nx]
    w_y005, w_y001   : omega at the same rows                            [Nx]
    th_bot   : the driver-wall row again (alignment key vs the S35 stash / th_final)
-- nothing returned to the ride changes.  The stash goes to a SEPARATE side
file p3e_<tagS>_k<kappa>_N<N>.npz; the S35 side file p3d_* is ALSO written by
the parent machinery (so the S35 keys are reproduced too); the p3 record is
renamed exactly as stage_p3 renames it.  The stage REFUSES to run if any target
filename exists (never overwrite a banked record).

bsq_solver.py, s20_sig2d.py and s35_p3d.py are NOT edited.

Stage:  p3e:<sigma>:<N>:<kappa>:<cfl>:<tag>[:A]
"""
import os
import sys
import time

import numpy as np
from scipy.fft import rfft

import bsq_solver
from bsq_solver import BSQ
import s35_p3d
from s35_p3d import BSQSD

LAST = []


class BSQSE(BSQSD):
    """BSQSD + interior-row / max-over-y theta observers, taken AFTER the parent."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.stash_e = []
        N = self.Ny
        kk = np.arange(N + 1)
        self._Tk = {y0: np.cos(kk * np.arccos(2.0 * y0 - 1.0)) for y0 in (0.05, 0.01)}
        # self-check of the evaluation formula at a CGL node (printed once)
        inode = int(np.argmin(np.abs(self.ch.y - 0.05)))
        self._Tk_node = np.cos(kk * np.arccos(self.ch.xi[inode]))
        self._inode = inode
        self._checked = False
        LAST.append(self)

    def _row(self, f, y0):
        return self._Tk[y0] @ self.ch.coeff(f)

    def diagnostics(self, w, th, psi=None):
        d = super().diagnostics(w, th, psi)           # BSQSD (which calls BSQS first): unchanged
        c_th = self.ch.coeff(th)
        if not self._checked:
            valn = self._Tk_node @ c_th
            err = float(np.abs(valn - th[self._inode]).max() / max(np.abs(th[self._inode]).max(), 1e-300))
            print("  [p3e] Chebyshev node-evaluation self-check at i=%d (y=%.5f): rel err %.2e" % (self._inode, self.ch.y[self._inode], err), flush=True)
            self._checked = True
        A_th = np.abs(rfft(th, axis=1, workers=self.wk))
        A_w = np.abs(rfft(w, axis=1, workers=self.wk))
        self.stash_e.append(dict(Eth=A_th.max(axis=0), iy_th=A_th.argmax(axis=0).astype(np.int32), iy_w=A_w.argmax(axis=0).astype(np.int32),
                                 th_y005=self._Tk[0.05] @ c_th, th_y001=self._Tk[0.01] @ c_th,
                                 w_y005=self._row(w, 0.05), w_y001=self._row(w, 0.01),
                                 th_bot=th[self.Ny].copy()))
        return d


def stage_p3e(sigma, N=256, kappa=1e-2, cfl=0.4, tag="", A=1.0e4):
    class _Sub(BSQ):
        def __new__(cls, *a, **kw):
            return BSQSE(*a, sigma=sigma, **kw)
    tagS = ("s%s" % ("%.3f" % sigma).replace(".", "p")) + (("_" + tag) if tag else "")
    ktag = ("%.6f" % kappa).replace(".", "p")
    asuf = "" if A == 1.0e4 else "_A%g" % A
    src = "p2_k%s_N%d%s_%s.npz" % (ktag, N, asuf, tagS)
    dst = "p3_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    side_d = "p3d_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    side_e = "p3e_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    for fn in (src, dst, side_d, side_e):
        if os.path.exists(fn):
            print("REFUSED: %s already exists (never overwrite a banked record; choose a fresh tag)" % fn, flush=True)
            sys.exit(2)
    old = bsq_solver.BSQ
    n0 = len(LAST); n0d = len(s35_p3d.LAST)
    try:
        bsq_solver.BSQ = _Sub
        print("stage p3e[sigma=%g]: the sigma-dial level + the S35 stash + the EXTENDED theta observers; "
              "ride loop = certified stage_p2 verbatim (class substitution as stage_p3)" % sigma, flush=True)
        bsq_solver.stage_p2(N=N, kappa=kappa, cfl=cfl, tag=tagS, A=A)
    finally:
        bsq_solver.BSQ = old
    if not os.path.exists(src):
        print("ERROR: expected %s not written" % src, flush=True)
        sys.exit(3)
    os.rename(src, dst)
    print("  record renamed -> %s" % dst, flush=True)
    S = LAST[-1]
    assert len(LAST) == n0 + 1 and len(s35_p3d.LAST) == n0d + 1, "exactly one solver instance expected per ride"
    st = S.stash; se = S.stash_e
    z = np.load(dst)
    t = np.asarray(z["t"])
    if len(st) != len(t) + 1 or len(se) != len(t) + 1:
        print("ALIGNMENT FAULT: %d / %d stash entries vs %d saved records + 1" % (len(st), len(se), len(t)), flush=True)
        sys.exit(4)
    sup_ok = all(st[i + 1]["sup_w"] == float(z["sup_w"][i]) for i in range(len(t)))
    dx_ok = all((st[i + 1]["delta_x"] == float(z["delta_x"][i])) or (np.isnan(st[i + 1]["delta_x"]) and np.isnan(z["delta_x"][i])) for i in range(len(t)))
    last_ok = np.array_equal(st[-1]["th_bot"], z["th_final"][N])
    last_ok_e = np.array_equal(se[-1]["th_bot"], z["th_final"][N])
    print("  stash alignment: sup_w bit-identical %s ; delta_x bit-identical %s ; last stash th_bot == th_final[Ny] %s (S35 stash) / %s (extended stash)"
          % (sup_ok, dx_ok, last_ok, last_ok_e), flush=True)
    # the S35 side file, EXACTLY as s35_p3d writes it (so the S35 keys are reproduced by this ride too)
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
    print("  side files saved -> %s and %s  [%d records x N=%d ; align_ok %s]" % (side_d, side_e, len(t), N, oute["align_ok"]), flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st.startswith("p3e"):
        p = st.split(":")
        stage_p3e(float(p[1]),
                  N=int(p[2]) if len(p) > 2 else 256,
                  kappa=float(p[3]) if len(p) > 3 else 1e-2,
                  cfl=float(p[4]) if len(p) > 4 else 0.4,
                  tag=p[5] if len(p) > 5 else "",
                  A=float(p[6]) if len(p) > 6 else 1.0e4)
    else:
        print("stages: p3e:<sigma>[:N[:kappa[:cfl[:tag[:A]]]]]")
