#!/usr/bin/env python
"""
s35_p3d.py -- THE ADDITIVE SAVED-FIELD DIAGNOSTIC for the sigma-dial ride.

The ride loop is bsq_solver.stage_p2 VERBATIM, reached by the SAME class
substitution s20_sig2d.stage_p3 uses (certified S20): the class BSQSD(BSQS)
overrides ONLY diagnostics(), calls the parent FIRST and then STASHES pure
observers of the fields the parent was handed --
    th_bot   : theta on the driver wall (y index Ny, y = 0)      [Nx]
    th_top   : theta on the far wall (y index 0, y = 1)           [Nx]
    Ew       : the parent's omega x-spectrum, max over y, normalised to its
               max (the object the parent's strip fit consumes)   [Nx/2+1]
    Ew_max   : that maximum (so the unnormalised spectrum is recoverable)
    Ew_wall  : |rfft_x omega| on the driver wall                  [Nx/2+1]
    sup_w, tail_x, delta_x, strip_r2 : the parent's own values, as
               ALIGNMENT KEYS (must match the saved p3 series exactly)
-- nothing returned to the ride changes, so the p3 npz and the printed lines
are untouched.  After the ride the stash is written to a SEPARATE side file
    p3d_<tagS>_k<kappa>_N<N>.npz
and the p3 record is renamed exactly as stage_p3 renames it.  The stage
REFUSES to run if its target p3 filename already exists (never overwrite a
banked npz -- the S27 rule made mechanical; stage_p3 would silently remove
and rename).

bsq_solver.py and s20_sig2d.py are NOT edited.

Stage:  p3d:<sigma>:<N>:<kappa>:<cfl>:<tag>[:A]     (the tag must NOT repeat
        the sigma prefix; kappa is the banked partner npz FIELD as repr)
"""
import os
import sys
import time

import numpy as np
from scipy.fft import rfft

import bsq_solver
from bsq_solver import BSQ
from s20_sig2d import BSQS

LAST = []          # the instances built during a ride (stage_p2 does not return S)


class BSQSD(BSQS):
    """BSQS + a pure-observer stash taken inside diagnostics()."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.stash = []
        LAST.append(self)

    def diagnostics(self, w, th, psi=None):
        d = super().diagnostics(w, th, psi)          # the parent FIRST, unchanged
        Ewraw = np.abs(rfft(w, axis=1, workers=self.wk)).max(axis=0)
        mx = max(Ewraw.max(), 1e-300)
        self.stash.append(dict(th_bot=th[self.Ny].copy(), th_top=th[0].copy(),
                               Ew=Ewraw / mx, Ew_max=float(mx),
                               Ew_wall=np.abs(rfft(w[self.Ny], workers=self.wk)),
                               sup_w=float(d['sup_w']), tail_x=float(d['tail_x']),
                               delta_x=float(d['delta_x']), strip_r2=float(d['strip_r2'])))
        return d


def stage_p3d(sigma, N=256, kappa=1e-2, cfl=0.4, tag="", A=1.0e4):
    class _Sub(BSQ):
        def __new__(cls, *a, **kw):
            return BSQSD(*a, sigma=sigma, **kw)
    tagS = ("s%s" % ("%.3f" % sigma).replace(".", "p")) + (("_" + tag) if tag else "")
    ktag = ("%.6f" % kappa).replace(".", "p")
    asuf = "" if A == 1.0e4 else "_A%g" % A
    src = "p2_k%s_N%d%s_%s.npz" % (ktag, N, asuf, tagS)
    dst = "p3_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    side = "p3d_%s_k%s_N%d%s.npz" % (tagS, ktag, N, asuf)
    for fn in (src, dst, side):
        if os.path.exists(fn):
            print("REFUSED: %s already exists (never overwrite a banked record; choose a fresh tag)" % fn, flush=True)
            sys.exit(2)
    old = bsq_solver.BSQ
    n0 = len(LAST)
    try:
        bsq_solver.BSQ = _Sub
        print("stage p3d[sigma=%g]: the sigma-dial level + ADDITIVE saved-field stash; "
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
    assert len(LAST) == n0 + 1, "exactly one solver instance expected per ride"
    st = S.stash
    z = np.load(dst)
    t = np.asarray(z["t"])
    # alignment: stash[0] is the t = 0 call (psi None); stash[1:] align with the saved records
    if len(st) != len(t) + 1:
        print("ALIGNMENT FAULT: %d stash entries vs %d saved records + 1" % (len(st), len(t)), flush=True)
        sys.exit(4)
    sup_ok = all(st[i + 1]["sup_w"] == float(z["sup_w"][i]) for i in range(len(t)))
    dx_ok = all((st[i + 1]["delta_x"] == float(z["delta_x"][i])) or (np.isnan(st[i + 1]["delta_x"]) and np.isnan(z["delta_x"][i])) for i in range(len(t)))
    last_ok = np.array_equal(st[-1]["th_bot"], z["th_final"][N])
    print("  stash alignment: sup_w bit-identical %s ; delta_x bit-identical %s ; last stash th_bot == th_final[Ny] %s"
          % (sup_ok, dx_ok, last_ok), flush=True)
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
    np.savez_compressed(side, **out)
    print("  side file saved -> %s  [%d records x N=%d ; align_ok %s]" % (side, len(t), N, out["align_ok"]), flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st.startswith("p3d"):
        p = st.split(":")
        stage_p3d(float(p[1]),
                  N=int(p[2]) if len(p) > 2 else 256,
                  kappa=float(p[3]) if len(p) > 3 else 1e-2,
                  cfl=float(p[4]) if len(p) > 4 else 0.4,
                  tag=p[5] if len(p) > 5 else "",
                  A=float(p[6]) if len(p) > 6 else 1.0e4)
    else:
        print("stages: p3d:<sigma>[:N[:kappa[:cfl[:tag[:A]]]]]")
