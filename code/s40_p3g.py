#!/usr/bin/env python
"""
s40_p3g.py -- THE ANISOTROPIC (Nx != Ny) PROFILE-STASH RIDE for the sigma-dial level.

The class chain is the S39 one UNCHANGED: BSQSF(BSQSE(BSQSD(BSQS(BSQ)))) -- bsq_solver's
BSQ takes Nx and Ny separately (stage_u7 rides 128 x 96), and every observer of the
chain indexes the wall by self.Ny and the x-modes by rfft over axis 1.  What passes ONE
N to both directions is the ride LOOP, bsq_solver.stage_p2 (S = BSQ(N, N, ...); the
initial-field tile shapes; the record's N field and file tag).  A stage with separate
(Nx, Ny) is therefore a COPY of that loop, not a class substitution (the S40 hunt hint
(c)), and its first obligation is to reproduce the square S39 x853f rides bit for bit at
Nx = Ny (U-0) before any anisotropic number is read.

THE COPY IS MECHANICAL: stage_p2's source is taken by inspect.getsource at import time,
SEVEN single-occurrence textual substitutions are applied (each asserted to occur exactly
once; the receipts s40_regcalc.txt print the unified diff), and the result is compiled
into bsq_solver's own namespace as stage_p2g -- so `BSQ` inside the loop resolves to the
substituted class exactly as it does for stage_p3 / p3d / p3e / p3f.  bsq_solver.py,
s20_sig2d.py, s35_p3d.py, s36_p3e.py, s39_p3f.py are NOT edited (runtime namespace only).

Side files: the p3 record p3_<tagS>_k<kappa>_N<grid>.npz, the S35 p3d, the S36 p3e and
the S39 p3f side files, written exactly as s39_p3f.stage_p3f writes them (N -> Ny at the
wall index; the record gains the fields Nx, Ny, grid).  <grid> = "512" when Nx == Ny
(the S39 naming reproduced) and "1024x512" otherwise.  The stage REFUSES to run if any
target filename exists.

Stage:  p3g:<sigma>:<Nx>:<Ny>:<kappa>:<cfl>:<tag>[:A]
"""
import inspect
import os
import sys

import numpy as np

import bsq_solver
from bsq_solver import BSQ
import s35_p3d
import s36_p3e
import s39_p3f
from s39_p3f import BSQSF, YG, COLS


def NTAG(nx, ny):
    return ("%d" % nx) if nx == ny else ("%dx%d" % (nx, ny))


SUBS = [
    ('def stage_p2(N=256, kappa=1e-2, cfl=0.4, tag="", A=1.0e4):',
     'def stage_p2g(Nx=256, Ny=256, kappa=1e-2, cfl=0.4, tag="", A=1.0e4):'),
    ("    S = BSQ(N, N, fast=True, qi=True, Lx=Lx, kappa=kappa)",
     "    S = BSQ(Nx, Ny, fast=True, qi=True, Lx=Lx, kappa=kappa)"),
    ("    X = np.tile(S.x, (N + 1, 1)); Y = np.tile(S.ch.y[:, None], (1, N))",
     "    X = np.tile(S.x, (Ny + 1, 1)); Y = np.tile(S.ch.y[:, None], (1, Nx))"),
    ('    tagS = ("k%s_N%d" % (("%.6f" % kappa).replace(".", "p"), N)',
     '    tagS = ("k%s_N%s" % (("%.6f" % kappa).replace(".", "p"), NTAG(Nx, Ny))'),
    ('    print(f"stage p2[{tagS}]: Lx=1/6, A={A:g}, N={N}, kappa={kappa:g}, cfl={cfl}"',
     '    print(f"stage p2g[{tagS}]: Lx=1/6, A={A:g}, Nx={Nx}, Ny={Ny}, kappa={kappa:g}, cfl={cfl}"'),
    ("               tcert=tcert, gcert=gcert, om_pk=om_pk, N=N, cfl=cfl, Lx=Lx, A=A,",
     "               tcert=tcert, gcert=gcert, om_pk=om_pk, N=Ny, cfl=cfl, Lx=Lx, A=A,"),
    ("               kappa=kappa, verdict=verdict, itx=itx, mean0=mean0, th2_0=th2_0,",
     "               kappa=kappa, verdict=verdict, itx=itx, mean0=mean0, th2_0=th2_0, Nx=Nx, Ny=Ny, grid=NTAG(Nx, Ny),"),
]

SRC_P2 = inspect.getsource(bsq_solver.stage_p2)
SRC_P2G = SRC_P2
for old, new in SUBS:
    assert SRC_P2G.count(old) == 1, "substitution anchor not unique: %r (count %d)" % (old, SRC_P2G.count(old))
    SRC_P2G = SRC_P2G.replace(old, new)
# no standalone N may survive in CODE except the keyword N=Ny (checked here and printed by the
# receipts); the one other hit is stage_p2's docstring line "Records: p2_k<kappa>_N<N>..." (text)
import re as _re
_left = [ln for ln in SRC_P2G.splitlines() if _re.search(r"(?<![A-Za-z0-9_])N(?![A-Za-z0-9_])", ln)]
assert _left == ['    Records: p2_k<kappa>_N<N>[_A<A>][_<tag>].npz (S11 filename rule)."""',
                 "               tcert=tcert, gcert=gcert, om_pk=om_pk, N=Ny, cfl=cfl, Lx=Lx, A=A,"], _left
bsq_solver.NTAG = NTAG
exec(compile(SRC_P2G, "<s40_p3g stage_p2g = stage_p2 copy>", "exec"), bsq_solver.__dict__)
stage_p2g = bsq_solver.stage_p2g


def stage_p3g(sigma, Nx=256, Ny=256, kappa=1e-2, cfl=0.4, tag="", A=1.0e4):
    """s39_p3f.stage_p3f with (Nx, Ny) in place of N: the ride through stage_p2g, then the
    S35 / S36 / S39 side files written exactly as stage_p3f writes them (wall index Ny)."""
    class _Sub(BSQ):
        def __new__(cls, *a, **kw):
            return BSQSF(*a, sigma=sigma, **kw)
    tagS = ("s%s" % ("%.3f" % sigma).replace(".", "p")) + (("_" + tag) if tag else "")
    ktag = ("%.6f" % kappa).replace(".", "p")
    asuf = "" if A == 1.0e4 else "_A%g" % A
    grid = NTAG(Nx, Ny)
    src = "p2_k%s_N%s%s_%s.npz" % (ktag, grid, asuf, tagS)
    dst = "p3_%s_k%s_N%s%s.npz" % (tagS, ktag, grid, asuf)
    side_d = "p3d_%s_k%s_N%s%s.npz" % (tagS, ktag, grid, asuf)
    side_e = "p3e_%s_k%s_N%s%s.npz" % (tagS, ktag, grid, asuf)
    side_f = "p3f_%s_k%s_N%s%s.npz" % (tagS, ktag, grid, asuf)
    for fn in (src, dst, side_d, side_e, side_f):
        if os.path.exists(fn):
            print("REFUSED: %s already exists (never overwrite a banked record; choose a fresh tag)" % fn, flush=True)
            sys.exit(2)
    old = bsq_solver.BSQ
    n0 = len(s39_p3f.LAST); n0e = len(s36_p3e.LAST); n0d = len(s35_p3d.LAST)
    try:
        bsq_solver.BSQ = _Sub
        print("stage p3g[sigma=%g, Nx=%d, Ny=%d]: the sigma-dial level + the S35 stash + the S36 extended observers + the PROFILE stash (YG %d rows, %d columns); "
              "ride loop = stage_p2g, the mechanical (Nx, Ny) copy of the certified stage_p2 (diff in s40_regcalc.txt)" % (sigma, Nx, Ny, len(YG), len(COLS)), flush=True)
        stage_p2g(Nx=Nx, Ny=Ny, kappa=kappa, cfl=cfl, tag=tagS, A=A)
    finally:
        bsq_solver.BSQ = old
    if not os.path.exists(src):
        print("ERROR: expected %s not written" % src, flush=True)
        sys.exit(3)
    os.rename(src, dst)
    print("  record renamed -> %s" % dst, flush=True)
    S = s39_p3f.LAST[-1]
    assert len(s39_p3f.LAST) == n0 + 1 and len(s36_p3e.LAST) == n0e + 1 and len(s35_p3d.LAST) == n0d + 1, "exactly one solver instance expected per ride"
    N = Ny
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
    print("  side files saved -> %s, %s and %s  [%d records x Nx=%d Ny=%d ; align_ok %s]" % (side_d, side_e, side_f, len(t), Nx, Ny, outf["align_ok"]), flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st.startswith("p3g"):
        p = st.split(":")
        stage_p3g(float(p[1]),
                  Nx=int(p[2]) if len(p) > 2 else 256,
                  Ny=int(p[3]) if len(p) > 3 else 256,
                  kappa=float(p[4]) if len(p) > 4 else 1e-2,
                  cfl=float(p[5]) if len(p) > 5 else 0.4,
                  tag=p[6] if len(p) > 6 else "",
                  A=float(p[7]) if len(p) > 7 else 1.0e4)
    elif st == "diff":
        import difflib
        sys.stdout.reconfigure(encoding="utf-8")
        print("".join(difflib.unified_diff(SRC_P2.splitlines(True), SRC_P2G.splitlines(True), "bsq_solver.stage_p2", "s40_p3g.stage_p2g", n=0)))
    else:
        print("stages: p3g:<sigma>:<Nx>:<Ny>[:kappa[:cfl[:tag[:A]]]] | diff")
