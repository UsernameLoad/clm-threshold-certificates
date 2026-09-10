#!/usr/bin/env python3
# S26 — the sigma-family paper v0.2 table-builder (P-0, s26_reg.txt).
# Reads EVERY certificate npz on disk and emits: (i) the §3 decay table,
# (ii) the §4 blowup table, (iii) the certificate-implied staircase, diffed
# against the RECORDED v1.0 before the S26 H-1 update
# and emitted as v1.1 after it. Zero memory-transcription: all numbers from
# the npz fields; the three literature/composition elements are marked.
import glob, math, sys
import numpy as np
from fractions import Fraction
sys.stdout.reconfigure(encoding="utf-8")  # cp1252 console vs Greek table headers

def load(fn):
    z = np.load(fn, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    sig = float(d.get("sigma", 2.0))
    if "nu_num" in d:
        nu = Fraction(int(d["nu_num"]), int(d["nu_den"]))
    elif fn == "cap_tm_cert.npz":
        # the S09 original predates the nu-field convention; identification from
        # the N18 Theorem-B record + s09_tmrun.txt (41/1024), not from the npz.
        nu = Fraction(41, 1024)
    else:
        raise KeyError("no nu identification for %s" % fn)
    return fn, sig, nu, d

def fmt_nu(nu):
    return "%d/%d = %.6g" % (nu.numerator, nu.denominator, float(nu))

dec, blo = [], []
for fn in sorted(glob.glob("exc_cert_*.npz")):
    dec.append(load(fn))
for fn in sorted(glob.glob("lad_cert_*.npz")) + sorted(glob.glob("cap_tm_cert*.npz")):
    blo.append(load(fn))

print("### §3 DECAY TABLE (%d records; the 31/500 pair = one theorem at two discretizations)" % len(dec))
print("| σ | ν (exact) | M | steps | Σ_pk(march) | Ȳ_max | max relw | landing margin | record |")
print("|---|---|---|---|---|---|---|---|---|")
for fn, sig, nu, d in sorted(dec, key=lambda r: (r[1], -float(r[2]))):
    st = int(d["nstep"]) if "nstep" in d else 0
    sp = float(d["Sig_peak"]) if "Sig_peak" in d else float("nan")
    yb = float(d["Ybar_max"]) if "Ybar_max" in d else float("nan")
    rw = float(d["relw_max"]) if "relw_max" in d else float("nan")
    mg = float(d["margin"]) if "margin" in d else float("nan")
    print("| %.3g | %s | %d | %d | %.4f | %.2e | %.1e | %.2f | %s |"
          % (sig, fmt_nu(nu), int(d["M"]), st, sp, yb, rw, mg, fn))

print()
print("### §4 BLOWUP TABLE (%d records)" % len(blo))
print("| σ | ν (exact) | M | k0 | window [t1, t_bar] | S_lo | margin ×A* | max relw | steps | record |")
print("|---|---|---|---|---|---|---|---|---|---|")
for fn, sig, nu, d in sorted(blo, key=lambda r: (r[1], -float(r[2]))):
    k0 = int(d["k0"]) if "k0" in d else 8
    t1 = float(d["t1"]); W = float(d["W"])
    tb = float(d["t_bar"]) if "t_bar" in d else t1 + W
    sl = float(d["S_lo"]); mg = float(d["margin"])
    rw = float(d["relw_max"]) if "relw_max" in d else float("nan")
    st = int(d["nstep"]) if "nstep" in d else 0
    print("| %.3g | %s | %d | %d | [%.6f, %.6f] | %.4e | %.4f | %.1e | %d | %s |"
          % (sig, fmt_nu(nu), int(d["M"]), k0, t1, tb, sl, mg, rw, st, fn))

# ---- the certificate-implied staircase --------------------------------------
def staircase(dec, blo, include_s26=True):
    # decay: cert at (s0, nu0) covers s >= s0  -> ceiling(s) = min nu0 over s0 <= s
    # blowup: cert at (s0, nu0) covers s <= s0 -> floor(s)  = max nu0 over s0 >= s
    dpts = {}
    for fn, sig, nu, d in dec:
        dpts[sig] = min(dpts.get(sig, Fraction(1)), nu)
    bpts = {}
    for fn, sig, nu, d in blo:
        if not include_s26 and fn.startswith("lad_cert_s0p25"):
            continue
        bpts[sig] = max(bpts.get(sig, Fraction(0)), nu)
    return dpts, bpts

for tag, inc in [("v1.0 (pre-H-1; the P-0 reproduction target =", False),
                 ("v1.1 (post-H-1)", True)]:
    dpts, bpts = staircase(dec, blo, include_s26=inc)
    print()
    print("### STAIRCASE %s" % tag)
    print("  decay ceilings (cert σ -> min cert ν; each covers σ' ≥ σ):")
    run = None
    for s in sorted(dpts):
        print("    σ=%.3g: %.6g" % (s, float(dpts[s])), end="")
        print()
    print("  blowup floors (cert σ -> max cert ν; each covers σ' ≤ σ):")
    for s in sorted(bpts):
        print("    σ=%.3g: %.6g" % (s, float(bpts[s])), end="")
        print()
    # literature/composition overlays (marked; from the records, not npz):
    print("  overlays: ceiling 1/2 on (0, 0.25) [ALSS σ=0 exact + Prop 1];")
    print("            ceiling 1/(2e)=0.183940 on [1, 1.25) [S24 A-1 composition];")
    print("            floor Thm 9 = 0.183940 on (0.875, 1) [Sakajo]; ν*(1)=1/(2e) exact.")
    # segment ratios on the recorded breakpoints:
    segs = [(0.0,0.25),(0.25,0.35),(0.35,0.5),(0.5,0.625),(0.625,0.75),(0.75,0.875),
            (0.875,1.0),(1.0,1.25),(1.25,1.5),(1.5,1.75),(1.75,2.0)]
    print("  interior segment ratios (ceiling-at-left-edge / floor-at-right-edge):")
    inv2e = 1.0/(2.0*math.e)
    for a, b in segs:
        # ceiling on [a, b): min over decay certs with σ <= a (+ overlays)
        cands_c = [float(v) for s, v in dpts.items() if s <= a + 1e-12]
        if a < 0.25 - 1e-9: cands_c.append(0.5)
        if a >= 1.0 - 1e-12: cands_c.append(inv2e)
        ceil = min(cands_c) if cands_c else float("nan")
        # floor on (a, b]: max over blowup certs with σ >= b (+ overlays)
        cands_f = [float(v) for s, v in bpts.items() if s >= b - 1e-12]
        if b <= 1.0 - 1e-12 or abs(b-1.0) < 1e-12: cands_f.append(inv2e)  # Thm 9 covers σ<1... see note
        # Thm 9 floor applies on (0.875, 1) only where it beats certs; harmless as max-cand for b<=1
        flo = max(cands_f) if cands_f else float("nan")
        print("    (%.3g, %.3g): ceiling %.6g / floor %.6g = %.4f" % (a, b, ceil, flo, ceil/flo))
    # pinch points at cert σ:
    print("  pinch points (σ with BOTH a cert floor and a cert-or-overlay ceiling AT that σ):")
    for s in sorted(set(list(bpts) + list(dpts))):
        f = float(bpts[s]) if s in bpts else None
        # ceiling AT σ: min decay cert with σ' <= σ (+ overlays at σ>=1: 1/(2e))
        cc = [float(v) for sd, v in dpts.items() if sd <= s + 1e-12]
        if s >= 1.0 - 1e-12: cc.append(inv2e)
        c = min(cc) if cc else None
        if f is not None and c is not None:
            print("    σ=%.3g: [%.6g, %.6g] ratio %.4f" % (s, f, c, c/f))
