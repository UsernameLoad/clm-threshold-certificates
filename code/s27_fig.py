#!/usr/bin/env python3
# S27 — the sigma-family paper §5 figure (; s27_reg.txt).
# Data discipline: the staircase is RE-DERIVED from the certificate npz records
# by the s26_tables.py load/staircase logic (replicated verbatim below because
# that file is a script, not an importable module) and ASSERTED against the
# emitted v1.1 digest values (s26_tables_out.txt) before drawing; the measured
# map is the paper §6 list (record-derived, transcribed with the file open).
# No new numbers enter at figure time.
import glob, math, sys
import numpy as np
from fractions import Fraction
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

def load(fn):
    z = np.load(fn, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    sig = float(d.get("sigma", 2.0))
    if "nu_num" in d:
        nu = Fraction(int(d["nu_num"]), int(d["nu_den"]))
    elif fn == "cap_tm_cert.npz":
        nu = Fraction(41, 1024)
    else:
        raise KeyError("no nu identification for %s" % fn)
    return fn, sig, nu, d

dec = [load(fn) for fn in sorted(glob.glob("exc_cert_*.npz"))]
blo = [load(fn) for fn in sorted(glob.glob("lad_cert_*.npz")) + sorted(glob.glob("cap_tm_cert*.npz"))]
dpts, bpts = {}, {}
for fn, sig, nu, d in dec:
    dpts[sig] = min(dpts.get(sig, Fraction(1)), nu)
for fn, sig, nu, d in blo:
    bpts[sig] = max(bpts.get(sig, Fraction(0)), nu)

# assert against the emitted v1.1 digest (s26_tables_out.txt)
DIGEST_PINCH = {0.25: (0.4, 0.43), 0.5: (0.3175, 0.3215), 0.625: (0.27, 0.29),
                0.75: (0.24, 0.25), 0.875: (0.2, 0.22), 1.25: (0.13, 0.137),
                1.5: (0.093, 0.1), 1.75: (0.066, 0.072), 2.0: (0.0449219, 0.052)}
for s, (flo, cei) in DIGEST_PINCH.items():
    assert abs(float(bpts[s]) - flo) < 5e-7, (s, float(bpts[s]), flo)
    assert abs(float(dpts[s]) - cei) < 5e-7, (s, float(dpts[s]), cei)
print("digest assertion passed: 9 certified pinch pairs match s26_tables_out.txt")

INV2E = 1.0/(2.0*math.e)
# staircase step functions over (0, 2] built from the cert points + overlays
bs = sorted(bpts)                     # blowup certs cover sigma' <= s
floor_steps = []                       # (s_left, s_right, value, kind)
prev = 0.0
for s in bs:
    floor_steps.append((prev, s, float(bpts[s]), "cert"))
    prev = s
# the (0.875, 1) Thm-9 overlay: floor 0.18394 (replaces the cert transport there)
FLOOR = []
for a, b, v, k in floor_steps:
    if (a, b) == (0.875, 1.25):
        FLOOR.append((0.875, 1.0, INV2E, "lit"))   # Thm 9 on (0.875, 1)
        FLOOR.append((1.0, 1.25, v, k))
    else:
        FLOOR.append((a, b, v, k))
ds = sorted(dpts)                     # decay certs cover sigma' >= s
CEIL = [(0.0, 0.25, 0.5, "lit")]      # ALSS sigma=0 + Prop 1
for i, s in enumerate(ds):
    b = ds[i+1] if i + 1 < len(ds) else 2.2
    v = min(float(dpts[t]) for t in ds if t <= s)
    if s == 0.875:                     # the [1, 1.25) A-1 riser splits this run
        CEIL.append((0.875, 1.0, v, "cert"))
        CEIL.append((1.0, 1.25, INV2E, "lit"))
    elif s == 1.25:
        CEIL.append((1.25, 1.5, v, "cert"))
    elif b <= 1.0 or s >= 1.25:
        CEIL.append((s, b, v, "cert"))
    else:
        CEIL.append((s, b, v, "cert"))

MEASURED = [  # the paper §6 map (record-derived; bracket midpoints / values)
    (0.10, 0.461125), (0.20, 0.4238125), (0.35, 0.370625), (0.50, 0.3205),
    (0.625, 0.28175), (0.75, 0.24615), (0.875, 0.2133125), (1.0, INV2E),
    (1.25, 0.13510), (1.375, 0.115241), (1.5, 0.098066), (1.75, 0.07062),
    (2.0, 0.0505674),
]

fig, ax = plt.subplots(figsize=(8.6, 5.4))
for a, b, v, k in FLOOR:
    ax.plot([a, b], [v, v], color="#b3452c", lw=2.4 if k == "cert" else 1.6,
            ls="-" if k == "cert" else "--", solid_capstyle="butt", zorder=3)
for a, b, v, k in CEIL:
    ax.plot([a, b], [v, v], color="#2c5f8a", lw=2.4 if k == "cert" else 1.6,
            ls="-" if k == "cert" else "--", solid_capstyle="butt", zorder=3)
for s, (flo, cei) in DIGEST_PINCH.items():
    ax.plot([s, s], [flo, cei], color="#444444", lw=1.1, zorder=4)
    ax.plot([s], [flo], marker="^", color="#b3452c", ms=5, zorder=5)
    ax.plot([s], [cei], marker="v", color="#2c5f8a", ms=5, zorder=5)
ax.plot([1.0], [INV2E], marker="*", color="#7a2ca0", ms=14, zorder=6,
        label=r"$\nu_*(1)=1/(2e)$ exact (Sakajo)")
ax.plot([0.0], [0.5], marker="*", color="#00814d", ms=14, zorder=6, clip_on=False,
        label=r"$\nu_*(0)=1/2$ exact (ALSS)")
mx = [p[0] for p in MEASURED]; my = [p[1] for p in MEASURED]
ax.plot(mx, my, color="#333333", lw=0.9, ls=":", zorder=2)
ax.plot(mx, my, "o", color="#333333", ms=3.4, zorder=5, label=r"measured $\nu_*(\sigma)$")
# quadrant arrows at the sigma=1.5 pair
ax.annotate("", xy=(1.9, 0.135), xytext=(1.52, 0.101),
            arrowprops=dict(arrowstyle="->", color="#2c5f8a", lw=1.2))
ax.annotate("decay transports\nup-right (Prop 1)", xy=(1.62, 0.147), fontsize=7.5, color="#2c5f8a")
ax.annotate("", xy=(1.1, 0.062), xytext=(1.48, 0.0915),
            arrowprops=dict(arrowstyle="->", color="#b3452c", lw=1.2))
ax.annotate("blowup transports\ndown-left", xy=(0.98, 0.049), fontsize=7.5, color="#b3452c")
ax.axvspan(2.0, 2.2, color="#dddddd", alpha=0.6, hatch="//", zorder=1)
ax.text(2.09, 0.30, "one-sided\nby choice", fontsize=7.5, ha="center", color="#555555")
ax.plot([2.0, 2.2], [0.052, 0.052], color="#2c5f8a", lw=2.4, zorder=3)
ax.set_yscale("log")
ax.set_xlim(0.0, 2.2); ax.set_ylim(0.028, 0.62)
ax.set_xlabel(r"dissipation order $\sigma$")
ax.set_ylabel(r"$\nu$  (log scale)")
ax.set_title("The certified two-sided staircase for $\\nu_*(\\sigma)$ — v1.1 "
             "(ten pinch points; every interior segment $\\leq 1.603$)", fontsize=10)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.grid(True, which="both", alpha=0.18)
fig.tight_layout()
fig.savefig("sigma-family-fig1.svg")
fig.savefig("sigma-family-fig1.png", dpi=180)
print("wrote sigma-family-fig1.svg / .png")
print("floor segments:", [(a, b, round(v, 6)) for a, b, v, k in FLOOR])
print("ceiling segments:", [(a, b, round(v, 6)) for a, b, v, k in CEIL])
