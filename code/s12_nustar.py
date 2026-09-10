#!/usr/bin/env python3
# s12_nustar.py -- S12 joint nu*(0) analysis: the S05-S07 seven decay points
# + the S12 deep points (M=8192/16384, _M-tagged records). Deterministic,
# stage-free (single pass); writes its findings to stdout (redirect to
# s12_nustar.txt as the record).
#
# Instruments (each independent):
#   (A) slaving inversion per point: delta_impl = sqrt(12 nu / Sigma_peak),
#       nu*_impl = nu/(1 + delta_impl)   [N16 slaving + N19 linear approach;
#       validated as the sharp instrument by the S12 sigma=1.5 exercise]
#   (B) pairwise gamma=2-anchored consistency: for each adjacent clean pair,
#       solve nu* from Sigma2/Sigma1 = (eps1/eps2)^2
#   (C) peak-time locus: t_pk = t0 - c*eps (the (2,3,1) time-shift law),
#       fit (t0, c, nu*) on the clean set by least squares over nu* grid
# Cleanliness: peaks above the front cap nu (M/2.7)^2 are flagged and EXCLUDED
# from certification-grade lines (N15; the S12 verdict-flip lesson).
import glob, os
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(DIR, "peak_nu*.npz"))):
    if f.endswith(".bak1") or ".bak" in f:
        continue
    d = np.load(f, allow_pickle=True)
    nu = float(d["nu"]); M = int(d["M"]) if "M" in d else 1024
    pk = float(d["peak"]); tpk = float(d["t_peak"])
    verdict = str(d["verdict"])
    cap = nu * (M / 2.7) ** 2
    clean = (pk <= cap) and verdict.startswith("DECAY")
    rows.append(dict(nu=nu, M=M, pk=pk, tpk=tpk, verdict=verdict,
                     cap=cap, clean=clean, f=os.path.basename(f)))

rows.sort(key=lambda r: -r["nu"])
print("== S12 nu*(0) joint analysis: collected records ==")
for r in rows:
    print("  nu=%.6f M=%5d  %-16s Sigma_pk=%.4e t_pk=%.4f  cap=%.2e  %s  [%s]"
          % (r["nu"], r["M"], r["verdict"], r["pk"], r["tpk"], r["cap"],
             "CLEAN" if r["clean"] else "dirty/blow", r["f"]))

cl = [r for r in rows if r["clean"]]
print("\n(A) slaving inversions (clean decay points, deepest last):")
inv = []
for r in cl:
    di = np.sqrt(12.0 * r["nu"] / r["pk"])
    ns = r["nu"] / (1.0 + di)
    inv.append((r["nu"], ns, di))
    print("  nu=%.6f: delta_impl=%.5f  nu*_impl=%.6f" % (r["nu"], di, ns))
if len(inv) >= 3:
    deep = [x[1] for x in inv if x[2] < 0.05]
    print("  deepest-3 mean: nu* = %.6f  (spread %.1e)"
          % (np.mean(deep[-3:]), np.ptp(deep[-3:])))

print("\n(B) pairwise gamma=2-anchored nu* (adjacent clean pairs):")
for i in range(len(cl) - 1):
    r1, r2 = cl[i], cl[i + 1]          # r1.nu > r2.nu
    R = (r2["pk"] / r1["pk"]) ** 0.5   # = eps1/eps2 under gamma=2
    if R <= 1.0:
        continue
    eps2 = (r1["nu"] - r2["nu"]) / (R - 1.0)
    print("  (%.6f, %.6f): nu* = %.6f" % (r1["nu"], r2["nu"], r2["nu"] - eps2))

print("\n(C) peak-time locus t_pk = t0 - c*eps over nu* grid (clean set):")
best = None
for ns in np.arange(0.05040, 0.05058, 2e-6):
    eps = np.array([r["nu"] for r in cl]) - ns
    if (eps <= 0).any():
        continue
    tp = np.array([r["tpk"] for r in cl])
    A = np.vstack([np.ones_like(eps), -eps]).T
    sol, res, *_ = np.linalg.lstsq(A, tp, rcond=None)
    rms = np.sqrt(res[0] / len(tp)) if len(res) else np.nan
    if best is None or (rms == rms and rms < best[0]):
        best = (rms, ns, sol)
for label, use in (("all-clean", cl),
                   ("near-field (Sigma_pk >= 1e3)", [r for r in cl if r["pk"] >= 1e3])):
    bb = None
    for ns in np.arange(0.05040, 0.05058, 1e-6):
        eps = np.array([r["nu"] for r in use]) - ns
        if (eps <= 0).any() or len(use) < 3:
            continue
        tp = np.array([r["tpk"] for r in use])
        A = np.vstack([np.ones_like(eps), -eps]).T
        sol, res, *_ = np.linalg.lstsq(A, tp, rcond=None)
        rms = np.sqrt(res[0] / len(tp)) if len(res) else np.inf
        if bb is None or rms < bb[0]:
            bb = (rms, ns, sol)
    if bb:
        print("  [%s, n=%d] best nu* = %.6f  t0 = %.4f  c = %.1f  rms = %.2e"
              % (label, len(use), bb[1], bb[2][0], bb[2][1], bb[0]))
print("\n(done)")
