#!/usr/bin/env python
"""
s39_alpha.py -- P-1's CONSISTENCY RECEIPT (seconds-class; a receipt for the
STATEMENT'S TRANSCRIPTION in the paper's normalisation, never evidence for the
lemma -- the lemma's proof is the induction written in the paper).

The truncated positive-mode hierarchy of the sigma-family paper,
    q_m' = -nu m^sigma q_m + (1/2) sum_{j=1}^{m-1} q_j q_{m-j},   q_1(0) = 1, q_m(0) = 0 (m >= 2),
integrated by classical RK4 with a fixed step at two exponents sigma_1 < sigma_2
on the SAME steps; the statement q_m^(sigma_1)(t) >= q_m^(sigma_2)(t) >= 0 is
checked at every mode m <= M and every sample time.  Design: s39_regcalc.txt
§4;.
"""
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
M, DT, T = 48, 1e-3, 3.0
PAIRS = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (0.75, 1.25), (1.0, 2.0), (1.375, 1.5)]
NUS = [0.05, 0.2, 0.5]
TOL = 1e-12


def rhs(q, lam):
    m = np.arange(1, M + 1)
    conv = np.zeros(M)
    for k in range(2, M + 1):                       # k = mode index; j = 1..k-1
        conv[k - 1] = 0.5 * np.dot(q[:k - 1], q[k - 2::-1])
    return -lam * q + conv


def rk4(lam, nsteps):
    q = np.zeros(M); q[0] = 1.0
    out = [q.copy()]
    for n in range(nsteps):
        k1 = rhs(q, lam); k2 = rhs(q + 0.5 * DT * k1, lam); k3 = rhs(q + 0.5 * DT * k2, lam); k4 = rhs(q + DT * k3, lam)
        q = q + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (n + 1) % 100 == 0:
            out.append(q.copy())
    return np.array(out)


print("== P-1 consistency receipt: q^(sigma_1)_m >= q^(sigma_2)_m >= 0 on the M=%d truncation, RK4 dt=%g, T=%g, samples every 100 steps ==" % (M, DT, T))
m = np.arange(1, M + 1, dtype=float)
nsteps = int(round(T / DT))
worst = 0.0; nonneg = True; rows = []
for s1, s2 in PAIRS:
    for nu in NUS:
        Q1 = rk4(nu * m ** s1, nsteps); Q2 = rk4(nu * m ** s2, nsteps)
        diff = Q1 - Q2
        scale = np.maximum(Q1, 1e-300)
        rel = diff / scale
        w = float(rel.min())                      # the most negative relative difference (0 at m = 1 exactly)
        worst = min(worst, w)
        nonneg = nonneg and bool((Q2 >= 0).all() and (Q1 >= 0).all())
        m1_equal = bool(np.array_equal(Q1[:, 0], Q2[:, 0]))
        # strictness report: the smallest relative difference over m >= 2 at T
        strict = float(rel[-1, 1:].min())
        s1sum, s2sum = float(Q1[-1].sum()), float(Q2[-1].sum())
        rows.append((s1, s2, nu, w, m1_equal, strict, s1sum, s2sum))
        print("  sigma (%5.3f, %5.3f) nu %.2f : min rel (q1-q2)/q1 over all m, t = %+.3e ; q_1 identical at both sigma %s ; min over m>=2 at T %+.3e ; sums at T %.4e / %.4e ; all q >= 0 %s"
              % (s1, s2, nu, w, m1_equal, strict, s1sum, s2sum, bool((Q1 >= 0).all() and (Q2 >= 0).all())))
letter = "CONSISTENT" if (worst >= -TOL and nonneg) else "INCONSISTENT"
print("  worst relative violation %+.3e against the bar -%g ; all modes non-negative %s -> P-1 CONSISTENCY LETTER: %s" % (worst, TOL, nonneg, letter))
