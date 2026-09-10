"""S20 --).

S19 (N28) MEASURED that a rank-M fixed-bar instrument's apparent blowup
threshold at sigma = 0 carries a negative bias closing like 1/M.  It never
asked what the constant IS.  The banked column says the constant is exact:
M*(1 - 2 nu*_M) = 12.6484 to six digits across four doublings.

This script tests the closed form derived in

    Sigma_M(t) -> Phi(x; c) = x e^{c-x}/(c-x),   x = M e^{-nu t},  c = M(1-2nu)
    peak at    c = x^2/(x-1),  i.e. x = 1+y and c = 1/y + 2 + y
    height     Phi_max = y e^{1 + 1/y}                      (M-INDEPENDENT)
    threshold  nu*_M(bar) = (1/2)(1 - c(bar)/M) + O(M^-2)
               with  w - ln w = ln(bar) - 1,  i.e. w = -W_{-1}(-e/bar),
               and   c(bar) = w + 2 + 1/w.

The exact sigma=0 routines are IMPORTED UNCHANGED from s19_sig0exact.py, so
stage `repro` is a genuine predecessor-reproduction obligation.

Stages:
  repro   reproduce s19_sig0flip.txt's nu*_M(bar=1e4) column digit-for-digit
  d1      the constant c(bar) at three bars not used elsewhere in this work
  d2      the similarity collapse at fixed c (the hard clause)
  d3      the peak-time law
  d4      the operational corollary (bar vs M)
"""
import sys
import time

import numpy as np
from scipy.special import lambertw

from s19_sig0exact import sigma_M, peak_exact, apparent_threshold

BANKED = [(1024, 0.493852), (2048, 0.496917), (4096, 0.498456),
          (8192, 0.499228), (16384, 0.499614), (32768, 0.499807),
          (65536, 0.499903)]


def c_of_bar(bar):
    """c(bar) = w + 2 + 1/w with w = -W_{-1}(-e/bar)  (w > 1)."""
    w = -lambertw(-np.e / bar, -1).real
    return w + 2.0 + 1.0 / w, w


def x_of_c(c):
    """the peak abscissa: root of c = x^2/(x-1) with x > 1
       (x = 1+y, y the smaller root of y^2 + (2-c)y + 1 = 0)."""
    y = 0.5 * ((c - 2.0) - np.sqrt((c - 2.0) ** 2 - 4.0))
    return 1.0 + y, y


def phi_max(c):
    """Phi_max = y e^{1+1/y}."""
    _, y = x_of_c(c)
    return y * np.exp(1.0 + 1.0 / y)


# --- POST-D2 CORRECTION: the exact similarity
# function keeps the -1 that the 3a registration dropped,
#     Phi(x;c) = x (e^{c-x} - 1)/(c - x),
# which is also the form that is regular at x = c (Sigma_M = uM when r = 1).
# The threshold constant it implies differs from the registered
# c = w + 2 + 1/w only at O(1/bar); both are reported.
def phi_exact(x, c):
    return x * (np.expm1(c - x)) / (c - x)


def phi_max_exact(c):
    from scipy.optimize import brentq
    f = lambda x: 1.0 / x - np.exp(c - x) / np.expm1(c - x) + 1.0 / (c - x)
    xp = brentq(f, 1.0 + 1e-12, c - 1e-12)
    return phi_exact(xp, c), xp


def c_of_bar_exact(bar):
    from scipy.optimize import brentq
    g = lambda c: np.log(phi_max_exact(c)[0]) - np.log(bar)
    return brentq(g, 2.0000001, 200.0)


def stage_repro():
    print("stage repro: s19_sig0flip.txt nu*_M(bar=1e4) column, digit for digit")
    ok = True
    for M, rec in BANKED:
        v = apparent_threshold(M, 1.0e4)
        hit = ("%.6f" % v) == ("%.6f" % rec)
        ok = ok and hit
        print("   M=%6d  recorded %.6f   recomputed %.6f   %s"
              % (M, rec, v, "MATCH" if hit else "MISMATCH"))
    print("   REPRO: %s" % ("PASS -- digit-for-digit"
                            if ok else "FAIL -- STOP AND DIAGNOSE"))
    return ok


def stage_d1():
    print("stage d1: the constant c(bar) at three NEW bars")
    print("   REGISTERED predictions (computed before any run):")
    for bar in (1.0e2, 1.0e6, 1.0e8):
        c, w = c_of_bar(bar)
        print("     bar=%.0e   w = -W_{-1}(-e/bar) = %.6f   c = w+2+1/w = %.6f"
              "   [post-D2 exact c = %.6f]" % (bar, w, c, c_of_bar_exact(bar)))
    print("   free POST-HOC row: bar=1e4 predicts c = %.6f against the banked"
          " 12.6484" % c_of_bar(1.0e4)[0])
    for bar in (1.0e2, 1.0e6, 1.0e8):
        cpred, _ = c_of_bar(bar)
        cex = c_of_bar_exact(bar)
        print("   --- bar = %.0e   (REGISTERED c = %.6f ; post-D2 exact c = %.6f) ---"
              % (bar, cpred, cex))
        prev = None
        for k in (12, 14, 16, 18, 20):
            M = 2 ** k
            t0 = time.time()
            v = apparent_threshold(M, bar)
            cm = M * (1.0 - 2.0 * v)
            d = cm - cpred
            rate = "" if prev is None else "   |dev| ratio %.2f" % (abs(prev) / abs(d))
            print("     M=2^%02d=%8d  nu*_M=%.10f  c_meas=%.6f  dev(reg)=%+.4f%%"
                  "  dev(exact)=%+.4f%%%s [%.0f s]"
                  % (k, M, v, cm, 100.0 * d / cpred,
                     100.0 * (cm - cex) / cex, rate, time.time() - t0), flush=True)
            prev = d


def stage_d2():
    print("stage d2: the similarity collapse at FIXED c = 8 (the hard clause)")
    c = 8.0
    xp, y = x_of_c(c)
    pm = phi_max(c)
    print("   REGISTERED: Phi_max(8) = y e^{1+1/y} = %.6f with x_pk = %.6f"
          % (pm, xp))
    print("   bars: <=1.0%% at M=4096, <=0.05%% at M=262144, monotone in M")
    prev = None
    for M in (4096, 16384, 65536, 262144):
        nu = 0.5 * (1.0 - c / M)
        pk, tpk = peak_exact(nu, M)
        dev = 100.0 * (pk - pm) / pm
        mono = "" if prev is None else ("  monotone" if abs(dev) < abs(prev)
                                        else "  NOT monotone")
        print("     M=%7d  nu=%.12f  Sigma_pk=%.6f  t_pk=%.4f  dev=%+.4f%%%s"
              % (M, nu, pk, tpk, dev, mono), flush=True)
        prev = dev


def stage_d3():
    print("stage d3: the peak-time law  t_pk = ln(M/x_pk)/nu")
    c = 8.0
    xp, _ = x_of_c(c)
    print("   REGISTERED: <=0.2%% at every point;  x_pk(c=8) = %.6f" % xp)
    for M in (4096, 16384, 65536, 262144):
        nu = 0.5 * (1.0 - c / M)
        pk, tpk = peak_exact(nu, M)
        pred = np.log(M / xp) / nu
        print("     M=%7d  t_pk measured %.5f   predicted %.5f   dev %+.4f%%"
              % (M, tpk, pred, 100.0 * (tpk - pred) / pred), flush=True)
    print("   free POST-HOC row (the banked S19 datum nu=0.499, M=4096):")
    c0 = 4096 * (1.0 - 2.0 * 0.499)
    x0, _ = x_of_c(c0)
    pk0, t0 = peak_exact(0.499, 4096)
    print("     c = M(1-2nu) = %.4f  ->  x_pk = %.6f" % (c0, x0))
    print("     t_pk: predicted %.4f  vs S19 record 16.3640   (measured here %.4f)"
          % (np.log(4096 / x0) / 0.499, t0))
    print("     Sigma_pk: predicted %.2f  vs S19 record 187.7508  (measured here %.4f)"
          % (phi_max(c0), pk0))


def stage_d4():
    print("stage d4: the operational corollary -- the bar buys logarithms,"
          " M buys linearly")
    print("     bar          c(bar)     M for a 0.1%% threshold bias")
    for e in (2, 3, 4, 6, 8, 10, 12):
        c, _ = c_of_bar(10.0 ** e)
        print("     1e%-2d       %8.4f       %10d" % (e, c, int(np.ceil(c / 0.002))))
    print("   (bias = c/(2M) relative to 1/2, so 0.1%% needs M = c/0.002.)")
    print("   c grows like ln bar + 2 ln ln bar: ten orders of magnitude of"
          " bar cost a factor %.2f in c."
          % (c_of_bar(1e12)[0] / c_of_bar(1e2)[0]))


if __name__ == "__main__":
    st = sys.argv[1]
    {"repro": stage_repro, "d1": stage_d1, "d2": stage_d2,
     "d3": stage_d3, "d4": stage_d4}[st]()
