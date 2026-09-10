"""S19 -- the sigma=0 closed form used as an EXACT model of the finite-M
verdict flip (the N15/N12 phenomenon, in closed form for the first time).

At sigma=0 the hierarchy is exactly solvable (S19 2a):
    q_m(t) = u (tau/2)^{m-1},   u = e^{-nu t},   tau = (1-u)/nu,
so the OBSERVED partial sum of a rank-M instrument is, exactly,
    Sigma_M(t) = u (1 - r^M)/(1 - r),   r(t) = tau/2 = (1-u)/(2 nu),
while the true sum is Sigma_inf = u/(1-r) for r<1 and +inf once r>=1.
r increases monotonically to r_inf = 1/(2 nu), so:
    TRUE blowup  <=>  r_inf > 1  <=>  nu < 1/2      (nu*_sigma(0) = 1/2)
    but a rank-M instrument sees only r^M, and when r_inf is only just above
    1 the factor r^M saturates at r_inf^M while the prefactor u decays like
    e^{-nu t} -- so Sigma_M peaks and then DIES.  A true blowup is reported
    as a metastable excursion followed by decay: N15, exactly solvable.

Everything below is closed-form arithmetic + verification rides.
"""
import sys

import numpy as np

from s19_sigsub import run_ride


def sigma_M(nu, t, M):
    u = np.exp(-nu * np.asarray(t, float))
    r = (1.0 - u) / (2.0 * nu)
    out = np.where(np.abs(r - 1.0) < 1e-14,
                   u * M,
                   u * (1.0 - r ** M) / (1.0 - r))
    return out


def peak_exact(nu, M, tmax=200.0, n=4000001):
    t = np.linspace(0.0, tmax, n)
    S = sigma_M(nu, t, M)
    i = int(np.argmax(S))
    return S[i], t[i]


def tbar_exact(nu, M, bar, tmax=200.0, n=4000001):
    """first t with Sigma_M(t) >= bar (exact closed form), else nan."""
    t = np.linspace(0.0, tmax, n)
    S = sigma_M(nu, t, M)
    idx = np.nonzero(S >= bar)[0]
    return float("nan") if len(idx) == 0 else t[idx[0]]


def apparent_threshold(M, bar, lo=0.40, hi=0.5, it=60):
    """the nu below which a rank-M, fixed-bar instrument reports BLOWUP."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        pk, _ = peak_exact(mid, M)
        if pk >= bar:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    BAR = 1.0e4

    if what in ("all", "book"):
        print("=== exact partial-sum bookkeeping for the S19 nu0 rides "
              "(M=4096, bar=1.0e4) ===")
        print("  measured values from s19_nu0.txt")
        meas = {0.450: ("BLOWUP", 1.000249e+04, 5.1403),
                0.480: ("BLOWUP", 1.000184e+04, 6.7797),
                0.499: ("DECAY", 1.877504e+02, 16.3639),
                0.501: ("DECAY", 1.000000e+00, 0.0000)}
        print("   nu      instrument verdict   exact t_bar(Sigma_M)  measured   dev")
        for nu in (0.450, 0.480):
            tb = tbar_exact(nu, 4096, BAR)
            print("   %.3f   %-8s             %10.4f       %10.4f   %+.4f"
                  % (nu, meas[nu][0], tb, meas[nu][2], meas[nu][2] - tb))
            tinf = -np.log(1 - 2 * nu) / nu
            print("           (true blowup time t* = %.4f; the full-sum t_bar was "
                  "%.4f -- the S19 2c registration used the FULL sum, which is "
                  "the wrong comparison for a rank-M instrument)"
                  % (tinf, -np.log(BAR * (1 - 2 * nu) / (BAR - 2 * nu)) / nu))
        print()
        print("   the nu=0.499 row: TRUE blowup (nu < 1/2) reported as DECAY")
        pk, tp = peak_exact(0.499, 4096)
        print("     exact  Sigma_M peak = %.6e at t = %.4f" % (pk, tp))
        print("     measured            = %.6e at t = %.4f" % (1.877504e+02, 16.3639))
        print("     deviation: %+.3f%% in amplitude, %+.4f in time"
              % (100 * (1.877504e+02 - pk) / pk, 16.3639 - tp))
        print()
        print("   the nu=0.501 row: TRUE decay, and the closed form says")
        pk, tp = peak_exact(0.501, 4096)
        print("     exact Sigma_M peak = %.8f at t = %.4f   (= Sigma(0) = 1, "
              "monotone) ; measured 1.000000 at t = 0.0000" % (pk, tp))

    if what in ("all", "flip"):
        print()
        print("=== the finite-M apparent threshold (the N15/N12 verdict flip, "
              "in closed form) ===")
        print("   a rank-M instrument with a fixed bar reports BLOWUP only for "
              "nu < nu*_M(bar):")
        print("     M      nu*_M(bar=1e4)     bias vs the true 1/2")
        for M in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
            na = apparent_threshold(M, BAR)
            print("   %6d      %.6f            %+.4f%%"
                  % (M, na, 100 * (na - 0.5) / 0.5))
        print("   => the bias is NEGATIVE (the instrument under-reports blowup: "
              "the front outruns M) and it closes like 1/M.")
        print()
        print("=== REGISTERED (before the rides): M-continuation at nu = 0.499 ===")
        for M in (4096, 8192, 16384):
            pk, tp = peak_exact(0.499, M)
            print("     M=%5d: exact Sigma_M peak = %.6e at t = %.4f  -> %s"
                  % (M, pk, tp, "BLOWUP" if pk >= BAR else "DECAY"))

    if what in ("all", "ride"):
        BAR2 = 1.0e3
        print()
        print("=== REGISTERED before the rides: M-continuation at nu=0.499, "
              "bar = 1.0e3 ===")
        print("   (bar lowered from 1e4 to 1e3 so the rides stay cheap; the "
              "flip is a verdict statement, not a magnitude one)")
        for M in (4096, 8192, 16384):
            pk, tp = peak_exact(0.499, M)
            tb = tbar_exact(0.499, M, BAR2)
            print("     M=%5d: exact peak %.6e at t=%.4f  -> %-6s"
                  "  exact t_bar(1e3) = %s"
                  % (M, pk, tp, "BLOWUP" if pk >= BAR2 else "DECAY",
                     "n/a (never reached)" if not np.isfinite(tb) else "%.4f" % tb))
        print()
        print("=== verification rides ===")
        for M in (8192, 16384):
            R = run_ride(0.0, 0.499, M, bar=BAR2, Tmax=400.0, fft=True)
            tb = tbar_exact(0.499, M, BAR2)
            print("   M=%5d: %-20s Sigma_peak=%.6e t_bar=%.4f  "
                  "[exact t_bar %.4f; dev %+.4f]  [%d steps, %.0f s]"
                  % (M, R["verdict"], R["peak"], R["t_bar"], tb,
                     R["t_bar"] - tb, R["nstep"], R["wall"]), flush=True)
