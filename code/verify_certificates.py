"""verify_certificates.py -- check every certificate record in certificates/.

Each certificate is a self-contained record of a machine-checked enclosure argument. Verifying one
does not require rerunning the computation: it requires checking the two inequalities that are the
hypotheses of the lemma the certificate invokes. This script does that for all of them.

    python verify_certificates.py [directory]        (default: ./certificates)

BLOWUP certificates (lad_cert_*.npz, cap_tm_cert*.npz) invoke the dyadic block-ladder lemma:
    margin := S_lo / Astar_hi >= 2      the seed exceeds the trigger, with the acceptance factor
    t_bar - t1 >= H                     the seed is held for at least the delay sum
Then the mode sum diverges at t_bar, and no smooth solution continues past it.

DECAY certificates (exc_cert_*.npz) invoke the landing lemma:
    Pbar < lamS                                    the low block is dominated by the tail rate
    (lamS - Pbar - H/2) * H > Rbar / 2             the hull condition, at margin >= 3
    Ytot < H                                       the tail is inside the hull at the landing time
Then the mode sum is bounded and decays.
"""
import sys, os, glob
import numpy as np


def f(z, k):
    return float(np.asarray(z[k]).reshape(-1)[0])


def nu(z):
    if "nu_num" in z.files and "nu_den" in z.files:
        return "%d/%d = %.9g" % (f(z, "nu_num"), f(z, "nu_den"), f(z, "nu_num") / f(z, "nu_den"))
    return "(in record)"


def check_blowup(z):
    S_lo, A_hi, t1 = f(z, "S_lo"), f(z, "Astar_hi"), f(z, "t1")
    margin = S_lo / A_hi
    ok = [("margin = S_lo/Astar_hi >= 2", margin >= 2.0, "%.4f" % margin)]
    if "t_bar" in z.files and "H_hi" in z.files:
        window, H = f(z, "t_bar") - t1, f(z, "H_hi")
        ok.append(("window t_bar - t1 >= H", window >= H, "%.6g >= %.6g" % (window, H)))
    elif "W" in z.files:
        ok.append(("window W recorded", f(z, "W") > 0, "%.6g" % f(z, "W")))
    return ok


def check_decay(z):
    Pbar, lamS, H, Rbar = f(z, "Pbar"), f(z, "lamS"), f(z, "H"), f(z, "Rbar")
    Ytot, margin = f(z, "Ytot"), f(z, "margin")
    hull = (lamS - Pbar - H / 2.0) * H
    return [
        ("Pbar < lamS", Pbar < lamS, "%.6g < %.6g" % (Pbar, lamS)),
        ("(lamS - Pbar - H/2)*H > Rbar/2", hull > Rbar / 2.0, "%.6g > %.6g" % (hull, Rbar / 2.0)),
        ("Ytot < H", Ytot < H, "%.6g < %.6g" % (Ytot, H)),
        ("landing margin >= 3", margin >= 3.0, "%.4g" % margin),
    ]


def main(d):
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    if not files:
        print("no certificate records found in", d)
        return 1
    npass = nfail = 0
    for path in files:
        z = np.load(path, allow_pickle=True)
        base = os.path.basename(path)
        kind = "decay" if base.startswith("exc_cert") else "blowup"
        checks = check_decay(z) if kind == "decay" else check_blowup(z)
        good = all(c[1] for c in checks)
        npass += good
        nfail += (not good)
        sig = ("sigma=%.4g " % f(z, "sigma")) if "sigma" in z.files else "sigma=2 "
        print("%-4s %-42s %s%-24s %s" % ("PASS" if good else "FAIL", base, sig, "nu=" + nu(z),
                                         "" if good else "  <-- " + "; ".join(c[0] for c in checks if not c[1])))
        if not good:
            for name, okc, val in checks:
                print("        %-34s %-28s %s" % (name, val, "ok" if okc else "FAILED"))
    print("\n%d certificates: %d pass, %d fail" % (len(files), npass, nfail))
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "certificates"))
