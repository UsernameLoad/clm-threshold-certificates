# Verifying the results

This note says what is in this repository and how to check it.

## 1. Contents

| Folder | Contents |
|---|---|
| `certificates/` | The 54 certificate records: 41 decay, 11 ladder, 2 earlier σ = 2 blowup. Each is a NumPy `.npz` holding the theorem's own parameters and margins. These *are* the theorems of Tables 1 and 2 of the first paper. |
| `code/` | The solvers, instruments and analysis scripts behind the published results, the table and figure generators, and the certificate verifier. |
| `records/` | Measurement outputs quoted in the papers. |
| `papers/` | LaTeX sources, PDFs and the figure. |
| `MANIFEST.txt` | Every file with its size and a SHA-256 prefix. |

Published papers by other authors are cited, not redistributed. The raw field archives from the
two-dimensional runs, about 1 GB, are too large to host here and are available on request.

## 2. Checking a certificate

A certificate is checkable in two lines, without rerunning anything. `code/verify_certificates.py`
does this for all 54 at once.

**Blowup certificates** (`lad_cert_*.npz`, `cap_tm_cert*.npz`). Read the fields and confirm

    S_lo / Astar_hi >= 2        and        t_bar - t1 >= H

Those two inequalities are the hypotheses of the ladder lemma (Lemma 4.2 of the first paper). The
lemma gives divergence of the mode sum at `t_bar`, and Proposition 2.1 turns that into the statement
that no smooth solution continues past `t_bar`.

**Decay certificates** (`exc_cert_*.npz`). Confirm

    Pbar < lambda_S             and        (lambda_S - Pbar - H/2) * H > Rbar / 2

at margin at least 3. Those are the hypotheses of the landing lemma (Lemma 3.2), which gives a bounded
and then decaying mode sum.

The design intent is that you do not have to trust the computation to check the result: the enclosure
argument is summarised in the record, and the two inequalities are arithmetic.

## 3. Reproducing a number

The tables are emitted from the certificate records by `code/s26_tables.py`, and the staircase figure
by `code/s27_fig.py`, which re-derives the staircase from the same records and asserts it against the
emitted tables before drawing. Measured values in Section 6 of the first paper come from the analysis
scripts in `code/`, run against the model hierarchy; the scripts are stage-based and deterministic, and
reruns are bit-identical.

Python 3.13 with NumPy, SciPy and mpmath. The certified interval paths additionally use python-flint
(Arb). Run scripts as `python <script> <stage>`.

## 4. Claims that were withdrawn during the work

These are the substantive corrections made before publication. Each is stated in the papers; they are
collected here so a reader can see them in one place.

- **A "valley" in a fitted analyticity-strip parameter**, reported in earlier drafts of the
  two-dimensional work, was an artifact of a degenerate two-parameter fit: the prefactor and
  exponential slots correlate at about 0.9 on that band, and under every one-shape variant the columns
  are monotone with no interior minimum. Every position statement resting on it is withdrawn. The bench
  note devotes a methods section to it.
- **A "sharp threshold" claim** was carried for a long time on a comparison lemma that does not give
  it. The comparison makes the blowup set an initial segment of the viscosity axis and the decay set a
  final segment, but does not force their endpoints to coincide. The first paper now states the
  threshold as that pair of endpoints, proves the half that is provable, and states the other half as
  open.
- **Two "×2 between grids" statements** were comparison artifacts: one compared bands that scaled with
  the grid instead of fixed physical modes, the other compared different times. Both are corrected in
  the bench note, together with the rules that prevent them.
- **A conjecture that a curvature constant equalled a squared exponent** was refuted by its own test
  and replaced by the measured map.

## 5. Scope

The mathematics here concerns one-dimensional model equations of the Constantin–Lax–Majda family and a
two-dimensional Boussinesq bench. **No claim is made about the three-dimensional Navier–Stokes
equations**, and no result here is offered as progress on that problem.

## 6. Authorship

Each paper carries an AI statement describing how it was produced and who is responsible for it. No AI
system is an author, in line with arXiv, COPE and publisher policy.
