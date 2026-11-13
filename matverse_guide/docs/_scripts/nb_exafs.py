"""Cells for tutorials/exafs.ipynb.

See _nbbuild.py for how these are turned into an executed notebook.
"""

from __future__ import annotations

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# EXAFS shell analysis

Extended X-ray absorption fine structure (EXAFS) is the oscillation of the
absorption coefficient above an edge. The photoelectron leaves the absorber,
backscatters off its neighbours and interferes with itself, so the
oscillation `chi(k)` is a sum of damped sines, one per scattering path:

$$
\\chi(k) = \\sum_j \\frac{N_j S_0^2 F_j(k)}{k R_j^2}\\,
e^{-2k^2\\sigma_j^2}\\, e^{-2R_j/\\lambda(k)}\\,
\\sin\\!\\big(2kR_j + \\delta_j(k)\\big)
$$

The amplitude `F_j`, phase `delta_j` and mean free path `lambda` of each path
come from a scattering code (FEFF), the structure from the fit: coordination
number `N`, distance `R`, disorder `sigma^2` and one edge shift `dE0` per
spectrum. `S0^2`, the many-body amplitude reduction, multiplies every `N`, so it
is calibrated once, on a reference whose `N` is known — a metal foil measured
under the same conditions ([Rehr & Albers 2000](https://doi.org/10.1103/RevModPhys.72.621);
[Newville, *Fundamentals of XAFS*](https://doi.org/10.2138/rmg.2014.78.2)).

The hard part is not the fit. It is deciding **which shells the data
support**. A spectrum measured to `k = 12` holds about
`2 dk dR / pi + 2 ≈ 19` independent numbers over `R = 1-4 A`
([Stern 1993](https://doi.org/10.1103/PhysRevB.48.9825)); every shell costs
three or four of them; two shells closer than `pi / (2 dk)` cannot be told
apart; and a catalogue of standards from a dozen reference compounds offers
far more combinations than the data can rank by eye. `mv.exp.exafs_*`
implements that decision as a stated procedure:

| question | function |
|---|---|
| what standards are there, and which belong together? | `mv.exp.exafs_paths` |
| what is `S0^2`? | `mv.exp.exafs_s02` |
| which shells, and what are N, R, sigma^2 with honest errors — for a whole series? | `mv.exp.exafs_series` |
| does this particular model fit, and how well? | `mv.exp.exafs_fit`, `mv.exp.exafs_agreement` |
| what are the shells (paths of one family merged)? | `mv.exp.exafs_shells` |
| paths from a structure; chi(k) from mu(E) | `mv.exp.exafs_feff_paths`, `mv.exp.exafs_chi` (xraylarch) |

The forward model is Larch's `FeffPathGroup` with linear interpolation of the
`feff.dat` columns ([Newville 2013](https://doi.org/10.1088/1742-6596/430/1/012007));
the test suite checks the two agree point by point."""),

    ("code", """\
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matverse as mv
from matverse import _exafs as X

mv.pl.set_style()"""),

    ("markdown", """\
## A path catalogue whose answers we know

Real work starts from FEFF calculations on candidate structures (Larch ships
`feff6l`; `larch.xafs.feff6l` runs it on a `feff.inp`). To be able to *check*
the analysis, this page uses smooth FEFF-like paths written in the same
`feff.dat` layout, so every answer is known. The catalogue is what a real one
looks like: each path has an id, an **equivalence group** (standards of one
chemical family — e.g. every Fe–O path, whatever compound it came from — are
interchangeable wherever their distance bounds let them reach a shell), its
scatterers, number of legs, degeneracy, `reff` and source compound.

Four compounds:

- `site` — a rigid molecular site: four N at 1.95 A, eight C at 2.98 A and
  the focusing N–C multiple-scattering path. Its shells exist **together**:
  the C shell has exactly twice the N coordination, and the multiple
  scattering follows the C shell.
- `metal` — the foil's own structure (8 + 6 metal neighbours).
- `oxide` — an oxide with a light first shell and a metal second shell.
- `sulfide` — a distractor."""),

    ("code", """\
CATALOG = {
    "site": [("N_site", "gN", "N", 2, 4.0, 1.95, 7), ("C_site", "gC", "C", 2, 8.0, 2.98, 6),
             ("MS_site", "gMS", "C-N", 3, 16.0, 3.17, 7)],
    "metal": [("M1", "gM", "Fe", 2, 8.0, 2.48, 26), ("M2", "gM", "Fe", 2, 6.0, 2.87, 26)],
    "oxide": [("O_ox", "gO", "O", 2, 6.0, 2.06, 8), ("M_ox", "gMox", "Fe", 2, 12.0, 3.05, 26)],
    "sulfide": [("S_sul", "gS", "S", 2, 6.0, 2.30, 16)],
}
work = Path(tempfile.mkdtemp())
rows = []
for comp, paths in CATALOG.items():
    for pid, grp, sc, nleg, deg, reff, z in paths:
        X.write_feffdat(work / f"{pid}.dat", X.synthetic_path(reff, z, degen=deg, nleg=nleg),
                        title=comp, scatterer=sc.split("-")[-1])
        rows.append(dict(path_id=pid, file=f"{pid}.dat", group=grp, scatterers=sc, nleg=nleg,
                         degeneracy=deg, reff_angstrom=reff, source_compound=comp))
pd.DataFrame(rows).to_csv(work / "path_catalog.csv", index=False)

lib = mv.exp.exafs_paths(work / "path_catalog.csv", sites=["site"], foil_compound="metal")
lib.table"""),

    ("markdown", """\
`exafs_paths` classifies every scatterer (light, metal, other — from the
periodic table, not a list), ties each multiple-scattering path to the
outermost single-scattering path of its compound that it visits, and builds a
**bundle** for each declared site and for the foil compound: the first two
shells with the coordination ratio the structure fixes. The site's outer shell
is `site_only` — it is never offered on its own, because a carbon shell at
2.98 A without the nitrogen shell that holds it is not that site.

Which compounds are sites is the analyst's statement, made from the
catalogue's provenance (an isolated molecular moiety, a macrocycle, a single
atom site), not from what the answer should be. The foil compound is the
structure of the reference foil."""),

    ("code", """\
lib.bundles"""),

    ("markdown", """\
## S0^2 from the foil

The foil's coordination numbers are crystallographic — for these paths the
degeneracies 8 and 6. Fixing them leaves `S0^2`, one edge shift, and a
distance and `sigma^2` per shell. The fit is in R space, k-weights 1, 2 and 3
at once (each weight emphasises a different part of the spectrum and together
they break the `N`–`sigma^2` correlation), and the error is the covariance
error **scaled by the square root of the reduced chi-square** counted over
independent points — the convention of Larch's `feffit` and Artemis. Without
that scaling an error computed from the stated point noise is too small by the
factor the model misses the data."""),

    ("code", """\
K = np.round(np.arange(2.5, 12.0 + 1e-9, 0.05), 4)
KF = np.round(np.arange(2.5, 14.0 + 1e-9, 0.05), 4)
S02_TRUE = 0.80
rng = np.random.default_rng(7)

def spectrum(rows, e0, k=K, noise=1e-4):
    chi = sum(X.path_chi(k, lib.paths[p], n, r, s2, e0, S02_TRUE) for p, n, r, s2 in rows)
    return {"k": k, "chi": chi + rng.normal(0, noise, k.size), "sigma": np.full(k.size, noise)}

foil = spectrum([("M1", 8, 2.47, 0.0050), ("M2", 6, 2.86, 0.0068)], e0=0.8, k=KF)
cal = mv.exp.exafs_s02(foil, lib)
print(f"S0^2 = {cal['s02']:.3f} +- {cal['s02_err']:.3f}  (true {S02_TRUE})")
pd.DataFrame(cal["paths"]).T"""),

    ("markdown", """\
The fitted foil `sigma^2` values are kept on the library: inside a sample, a
metallic phase of the same element is never **more** ordered than the bulk
foil measured under the same conditions, so they become the lower bound of the
metal bundle's disorder.

## One stated model: `exafs_fit`

When the model is known, fit it. A term is a path id, `'<path>+ms'` (the path
with its tied multiple scattering), or `'site:<compound>'` (the bundle). Tied
members carry the propagated error: the C shell's `N` error is twice the N
shell's because its `N` is twice the N shell's."""),

    ("code", """\
oxide_true = [("O_ox", 5.2, 2.03, 0.0060), ("M_ox", 7.0, 3.08, 0.0085)]
ox = spectrum(oxide_true, e0=2.0)
fit = mv.exp.exafs_fit(ox, lib, ["O_ox", "M_ox"], s02=cal["s02"])
print(f"chi2_nu = {fit['redchi']:.2f}  n_idp = {fit['n_idp']:.1f}  parameters = {fit['n_params']}")
fit["table"]"""),

    ("markdown", """\
`exafs_agreement` measures how well **any** table of rows reproduces a
spectrum: a normalised k-space residual at each k-weight and the complex
R-space residual `|F_model - F_data| / |F_data|`. Unobserved points (a
monochromator glitch) are bridged by straight lines in both data and model
before the transform, so a gap never enters as a step."""),

    ("code", """\
mv.exp.exafs_agreement(ox, fit["table"], lib, s02=cal["s02"])"""),

    ("markdown", """\
## Which shells? The selection procedure

`exafs_series` does for every spectrum what an experienced analyst does by
hand, but every decision is a stated test:

1. **First shells first.** Terms (a path, a path with its tied multiple
   scattering, or a bundle) are added one at a time while the R window widens;
   a path whose `|FT|` peak would lie beyond the current window is not offered
   yet. A step must lower the BIC computed against the stated noise.
2. **Physical rules.** A shell of fewer than half a neighbour is below
   detection; a parameter on its bound is not determined; two shells of
   different terms closer than the resolution `pi / (2 dk)` are not both kept
   — except a light and a metal backscatterer, whose amplitudes and phases
   differ enough to separate (half the resolution still applies); a complete
   model keeps a margin of degrees of freedom below the independent points.
3. **Model choice.** Complete models compete by BIC on chi-square scaled by
   the best model's reduced chi-square: catalogue standards transfer
   imperfectly and point uncertainties understate the error, so the misfit,
   not the stated noise, sets the scale. Within two BIC units the simpler
   model wins.
4. **Clean-up.** Every simplification a likelihood-ratio test cannot reject
   is taken; a one-shell standard is swapped for another that reaches the same
   distance and fits better; a term left on a bound is replaced by another
   standard of its **equivalence group**, or dropped only if the
   likelihood-ratio test says the data do not need it.
5. **Edge shift.** Each spectrum's `dE0` is poorly determined and correlated
   with every `R`, so the search runs in overlapping `dE0` windows and the
   pooled candidates are refitted with the shift free.
6. **Series.** A spectrum read alone cannot determine a two-phase sample. A
   sample whose model holds no site is a **reference** for its phase, and
   every sample is re-read as site + reference phase with the shares adding
   up to one (mass balance). The phase keeps its reference's distances,
   disorder and coordination ratios; its metal neighbours share one size
   factor (small particles have fewer of them). That reading replaces the
   sample's own model unless the own model fits significantly better.

The **site fraction** is the site's first-shell `N` over its degeneracy.

Three samples: a pure site, the pure oxide, and a 35 % site / 65 % oxide
mixture whose oxide particles are smaller (fewer metal neighbours)."""),

    ("code", """\
site_true = [("N_site", 4.0, 1.97, 0.0040), ("C_site", 8.0, 3.00, 0.0060), ("MS_site", 16.0, 3.19, 0.0060)]
f = 0.35
mix_true = ([(p, f * n, r, s) for p, n, r, s in site_true]
            + [("O_ox", (1 - f) * 5.2, 2.03, 0.0060), ("M_ox", (1 - f) * 0.8 * 7.0, 3.08, 0.0085)])
data = {"P1": spectrum(site_true, e0=1.0), "P2": ox, "P3": spectrum(mix_true, e0=1.5)}

res = mv.exp.exafs_series(data, lib, foil=foil, kmin=3.0, rmin=1.0, rmax=3.6,
                          e0_starts=(0.0, 3.0), beam=3, n_jobs=1)
res["table"]"""),

    ("code", """\
pd.DataFrame({sid: {"reading": v["reading"], "reference": v.get("reference"),
                    "site fraction": res["site_fraction"][sid], "chi2_nu": round(v["redchi"], 2),
                    "n_idp": round(v["n_idp"], 1), "R-space misfit": round(v["agreement"]["rspace"], 4)}
              for sid, v in res["samples"].items()}).T"""),

    ("markdown", """\
`P3` was read as the site plus `P2`'s phase. Its site fraction sits close to
the true 0.35, and the oxide rows carry the complementary share: the light
first shell exactly `(1 - f)` of `P2`'s, the metal shell additionally scaled
by the size factor. A shell-by-shell fit of `P3` alone has no reason to
respect that balance.

## Shells, and how they are graded

Rows of one equivalence group whose sorted distances differ by less than a
link distance are one shell: total `N`, `N`-weighted `R`, and an effective
`sigma^2` that includes the spread of the members' distances. Errors follow to
first order."""),

    ("code", """\
mv.exp.exafs_shells(res["table"], lib, link=0.25)"""),

    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for sid, ax in zip(("P1", "P3"), axes):
    s = data[sid]
    mine = res["table"][res["table"].sample_id == sid]
    model = X.rows_chi(lib, zip(mine.path_id, mine.coordination_number, mine.distance_angstrom,
                                mine.sigma2_angstrom2, mine.delta_e0_ev), s["k"], res["s02"])
    r = np.arange(0, 5, 0.02)
    w = X.hanning(s["k"], 3.0, s["k"].max(), 1.0) * s["k"] ** 2 * 0.05
    op = np.exp(2j * np.outer(r, s["k"])) * w
    ax.plot(r, np.abs(op @ s["chi"]), label="data")
    ax.plot(r, np.abs(op @ model), "--", label="model")
    ax.set(xlabel="R (A, not phase-corrected)", ylabel="|FT[k^2 chi]|", title=sid)
    ax.legend()
fig.tight_layout()"""),

    ("markdown", """\
## Real inputs: paths from a structure, chi(k) from mu(E)

Two Larch steps sit in front of everything above.

`mv.exp.exafs_feff_paths` writes the FEFF input for one absorber in a
crystal structure (Larch's `structure2feffinp`), runs the FEFF6L that ships
with xraylarch, and returns a catalogue in the layout `exafs_paths` reads —
with the group set to the scatterer family, so standards from different
compounds that describe the same kind of neighbour share a group.
Catalogues of several structures are concatenated.

`mv.exp.exafs_chi` normalises a measured `mu(E)` (`pre_edge`) and removes the
background with AUTOBK, returning `chi(k)` on a uniform grid and a per-point
noise estimate from the high-R transform. Calibrate the energy axis against
the foil first."""),

    ("code", """\
try:
    from pymatgen.core import Lattice, Structure
    bcc = Structure.from_spacegroup("Im-3m", Lattice.cubic(2.8665), ["Fe"], [[0, 0, 0]])
    cat = mv.exp.exafs_feff_paths(bcc, "Fe", work / "feff_bcc", compound="bcc_Fe",
                                  cluster_size=5.0, rmax=4.2)
    display(cat[["path_id", "group", "scatterers", "nleg", "degeneracy", "reff_angstrom", "amp_ratio"]])

    e0 = 7112.0
    energy = np.arange(e0 - 200, e0 + 800, 0.5)
    kk = np.sqrt(np.clip(energy - e0, 0, None) * X.ETOK)
    chi_true = X.path_chi(np.maximum(kk, 0.05), lib.paths["O_ox"], 6.0, 2.03, 0.006, 0.0, S02_TRUE) * (energy > e0 + 5)
    step = 0.5 + np.arctan((energy - e0) / 2.0) / np.pi
    mu = 0.1 - 1e-5 * (energy - e0) + step * (1 - 1e-4 * np.clip(energy - e0, 0, None)) * (1 + chi_true)
    ext = mv.exp.exafs_chi(energy, mu, e0=e0, rbkg=1.0)
    print(f"E0 = {ext['e0']:.1f} eV, edge step {ext['edge_step']:.3f}, noise ~ {ext['sigma'][0]:.1e}")
except ImportError as exc:
    print("xraylarch not installed:", exc)"""),

    ("markdown", """\
## Pitfalls

- **`S0^2` and `N` are one number.** Floating both in a sample fit returns
  their product. Calibrate `S0^2` on the foil with the crystallographic `N`
  and keep it fixed.
- **`N` and `sigma^2` are correlated** (both set the amplitude). Fitting
  k-weights 1–3 together reduces the correlation; a shell with `sigma^2` on
  its upper bound and a small `N` is not detected, it is undetermined.
- **Resolution is `pi / (2 dk)`.** With `k = 3–12` two shells closer than
  0.17 A are one shell; reporting both is over-interpretation.
- **Errors without the reduced chi-square** are too small whenever the model
  misses the data by more than the noise — which is always, with transferred
  standards.
- **Equivalence groups are not decoration.** When a shell's standard ends on
  a bound, try the other standards of its group before concluding the shell
  is absent; a standard from another family that "fits as well" is a
  statement about the data's resolving power, not about the chemistry.
- **One spectrum cannot split two phases** whose shells overlap; a series
  with pure-phase references and mass balance can.
- **Runtime** grows with catalogue size x samples x edge-shift windows. For
  ~40 paths and 6 spectra expect minutes to tens of minutes on 4 CPUs: run the
  campaign once and save its table.

## References

- J. J. Rehr, R. C. Albers, *Rev. Mod. Phys.* **72**, 621 (2000).
- E. A. Stern, *Phys. Rev. B* **48**, 9825 (1993) — the number of independent points.
- M. Newville, *J. Phys.: Conf. Ser.* **430**, 012007 (2013) — Larch.
- B. Ravel, M. Newville, *J. Synchrotron Rad.* **12**, 537 (2005) — Athena/Artemis, `feffit` conventions.
- International X-ray Absorption Society, *Error reporting recommendations* (2000)."""),
]
