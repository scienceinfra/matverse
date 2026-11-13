"""EXAFS shell analysis engine behind the ``mv.exp.exafs_*`` functions.

Everything here is a function of the inputs: FEFF paths (``feff.dat`` files or
arrays with the same columns) described by a path catalog, chi(k) spectra with
their pointwise uncertainties, and the reference foil. No element, compound,
distance or sample is named in the code; which compounds are rigid "sites" and
which compound is the foil comes from the caller.

Physics
-------
``path_chi`` is Larch's ``FeffPathGroup`` calculation with linear
interpolation of the ``feff.dat`` columns at ``q = sqrt(k^2 - ETOK * dE0)``::

    chi_j = Im{ N S0^2 amp / (q R^2) exp(-2 reff Im(p) - 2 p^2 sigma2
                + i (2 q reff + pha + 2 p (R - reff - 2 sigma2 / reff))) }

Fitting is done in R space (real and imaginary parts of the complex Fourier
transform over ``[rmin, rmax]``), several k-weights at once, each weighted by
the noise its transform carries. Points flagged unobserved are bridged by
straight lines in data and model alike, so a gap never enters the transform as
a step. chi^2 is rescaled to the number of independent points
``n_idp = 2 dk dR / pi + offset`` (Larch: offset 1), and every 1-sigma error is
the covariance error scaled by ``sqrt(chi^2_nu)``.

Model selection
---------------
A model is a list of *terms*. A term is a single-scattering path, or a
single-scattering path with a multiple-scattering path of the same compound
whose amplitude, distance shift and disorder are tied to it, or a *site
bundle*: two shells of one compound whose coordination ratio that compound's
structure fixes (a rigid molecular site, with its focusing multiple
scattering; or the foil's own metal, whose shells may not be more ordered than
the foil). Terms are added one at a time in a widening R window (first shells
first), each step judged by the noise-referenced BIC, physical bounds, the
detection limit (half a neighbour) and the resolution rule (two shells of
separate terms closer than pi/(2 dk) are not both kept, except a light and a
metal backscatterer, which differ in phase and amplitude). Complete models
compete by BIC on chi^2 scaled by the best model's reduced chi^2 (the
standards transfer imperfectly, so the misfit, not only the stated noise, sets
the scale); then a bidirectional clean-up takes every simplification a
likelihood-ratio test cannot reject, and swaps a one-shell standard for another
that reaches the same distance and fits better. Each spectrum has its own edge
shift, poorly determined, so the search runs in overlapping edge-shift windows
and the pooled candidates are refitted with the shift free.

Series
------
A sample whose model holds no site is a reference for its phase. Each sample
can then be read as "site + reference phase" with the shares of the absorber
adding up to one (mass balance): the phase keeps its reference's distances,
disorder and coordination ratios, its metal neighbours share one size factor
(small particles have fewer neighbours), the site's share and geometry and the
edge shift are free. That reading replaces the sample's own model unless the
own model fits significantly better (likelihood-ratio test on chi^2 scaled by
the smaller misfit and at least the median misfit of the series).
"""

from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

#: 2 m_e / hbar^2 in eV^-1 A^-2 (k^2 = ETOK * (E - E0)).
ETOK = 0.26246842396479836


# --------------------------------------------------------------------- options
@dataclass
class Options:
    """Every number the analysis uses, with its reason.

    Transform / fit window
        kmin, kmax (None: end of each spectrum), window_dk (sin^2 taper width,
        also the end taper), rmin, rmax, rstep, kweights.
    Bounds (the physically admissible ranges)
        n_bounds (single / multiple scattering), dr_max (|R - reff|),
        s2_bounds, e0_bounds.
    Selection
        r_steps: Rmax of the early selection steps before the full window
        (first shells first); ft_offset: a path's |FT| peak lies roughly this
        far below its distance (paths whose peak is beyond the window are not
        offered yet); same_class_gap: two standards of one scatterer class
        closer than this are one shell; bundle_span: a site bundle joins a
        compound's first shell to its next shell within this span; beam:
        distinct structural hypotheses kept per step; nyquist_margin: a
        complete model keeps this many degrees of freedom; p_accept: the
        likelihood-ratio level for a simplification; bic_tie: within this many
        BIC units the model with fewer parameters wins; min_n: detection
        limit (neighbours); weak_n / weak_s2: a forward step may not add a
        shell of fewer than weak_n neighbours with sigma^2 above weak_s2;
        n_pinned: a coordination number above this fraction of its bound is
        pinned there (undetermined).
    Edge shift
        e0_starts: centres of the overlapping edge-shift windows, e0_window
        their half width, e0_final_window the half width of the last clean-up.
    Series
        min_share: each phase of a two-phase reading holds at least this share
        of the absorber; size_bounds: the factor on a phase's metal neighbours
        relative to its reference.
    """

    kmin: float = 3.0
    kmax: float | None = None
    window_dk: float = 1.0
    rmin: float = 1.0
    rmax: float = 4.0
    rstep: float = 0.02
    kweights: tuple = (1, 2, 3)
    nidp_offset: float = 2.0   # Stern 1993; Larch feffit uses 1 (chi^2 / error normalisation only)
    n_bounds: dict = field(default_factory=lambda: {"ss": (0.2, 12.0), "ms": (0.5, 48.0)})
    dr_max: dict = field(default_factory=lambda: {"ss": 0.12, "ms": 0.30})
    s2_bounds: tuple = (0.0010, 0.0250)
    e0_bounds: tuple = (-10.0, 10.0)
    r_steps: tuple = (2.6, 3.2)
    ft_offset: float = 0.45
    same_class_gap: float = 0.35
    bundle_span: float = 1.2
    beam: int = 6
    nyquist_margin: float = 5.0
    p_accept: float = 1e-5
    bic_tie: float = 2.0
    min_n: float = 0.5
    weak_n: float = 4.0
    weak_s2: float = 0.015
    n_pinned: float = 0.98
    n_candidates: int = 6
    e0_starts: tuple = (-3.0, 0.0, 3.0, 6.0)
    e0_window: float = 1.5
    e0_final_window: float = 0.5
    min_share: float = 0.1
    size_bounds: tuple = (0.2, 1.5)
    use_ms: bool = True

    @property
    def r_windows(self):
        steps = tuple(min(r, self.rmax) for r in self.r_steps)
        return steps + (self.rmax,) * 3


def make_options(**kw) -> Options:
    unknown = set(kw) - set(Options.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown EXAFS option(s) {sorted(unknown)}; "
                         f"known: {sorted(Options.__dataclass_fields__)}")
    return Options(**kw)


# --------------------------------------------------------------------- paths
def read_feffdat(source) -> dict:
    """Parse a FEFF ``feff.dat`` (FEFF6/8 layout) into arrays.

    Returns ``{'reff', 'degen', 'nleg', 'k', 'pha', 'amp', 'lam', 'rep',
    'atoms'}`` with ``pha = real[2*phc] + phase[feff]`` and
    ``amp = mag[feff] * red factor``, the combination Larch uses.
    """
    lines = Path(source).read_text().splitlines()
    head_i = next(i for i, ln in enumerate(lines) if "nleg, deg, reff" in ln)
    head = lines[head_i].split()
    start = next(i for i, ln in enumerate(lines) if "real[2*phc]" in ln) + 1
    atoms = []
    for ln in lines[head_i + 2:start - 1]:
        parts = ln.split()
        if len(parts) >= 6:
            atoms.append(parts[5])
    d = np.array([[float(x) for x in ln.split()] for ln in lines[start:] if ln.strip()])
    return {"reff": float(head[2]), "degen": float(head[1]), "nleg": int(float(head[0])),
            "k": d[:, 0], "pha": d[:, 1] + d[:, 3], "amp": d[:, 2] * d[:, 4],
            "lam": d[:, 5], "rep": d[:, 6], "atoms": atoms}


def path_chi(k, p, n, r, s2, e0, s02):
    """chi(k) of one path (Larch FeffPathGroup, linear interpolation)."""
    k = np.asarray(k, float)
    en = k * k - e0 * ETOK
    en = np.where(np.abs(en) < 1e-6, 1e-6, en)
    q = np.sign(en) * np.sqrt(np.abs(en))
    pha, amp = np.interp(q, p["k"], p["pha"]), np.interp(q, p["k"], p["amp"])
    rep, lam = np.interp(q, p["k"], p["rep"]), np.interp(q, p["k"], p["lam"])
    pp = (rep + 1j / lam) ** 2
    pc = np.sqrt(pp)
    reff = p["reff"]
    c = np.exp(-2 * reff * pc.imag - 2 * pp * s2
               + 1j * (2 * q * reff + pha + 2 * pc * (r - reff - 2 * s2 / reff)))
    return (n * s02 * amp * c / (q * r * r)).imag


def scatterer_class(symbol: str) -> str:
    """'light' (Z <= 9), 'metal', or 'nonmetal' (heavier p-block anions)."""
    sym = symbol.strip().capitalize()
    try:
        from pymatgen.core.periodic_table import Element
        el = Element(sym)
        z, metal = el.Z, bool(el.is_metal)
    except Exception:  # pragma: no cover - pymatgen missing
        from ase.data import atomic_numbers
        z = atomic_numbers[sym]
        metal = z in set(range(3, 5)) | set(range(11, 14)) | set(range(19, 33)) | \
            set(range(37, 52)) | set(range(55, 85)) | set(range(87, 119))
    if z <= 9:
        return "light"
    return "metal" if metal else "nonmetal"


_ALIASES = {
    "path_id": ("path_id", "id", "label", "name", "path"),
    "file": ("file", "feffdat", "filename", "feff_file"),
    "group": ("group", "equivalence_group", "family"),
    "scatterers": ("scatterers", "scatterer", "atoms"),
    "nleg": ("nleg", "n_leg", "legs"),
    "degeneracy": ("degeneracy", "degen", "deg", "n"),
    "reff": ("reff_angstrom", "reff", "r_eff", "distance"),
    "compound": ("source_compound", "compound", "structure", "source"),
}


def _col(row: dict, key: str, default=None):
    low = {k.lower().strip(): v for k, v in row.items()}
    for name in _ALIASES[key]:
        if name in low and low[name] not in (None, ""):
            return low[name]
    return default


class Library:
    """A catalog of FEFF paths with equivalence groups, compounds and terms.

    ``catalog``: list of dicts / DataFrame / CSV path with (aliases accepted)
    ``path_id, file, group, scatterers, nleg, degeneracy, reff, compound``;
    ``group`` defaults to the scatterer (plus ``-ms``), ``compound`` to ''.
    ``paths``: optional ``{path_id: feffdat dict}`` instead of files.
    ``sites``: compounds that are rigid sites (coordination ratios fixed by
    the compound, focusing multiple scattering tied); ``foil_compound``: the
    foil's structure (a metal bundle whose shells are at least as disordered
    as the foil's).
    """

    def __init__(self, catalog, base_dir=None, paths=None, sites=(), foil_compound=None, bundle_span=1.2):
        rows = _catalog_rows(catalog)
        base = Path(base_dir) if base_dir else None
        if base is None and isinstance(catalog, (str, Path)):
            base = Path(catalog).parent
        self.info, self.paths = {}, {}
        for r in rows:
            pid = str(_col(r, "path_id"))
            if paths and pid in paths:
                p = dict(paths[pid])
            else:
                f = _col(r, "file")
                if f is None:
                    raise ValueError(f"path {pid}: no file and no arrays given")
                fp = Path(f)
                if not fp.is_absolute() and base is not None:
                    fp = base / fp
                p = read_feffdat(fp)
            deg = _col(r, "degeneracy")
            reff = _col(r, "reff")
            nleg = _col(r, "nleg")
            p["reff"] = float(reff) if reff is not None else p["reff"]
            p["degen"] = float(deg) if deg is not None else p.get("degen", 1.0)
            p["nleg"] = int(float(nleg)) if nleg is not None else p.get("nleg", 2)
            sc = _col(r, "scatterers")
            if sc is None:
                atoms = p.get("atoms") or []
                sc = "-".join(dict.fromkeys(atoms[1:])) if len(atoms) > 1 else "X"
            grp = _col(r, "group") or (sc + ("-ms" if p["nleg"] > 2 else ""))
            self.info[pid] = {"path_id": pid, "group": str(grp), "scatterers": str(sc),
                              "nleg": p["nleg"], "degeneracy": p["degen"], "reff": p["reff"],
                              "compound": str(_col(r, "compound", "") or "")}
            self.paths[pid] = p
        self.kind = {}
        for pid, inf in self.info.items():
            if inf["nleg"] > 2:
                self.kind[pid] = "ms"
            else:
                self.kind[pid] = scatterer_class(inf["scatterers"].split("-")[-1])
        compounds = {inf["compound"] for inf in self.info.values()}
        for c in list(sites) + ([foil_compound] if foil_compound else []):
            if c not in compounds:
                raise ValueError(f"compound {c!r} is not in the catalog; compounds: {sorted(compounds)}")
        self.sites = tuple(sites)
        self.foil_compound = foil_compound
        # each multiple-scattering path is tied to the outermost single-scattering
        # path of the same compound that it visits
        self.partner = {}
        for pid, inf in self.info.items():
            if self.kind[pid] != "ms":
                continue
            atoms = set(inf["scatterers"].split("-"))
            same = [q for q, s in self.info.items() if self.kind[q] != "ms" and s["compound"] == inf["compound"]
                    and s["compound"] and s["scatterers"] in atoms and s["reff"] <= inf["reff"]]
            if same:
                ss = max(same, key=lambda q: self.info[q]["reff"])
                self.partner.setdefault(ss, []).append(pid)
        self.single = sorted(p for p in self.paths if self.kind[p] != "ms")
        self.foil_s2 = {}
        self.bundles, self.site_only = [], set()
        for compound in sorted(set(self.sites) | ({foil_compound} if foil_compound else set())):
            ss = sorted((q for q in self.single if self.info[q]["compound"] == compound),
                        key=lambda q: self.info[q]["reff"])
            if len(ss) < 2:
                continue
            first, second = ss[0], ss[1]
            if self.info[second]["reff"] - self.info[first]["reff"] > bundle_span:
                continue
            if self.kind[first] != self.kind[second]:
                continue  # a fixed ratio between unlike scatterers only holds for that very compound
            ratio = self.paths[second]["degen"] / self.paths[first]["degen"]
            core = ((first, 1.0, 0), (second, ratio, 1))
            if compound in self.sites:
                ms = tuple((m, self.paths[m]["degen"] / self.paths[first]["degen"], 1)
                           for m in self.partner.get(second, []))
                self.bundles.append(core + ms)
                self.site_only.add(second)  # a site's outer shell exists only with the site
            else:
                self.bundles.append(core)

    # ----- terms
    def moves(self, use_ms=True):
        out = [((p, 1.0, 0),) for p in self.single if p not in self.site_only]
        out += self.bundles
        if not use_ms:
            out = [t for t in out if not any(self.kind[m] == "ms" for m, _, _ in t)]
        return out

    def owner(self, term, group):
        return next(m for m, _, g in term if g == group and self.kind[m] != "ms")

    def n_groups(self, term):
        return 1 + max(g for _, _, g in term)

    def n_par(self, term):
        return 1 + 2 * self.n_groups(term)

    def lead_reff(self, term):
        return self.paths[term[0][0]]["reff"]

    def singles_of(self, term):
        return [m for m, _, _ in term if self.kind[m] != "ms"]

    def members(self, term, xt):
        out = []
        for m, ratio, g in term:
            own = self.owner(term, g)
            r_g, s2_g = xt[1 + 2 * g], xt[2 + 2 * g]
            out.append((m, ratio * xt[0], self.paths[m]["reff"] + (r_g - self.paths[own]["reff"]), s2_g))
        return out

    def term_chi(self, k, term, xt, e0, s02):
        chi = np.zeros_like(k)
        for m, n, r, s2 in self.members(term, xt):
            chi += path_chi(k, self.paths[m], n, r, s2, e0, s02)
        return chi

    def is_bundle_of(self, term, compound):
        return self.n_groups(term) > 1 and self.info[term[0][0]]["compound"] == compound

    def s2_floor(self, term, group, opt):
        own = self.owner(term, group)
        if (self.n_groups(term) > 1 and self.foil_compound
                and self.info[own]["compound"] == self.foil_compound and own in self.foil_s2):
            return max(opt.s2_bounds[0], self.foil_s2[own])
        return opt.s2_bounds[0]

    def bounds(self, term, opt, dr_max=None):
        dr_max = opt.dr_max["ss"] if dr_max is None else dr_max
        n_lo, n_hi = 0.0, np.inf
        for m, ratio, _ in term:
            lo, hi = opt.n_bounds["ms" if self.kind[m] == "ms" else "ss"]
            n_lo, n_hi = max(n_lo, lo / ratio), min(n_hi, hi / ratio)
        lo, hi = [n_lo], [n_hi]
        for g in range(self.n_groups(term)):
            reff = self.paths[self.owner(term, g)]["reff"]
            lo += [reff - dr_max, self.s2_floor(term, g, opt)]
            hi += [reff + dr_max, opt.s2_bounds[1]]
        return lo, hi

    def start(self, term, dr):
        x = [min(self.paths[term[0][0]]["degen"], 4.0)]
        for g in range(self.n_groups(term)):
            x += [self.paths[self.owner(term, g)]["reff"] + dr, 0.006]
        return x


def _catalog_rows(catalog):
    if isinstance(catalog, (str, Path)):
        with open(catalog, newline="") as f:
            return list(csv.DictReader(f))
    try:
        import pandas as pd
        if isinstance(catalog, pd.DataFrame):
            return catalog.to_dict("records")
    except ImportError:  # pragma: no cover
        pass
    return [dict(r) for r in catalog]


# --------------------------------------------------------------------- transform
def hanning(k, kmin, kmax, dk):
    w = np.zeros_like(k)
    w[(k >= kmin + dk) & (k <= kmax - dk)] = 1.0
    up, dn = (k >= kmin) & (k < kmin + dk), (k > kmax - dk) & (k <= kmax)
    w[up] = np.sin(0.5 * np.pi * (k[up] - kmin) / dk) ** 2
    w[dn] = np.sin(0.5 * np.pi * (kmax - k[dn]) / dk) ** 2
    return w


def fill_matrix(k, observed):
    """Linear operator bridging unobserved points by straight lines."""
    obs = np.flatnonzero(observed)
    m = np.zeros((k.size, k.size))
    for i in range(k.size):
        if observed[i]:
            m[i, i] = 1.0
            continue
        j = np.searchsorted(obs, i)
        lo, hi = obs[max(j - 1, 0)], obs[min(j, obs.size - 1)]
        t = 0.0 if hi == lo else (k[i] - k[lo]) / (k[hi] - k[lo])
        m[i, lo] += 1.0 - t
        m[i, hi] += t
    return m


class Transform:
    """Complex FT at several k-weights over [rmin, rmax]; data and model share
    the gap bridging; each k-weight's residual is scaled by its noise."""

    def __init__(self, k, sigma, kmax, opt: Options, kmin=None, rmin=None, rmax=None, observed=None):
        kmin = opt.kmin if kmin is None else kmin
        rmin = opt.rmin if rmin is None else rmin
        rmax = opt.rmax if rmax is None else rmax
        r = np.arange(rmin, rmax + 1e-9, opt.rstep)
        dk = k[1] - k[0]
        w = hanning(k, kmin, kmax, opt.window_dk)
        base = np.exp(2j * np.outer(r, k)) * dk
        fill = np.eye(k.size) if observed is None else fill_matrix(k, np.asarray(observed, bool))
        self.ops = [(base * (w * k ** kw)) @ fill for kw in opt.kweights]
        self.eps = [np.sqrt(np.sum((w * k ** kw * sigma * dk) ** 2) / 2.0) for kw in opt.kweights]
        self.n_idp = 2 * (kmax - kmin) * (rmax - rmin) / np.pi + opt.nidp_offset
        self.n_pts = 2 * r.size * len(opt.kweights)
        self.r = r

    def data(self, chi):
        return [op @ chi for op in self.ops]

    def resid(self, chi_model, data_ft):
        out = []
        for op, eps, dft in zip(self.ops, self.eps, data_ft):
            z = (op @ chi_model - dft) / eps
            out += [z.real, z.imag]
        return np.concatenate(out)


def estimate_sigma(k, chi, kmin=3.0, kmax=None, kweight=2):
    """Constant per-point noise of chi(k) from the |FT| at 15-25 A (where no
    signal lives), the convention of Larch's estimate_noise."""
    k = np.asarray(k, float)
    chi = np.nan_to_num(np.asarray(chi, float))
    kmax = float(k.max()) if kmax is None else kmax
    dk = k[1] - k[0]
    w = hanning(k, kmin, kmax, 1.0)
    r = np.arange(15.0, 25.0, 0.05)
    ft = (np.exp(2j * np.outer(r, k)) * dk) @ (w * k ** kweight * chi)
    eps_r = np.sqrt(np.mean(np.abs(ft) ** 2))
    # one point's noise sigma gives |FT| rms sqrt(sum (w k^kw sigma dk)^2)
    norm = np.sqrt(np.sum((w * k ** kweight * dk) ** 2))
    return np.full_like(k, eps_r / max(norm, 1e-30))


# --------------------------------------------------------------------- fits
class Fit:
    def __init__(self, terms, x, chi2, lib):
        self.terms, self.x, self.chi2 = terms, np.asarray(x, float), chi2
        self.slices, i = [], 0
        for t in terms:
            self.slices.append(slice(i, i + lib.n_par(t)))
            i += lib.n_par(t)

    def xt(self, i):
        return self.x[self.slices[i]]


class Context:
    """One spectrum with its transforms (cached per Rmax)."""

    def __init__(self, sid, lib, k, chi, sigma, observed, opt: Options):
        self.sid, self.lib, self.opt = sid, lib, opt
        self.k = np.asarray(k, float)
        self.obs = np.asarray(observed, bool)
        self.chi = np.where(self.obs, np.nan_to_num(np.asarray(chi, float)), 0.0)
        self.sigma = np.asarray(sigma, float)
        self.kmax = float(opt.kmax) if opt.kmax is not None else float(self.k[self.obs].max())
        self.res = np.pi / (2 * (self.kmax - opt.kmin))
        self._win = {}

    def window(self, rmax):
        if rmax not in self._win:
            tr = Transform(self.k, self.sigma, self.kmax, self.opt, rmax=rmax, observed=self.obs)
            self._win[rmax] = (tr, tr.data(self.chi))
        return self._win[rmax]


def term_bounds(lib, terms, opt, e0_bounds):
    lo, hi = [], []
    for t in terms:
        tl, th = lib.bounds(t, opt)
        lo += tl
        hi += th
    return np.array(lo + [e0_bounds[0]]), np.array(hi + [e0_bounds[1]])


def model_chi(ctx, terms, x, s02):
    lib, k = ctx.lib, ctx.k
    chi, i = np.zeros_like(k), 0
    for t in terms:
        size = lib.n_par(t)
        chi += lib.term_chi(k, t, x[i:i + size], x[-1], s02)
        i += size
    return chi


def fit_terms(ctx, terms, x0, s02, e0_bounds=None, rmax=None, max_nfev=400, want_res=False):
    opt, lib = ctx.opt, ctx.lib
    e0_bounds = opt.e0_bounds if e0_bounds is None else e0_bounds
    rmax = opt.rmax if rmax is None else rmax
    tr, dft = ctx.window(rmax)
    lo, hi = term_bounds(lib, terms, opt, e0_bounds)
    x0 = np.clip(np.asarray(x0, float), lo + 1e-9, hi - 1e-9)

    def resid(x):
        return tr.resid(model_chi(ctx, terms, x, s02), dft)

    res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=max_nfev)
    fit = Fit(tuple(terms), res.x, float(np.sum(res.fun ** 2) * tr.n_idp / tr.n_pts), lib)
    fit.res = ctx.res
    if want_res:
        return fit, res, tr
    return fit


def scaled_covariance(res, tr, n_par=None):
    """Covariance of a least_squares result on a Transform residual, scaled by
    the reduced chi^2 counted over independent points (Larch / Artemis)."""
    jac = res.jac * np.sqrt(tr.n_idp / tr.n_pts)
    chi2 = float(np.sum(res.fun ** 2) * tr.n_idp / tr.n_pts)
    n_par = len(res.x) if n_par is None else n_par
    redchi = chi2 / max(tr.n_idp - n_par, 1.0)
    return np.linalg.pinv(jac.T @ jac) * redchi


def scaled_errors(res, tr):
    return np.sqrt(np.clip(np.diag(scaled_covariance(res, tr)), 0.0, None))


def physical(fit, lib, opt, complete=True):
    """Detection, resolution and bound rules for a fitted model."""
    resolution = getattr(fit, "res", 0.2)
    dr_sel = opt.dr_max["ss"]
    shells = []
    for i, t in enumerate(fit.terms):
        for m, n, r, s2 in lib.members(t, fit.xt(i)):
            if lib.kind[m] != "ms":
                shells.append((i, r, lib.kind[m]))
    for a in range(len(shells)):
        for b in range(a + 1, len(shells)):
            (ia, ra, ka), (ib, rb, kb) = shells[a], shells[b]
            gap = abs(ra - rb)
            if ia != ib and (gap < resolution / 2 or (gap < resolution and {ka, kb} != {"light", "metal"})):
                return False
    for i, t in enumerate(fit.terms):
        xt = fit.xt(i)
        for g in range(lib.n_groups(t)):
            s2 = xt[2 + 2 * g]
            n_shell = next(r for m, r, gg in t if gg == g and lib.kind[m] != "ms") * xt[0]
            if n_shell < opt.min_n and g == 0:
                return False
            floored = lib.s2_floor(t, g, opt) > opt.s2_bounds[0]
            pinned_low = s2 < 1.2 * opt.s2_bounds[0] and not floored
            pinned_high = s2 > 0.95 * opt.s2_bounds[1]
            if g == 0 and (pinned_low or pinned_high):
                return False
            if g == 1 and complete and pinned_high:
                return False
            owner = lib.owner(t, g)
            if abs(xt[1 + 2 * g] - lib.paths[owner]["reff"]) > 0.97 * dr_sel:
                return False
            if not complete and g == 0 and n_shell < opt.weak_n and s2 > opt.weak_s2:
                return False
    return True


def determined(fit, lib, opt):
    return all(fit.xt(i)[0] <= opt.n_pinned * lib.bounds(t, opt)[1][0] for i, t in enumerate(fit.terms))


def n_params(fit, lib):
    """Free parameters, counting each tied multiple-scattering member as one modelling choice."""
    return len(fit.x) + sum(1 for t in fit.terms for m, _, _ in t if lib.kind[m] == "ms")


def bic(fit, n_idp, lib, scale=1.0):
    return fit.chi2 / scale + n_params(fit, lib) * np.log(n_idp)


def supported(fit, n_idp, lib, opt):
    """Degrees-of-freedom rule: a complete model keeps ``nyquist_margin``
    independent points free. The count here is always Stern's
    ``2 dk dR / pi + 2``, whatever offset normalises chi^2 (``nidp_offset``
    only changes the error / chi^2 convention, not which models are allowed)."""
    return n_params(fit, lib) <= (n_idp - opt.nidp_offset + 2.0) - opt.nyquist_margin


def signature(fit, lib):
    return tuple(sorted(lib.info[m]["group"] for t in fit.terms for m, _, _ in t))


def diverse(children, lib):
    out, seen = [], set()
    for f in sorted(children, key=lambda f: f.chi2):
        sig = signature(f, lib)
        if sig not in seen:
            seen.add(sig)
            out.append(f)
    return out


def improves(parent, child, n_idp, lib, opt):
    if not physical(child, lib, opt, complete=False):
        return False
    return child.chi2 + len(child.x) * np.log(n_idp) < parent.chi2 + len(parent.x) * np.log(n_idp)


def allowed(lib, terms, move, opt):
    present = [m for t in terms for m in lib.singles_of(t)]
    for ss in lib.singles_of(move):
        for other in present:
            if other == ss:
                return False
            if lib.kind[other] == lib.kind[ss] and abs(lib.paths[other]["reff"] - lib.paths[ss]["reff"]) < opt.same_class_gap:
                return False
    return True


def empty(ctx, rmax, e0):
    tr, dft = ctx.window(rmax)
    fit = Fit((), [e0], float(np.sum(tr.resid(np.zeros_like(ctx.k), dft) ** 2) * tr.n_idp / tr.n_pts), ctx.lib)
    fit.res = ctx.res
    return fit


def select(ctx, s02, e0_bounds):
    """Beam search over terms with a growing window; returns
    (best terms, best x, candidate list, misfit scale)."""
    lib, opt = ctx.lib, ctx.opt
    e0_start = 0.5 * (e0_bounds[0] + e0_bounds[1])
    windows = opt.r_windows
    beam, accepted, seen = [empty(ctx, windows[0], e0_start)], [], set()
    for step, rmax in enumerate(windows):
        tr, _ = ctx.window(rmax)
        if step and rmax != windows[step - 1]:
            beam = [fit_terms(ctx, f.terms, f.x, s02, e0_bounds, rmax) if f.terms
                    else empty(ctx, rmax, e0_start) for f in beam]
        children = []
        for parent in beam:
            for move in lib.moves(opt.use_ms):
                if lib.lead_reff(move) - opt.ft_offset > rmax or not allowed(lib, parent.terms, move, opt):
                    continue
                terms = parent.terms + (move,)
                key = (tuple(sorted(terms, key=str)), rmax)
                if key in seen:
                    continue
                seen.add(key)
                trial = None
                for dr in (-0.08, 0.0, 0.08):
                    x0 = list(parent.x[:-1]) + lib.start(move, dr) + [parent.x[-1]]
                    fit = fit_terms(ctx, terms, x0, s02, e0_bounds, rmax)
                    if trial is None or fit.chi2 < trial.chi2:
                        trial = fit
                if improves(parent, trial, tr.n_idp, lib, opt):
                    children.append(trial)
        if children:
            beam = diverse(children, lib)[:opt.beam]
            for b in lib.bundles:
                holding = [f for f in children if b in f.terms]
                if holding and not any(b in f.terms for f in beam):
                    beam.append(min(holding, key=lambda f: f.chi2))
            accepted += beam
        elif rmax == opt.rmax:
            break
    if not accepted:
        return (), np.array([e0_start]), [], 1.0
    full_tr, _ = ctx.window(opt.rmax)
    finals, done = [], set()
    for f in accepted:
        key = tuple(sorted(f.terms, key=str))
        if key in done:
            continue
        done.add(key)
        g = fit_terms(ctx, f.terms, f.x, s02, e0_bounds)
        if physical(g, lib, opt) and determined(g, lib, opt) and supported(g, full_tr.n_idp, lib, opt):
            finals.append(g)
    if not finals:
        return accepted[-1].terms, accepted[-1].x, [(accepted[-1].terms, accepted[-1].x)], 1.0
    scale = max(1.0, min(g.chi2 / max(full_tr.n_idp - n_params(g, lib), 1.0) for g in finals))
    top = min(bic(g, full_tr.n_idp, lib, scale) for g in finals)
    near = [g for g in finals if bic(g, full_tr.n_idp, lib, scale) <= top + opt.bic_tie]
    best = min(near, key=lambda g: (n_params(g, lib), bic(g, full_tr.n_idp, lib, scale)))
    best = refine(ctx, best, s02, e0_bounds, full_tr.n_idp)
    scale = max(1.0, best.chi2 / max(full_tr.n_idp - n_params(best, lib), 1.0))
    key = tuple(sorted(best.terms, key=str))
    ranked = sorted(finals, key=lambda g: bic(g, full_tr.n_idp, lib, scale))
    cands = [(best.terms, best.x)] + [(g.terms, g.x) for g in ranked if tuple(sorted(g.terms, key=str)) != key]
    return best.terms, best.x, cands[:opt.n_candidates], scale


def adequate(simple, full, n_idp, lib, opt):
    """The simpler model is not significantly worse (likelihood-ratio test)."""
    if not physical(simple, lib, opt) or not determined(simple, lib, opt):
        return False
    extra = n_params(full, lib) - n_params(simple, lib)
    if extra <= 0:
        return simple.chi2 <= full.chi2
    scale = max(1.0, full.chi2 / max(n_idp - n_params(full, lib), 1.0))
    return (simple.chi2 - full.chi2) / scale < chi2_dist.ppf(1.0 - opt.p_accept, extra)


def reduced(lib, term, xt):
    out = []
    if lib.n_groups(term) > 1:
        out.append((tuple(m for m in term if m[2] == 0), list(xt[:3])))
        outer = [m for m in term if m[2] == 1]
        lead = next(m for m in outer if lib.kind[m[0]] != "ms")
        rest = [m for m in outer if m is not lead]
        out.append((((lead[0], 1.0, 0),) + tuple((m, r / lead[1], 0) for m, r, _ in rest),
                    [lead[1] * xt[0], xt[3], xt[4]]))
    if any(lib.kind[m] == "ms" for m, _, _ in term) and not any(m in lib.site_only for m, _, _ in term):
        no_ms = tuple(m for m in term if lib.kind[m[0]] != "ms")
        out.append((no_ms, list(xt[:lib.n_par(no_ms)])))
    return out


def tied_alternatives(lib, term, xt):
    out, present = [], {m for m, _, _ in term}
    for ss, ratio, g in term:
        if lib.kind[ss] == "ms":
            continue
        for ms in lib.partner.get(ss, []):
            if ms not in present:
                out.append((term + ((ms, ratio * lib.paths[ms]["degen"] / lib.paths[ss]["degen"], g),), list(xt)))
    return out


def simplifications(lib, fit):
    terms = list(fit.terms)
    xs = [list(fit.xt(i)) for i in range(len(terms))]

    def pack(items):
        return tuple(t for t, _ in items), [v for _, x in items for v in x] + [fit.x[-1]]

    out = []
    for i, t in enumerate(terms):
        out.append(pack([(terms[j], xs[j]) for j in range(len(terms)) if j != i]))
        for r, xr in reduced(lib, t, xs[i]):
            out.append(pack([(r, xr) if j == i else (terms[j], xs[j]) for j in range(len(terms))]))
            for k in range(len(terms)):
                if k != i:
                    out.append(pack([(r, xr) if j == i else (terms[j], xs[j]) for j in range(len(terms)) if j != k]))
        for r, xr in tied_alternatives(lib, t, xs[i]):
            for k in range(len(terms)):
                if k != i:
                    out.append(pack([(r, xr) if j == i else (terms[j], xs[j]) for j in range(len(terms)) if j != k]))
    return out


def swaps(lib, fit, i, opt):
    t = fit.terms[i]
    if lib.n_groups(t) != 1:
        return []
    r_fit = fit.xt(i)[1]
    dr = opt.dr_max["ss"]
    ones = [m for m in lib.moves(opt.use_ms) if lib.n_groups(m) == 1 and m != t
            and abs(lib.paths[m[0][0]]["reff"] - r_fit) <= dr]
    bundles = [b for b in lib.bundles if lib.kind[b[0][0]] == lib.kind[t[0][0]] == lib.kind[lib.owner(b, 1)]
               and abs(lib.paths[b[0][0]]["reff"] - r_fit) <= dr]
    return ones + bundles


def refine(ctx, fit, s02, e0_bounds, n_idp):
    """Bidirectional clean-up until stable: simplest adequate simplification,
    then standard swaps that fit better without extra parameters or whose
    extra tied path is significant."""
    lib, opt = ctx.lib, ctx.opt
    changed = True
    while changed:
        changed = False
        best = None
        for terms, x0 in simplifications(lib, fit):
            if not terms:
                continue
            trial = fit_terms(ctx, terms, x0, s02, e0_bounds)
            if adequate(trial, fit, n_idp, lib, opt) and (
                    best is None or (n_params(trial, lib), trial.chi2) < (n_params(best, lib), best.chi2)):
                best = trial
        if best is not None:
            fit, changed = best, True
            continue
        for i, t in enumerate(fit.terms):
            for new in swaps(lib, fit, i, opt):
                terms = fit.terms[:i] + (new,) + fit.terms[i + 1:]
                if any(not allowed(lib, terms[:j] + terms[j + 1:], terms[j], opt) for j in range(len(terms))):
                    continue
                trial = None
                r_fit = fit.xt(i)[1]
                for dr in (-0.03, 0.0, 0.03):
                    xt_new = [min(lib.paths[new[0][0]]["degen"], 4.0), r_fit + dr, 0.006]
                    if lib.n_groups(new) > 1:
                        xt_new += [lib.paths[lib.owner(new, 1)]["reff"] + dr, 0.006]
                    x0 = list(fit.x[:fit.slices[i].start]) + xt_new + list(fit.x[fit.slices[i].stop:])
                    f = fit_terms(ctx, terms, x0, s02, e0_bounds)
                    if trial is None or f.chi2 < trial.chi2:
                        trial = f
                if not physical(trial, lib, opt) or not determined(trial, lib, opt) or not supported(trial, n_idp, lib, opt):
                    continue
                simpler_better = trial.chi2 < fit.chi2 and n_params(trial, lib) <= n_params(fit, lib)
                richer_needed = n_params(trial, lib) > n_params(fit, lib) and not adequate(fit, trial, n_idp, lib, opt)
                if simpler_better or richer_needed:
                    fit, changed = trial, True
                    break
            if changed:
                break
    return fit


def pooled_refit(ctx, s02, cands):
    lib, opt = ctx.lib, ctx.opt
    best = {}
    for terms, x in cands:
        key = tuple(sorted(terms, key=str))
        for e0 in sorted({float(x[-1])} | set(opt.e0_starts)):
            fit = fit_terms(ctx, terms, list(x[:-1]) + [e0], s02, opt.e0_bounds)
            if physical(fit, lib, opt) and determined(fit, lib, opt) and (key not in best or fit.chi2 < best[key].chi2):
                best[key] = fit
    return list(best.values())


def final_fit(ctx, terms, x0, s02, e0_bounds):
    fit, res, tr = fit_terms(ctx, terms, x0, s02, e0_bounds, max_nfev=3000, want_res=True)
    return fit, scaled_errors(res, tr)


def weak(lib, term, xt, opt):
    if xt[0] > opt.n_pinned * lib.bounds(term, opt)[1][0]:
        return True
    for g in range(lib.n_groups(term)):
        s2 = xt[2 + 2 * g]
        n_shell = next(r for m, r, gg in term if gg == g and lib.kind[m] != "ms") * xt[0]
        if (g == 0 and s2 < 1.2 * opt.s2_bounds[0] and lib.s2_floor(term, g, opt) <= opt.s2_bounds[0]) \
                or s2 > 0.95 * opt.s2_bounds[1]:
            return True
        if g == 0 and n_shell < opt.min_n:
            return True
        if abs(xt[1 + 2 * g] - lib.paths[lib.owner(term, g)]["reff"]) > 0.97 * opt.dr_max["ss"]:
            return True
    return False


def clamp(bounds, opt):
    lo, hi = max(bounds[0], opt.e0_bounds[0]), min(bounds[1], opt.e0_bounds[1])
    return (lo, hi) if hi > lo else tuple(opt.e0_bounds)


def rescue(ctx, fit, i, s02, e0_bounds, exclude):
    lib, opt = ctx.lib, ctx.opt
    t = fit.terms[i]
    r_fit, group = fit.xt(i)[1], lib.info[t[0][0]]["group"]
    best = None
    for new in lib.moves(opt.use_ms):
        if new == t or new in exclude or lib.n_groups(new) != lib.n_groups(t) or lib.info[new[0][0]]["group"] != group \
                or abs(lib.paths[new[0][0]]["reff"] - r_fit) > opt.dr_max["ss"]:
            continue
        terms = fit.terms[:i] + (new,) + fit.terms[i + 1:]
        if any(not allowed(lib, terms[:j] + terms[j + 1:], terms[j], opt) for j in range(len(terms))):
            continue
        xt_new = [fit.xt(i)[0], r_fit, max(fit.xt(i)[2], 0.003)]
        if lib.n_groups(new) > 1:
            xt_new += [lib.paths[lib.owner(new, 1)]["reff"] + r_fit - lib.paths[new[0][0]]["reff"], max(fit.xt(i)[4], 0.003)]
        x0 = list(fit.x[:fit.slices[i].start]) + xt_new + list(fit.x[fit.slices[i].stop:])
        trial = fit_terms(ctx, terms, x0, s02, e0_bounds)
        if physical(trial, lib, opt) and determined(trial, lib, opt) and not weak(lib, new, trial.xt(i), opt) \
                and (best is None or trial.chi2 < best.chi2):
            best = trial
    return best


def finish(ctx, s02, terms, x, e0_bounds):
    """Final clean-up at the sample's edge shift; returns (fit, errors)."""
    lib, opt = ctx.lib, ctx.opt
    full_tr, _ = ctx.window(opt.rmax)
    fit = fit_terms(ctx, terms, x, s02, e0_bounds)
    fit = refine(ctx, fit, s02, e0_bounds, full_tr.n_idp)
    replaced_terms, needed = set(), set()
    while True:
        fit, err = final_fit(ctx, fit.terms, fit.x, s02, e0_bounds)
        drop = [i for i, t in enumerate(fit.terms) if t not in needed and weak(lib, t, fit.xt(i), opt)]
        if not drop or len(fit.terms) == 1:
            return fit, err
        replaced_terms.add(fit.terms[drop[0]])
        replaced = rescue(ctx, fit, drop[0], s02, e0_bounds, replaced_terms)
        if replaced is not None:
            fit = replaced
            continue
        keep = [i for i in range(len(fit.terms)) if i != drop[0]]
        smaller = fit_terms(ctx, tuple(fit.terms[i] for i in keep),
                            [v for i in keep for v in fit.xt(i)] + [fit.x[-1]], s02, e0_bounds)
        if adequate(smaller, fit, full_tr.n_idp, lib, opt):
            fit = smaller
        else:
            needed.add(fit.terms[drop[0]])


# --------------------------------------------------------------------- foil
def calibrate_s02(lib, k, chi, sigma, coordination, opt: Options, kmin=3.0, kmax=None,
                  rmin=1.5, rmax=3.0, s02_bounds=(0.3, 1.5)):
    """S0^2 from a reference of known coordination numbers.

    ``coordination``: ``{path_id: N}`` (the reference's crystallographic
    neighbour counts). Every path gets its own distance and sigma^2, one edge
    shift is shared. Returns a dict with s02, its scaled 1-sigma error, the
    per-path R / sigma^2 and their errors, chi^2_nu and n_idp.
    """
    k = np.asarray(k, float)
    chi = np.asarray(chi, float)
    sigma = np.asarray(sigma, float)
    kmax = float(k.max()) - 0.5 if kmax is None else float(kmax)
    tr = Transform(k, sigma, kmax, opt, kmin=kmin, rmin=rmin, rmax=rmax)
    dft = tr.data(chi)
    ids = list(coordination)
    dr = opt.dr_max["ss"]

    def resid(x):
        model = np.zeros_like(k)
        for i, pid in enumerate(ids):
            model += path_chi(k, lib.paths[pid], float(coordination[pid]), x[2 + 2 * i], x[3 + 2 * i], x[1], x[0])
        return tr.resid(model, dft)

    x0 = [0.9, 0.0] + sum(([lib.paths[p]["reff"], 0.005] for p in ids), [])
    lo = [s02_bounds[0], opt.e0_bounds[0]] + sum(([lib.paths[p]["reff"] - dr, opt.s2_bounds[0]] for p in ids), [])
    hi = [s02_bounds[1], opt.e0_bounds[1]] + sum(([lib.paths[p]["reff"] + dr, opt.s2_bounds[1]] for p in ids), [])
    res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac")
    err = scaled_errors(res, tr)
    lib.foil_s2 = {pid: float(res.x[3 + 2 * i]) for i, pid in enumerate(ids)}
    chi2 = float(np.sum(res.fun ** 2) * tr.n_idp / tr.n_pts)
    return {"s02": float(res.x[0]), "s02_err": float(err[0]), "delta_e0": float(res.x[1]),
            "delta_e0_err": float(err[1]),
            "paths": {pid: {"N": float(coordination[pid]), "R": float(res.x[2 + 2 * i]), "R_err": float(err[2 + 2 * i]),
                            "sigma2": float(res.x[3 + 2 * i]), "sigma2_err": float(err[3 + 2 * i])}
                      for i, pid in enumerate(ids)},
            "chi2": chi2, "redchi": chi2 / max(tr.n_idp - len(res.x), 1.0), "n_idp": float(tr.n_idp),
            "window": {"kmin": kmin, "kmax": kmax, "rmin": rmin, "rmax": rmax, "kweights": list(opt.kweights)}}


def default_foil_coordination(lib, compound, n_shells=2):
    """The first ``n_shells`` single-scattering paths of the foil compound with
    their crystallographic degeneracy as N."""
    ss = sorted((p for p in lib.single if lib.info[p]["compound"] == compound), key=lambda p: lib.info[p]["reff"])
    if not ss:
        raise ValueError(f"no single-scattering path of compound {compound!r} in the catalog")
    return {p: lib.paths[p]["degen"] for p in ss[:n_shells]}


# --------------------------------------------------------------------- site fraction / rows
def site_fraction(lib, fit, compound):
    """Fraction of the absorber in a site: the site's first-shell N over its
    crystallographic degeneracy, when the model holds the site (its outer shell
    present); else 0."""
    rows = [(m, n) for i, t in enumerate(fit.terms) for m, n, _, _ in lib.members(t, fit.xt(i))]
    core = {m: n for m, n in rows if lib.info[m]["compound"] == compound and lib.kind[m] != "ms"}
    ss = sorted((p for p in lib.single if lib.info[p]["compound"] == compound), key=lambda p: lib.info[p]["reff"])
    if not ss or not any(m in lib.site_only for m in core):
        return 0.0
    first = ss[0]
    if first in core:
        f = core[first] / lib.paths[first]["degen"]
    else:
        outer = next(m for m in core if m in lib.site_only)
        f = core[outer] / lib.paths[outer]["degen"]
    return float(min(max(f, 0.0), 1.0))


def phase_rows(lib, fit, err):
    rows = []
    for i, t in enumerate(fit.terms):
        et = err[fit.slices[i]]
        for (m, ratio, g), (_, n, r, s2) in zip(t, lib.members(t, fit.xt(i))):
            rows.append((m, n, ratio * et[0], r, et[1 + 2 * g], s2, et[2 + 2 * g]))
    return rows


# --------------------------------------------------------------------- series
def site_term(lib, compound):
    return next(b for b in lib.bundles if lib.info[b[0][0]]["compound"] == compound)


def mixture_fit(ctx, site, phase, s02):
    """One sample as a site bundle + a reference phase; x = [site N, R/s2 per
    site shell..., size, e0]. Shares add to one: the phase holds 1 - N/deg."""
    lib, opt, k = ctx.lib, ctx.opt, ctx.k
    tr, dft = ctx.window(opt.rmax)
    deg = lib.paths[site[0][0]]["degen"]
    npar = lib.n_par(site)
    lo, hi = lib.bounds(site, opt)
    lo = np.array([deg * opt.min_share] + lo[1:] + [opt.size_bounds[0], opt.e0_bounds[0]])
    hi = np.array([deg * (1.0 - opt.min_share)] + hi[1:] + [opt.size_bounds[1], opt.e0_bounds[1]])
    phase_chi_cache = {}

    def resid(x):
        chi = lib.term_chi(k, site, x[:npar], x[-1], s02)
        share = 1.0 - x[0] / deg
        for pid, n, _, r, _, s2, _ in phase:
            f = share * (x[npar] if lib.kind[pid] == "metal" else 1.0)
            chi = chi + path_chi(k, lib.paths[pid], f * n, r, s2, x[-1], s02)
        return tr.resid(chi, dft)

    best = None
    for e0 in opt.e0_starts:
        for frac in (0.25, 0.5, 0.75):
            x0 = [frac * deg]
            for g in range(lib.n_groups(site)):
                x0 += [lib.paths[lib.owner(site, g)]["reff"] + (0.04 if g == 0 else 0.02), 0.006]
            x0 += [0.9, e0]
            res = least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), x_scale="jac", max_nfev=400)
            chi2 = float(np.sum(res.fun ** 2) * tr.n_idp / tr.n_pts)
            if best is None or chi2 < best["chi2"]:
                best = {"chi2": chi2, "x": res.x, "res": res}
    best["cov"] = scaled_covariance(best.pop("res"), tr)
    best["err"] = np.sqrt(np.clip(np.diag(best["cov"]), 0.0, None))
    best["site"] = site
    best["deg"] = deg
    del phase_chi_cache
    return best


def mixture_rows(lib, mix, phase, opt):
    """Rows of a two-phase reading: the site's, and the reference phase's at
    its share (metal neighbours times the size factor), with the share's and
    the factor's errors propagated from the fit covariance."""
    site, x, err, cov, deg = mix["site"], mix["x"], mix["err"], mix["cov"], mix["deg"]
    npar = lib.n_par(site)
    rows = [(m, n, ratio * err[0], r, err[1 + 2 * g], s2, err[2 + 2 * g])
            for (m, ratio, g), (_, n, r, s2) in zip(site, lib.members(site, x[:npar]))]
    share = 1.0 - x[0] / deg
    for pid, n, dn, r, dr, s2, ds2 in phase:
        if lib.kind[pid] == "metal":
            factor, grad = share * x[npar], np.array([-x[npar] / deg, share])
            dfactor = float(np.sqrt(max(grad @ cov[np.ix_([0, npar], [0, npar])] @ grad, 0.0)))
        else:
            factor, dfactor = share, err[0] / deg
        lo = opt.n_bounds["ms" if lib.kind[pid] == "ms" else "ss"][0]
        rows.append((pid, max(lo, factor * n), float(np.sqrt((n * dfactor) ** 2 + (factor * dn) ** 2)), r, dr, s2, ds2))
    return rows


def stands(mix, chi2, n_own, n_idp, misfit, opt):
    n_mix = len(mix["x"]) + sum(1 for m, _, _ in mix["site"] if mix.get("lib_kind", {}).get(m) == "ms")
    scale = max(1.0, misfit, min(mix["chi2"] / max(n_idp - n_mix, 1.0), chi2 / max(n_idp - n_own, 1.0)))
    return (mix["chi2"] - chi2) / scale < chi2_dist.ppf(1.0 - opt.p_accept, max(n_own - n_mix, 1))


# --------------------------------------------------------------------- campaign
def _select_job(args):
    ctx, s02, bounds = args
    terms, x, cands, scale = select(ctx, s02, bounds)
    return ctx.sid, terms, x, cands, scale, bounds


def _pooled_job(args):
    ctx, s02, cands = args
    return ctx.sid, pooled_refit(ctx, s02, cands)


def _finish_job(args):
    ctx, s02, terms, x, bounds = args
    fit, err = finish(ctx, s02, terms, x, bounds)
    return ctx.sid, fit, err


def _mixture_job(args):
    ctx, site, phase, s02, ref = args
    return ctx.sid, ref, mixture_fit(ctx, site, phase, s02)


def available_cpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover
        return max(1, os.cpu_count() or 1)


def _map(fn, jobs, n_jobs, log=None, label=""):
    """Run jobs (serially or in forked workers), results in input order; logs
    one line per finished job so a long campaign never goes silent."""
    import time
    t0 = time.time()

    def note(i, res):
        if log is not None:
            log(f"[exafs] {label} {i}/{len(jobs)} done ({res[0] if isinstance(res, tuple) else ''}, "
                f"{time.time() - t0:.0f} s)")

    if n_jobs <= 1 or len(jobs) <= 1:
        out = []
        for i, j in enumerate(jobs, 1):
            out.append(fn(j))
            note(i, out[-1])
        return out
    import multiprocessing as mp
    from concurrent.futures import as_completed
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover
        ctx = None
    out = [None] * len(jobs)
    with ProcessPoolExecutor(min(n_jobs, len(jobs)), mp_context=ctx) as ex:
        futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
        for n, fut in enumerate(as_completed(futs), 1):
            out[futs[fut]] = fut.result()
            note(n, out[futs[fut]])
    return out


def analyse(lib, ctxs, s02, n_jobs=1, log=print):
    """Each sample on its own: sample -> (final fit, parameter errors)."""
    opt = next(iter(ctxs.values())).opt
    sids = sorted(ctxs)
    jobs = [(ctxs[sid], s02, clamp((c - opt.e0_window, c + opt.e0_window), opt)) for sid in sids for c in opt.e0_starts]
    log(f"[exafs] shell selection: {len(sids)} spectra x {len(opt.e0_starts)} edge-shift windows "
        f"on {n_jobs} worker(s)")
    runs = {}
    for sid, terms, x, cands, scale, bounds in _map(_select_job, jobs, n_jobs, log, "selection"):
        if terms:
            fit = fit_terms(ctxs[sid], terms, x, s02, bounds)
            runs.setdefault(sid, []).append((fit, cands, scale, bounds))
    pools = {sid: [c for _, cands, _, _ in runs.get(sid, []) for c in cands] for sid in sids}
    selected = {}
    for sid, fits in _map(_pooled_job, [(ctxs[sid], s02, pools[sid]) for sid in sids], n_jobs, log, "pooled refit"):
        ctx = ctxs[sid]
        n_idp = ctx.window(opt.rmax)[0].n_idp
        if not fits:
            if not runs.get(sid):
                raise RuntimeError(f"{sid}: no model could be built from the catalog")
            fits = [min(runs[sid], key=lambda r: bic(r[0], n_idp, lib, min(q[2] for q in runs[sid])))[0]]
        scale = max(1.0, min(f.chi2 / max(n_idp - n_params(f, lib), 1.0) for f in fits))
        top = min(bic(f, n_idp, lib, scale) for f in fits)
        near = [f for f in fits if bic(f, n_idp, lib, scale) <= top + opt.bic_tie]
        fit = min(near, key=lambda f: (n_params(f, lib), bic(f, n_idp, lib, scale)))
        selected[sid] = (fit.terms, fit.x)
    out = {}
    jobs = [(ctxs[sid], s02, selected[sid][0], selected[sid][1],
             clamp((selected[sid][1][-1] - opt.e0_final_window, selected[sid][1][-1] + opt.e0_final_window), opt))
            for sid in sids]
    for sid, fit, err in _map(_finish_job, jobs, n_jobs, log, "clean-up"):
        out[sid] = (fit, err)
    return out


def read_as_series(lib, ctxs, s02, own, phases, n_jobs=1, log=print):
    """sample -> best two-phase reading that stands against its own model.
    ``own``: {sid: (chi2, n_params)}; ``phases``: {sid: rows} of site-free samples."""
    opt = next(iter(ctxs.values())).opt
    n_idp = {sid: ctxs[sid].window(opt.rmax)[0].n_idp for sid in own}
    misfit = float(np.median([chi2 / max(n_idp[sid] - n, 1.0) for sid, (chi2, n) in own.items()]))
    sites = [site_term(lib, c) for c in lib.sites if any(lib.is_bundle_of(b, c) for b in lib.bundles)]
    if not sites or not phases:
        return {}
    kind_map = dict(lib.kind)
    cache = {}

    def readings(sids, references):
        jobs = [(ctxs[sid], site, phase, s02, (other, si)) for sid in sids for other, phase in references.items()
                if other != sid for si, site in enumerate(sites) if (sid, other, si) not in cache]
        for sid, ref, mix in _map(_mixture_job, jobs, n_jobs, log, "series reading"):
            mix["lib_kind"] = kind_map
            mix["reference"] = ref[0]
            cache[(sid, ref[0], ref[1])] = mix
        out = {}
        for sid in sids:
            fits = [dict(cache[(sid, other, si)], phase=references[other]) for other in references if other != sid
                    for si in range(len(sites))]
            out[sid] = [m for m in fits if stands(m, *own[sid], n_idp[sid], misfit, opt)]
        return out

    references = dict(phases)
    pure = readings(sorted(phases), references)
    for sid in sorted(phases):
        if pure[sid]:
            log(f"[exafs] {sid} reads as a mixture of the others: not a reference")
            del references[sid]
    if not references:
        return {}
    mixed = {}
    for sid, fits in readings(sorted(own), references).items():
        if fits:
            mixed[sid] = min(fits, key=lambda m: m["chi2"])
    return mixed


def load_spectra(spectra):
    """{sid: {'k','chi','sigma','observed'}} from a long CSV / DataFrame with
    columns sample_id, k, chi, sigma (optional), observed (optional), or from
    such a dict already."""
    if isinstance(spectra, dict):
        out = {}
        for sid, s in spectra.items():
            out[str(sid)] = _spectrum_dict(s)
        return out
    rows = _catalog_rows(spectra)
    out = {}
    for r in rows:
        low = {k.lower().strip(): v for k, v in r.items()}
        sid = str(next(low[c] for c in ("sample_id", "sample", "id") if c in low))
        s = out.setdefault(sid, {"k": [], "chi": [], "sigma": [], "observed": []})
        kv = next(low[c] for c in ("k_inv_angstrom", "k", "wavenumber") if c in low)
        s["k"].append(float(kv))
        obs = low.get("observed", 1)
        obs = str(obs).strip().lower() not in ("0", "false", "no", "nan", "")
        cv = low.get("chi")
        cv = float("nan") if cv in (None, "") else float(cv)
        obs = obs and math.isfinite(cv)
        s["chi"].append(cv if obs else 0.0)
        sg = low.get("sigma", low.get("chi_err", low.get("uncertainty")))
        s["sigma"].append(float(sg) if sg not in (None, "") else float("nan"))
        s["observed"].append(obs)
    return {sid: _spectrum_dict(s) for sid, s in out.items()}


def _spectrum_dict(s):
    if isinstance(s, (tuple, list)) and len(s) in (2, 3):
        s = dict(zip(("k", "chi", "sigma"), s))
    k = np.asarray(s["k"], float)
    chi = np.asarray(s["chi"], float)
    obs = np.asarray(s.get("observed", np.isfinite(chi)), bool) & np.isfinite(chi)
    sigma = s.get("sigma")
    sigma = None if sigma is None else np.asarray(sigma, float)
    if sigma is None or not np.all(np.isfinite(sigma[obs])):
        sigma = estimate_sigma(k, np.where(obs, chi, 0.0))
    if not np.allclose(np.diff(k), k[1] - k[0], rtol=1e-3):
        raise ValueError("chi(k) must be on a uniform k grid (resample first, e.g. 0.05 A^-1)")
    return {"k": k, "chi": np.where(obs, chi, 0.0), "sigma": sigma, "observed": obs}


# --------------------------------------------------------------------- metrics
def agreement(k, chi_data, chi_model, observed=None, sigma=None, kmin=3.0, kmax=None, window_dk=1.0,
              rmin=1.0, rmax=4.0, rstep=0.02, kweights=(1, 2, 3)):
    """Normalised k-space residuals per k-weight (observed points, kmin..kmax)
    and the complex R-space residual (|F_model - F_data| / |F_data|, gaps
    bridged linearly in both)."""
    k = np.asarray(k, float)
    d = np.asarray(chi_data, float)
    m = np.asarray(chi_model, float)
    obs = np.isfinite(d) if observed is None else (np.asarray(observed, bool) & np.isfinite(d))
    sig = np.ones_like(k) if sigma is None else np.asarray(sigma, float)
    kmax = float(k[obs].max()) if kmax is None else float(kmax)
    sel = obs & (k >= kmin - 1e-9) & (k <= kmax + 1e-9)
    out = {}
    for w in kweights:
        num = np.sqrt(np.mean(((k[sel] ** w) * (d[sel] - m[sel]) / sig[sel]) ** 2))
        den = np.sqrt(np.mean(((k[sel] ** w) * d[sel] / sig[sel]) ** 2))
        out[f"k{w}"] = float(num / den) if den > 0 else float("nan")
    fill = fill_matrix(k, obs)
    dd, mm = fill @ np.where(obs, d, 0.0), fill @ np.where(obs, m, 0.0)
    r = np.arange(rmin, rmax + 1e-9, rstep)
    win = hanning(k, kmin, kmax, window_dk)
    dk = k[1] - k[0]
    op = np.exp(2j * np.outer(r, k)) * (win * k ** 2) * dk
    fd, fm = op @ dd, op @ mm
    out["rspace"] = float(np.linalg.norm(fm - fd) / np.linalg.norm(fd))
    return out


def rows_chi(lib, rows, k, s02):
    """chi(k) of a table of rows (path_id, N, R, sigma2, e0)."""
    k = np.asarray(k, float)
    chi = np.zeros_like(k)
    for pid, n, r, s2, e0 in rows:
        chi += path_chi(k, lib.paths[pid], n, r, s2, e0, s02)
    return chi


# --------------------------------------------------------------------- shells
def merge_shells(rows, lib, link=0.25):
    """Rows of one sample grouped into graded shells: rows of one equivalence
    group whose sorted distances differ by less than ``link``; total N,
    N-weighted R, effective sigma^2 (with the spread), errors to first order."""
    out = []
    groups = sorted({lib.info[r["path_id"]]["group"] for r in rows})
    for grp in groups:
        mem = sorted((r for r in rows if lib.info[r["path_id"]]["group"] == grp), key=lambda r: r["R"])
        cur = [mem[0]]
        for r in mem[1:]:
            if r["R"] - cur[-1]["R"] < link:
                cur.append(r)
            else:
                out.append(_merge(grp, cur))
                cur = [r]
        out.append(_merge(grp, cur))
    return sorted(out, key=lambda s: s["R"])


def _merge(group, rows):
    n = sum(r["N"] for r in rows)
    rr = sum(r["N"] * r["R"] for r in rows) / n
    s2 = sum(r["N"] * (r["sigma2"] + (r["R"] - rr) ** 2) for r in rows) / n
    w = [r["N"] / n for r in rows]
    return {"group": group, "N": n, "R": rr, "sigma2": s2,
            "N_err": math.sqrt(sum(r["N_err"] ** 2 for r in rows)),
            "R_err": math.sqrt(sum((wi * r["R_err"]) ** 2 + ((r["R"] - rr) / n * r["N_err"]) ** 2
                                   for wi, r in zip(w, rows))),
            "sigma2_err": math.sqrt(sum((wi * r["sigma2_err"]) ** 2 + (2 * wi * (r["R"] - rr) * r["R_err"]) ** 2
                                        + ((r["sigma2"] + (r["R"] - rr) ** 2 - s2) / n * r["N_err"]) ** 2
                                        for wi, r in zip(w, rows))),
            "paths": [r["path_id"] for r in rows]}


# --------------------------------------------------------------------- synthetic paths (tests / tutorial)
def synthetic_path(reff, z_scatterer, degen=1.0, nleg=2, kmax=20.0, kstep=0.05, phase_shift=0.0):
    """A smooth, FEFF-like path for tests and teaching: backscattering
    amplitude peaked at a k that grows with Z, a phase that decreases with k,
    a mean free path ~ k / (Gamma + small k^2 term). Not a substitute for FEFF."""
    k = np.arange(0.0, kmax + 1e-9, kstep)
    kk = np.maximum(k, 0.05)
    z = float(z_scatterer)
    kpk = 2.0 + 0.12 * z
    amp = (0.25 + 0.012 * z) * np.exp(-((kk - kpk) / (3.0 + 0.05 * z)) ** 2) + 0.02 * z / (1 + kk)
    amp = amp * (0.9 if nleg > 2 else 1.0)
    pha = -0.3 * kk + 1.5 - 0.03 * z + phase_shift + 0.4 * np.sin(kk / (1.5 + 0.05 * z))
    lam = 5.0 + 2.0 * kk / (0.3 + 0.004 * kk ** 2) / 4.0
    rep = np.sqrt(kk ** 2 + 0.5)
    return {"reff": float(reff), "degen": float(degen), "nleg": int(nleg), "k": k, "pha": pha,
            "amp": amp, "lam": lam, "rep": rep, "atoms": []}


def write_feffdat(path, p, title="synthetic", absorber="X", scatterer="Y"):
    """Write a path dict in the FEFF6 feff.dat column layout (readable by
    read_feffdat and by Larch's FeffDatFile)."""
    lines = [f" {title}", " Abs   Z=26 Rmt= 1.000 Rnm= 1.300 K shell",
             " Gam_ch=1.325E+00 H-L exch", " Mu=-2.440E+00 kf=2.816E+00 Vint=-3.266E+01 Rs_int= 1.288",
             " Path    1      icalc       2", " " + "-" * 79,
             f"   {p['nleg']}   {p['degen']:.3f}   {p['reff']:.4f}    {p['reff'] / 0.529177:.4f}   -2.43996 nleg, deg, reff, rnrmav(bohr), edge",
             "        x         y         z   pot at#",
             f"     0.0000    0.0000    0.0000  0  26 {absorber}     absorbing atom",
             f"     {p['reff']:.4f}    0.0000    0.0000  1   7 {scatterer}",
             "    k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda      real[p]@#"]
    for i in range(len(p["k"])):
        lines.append(f"  {p['k'][i]:7.4f} {0.0: .4e} {p['amp'][i]: .4e} {p['pha'][i]: .4e} {1.0: .4e} "
                     f"{p['lam'][i]: .4e} {p['rep'][i]: .4e}")
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- whole campaign
def campaign(lib, spectra, opt: Options, s02, n_jobs=1, series=True, log=print):
    """Shell selection for every spectrum, then the series reading.

    Returns ``(rows, info)``: rows ``(sid, path_id, N, dN, R, dR, s2, ds2, e0,
    de0)`` and per-sample info (model, chi^2, reduced chi^2, n_idp, number of
    parameters, site fractions, whether the series reading replaced it)."""
    ctxs = {sid: Context(sid, lib, s["k"], s["chi"], s["sigma"], s["observed"], opt) for sid, s in spectra.items()}
    finals = analyse(lib, ctxs, s02, n_jobs=n_jobs, log=log)
    own = {sid: (fit.chi2, n_params(fit, lib)) for sid, (fit, _) in finals.items()}
    fractions = {sid: {c: site_fraction(lib, fit, c) for c in lib.sites} for sid, (fit, _) in finals.items()}
    mixed = {}
    if series and lib.sites:
        phases = {sid: phase_rows(lib, *finals[sid]) for sid in finals
                  if all(v == 0.0 for v in fractions[sid].values())}
        mixed = read_as_series(lib, ctxs, s02, own, phases, n_jobs=n_jobs, log=log)
    rows, info = [], {}
    for sid in sorted(finals):
        fit, err = finals[sid]
        n_idp = ctxs[sid].window(opt.rmax)[0].n_idp
        rec = {"n_idp": float(n_idp), "kmax": ctxs[sid].kmax, "resolution": ctxs[sid].res,
               "own_model": [[m for m, _, _ in t] for t in fit.terms], "own_chi2": fit.chi2,
               "own_n_params": n_params(fit, lib),
               "own_redchi": fit.chi2 / max(n_idp - n_params(fit, lib), 1.0)}
        if sid in mixed:
            mix = mixed[sid]
            e0, e0_err = float(mix["x"][-1]), float(mix["err"][-1])
            prow = mixture_rows(lib, mix, mix["phase"], opt)
            comp = lib.info[mix["site"][0][0]]["compound"]
            frac = {c: 0.0 for c in lib.sites}
            frac[comp] = float(mix["x"][0] / mix["deg"])
            rec.update(reading="series", reference=mix["reference"], chi2=mix["chi2"],
                       n_params=len(mix["x"]) + 1, size_factor=float(mix["x"][lib.n_par(mix["site"])]))
            log(f"[exafs] {sid}: read as site {comp!r} + the phase of {mix['reference']} "
                f"(site fraction {frac[comp]:.3f}, chi2 {mix['chi2']:.1f} vs own {fit.chi2:.1f})")
        else:
            e0, e0_err = float(fit.x[-1]), float(err[-1])
            prow = phase_rows(lib, fit, err)
            frac = fractions[sid]
            rec.update(reading="own", chi2=fit.chi2, n_params=n_params(fit, lib))
        rec["redchi"] = rec["chi2"] / max(n_idp - rec["n_params"], 1.0)
        rec["site_fraction"] = frac
        info[sid] = rec
        rows += [(sid, m, n, dn, r, dr, s2, ds2, e0, e0_err) for m, n, dn, r, dr, s2, ds2 in prow]
    return rows, info
