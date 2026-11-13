"""EXAFS shell analysis (mv.exp.exafs_*) on synthetic FEFF-path data.

The paths are smooth FEFF-like functions written in the feff.dat layout
(``_exafs.synthetic_path`` / ``write_feffdat``), so every answer is known:
the foil's S0^2, each sample's shells, and the site fraction of a mixture.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

import matverse as mv
from matverse import _exafs as X

K = np.round(np.arange(2.5, 12.0 + 1e-9, 0.05), 4)
S02 = 0.8
NOISE = 2e-4

# compound -> [(path_id, group, scatterers, nleg, degeneracy, reff, Z)]
CATALOG = {
    "site": [("N_site", "gN", "N", 2, 4.0, 1.95, 7),
             ("C_site", "gC", "C", 2, 8.0, 2.98, 6),
             ("MS_site", "gMS", "C-N", 3, 16.0, 3.17, 7)],
    "metal": [("M1", "gM", "Fe", 2, 8.0, 2.48, 26),
              ("M2", "gM", "Fe", 2, 6.0, 2.87, 26)],
    "oxide": [("O_ox", "gO", "O", 2, 6.0, 2.06, 8),
              ("M_ox", "gMox", "Fe", 2, 12.0, 3.05, 26)],
    "sulfide": [("S_sul", "gS", "S", 2, 6.0, 2.30, 16)],
}


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    d = tmp_path_factory.mktemp("feff")
    rows = []
    for comp, paths in CATALOG.items():
        for pid, grp, sc, nleg, deg, reff, z in paths:
            p = X.synthetic_path(reff, z, degen=deg, nleg=nleg)
            X.write_feffdat(d / f"{pid}.dat", p, title=comp, scatterer=sc.split("-")[-1])
            rows.append({"path_id": pid, "file": f"{pid}.dat", "group": grp, "scatterers": sc,
                         "nleg": nleg, "degeneracy": deg, "reff_angstrom": reff, "source_compound": comp})
    pd.DataFrame(rows).to_csv(d / "catalog.csv", index=False)
    return mv.exp.exafs_paths(d / "catalog.csv", sites=["site"], foil_compound="metal")


def _chi(lib, rows, e0=0.0, s02=S02, k=K):
    return sum(X.path_chi(k, lib.paths[p], n, r, s2, e0, s02) for p, n, r, s2 in rows)


def _noisy(chi, seed, noise=NOISE):
    rng = np.random.default_rng(seed)
    return chi + rng.normal(0, noise, chi.size), np.full(chi.size, noise)


class TestPaths:
    def test_catalog_kinds_ties_and_bundles(self, library):
        t = library.table.set_index("path_id")
        assert t.loc["N_site", "kind"] == "light"
        assert t.loc["M1", "kind"] == "metal"
        assert t.loc["S_sul", "kind"] == "nonmetal"
        assert t.loc["MS_site", "kind"] == "ms" and t.loc["MS_site", "tied_to"] == "C_site"
        assert bool(t.loc["C_site", "site_only"])
        firsts = sorted(b[0][0] for b in library.bundles)
        assert firsts == ["M1", "N_site"]
        site = next(b for b in library.bundles if b[0][0] == "N_site")
        assert dict((m, r) for m, r, _ in site) == {"N_site": 1.0, "C_site": 2.0, "MS_site": 4.0}

    def test_feffdat_roundtrip(self, library, tmp_path):
        p = library.paths["O_ox"]
        X.write_feffdat(tmp_path / "x.dat", p)
        q = X.read_feffdat(tmp_path / "x.dat")
        assert q["reff"] == pytest.approx(p["reff"], abs=1e-4)
        assert np.allclose(q["amp"], p["amp"], rtol=1e-3)
        assert np.allclose(q["pha"], p["pha"], atol=1e-3)

    def test_forward_model_matches_larch(self, library, tmp_path):
        larch_feffdat = pytest.importorskip("larch.xafs.feffdat")
        X.write_feffdat(tmp_path / "feff0001.dat", library.paths["O_ox"], scatterer="O")
        pg = larch_feffdat.feffpath(str(tmp_path / "feff0001.dat"), degen=4.5, s02=0.83,
                                    deltar=0.03, sigma2=0.006, e0=2.5)
        pg._calc_chi(k=K, interp="linear")
        p = X.read_feffdat(tmp_path / "feff0001.dat")
        mine = X.path_chi(K, p, 4.5, p["reff"] + 0.03, 0.006, 2.5, 0.83)
        # Larch overwrites the first point by linear extrapolation; compare the rest
        assert np.allclose(mine[1:], pg.chi[1:], atol=2e-6)


class TestS02:
    def test_foil_recovers_s02_with_scaled_error(self, library):
        rows = [("M1", 8.0, 2.47, 0.005), ("M2", 6.0, 2.86, 0.007)]
        k = np.round(np.arange(2.5, 14.0 + 1e-9, 0.05), 4)
        chi, sig = _noisy(_chi(library, rows, e0=1.0, k=k), 0)
        cal = mv.exp.exafs_s02({"k": k, "chi": chi, "sigma": sig}, library)
        assert set(cal["paths"]) == {"M1", "M2"}
        assert cal["paths"]["M1"]["N"] == 8.0
        assert abs(cal["s02"] - S02) < max(3 * cal["s02_err"], 0.02)
        assert 0 < cal["s02_err"] < 0.1
        # the foil's sigma^2 becomes the floor of the metal bundle
        assert library.foil_s2["M1"] == pytest.approx(0.005, abs=0.001)


class TestFit:
    def test_stated_model_recovers_parameters(self, library):
        truth = [("O_ox", 5.0, 2.03, 0.006), ("M_ox", 6.0, 3.08, 0.009)]
        chi, sig = _noisy(_chi(library, truth, e0=2.0), 1)
        fit = mv.exp.exafs_fit({"k": K, "chi": chi, "sigma": sig}, library, ["O_ox", "M_ox"], S02)
        tab = fit["table"].set_index("path_id")
        for pid, n, r, s2 in truth:
            assert abs(tab.loc[pid, "distance_angstrom"] - r) < 0.01
            assert abs(tab.loc[pid, "coordination_number"] - n) < max(4 * tab.loc[pid, "coordination_number_err"], 0.5)
            assert abs(tab.loc[pid, "sigma2_angstrom2"] - s2) < 0.002
        assert abs(tab["delta_e0_ev"].iloc[0] - 2.0) < 1.0
        assert fit["agreement"]["rspace"] < 0.05

    def test_errors_scale_with_sqrt_reduced_chi2(self, library):
        """Scaled errors do not depend on the stated noise level."""
        truth = [("O_ox", 5.0, 2.03, 0.006)]
        chi, sig = _noisy(_chi(library, truth), 2)
        a = mv.exp.exafs_fit({"k": K, "chi": chi, "sigma": sig}, library, ["O_ox"], S02)
        b = mv.exp.exafs_fit({"k": K, "chi": chi, "sigma": sig / 4}, library, ["O_ox"], S02)
        ea = a["table"]["coordination_number_err"].iloc[0]
        eb = b["table"]["coordination_number_err"].iloc[0]
        assert ea == pytest.approx(eb, rel=0.05)
        assert b["redchi"] == pytest.approx(16 * a["redchi"], rel=0.05)

    def test_tied_members_carry_propagated_errors(self, library):
        truth = [("N_site", 4.0, 1.97, 0.004), ("C_site", 8.0, 3.00, 0.006), ("MS_site", 16.0, 3.19, 0.006)]
        chi, sig = _noisy(_chi(library, truth), 3)
        fit = mv.exp.exafs_fit({"k": K, "chi": chi, "sigma": sig}, library, ["site:site"], S02)
        tab = fit["table"].set_index("path_id")
        assert tab.loc["C_site", "coordination_number"] == pytest.approx(2 * tab.loc["N_site", "coordination_number"])
        assert tab.loc["C_site", "coordination_number_err"] == pytest.approx(2 * tab.loc["N_site", "coordination_number_err"])
        assert tab.loc["MS_site", "sigma2_angstrom2"] == pytest.approx(tab.loc["C_site", "sigma2_angstrom2"])
        assert abs(tab.loc["N_site", "coordination_number"] - 4.0) < 0.5


class TestAgreementAndShells:
    def test_agreement_with_gaps(self, library):
        truth = [("O_ox", 5.0, 2.03, 0.006)]
        chi = _chi(library, truth)
        obs = np.ones(K.size, bool)
        obs[60:66] = False
        rows = pd.DataFrame({"path_id": ["O_ox"], "coordination_number": [5.0], "distance_angstrom": [2.03],
                             "sigma2_angstrom2": [0.006], "delta_e0_ev": [0.0]})
        a = mv.exp.exafs_agreement({"k": K, "chi": np.where(obs, chi, np.nan), "observed": obs,
                                    "sigma": np.full(K.size, NOISE)}, rows, library, S02)
        assert a["k2"] < 1e-9 and a["rspace"] < 1e-9
        rows2 = rows.assign(coordination_number=[4.0])
        b = mv.exp.exafs_agreement({"k": K, "chi": chi, "sigma": np.full(K.size, NOISE)}, rows2, library, S02)
        assert b["k2"] == pytest.approx(0.2, rel=1e-6) and b["rspace"] == pytest.approx(0.2, rel=1e-6)

    def test_shells_merge_within_group(self, library):
        tab = pd.DataFrame([
            ["A", "M1", 4.0, 0.4, 2.48, 0.01, 0.005, 0.001, 0.0, 0.1],
            ["A", "M2", 3.0, 0.3, 2.60, 0.01, 0.006, 0.001, 0.0, 0.1],
            ["A", "O_ox", 2.0, 0.2, 2.05, 0.01, 0.004, 0.001, 0.0, 0.1],
        ], columns=mv.exp._EXAFS_COLUMNS)
        sh = mv.exp.exafs_shells(tab, library).set_index("group")
        assert sh.loc["gM", "N"] == pytest.approx(7.0)
        rbar = (4 * 2.48 + 3 * 2.60) / 7
        assert sh.loc["gM", "R"] == pytest.approx(rbar)
        assert sh.loc["gM", "sigma2"] == pytest.approx((4 * (0.005 + (2.48 - rbar) ** 2) + 3 * (0.006 + (2.60 - rbar) ** 2)) / 7)
        assert sh.loc["gO", "N"] == 2.0


class TestSeries:
    def test_selection_and_mixture_reading(self, library, tmp_path):
        """A pure site, a pure oxide, and a 40/60 mixture: the selection finds
        the site bundle and the oxide shells, and the series reading gives the
        mixture's site fraction with the shares adding up to one."""
        site = [("N_site", 4.0, 1.97, 0.004), ("C_site", 8.0, 3.00, 0.006), ("MS_site", 16.0, 3.19, 0.006)]
        oxide = [("O_ox", 6.0, 2.03, 0.006), ("M_ox", 8.0, 3.08, 0.008)]
        f = 0.4
        mix = [(p, f * n, r, s) for p, n, r, s in site] + [(p, (1 - f) * n, r, s) for p, n, r, s in oxide]
        foil_rows = [("M1", 8.0, 2.47, 0.005), ("M2", 6.0, 2.86, 0.007)]
        kf = np.round(np.arange(2.5, 14.0 + 1e-9, 0.05), 4)
        foil, fsig = _noisy(_chi(library, foil_rows, k=kf), 10)
        lines = []
        for sid, rows, e0, seed in (("A", site, 1.0, 11), ("B", oxide, 2.0, 12), ("C", mix, 1.5, 13)):
            chi, sig = _noisy(_chi(library, rows, e0=e0), seed, noise=1e-4)
            lines += [{"sample_id": sid, "k_inv_angstrom": k, "chi": c, "sigma": s, "observed": 1}
                      for k, c, s in zip(K, chi, sig)]
        pd.DataFrame(lines).to_csv(tmp_path / "spectra.csv", index=False)
        res = mv.exp.exafs_series(tmp_path / "spectra.csv", library,
                                  foil={"k": kf, "chi": foil, "sigma": fsig},
                                  n_jobs=1, verbose=False, e0_starts=(0.0, 3.0), rmax=3.6, beam=3)
        assert abs(res["s02"] - S02) < 0.05
        tab = res["table"]
        a = set(tab[tab.sample_id == "A"].path_id)
        b = set(tab[tab.sample_id == "B"].path_id)
        assert {"N_site", "C_site"} <= a
        assert {"O_ox", "M_ox"} <= b and not ({"N_site", "C_site"} & b)
        assert res["site_fraction"]["A"] == pytest.approx(1.0, abs=0.15)
        assert res["site_fraction"]["B"] == 0.0
        assert res["site_fraction"]["C"] == pytest.approx(f, abs=0.1)
        c = tab[tab.sample_id == "C"].set_index("path_id")
        share_site = c.loc["N_site", "coordination_number"] / 4.0
        share_ox = c.loc["O_ox", "coordination_number"] / res["table"].query("sample_id == 'B'").set_index(
            "path_id").loc["O_ox", "coordination_number"]
        assert share_site + share_ox == pytest.approx(1.0, abs=1e-6)
        for sid in "ABC":
            assert res["samples"][sid]["agreement"]["rspace"] < 0.1
        assert (tab.groupby("sample_id").delta_e0_ev.nunique() == 1).all()
        assert (tab[[c for c in tab.columns if c.endswith("_err")]] >= 0).all().all()


class TestLarchBridges:
    def test_feff_paths_from_structure(self, tmp_path):
        pytest.importorskip("larch.xafs.feffrunner")
        from pymatgen.core import Lattice, Structure
        s = Structure.from_spacegroup("Im-3m", Lattice.cubic(2.8665), ["Fe"], [[0, 0, 0]])
        cat = mv.exp.exafs_feff_paths(s, "Fe", tmp_path / "bcc", compound="bcc", cluster_size=5.0, rmax=4.2)
        first = cat.sort_values("reff_angstrom").iloc[0]
        assert first.degeneracy == 8.0 and first.nleg == 2 and first.scatterers == "Fe"
        assert first.reff_angstrom == pytest.approx(2.8665 * 3 ** 0.5 / 2, abs=2e-3)
        assert (cat.nleg > 2).any() and set(cat.group) >= {"Fe-Fe", "Fe-Fe-ms"}
        lib = mv.exp.exafs_paths(cat, foil_compound="bcc")
        assert [b[0][0] for b in lib.bundles] == [first.path_id]

    def test_chi_from_mu(self, library):
        pytest.importorskip("larch.xafs")
        e0 = 7112.0
        energy = np.arange(e0 - 200, e0 + 900, 0.5)
        kk = np.sqrt(np.clip(energy - e0, 0, None) * X.ETOK)
        truth = [("O_ox", 6.0, 2.03, 0.006)]
        chi_true = _chi(library, truth, k=np.maximum(kk, 0.05)) * (energy > e0 + 5)
        step = 0.5 + np.arctan((energy - e0) / 2.0) / np.pi
        mu = 0.1 - 1e-5 * (energy - e0) + step * (1.0 - 1e-4 * np.clip(energy - e0, 0, None)) * (1 + chi_true)
        s = mv.exp.exafs_chi(energy, mu, e0=e0, rbkg=1.0)
        sel = (s["k"] >= 3) & (s["k"] <= 12)
        ref = _chi(library, truth, k=s["k"][sel])
        assert np.corrcoef(s["chi"][sel] * s["k"][sel] ** 2, ref * s["k"][sel] ** 2)[0, 1] > 0.95
        assert np.allclose(np.diff(s["k"]), 0.05)
