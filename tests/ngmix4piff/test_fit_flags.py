import fitsio
import galsim
import ngmix
import numpy as np
import piff
import pytest
from ngmix.admom.admom import AdmomFitter, AdmomResult
from ngmix.gaussmom import GaussMom

from ngmix4piff.hom import AdmomFitterHOM
from ngmix4piff.ngmix_utils import make_observations, setup_am_runner
from ngmix4piff.piff_interface import NgmixCatalog

SCALE = 0.11
STAMP_SIZE = 25
# Stars are told apart in the patched functions by their peak pixel value.
FLUXES = [3000.0, 2000.0, 1000.0]
GMIX_RANGE_ERROR = ngmix.flags.GMIX_RANGE_ERROR


def _make_image(flux):
    image = galsim.ImageD(STAMP_SIZE, STAMP_SIZE, scale=SCALE)
    galsim.Gaussian(fwhm=0.3, flux=flux).drawImage(image)
    return image


def _is_star(obs, i):
    return np.isclose(obs.image.max(), _make_image(FLUXES[i]).array.max())


def _patch_admom(monkeypatch, bad_star=None, ncalls=None):
    """Make adaptive moments converge to |e| = 1 exactly, as in issue #5.

    Applies to observations of ``bad_star``, or to the first ``ncalls`` calls.
    """
    orig_go = AdmomFitter.go
    calls = {"n": 0}

    def go(self, obs, guess):
        res = orig_go(self, obs=obs, guess=guess)
        calls["n"] += 1
        bad = _is_star(obs, bad_star) if bad_star is not None else calls["n"] <= ncalls
        if bad and res["flags"] == 0:
            # get_gmix() computes e1 = pars[2] / pars[4].
            res["pars"][2] = res["pars"][4]
            res["pars"][3] = 0.0
            res["e1"], res["e2"] = 1.0, 0.0
        return res

    monkeypatch.setattr(AdmomFitter, "go", go)


def test_admom_hom_flags(monkeypatch):
    """A fit that converges to |e| = 1 is flagged instead of raising."""
    image = _make_image(FLUXES[0])
    weight = galsim.ImageD(STAMP_SIZE, STAMP_SIZE, init_value=1.0, scale=SCALE)
    obs = make_observations(image, weight, image.true_center)

    _patch_admom(monkeypatch, ncalls=1)
    fitter = AdmomFitterHOM(with_higher_order=True, rng=np.random.RandomState(1))
    res = fitter.go(obs, guess=0.1)
    assert res["flags"] & GMIX_RANGE_ERROR

    # Only the first try fails, so a runner with retries recovers.
    _patch_admom(monkeypatch, ncalls=1)
    runner, _ = setup_am_runner({"model": "am", "ntry": 3}, do_hom=True, seed=1)
    res = runner.go(obs)
    assert res["flags"] == 0
    assert res["hom_flags"] == 0
    assert np.isfinite(res["M22"])


@pytest.mark.parametrize("do_hom", [True, False])
def test_compute_catalog(monkeypatch, tmp_path, do_hom):
    """The catalog is written with flags and NaNs for failed measurements."""
    stars = []
    for i, flux in enumerate(FLUXES):
        image = _make_image(flux)
        weight = galsim.ImageD(image.bounds, init_value=1.0)
        x, y = image.true_center.x, image.true_center.y
        stars.append(
            piff.Star.makeTarget(
                x=x, y=y, image=image, weight=weight, properties={"chipnum": i}
            )
        )

    # Star 0: the am and wmom fits converge to |e| = 1. The am fitter flags
    # itself; wmom has no gmix to check, so this tests the backstop in
    # NgmixCatalog.compute around e1e2_to_g1g2.
    _patch_admom(monkeypatch, bad_star=0)
    orig_wmom_go = GaussMom.go

    def wmom_go(self, obs):
        res = orig_wmom_go(self, obs)
        if _is_star(obs, 0):
            res["e1"], res["e2"] = 1.0, 0.0
        return res

    monkeypatch.setattr(GaussMom, "go", wmom_go)
    # Star 1: the higher order moments fail but the am fit itself is fine.
    # Patch only the gmix from the am result; wmom also uses get_weighted_moments.
    orig_get_gmix = AdmomResult.get_gmix

    def get_gmix(self):
        gmix = orig_get_gmix(self)
        if _is_star(self._obs, 1):
            orig_moments = gmix.get_weighted_moments

            def get_weighted_moments(obs, **kwargs):
                res = orig_moments(obs, **kwargs)
                res["flags"] = ngmix.flags.NONPOS_VAR
                return res

            gmix.get_weighted_moments = get_weighted_moments
        return gmix

    monkeypatch.setattr(AdmomResult, "get_gmix", get_gmix)

    fitters = [{"model": "am", "ntry": 1}, {"model": "wmom", "weight": {"fwhm": 0.3}}]
    file_name = str(tmp_path / "ngmix.fits")
    stat = NgmixCatalog(fitters=fitters, seed=1, do_hom=do_hom, file_name=file_name)
    stat.compute(None, stars)
    stat.write()
    cat = fitsio.read(file_name)

    assert len(cat) == 3
    measured = ["g1", "g2", "T", "flux", "snr"]
    hom = ["g41", "g42", "T4", "h41", "h42"]
    if do_hom:
        assert "am_hom_flags_data" in cat.dtype.names
    else:
        assert not any("hom_flags" in name for name in cat.dtype.names)

    # Star 0: flagged, NaN values.
    for runner in ["am", "wmom"]:
        assert cat[f"{runner}_flags_data"][0] & GMIX_RANGE_ERROR
        for col in measured + (hom if do_hom else []):
            assert np.isnan(cat[f"{runner}_{col}_data"][0])
        if do_hom:
            assert cat[f"{runner}_hom_flags_data"][0] & GMIX_RANGE_ERROR

    # Star 1: the am fit is good; only the higher order moments are flagged.
    assert cat["am_flags_data"][1] == 0
    for col in measured:
        assert np.isfinite(cat[f"am_{col}_data"][1])
    if do_hom:
        assert cat["am_hom_flags_data"][1] == ngmix.flags.NONPOS_VAR
        for col in hom:
            assert np.isnan(cat[f"am_{col}_data"][1])

    # Star 2: everything good.
    assert cat["am_flags_data"][2] == 0
    for col in measured + (hom if do_hom else []):
        assert np.isfinite(cat[f"am_{col}_data"][2])
    if do_hom:
        assert cat["am_hom_flags_data"][2] == 0

    # wmom measures its own moments, so stars 1 and 2 are unaffected.
    assert np.all(cat["wmom_flags_data"][1:] == 0)
    for col in measured + (hom if do_hom else []):
        assert np.all(np.isfinite(cat[f"wmom_{col}_data"][1:]))
    if do_hom:
        assert np.all(cat["wmom_hom_flags_data"][1:] == 0)

    assert np.array_equal(cat["chipnum"], [0, 1, 2])
