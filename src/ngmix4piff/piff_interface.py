import numpy as np
import galsim
from piff.stats import Stats
from piff.config import LoggerWrapper
from piff import __version__ as piff_version
from ngmix.shape import e1e2_to_g1g2
import fitsio

from .ngmix_utils import make_observations, get_runners
from . import __version__ as ngmix4piff_version


def get_runner_output_dtype(runner_name, kinds):
    """
    Build output dtypes for a single runner.

    Parameters
    ------------
    runner_name: str
        Name of the configured runner. This prefix is used in output column
        names.
    kinds: list[str]
        Measurement kinds to include (for example, ``data`` and ``model``).

    Returns
    --------
    list[tuple[str, type]]
        List of ``(column_name, numpy_dtype)`` entries for structured-array
        output.
    """
    dtypes = []
    for kind in kinds:
        dtypes += [
            (f"{runner_name}_flags_{kind}", np.int32),
            (f"{runner_name}_g1_{kind}", np.float64),
            (f"{runner_name}_g2_{kind}", np.float64),
            (f"{runner_name}_T_{kind}", np.float64),
            (f"{runner_name}_flux_{kind}", np.float64),
            (f"{runner_name}_snr_{kind}", np.float64),
            # (f"{runner_name}_g1_err_{kind}", np.float64),
            # (f"{runner_name}_g2_err_{kind}", np.float64),
            # (f"{runner_name}_T_err_{kind}", np.float64),
            # (f"{runner_name}_flux_err_{kind}", np.float64),
            # (f"{runner_name}_chisq_{kind}", np.float64),
        ]
    return dtypes


def get_output_empty(nobj, runner_names, kinds):
    """
    Allocate an empty structured output catalog.

    Parameters
    ------------
    nobj: int
        Number of objects (stars) in the output catalog.
    runner_names: iterable[str]
        Runner names to include in the catalog schema.
    kinds: list[str]
        Measurement kinds to include (for example, ``data`` and ``model``).

    Returns
    --------
    numpy.ndarray
        Zero-initialized structured array with base metadata and per-runner
        measurement columns.
    """
    dtypes = [
        ("u", np.float64),
        ("v", np.float64),
        ("x", np.float64),
        ("y", np.float64),
        ("ra", np.float64),
        ("dec", np.float64),
        ("reserve", bool),
        ("flag_psf", np.int32),
        ("chipnum", np.float64),
    ]
    for runner_name in runner_names:
        dtypes += get_runner_output_dtype(runner_name, kinds)

    output_cat = np.zeros(nobj, dtype=dtypes)
    return output_cat


class NgmixCatalog(Stats):
    """
    Piff stats plugin that measures stars with ngmix runners.

    Parameters
    ------------
    fitters: list[dict] | None
        Fitter configuration dictionaries used to build ngmix runners.
    seed: int | None
        Optional default random seed forwarded to runner setup.
    file_name: str | None
        Default output FITS file path used by :meth:`write`.
    model_properties: dict | None
        Optional properties applied to stars before drawing model stars.
    logger: object | None
        Unused initializer logger argument kept for PIFF config
        compatibility.
    """

    _type_name = "NgmixCatalog"

    def __init__(
        self,
        fitters=None,
        seed=None,
        file_name=None,
        model_properties=None,
        logger=None,
    ):

        self.file_name = file_name
        self.model_properties = model_properties

        self.runners = get_runners(fitters, seed=seed)

    def compute(self, psf, stars, logger=None):
        """Measure configured ngmix moments on data and optional model stars.

        Parameters
        ------------
        psf: object | None
            PIFF PSF model. When provided, model stars are drawn and measured
            in addition to data stars.
        stars: list
            PIFF star objects to process.
        logger: object | None
            Logger used for debug messages during model-star generation.

        Returns
        --------
        None
            Results are stored in ``self.output_cat``.
        """

        kinds = ["data"]
        if psf is not None:
            kinds.append("model")

        n_stars = len(stars)
        self.output_cat = get_output_empty(n_stars, self.runners.keys(), kinds)

        for i in range(n_stars):
            star = stars[i]
            image, weight, image_pos = star.data.getImage()

            data_obs = make_observations(
                image, weight, image_pos, logger=logger
            )
            for runner_name, runner in self.runners.items():
                res_ = runner.go(data_obs)
                self._add_result(i, res_, runner_name, kind="data")

        if psf is not None:
            logger.debug("Generating and Measuring Model Stars")
            if self.model_properties is not None:
                stars = [
                    star.withProperties(**self.model_properties)
                    for star in stars
                ]
                psf.interpolateStarList(stars, inplace=True)
            model_stars = psf.drawStarList(stars)
            for i in range(n_stars):
                model_star = model_stars[i]
                image, weight, image_pos = model_star.data.getImage()
                model_obs = make_observations(
                    image, weight, image_pos, logger=logger
                )
                for runner_name, runner in self.runners.items():
                    res_ = runner.go(model_obs)
                    self._add_result(i, res_, runner_name, kind="model")

        # Build the columns for the output catalog
        if isinstance(stars[0].image.wcs, galsim.wcs.CelestialWCS):
            ra = np.array(
                [
                    star.image.wcs.toWorld(star.image_pos).ra.deg
                    for star in stars
                ]
            )
            dec = np.array(
                [
                    star.image.wcs.toWorld(star.image_pos).dec.deg
                    for star in stars
                ]
            )
        else:
            ra = np.zeros(len(stars))
            dec = np.zeros(len(stars))
        positions = np.array(
            [
                (star.data.properties["u"], star.data.properties["v"])
                for star in stars
            ]
        )
        self.output_cat["u"] = positions[:, 0]  # u
        self.output_cat["v"] = positions[:, 1]  # v
        self.output_cat["x"] = np.array([star.image_pos.x for star in stars])
        self.output_cat["y"] = np.array([star.image_pos.y for star in stars])
        self.output_cat["ra"] = ra
        self.output_cat["dec"] = dec
        self.output_cat["reserve"] = np.array(
            [s.is_reserve for s in stars],
            dtype=self.output_cat["reserve"].dtype,
        )  # reserve
        self.output_cat["flag_psf"] = np.array(
            [s.is_flagged for s in stars],
            dtype=self.output_cat["flag_psf"].dtype,
        )  # flag_psf
        self.output_cat["chipnum"] = np.array(
            [s.chipnum for s in stars], dtype=self.output_cat["chipnum"].dtype
        )  # chipnum

    def _add_result(self, i, res_, runner_name, kind="data"):
        """
        Store one runner result into the output catalog.

        Parameters
        ------------
        i: int
            Index of the star row to update.
        res_: dict
            Result dictionary returned by a runner.
        runner_name: str
            Name of the runner used to build output column names.
        kind: str
            Measurement kind suffix (for example, ``data`` or ``model``).
        """
        if res_["flags"] != 0:
            self.output_cat[f"{runner_name}_flags_{kind}"][i] = res_["flags"]
            return
        if runner_name in ["wmom", "am"]:
            g1, g2 = e1e2_to_g1g2(res_["e1"], res_["e2"])
        else:
            g1, g2 = res_["g"]
        self.output_cat[f"{runner_name}_g1_{kind}"][i] = g1
        self.output_cat[f"{runner_name}_g2_{kind}"][i] = g2
        self.output_cat[f"{runner_name}_T_{kind}"][i] = res_["T"]
        self.output_cat[f"{runner_name}_flux_{kind}"][i] = res_["flux"]
        self.output_cat[f"{runner_name}_snr_{kind}"][i] = res_["s2n"]

    def write(self, file_name=None, logger=None):
        """
        Write the computed catalog to a FITS table.

        Parameters
        ------------
        file_name: str | None
            Output FITS path. If ``None``, uses ``self.file_name``.
        logger: object | None
            Logger passed through PIFF's ``LoggerWrapper``.
        """

        logger = LoggerWrapper(logger)
        if file_name is None:
            file_name = self.file_name
        if file_name is None:
            raise ValueError(f"No file_name specified for {self._type_name}")
        if not hasattr(self, "output_cat"):
            raise RuntimeError("Must call compute before calling write")

        logger.info("Writing Ngmix catalog to file %s", file_name)

        header = {
            "piff_version": piff_version,
            "ngmix4piff_version": ngmix4piff_version,
        }
        with fitsio.FITS(file_name, "rw", clobber=True) as f:
            f.write_table(self.output_cat, header=header)
