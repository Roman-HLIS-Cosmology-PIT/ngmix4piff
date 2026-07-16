import numpy as np

import ngmix
from ngmix.runners import PSFRunner, run_psf_fitter
from ngmix.guessers import GMixPSFGuesser, CoellipPSFGuesser


def make_observations(image, weight, image_pos, logger=None):
    """
    Create an ``ngmix.Observation`` from image data and WCS information.

    Parameters
    ------------
    image: galsim.Image
        Image cutout containing the star data.
    weight: galsim.Image
        Weight map corresponding to ``image``.
    image_pos: galsim.PositionD
        Position of the star in image coordinates.
    logger: object | None
        Optional logger argument reserved for interface consistency.

    Returns
    --------
    ngmix.Observation
        Observation object populated with image, weight, and Jacobian.
    """

    image_shape = image.array.shape
    galsim_jac = image.wcs.jacobian(image_pos=image_pos)
    ngmix_jac = ngmix.Jacobian(
        row=(image_shape[0] - 1) / 2.0,
        col=(image_shape[1] - 1) / 2.0,
        wcs=galsim_jac,
    )

    obs = ngmix.Observation(
        image=image.array,
        weight=weight.array,
        jacobian=ngmix_jac,
    )
    return obs


def get_runners(fitters, seed=None):
    """
    Build a mapping of configured fitter runners.

    Parameters
    ------------
    fitters: list[dict]
        List of fitter configuration dictionaries.
    seed: int | None
        Optional default seed used by fitters that require randomness.

    Returns
    --------
    dict[str, PSFRunner]
        Dictionary keyed by runner name with initialized runner instances.
    """
    if not isinstance(fitters, list):
        raise ValueError("fitters must be a list")
    runners = {}
    for fitter_config in fitters:
        if not isinstance(fitter_config, dict):
            raise ValueError("fitter_config must be a dict")
        if fitter_config["model"] == "wmom":
            runner, runner_name = setup_wmom_runner(fitter_config)
        elif fitter_config["model"] == "am":
            runner, runner_name = setup_am_runner(fitter_config, seed=seed)
        elif fitter_config["model"] == "gauss":
            runner, runner_name = setup_gauss_runner(fitter_config, seed=seed)
        else:
            raise NotImplementedError(
                f"Fitter model {fitter_config['model']} not implemented"
            )
        runners[runner_name] = runner
    return runners


def setup_wmom_runner(fitter_config):
    """
    Configure a weighted-moments runner.

    Parameters
    ------------
    fitter_config: dict
        Configuration dictionary for the ``wmom`` model.

    Returns
    --------
    tuple[PSFRunner, str]
        Configured runner and its output name.
    """
    fitter = ngmix.gaussmom.GaussMom(fwhm=fitter_config["weight"]["fwhm"])
    runner = PSFRunner(fitter, ntry=1)
    runner_name = fitter_config.get("name", "wmom")
    return runner, runner_name


def setup_am_runner(fitter_config, seed=None):
    """
    Configure an adaptive-moments runner.

    Parameters
    ------------
    fitter_config: dict
        Configuration dictionary for the ``am`` model.
    seed: int | None
        Optional seed used when not specified in ``fitter_config``.

    Returns
    --------
    tuple[PSFRunner, str]
        Configured runner and its output name.
    """

    seed = fitter_config.get("seed", seed)
    if seed is None:
        raise ValueError(
            "seed must be provided for am fitter either NgmixCatalog level or "
            "the fitter level"
        )
    print(f"Setting up am runner with seed {seed}")
    rng = np.random.RandomState(seed)
    guesser = GMixPSFGuesser(
        rng=rng,
        ngauss=1,
        guess_from_moms=True,
    )
    fitter = ngmix.admom.AdmomFitter(rng=rng)
    runner = PSFRunner(
        fitter=fitter, guesser=guesser, ntry=fitter_config.get("ntry", 1)
    )
    runner_name = fitter_config.get("name", "am")
    return runner, runner_name


def setup_gauss_runner(fitter_config, seed=None):
    """
    Configure a Gaussian or coelliptical Gaussian runner.

    Parameters
    ------------
    fitter_config: dict
        Configuration dictionary for the ``gauss`` model.
    seed: int | None
        Optional seed used when not specified in ``fitter_config``.

    Returns
    --------
    tuple[PSFRunner, str]
        Configured runner and its output name.
    """
    seed = fitter_config.get("seed", seed)
    if seed is None:
        raise ValueError(
            "seed must be provided for gauss fitter either NgmixCatalog level "
            "or the fitter level"
        )
    print(f"Setting up gauss runner with seed {seed}")
    rng = np.random.RandomState(seed)
    ngauss = fitter_config.get("ngauss", 1)
    use_em = fitter_config.get("em", False)
    if not use_em:
        if ngauss == 1:
            fitter = ngmix.fitting.Fitter(model="gauss")
        elif ngauss > 1:
            fitter = ngmix.fitting.CoellipFitter(ngauss=ngauss)
        else:
            raise ValueError("ngauss must be >= 1")
        guesser = CoellipPSFGuesser(
            rng=rng, ngauss=ngauss, guess_from_moms=True
        )
        runner = PSFRunner(
            fitter=fitter, guesser=guesser, ntry=fitter_config.get("ntry", 1)
        )
    else:
        fitter = ngmix.em.EMFitter()
        guesser = GMixPSFGuesser(rng=rng, ngauss=ngauss, guess_from_moms=True)
        runner = EMRunner(
            fitter=fitter, guesser=guesser, ntry=fitter_config.get("ntry", 1)
        )
    runner_name = fitter_config.get("name", f"gauss{ngauss}")
    return runner, runner_name


class EMRunner(PSFRunner):
    """
    PSF runner variant that adapts EM fitter output fields.
    """

    def go(self, obs):
        """
        Run EM fitting and get best fit parameters.

        Parameters
        ------------
        obs: ngmix.Observation
            Observation to fit.

        Returns
        --------
        dict
            Fitter result dictionary, including standardized ``g``, ``T``,
            ``flux``, and ``s2n`` values when fitting succeeds.
        """

        res = run_psf_fitter(
            obs=obs,
            fitter=self.fitter,
            guesser=self.guesser,
            ntry=self.ntry,
            set_result=self.set_result,
        )

        if res["flags"] == 0:
            gm = res.get_gmix()
            g1, g2, T = gm.get_g1g2T()
            flux = gm.get_flux()
            res["g"] = np.array([g1, g2])
            res["T"] = T
            res["flux"] = flux
            res["s2n"] = gm.get_model_s2n(obs)
            # res["T_err"] = -10.0
            # res["flux_err"] = -10.0
            # res["g_err"] = np.array([-10.0, -10.0])
        return res
