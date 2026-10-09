import ngmix
from ngmix.admom.admom import AdmomFitter
from ngmix.fitting import Fitter
from ngmix.gexceptions import GMixRangeError


def parse_hom_momoents(res):
    """
    Parse higher order moments from the result dictionary.
    We use the same mapping as in Piff:
    m['M22']/m['M11']  # T4
    m['M31']/m['M11']**2  # g41
    m['M13']/m['M11']**2  # g42
    m['M40']/m['M11']**2  # h41
    m['M04']/m['M11']**2  # h42
    with M['04'] = M['M14'] in ngmix.

    Parameters
    ----------
    res: dict
        Result dictionary from the adaptive moments fitter.

    Returns
    -------
    hom_res: dict
        Dictionary containing higher order moments
    """

    hom_res = {}
    hom_res["T4"] = res["M22"] / res["M11"]
    hom_res["g41"] = res["M31"] / res["M11"] ** 2 - 3.0 * res["e1"]
    hom_res["g42"] = res["M13"] / res["M11"] ** 2 - 3.0 * res["e2"]
    hom_res["h41"] = res["M40"] / res["M11"] ** 2
    hom_res["h42"] = res["M14"] / res["M11"] ** 2
    return hom_res


def add_hom_results(result, obs, with_higher_order):
    """
    Check that a successful fit yields a valid gmix, and optionally add
    higher order moments measured with that gmix as the weight.

    ngmix does not check that a converged fit has |e| < 1, so building the
    gmix can raise a ``GMixRangeError``. In that case the fit is flagged with
    ``ngmix.flags.GMIX_RANGE_ERROR`` instead of raising, which also lets the
    runner retry with a new guess.

    Parameters
    ----------
    result: dict
        Fit result. Must provide ``get_gmix`` when ``result["flags"] == 0``.
    obs: Observation
        ngmix.Observation that was fit.
    with_higher_order: bool
        If set to True, add the higher order moments to the result.

    Returns
    -------
    result: dict
        The fit result, including higher order moments if requested.
    """
    if result["flags"] != 0:
        return result
    try:
        gmix = result.get_gmix()
    except GMixRangeError:
        result["flags"] |= ngmix.flags.GMIX_RANGE_ERROR
        return result
    if not with_higher_order:
        return result

    hom_results = gmix.get_weighted_moments(
        obs=obs,
        with_higher_order=with_higher_order,
    )
    hom_results.update(result)
    return hom_results


class AdmomFitterHOM(AdmomFitter):
    """
    Measure adaptive moments for the input observation from NGMix.
    We have added the option to return higher order moments using the derived
    adaptive weight function. The second order quantities such as e1, e2, T,
    flux, etc comes from the adaptive moments.

    parameters
    ----------
    with_higher_order: bool, optional
        If set to True, return higher order moments in the sums/sums_cov
        arrays.  See ngmix.moments.MOMENTS_NAME_MAP for a map between
        name and index.
    maxiter: integer, optional
        Maximum number of iterations, default 200
    etol: float, optional
        absolute tolerance in e1 or e2 to determine convergence,
        default 1.0e-5
    Ttol: float, optional
        relative tolerance in T <x^2> + <y^2> to determine
        convergence, default 1.0e-3
    shiftmax: float, optional
        Largest allowed shift in the centroid, relative to
        the initial guess.  Default 5.0 (5 pixels if the jacobian
        scale is 1)
    cenonly: bool, optional
        If set to True, only vary the center
    rng: np.random.RandomState
        Random state for creating full gaussian guesses based
        on a T guess
    """

    def __init__(
        self,
        with_higher_order=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.with_higher_order = with_higher_order

    def go(self, obs, guess):
        """
        run the adpative moments

        parameters
        ----------
        obs: Observation
            ngmix.Observation
        guess: ngmix.GMix or a float
            A guess for the fitter.  Can be a full gaussian mixture or a single
            value for T, in which case the rest of the parameters for the
            gaussian are generated.
        """
        result = super().go(obs=obs, guess=guess)
        return add_hom_results(result, obs, self.with_higher_order)


class FitterHOM(Fitter):
    """
    A class for doing a fit using levenberg marquardt.
    We have added the option to return higher order moments using the derived
    best fitted gmix.

    Parameters
    ----------
    model: str
        The model to fit
    prior: ngmix prior
        A prior for fitting
    fit_pars: dict
        Parameters to send to the leastsq fitting routine
    with_higher_order: bool, optional
        If set to True, return higher order moments in the sums/sums_cov
        arrays.  See ngmix.moments.MOMENTS_NAME_MAP for a map between
        name and index.
    """

    def __init__(
        self,
        with_higher_order=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.with_higher_order = with_higher_order

    def go(self, obs, guess):
        """
        Run leastsq and set the result

        Parameters
        ----------
        obs: Observation, ObsList, or MultiBandObsList
            Observation(s) to fit
        guess: array
            Array of initial parameters for the fit

        Returns
        --------
        a dict-like which contains the result as well as functions used for the
        fitting.

        """

        result = super().go(obs=obs, guess=guess)
        return add_hom_results(result, obs, self.with_higher_order)
