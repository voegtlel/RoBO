# encoding=utf8
"""
This module contains the optimizers that will be used
when looking for the maxima of the acquisition functions.

The following optimizers are currently defined:

* DIRECT: Python wrapper to the DIRECT (DIviding RECTanlges) algorithm.
* cma: Covariance Matrix Adaptation, a stochastic numerical
  optimization algorithm for "difficult" optimization
  problems in Python. This works for problems with dimensionality strictly greater than one only.
* grid_search: A simple implementation of constrained grid search in 1-D.
* scipy.optimize.minimize: SciPy's built-in interface to numerous solvers.
  The ones available for constrained minimization problems are: L-BFGS-B, TNC, COBYLA, SLSQP.


"""


import sys
import StringIO
import numpy as np
import scipy
import emcee
import cma
import DIRECT


def _direct_acquisition_fkt_wrapper(acq_f):
    def _l(x, user_data):
        return -acq_f(np.array([x])), 0
    return _l


def direct(acquisition_fkt, X_lower, X_upper, n_func_evals=1000, n_iters=2000):
    x, fmin, ierror = DIRECT.solve(_direct_acquisition_fkt_wrapper(acquisition_fkt), l=[
                                   X_lower], u=[X_upper], maxT=n_iters, maxf=n_func_evals)
    return np.array([x])


def _cma_fkt_wrapper(acq_f, derivative=False):
    def _l(x, *args, **kwargs):
        x = np.array([x])
        return -acq_f(x, derivative=derivative, *args, **kwargs)[0]
    return _l


def cmaes(acquisition_fkt, X_lower, X_upper, verbose=False):

    if X_lower.shape[0] == 1:
        raise RuntimeError("CMAES does not works in a one dimensional function space")

    if not verbose:
        stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        res = cma.fmin(
            _cma_fkt_wrapper(acquisition_fkt),
            (X_upper + X_lower) * 0.5,
            0.6,
            options={
                "bounds": [
                    X_lower,
                    X_upper],
                "verbose": 0,
                "verb_log": sys.maxsize})
        sys.stdout = stdout
    else:
        res = cma.fmin(
            _cma_fkt_wrapper(acquisition_fkt),
            (X_upper + X_lower) * 0.5,
            0.6,
            options={
                "bounds": [
                    X_lower,
                    X_upper],
                "verbose": 0,
                "verb_log": sys.maxsize})
    return np.array([res[0]])


def grid_search(acquisition_fkt, X_lower, X_upper, resolution=1000):

    if X_lower.shape[0] > 1:
        raise RuntimeError("Grid search works just for one dimensional functions")
    x = np.linspace(X_lower[0], X_upper[0], resolution).reshape((resolution, 1, 1))
    # y = array(map(acquisition_fkt, x))
    ys = np.zeros([resolution])
    for i in range(resolution):
        ys[i] = acquisition_fkt(x[i])
    y = np.array(ys)
    x_star = x[y.argmax()]
    return x_star


def _scipy_optimizer_fkt_wrapper(acq_f, derivative=True):
    def _l(x, *args, **kwargs):
        x = np.array([x])
        if np.any(np.isnan(x)):
            # raise Exception("oO")

            if derivative:
                return np.inf, np.zero_like(x)
            else:
                return np.inf
        a = acq_f(x, derivative=derivative, *args, **kwargs)

        if derivative:
            # print -a[0][0], -a[1][0][0, :]
            return -a[0][0], -a[1][0][0, :]

        else:
            return -a[0]
    return _l


def stochastic_local_search(acquisition_fkt, X_lower, X_upper, Ne=20, starts=None):
    if hasattr(acquisition_fkt, "_get_most_probable_minimum"):
        xx = acquisition_fkt._get_most_probable_minimum()
    else:
        xx = np.add(
            np.multiply(
                (X_lower - X_upper),
                np.random.uniform(
                    size=(
                        1,
                        X_lower.shape[0]))),
            X_lower)

    def fun_p(x):
        acq_v = acquisition_fkt(np.array([x]))[0]
        log_acq_v = np.log(acq_v) if acq_v > 0 else -np.inf

        return log_acq_v
    sc_fun = _scipy_optimizer_fkt_wrapper(acquisition_fkt, False)
    S0 = 0.5 * np.linalg.norm(X_upper - X_lower)
    D = X_lower.shape[0]
    Xstart = np.zeros((Ne, D))

    restarts = np.zeros((Ne, D))
    if starts is None and hasattr(acquisition_fkt, "BestGuesses"):
        starts = acquisition_fkt.BestGuesses
    if starts is not None and Ne > starts.shape[0]:
        restarts[starts.shape[0]:Ne, ] = X_lower + \
            (X_upper - X_lower) * np.random.uniform(size=(Ne - starts.shape[0], D))
    elif starts is not None:
        restarts[0:Ne] = starts[0:Ne]
    else:
        restarts = X_lower + (X_upper - X_lower) * np.random.uniform(size=(Ne, D))

    sampler = emcee.EnsembleSampler(Ne, D, fun_p)
    Xstart, logYstart, _ = sampler.run_mcmc(restarts, 20)
    search_cons = []
    for i in range(0, X_lower.shape[0]):
        xmin = X_lower[i]
        xmax = X_upper[i]
        search_cons.append({'type': 'ineq',
                            'fun': lambda x: x - xmin})
        search_cons.append({'type': 'ineq',
                            'fun': lambda x: xmax - x})
    search_cons = tuple(search_cons)
    minima = []
    jacobian = False
    i = 0
    while i < Ne:
        try:
            minima.append(
                scipy.optimize.minimize(
                    fun=sc_fun,
                    x0=Xstart[
                        i,
                        np.newaxis],
                    jac=jacobian,
                    method='L-BFGS-B',
                    constraints=search_cons,
                    options={
                        'ftol': np.spacing(1),
                        'maxiter': 20}))
            i += 1
        # no derivatives
        except BayesianOptimizationError as e:
            if e.errno == BayesianOptimizationError.NO_DERIVATIVE:
                jacobian = False
                sc_fun = _scipy_optimizer_fkt_wrapper(acquisition_fkt, False)
            else:
                raise e
    # X points:
    Xend = np.array([res.x for res in minima])
    # Objective function values:
    Xdh = np.array([res.fun for res in minima])
    new_x = Xend[np.nanargmin(Xdh)]
    if len(new_x.shape):
        new_x = np.array([new_x])
    return new_x
