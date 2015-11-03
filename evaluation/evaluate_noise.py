'''
Created on 28.10.2015

@author: Lukas Voegtle
'''
import plot_helper
import setup_logger
import pickle
import cma
import george
from robo.maximizers.direct import Direct
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.priors import default_priors
from robo.priors.base_prior import BasePrior
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.task.branin import Branin
from robo.task.noise_task import NoiseTask

import GPy
import matplotlib.pyplot as plt
import numpy as np

from robo.models.gpy_model import GPyModel
from robo.acquisition.ei import EI
from robo.maximizers.grid_search import GridSearch
from robo.recommendation.incumbent import compute_incumbent
from robo.task.base_task import BaseTask
from robo.visualization.plotting import plot_objective_function, plot_model,\
    plot_acquisition_function


# The optimization function that we want to optimize.
# It gets a numpy array with shape (N,D) where N >= 1 are the number of
# datapoints and D are the number of features

tasks = [(Branin(), "branin"), (NoiseTask(Branin(), 0.1), "branin_0.1"), (NoiseTask(Branin(), 1), "branin_1"), (NoiseTask(Branin(), 10), "branin_10")]


class MyPrior(BasePrior):

    def __init__(self, n_dims):
        super(MyPrior, self).__init__()
        # The number of hyperparameters
        self.n_dims = n_dims
        # Prior for the Matern52 lengthscales
        self.tophat = default_priors.TophatPrior(-2, 2)
        # Prior for the covariance amplitude
        self.ln_prior = default_priors.LognormalPrior(mean=0.0, sigma=1.0)
        # Prior for the noise
        self.horseshoe = default_priors.HorseshoePrior(scale=0.1)

    def lnprob(self, theta):
        lp = 0
        # Covariance amplitude
        lp += self.ln_prior.lnprob(theta[0])
        # Lengthscales
        lp += self.tophat.lnprob(theta[1:-1])
        # Noise
        lp += self.horseshoe.lnprob(theta[-1])

        return lp

    def sample_from_prior(self, n_samples):
        p0 = np.zeros([n_samples, self.n_dims])
        # Covariance amplitude
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)
        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)
                              for _ in range(1, (self.n_dims - 1))]).T
        p0[:, 1:(self.n_dims - 1)] = ls_sample
        # Noise
        p0[:, -1] = self.horseshoe.sample_from_prior(n_samples)

        return p0


def global_optimize_posterior(model, X_lower, X_upper, startpoint):
    def f(x):
        mu, var = model.predict(x[np.newaxis, :])
        return (mu + np.sqrt(var))[0, 0]
    # Use CMAES to optimize the posterior mean + std
    res = cma.fmin(f, startpoint, 0.6, options={"bounds": [X_lower, X_upper]})
    return res[0], np.array([res[1]])


burnin = 100
chain_length = 200
n_hypers = 20


# Create figure with subplots

for task, task_name in tasks:
    # Make 10 restarts
    for restart_idx in range(10):
        # Allow j restarts on failure
        for retry_idx in range(10):
            cov_amp = 1.0
            config_kernel = george.kernels.Matern52Kernel(np.ones([task.n_dims]) * 0.5,
                                                           ndim=task.n_dims)

            noise_kernel = george.kernels.WhiteKernel(0.01, ndim=task.n_dims)
            kernel = cov_amp * (config_kernel + noise_kernel)

            prior = MyPrior(len(kernel))

            model = GaussianProcessMCMC(kernel, prior=prior, burnin=burnin,
                                        chain_length=chain_length, n_hypers=n_hypers)

            acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower,
                                  compute_incumbent=compute_incumbent, par=0.1)

            maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)

            bo = BayesianOptimization(acquisition_func=acquisition_func,
                                      model=model,
                                      maximize_func=maximizer,
                                      task=task,
                                      recommendation_strategy=global_optimize_posterior,
                                      save_dir="%s_%i" % (task_name, restart_idx))
            try:
                bo.run(60)
            except ValueError:
                continue
            fig = plt.figure(figsize=(15, 10))
            plt.hold(True)
            ax_real = plt.subplot2grid((2, 2), (0, 0))
            ax_predicted = plt.subplot2grid((2, 2), (0, 1))
            ax_perf = plt.subplot2grid((2, 2), (1, 0), colspan=2)

            plot_helper.plot_2d_contour(ax_real, ax_predicted, ax_perf, task, bo)

            plt.tight_layout()
            plt.savefig('%s_%i/evaluate_noise.svg' % (task_name, restart_idx))

#plt.show(block=True)
