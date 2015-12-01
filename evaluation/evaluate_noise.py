'''
Created on 28.10.2015

@author: Lukas Voegtle
'''
import setup_logger

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import plot_helper

import cma
import george
from robo.maximizers.direct import Direct
from robo.acquisition.integrated_acquisition import IntegratedAcquisition
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.priors import default_priors
from robo.priors.base_prior import BasePrior
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.task.branin import Branin
from robo.task.noise_task import NoiseTask

import numpy as np

from robo.acquisition.ei import EI
from robo.recommendation.incumbent import compute_incumbent

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", choices=['branin'], help="The task to perform (branin)", default="branin")
parser.add_argument("-i", "--incumbent", choices=['best', 'posterior_mean', 'posterior_mean_and_std'],
                    help="Type of incumbent", default="best")
parser.add_argument("-n", "--noise", type=float, help="Amount (variance) of noise to add", default=1)
parser.add_argument("-u", "--numsteps", type=int, help="Number of iterations", default=20)
parser.add_argument("-d", "--savedir", help="Directory to save the result to", default="")
parser.add_argument("-s", "--seed", type=int, help="Seed (random if not specified)", default=None)
args = parser.parse_args()

# The optimization function that we want to optimize.
# It gets a numpy array with shape (N,D) where N >= 1 are the number of
# datapoints and D are the number of features

task = None
task_name = args.task
savedir = args.savedir
numsteps = args.numsteps
if args.task == 'branin':
    task = Branin()
else:
    parser.usage()
    exit(-1)
if args.noise > 0:
    task = NoiseTask(task, args.noise)
    task_name += '_%d' % args.noise
if not savedir:
    savedir = "%s" % (task_name)
if args.seed is not None:
    np.random.seed(args.seed)

print "Arguments:"
print "Task: %s" % task_name
print "Incumbent: %s" % args.incumbent
print "Noise: %f" % args.noise
print "Savedir: %s" % savedir
print "Numsteps: %i" % numsteps
if args.seed is not None:
    print "Seed: %i" % args.seed
else:
    print "Seed: Random"


class MyPrior(BasePrior):
    """
    Custom prior with three base priors
    """
    def __init__(self, n_dims):
        super(MyPrior, self).__init__()
        # The number of hyperparameters
        self.n_dims = n_dims
        # Prior for the Matern52 lengthscales
        self.tophat = default_priors.TophatPrior(-5, 1)
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
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)[:, 0]
        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)[:, 0]
                              for _ in range(1, (self.n_dims - 1))]).T
        p0[:, 1:(self.n_dims - 1)] = ls_sample
        # Noise
        p0[:, -1] = self.horseshoe.sample_from_prior(n_samples)[:, 0]

        return p0


def get_recommendation_strategy(strategy):
    """
    Gets the recommendation strategy for a name
    """
    if strategy == 'best':
        return compute_incumbent
    elif strategy == 'posterior_mean' or strategy == 'posterior_mean_and_std':
        if strategy == 'posterior_mean':
            def f(x):
                """
                Function for optimization returns (mean)
                """
                mu, var = model.predict(x[np.newaxis, :])
                return mu[0]
        elif strategy == 'posterior_mean_and_std':
            def f(x):
                """
                Function for optimization returns (mean + sqrt(variance))
                """
                mu, var = model.predict(x[np.newaxis, :])
                return (mu + np.sqrt(var))[0, 0]

        def global_optimize_posterior(model, X_lower, X_upper, startpoint):
            """
            Optimizes the incumbent
            """
            # Use CMAES to optimize the posterior mean + std
            res = cma.fmin(f, startpoint, 0.6, options={"bounds": [X_lower, X_upper]})
            return res[0], np.array([res[1]])
        return global_optimize_posterior


burnin = 100
chain_length = 200
n_hypers = 20

cov_amp = 1.0
config_kernel = george.kernels.Matern52Kernel(np.ones([task.n_dims]),
                                               ndim=task.n_dims)

noise_kernel = george.kernels.WhiteKernel(0.01, ndim=task.n_dims)
kernel = cov_amp * (config_kernel + noise_kernel)

prior = MyPrior(len(kernel))

model = GaussianProcessMCMC(kernel, prior=prior, burnin=burnin,
                            chain_length=chain_length, n_hypers=n_hypers)

acq_func = EI(model, X_lower=task.X_lower, X_upper=task.X_upper,
              compute_incumbent=compute_incumbent, par=0.1)
acquisition_func = IntegratedAcquisition(model, acq_func, task.X_lower, task.X_upper)

maximizer = Direct(acquisition_func, task.X_lower, task.X_upper)

pltFig = plt.figure(figsize=(15, 10))


class BOStepped(BayesianOptimization):
    """
    BO with iteration plotting and incumbent history
    """
    def __init__(self, *args, **kwargs):
        super(BOStepped, self).__init__(*args, **kwargs)
        self.incumbents = None

    def iterate(self, it):
        """
        Custom iteration delegates to the default iteration and stores metadata
        """
        global pltFig
        super(BOStepped, self).iterate(it)

        # Perform metadata storage/plot generation
        print "--------- PLOTTING ------------"
        if self.incumbents is None:
            self.incumbents = np.zeros((1, self.task.n_dims))
            self.incumbents[0, :] = self.incumbent
        else:
            self.incumbents = np.append(self.incumbents, np.array(self.incumbent, ndmin=2), axis=0)

        plot_helper.plot_2d_contour(task, model, self.incumbents, pltFig, True, True, True, True)

        plt.tight_layout()
        plt.savefig('%s/plot_it_%i.svg' % (savedir, it))


bo = BOStepped(acquisition_func=acquisition_func,
               model=model,
               maximize_func=maximizer,
               task=task,
               recommendation_strategy=get_recommendation_strategy(args.incumbent),
               save_dir=savedir)
bo.run(numsteps)
plot_helper.plot_2d_contour(task, model, bo.incumbents, pltFig, True, True, True, True)

plt.tight_layout()
plt.savefig('%s/plot.svg' % savedir)

#plt.show(block=True)
