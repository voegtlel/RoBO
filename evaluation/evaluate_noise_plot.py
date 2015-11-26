'''
Created on 24.11.2015

@author: Lukas Voegtle
'''
import setup_logger

import matplotlib as mpl
#mpl.use('Agg')

import plot_helper
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
import glob

from robo.models.gpy_model import GPyModel
from robo.acquisition.ei import EI
from robo.maximizers.grid_search import GridSearch
from robo.recommendation.incumbent import compute_incumbent
from robo.task.base_task import BaseTask
from robo.visualization.plotting import plot_objective_function, plot_model,\
    plot_acquisition_function
from robo.util.output_reader import OutputReader
import os.path
import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--savedir", help="Directory where to load the result from", default="")
args = parser.parse_args()

# The optimization function that we want to optimize.
# It gets a numpy array with shape (N,D) where N >= 1 are the number of
# datapoints and D are the number of features

savedir = args.savedir

reader = OutputReader()

resultPaths = glob.glob(os.path.join(savedir, '**', 'results.csv'))

dirnameRe = re.compile(r'(?P<task>\w+)_(?P<noise>[\d.]+)_(?P<run>\d+)')

resultGroups = dict()

for resultPath in resultPaths:
    dirname = os.path.basename(os.path.dirname(resultPath))
    dirnameMatch = dirnameRe.match(dirname)
    if not dirnameMatch:
        raise Exception('Dir does not match: ' + dirname)
    groups = dirnameMatch.groupdict()
    task = groups['task']
    if task <> 'branin':
        raise Exception("Invalid task: " + task)
    noise = groups['noise']
    if not resultGroups.has_key(noise):
        resultGroups[noise] = {'paths': [], 'incumbents': None, 'incumbent_vals': None}
    resultGroups[noise]['paths'].append(resultPath)

task = Branin()

for noise, resultGroup in resultGroups.iteritems():
    resultGroup['incumbents'] = np.zeros((len(resultGroup['paths']), 100, 2))
    resultGroup['incumbent_vals'] = np.zeros((len(resultGroup['paths']), 100))

    for i, resultPath in enumerate(resultGroup['paths']):
        results = reader.read_results_file(resultPath)
        while len(results['incumbent']) < 100:
            results['incumbent'].append(results['incumbent'][-1])
        resultGroup['incumbents'][i, :, :] = results['incumbent']

    for i in range(len(resultGroup['paths'])):
        for j in range(100):
            resultGroup['incumbent_vals'][i, j] = task.evaluate_test(resultGroup['incumbents'][i, j, np.newaxis]) - task.fopt
    mean = np.mean(resultGroup['incumbent_vals'], 0)

    plt.figure()
    plt.title("Noise: %s" % noise)
    plt.plot(mean)
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('regret (f(inc)-fopt)')
    plt.savefig('accum_%s.svg' % noise)

plt.show(block=True)