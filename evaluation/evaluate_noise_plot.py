'''
Created on 24.11.2015

@author: Lukas Voegtle
'''
import setup_logger

import matplotlib as mpl
mpl.use('Agg')

from robo.task.branin import Branin

import matplotlib.pyplot as plt
import numpy as np
import glob

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

dirnameRe = re.compile(r'^(?P<task>[a-zA-Z]+)_(?P<incumbent>[\w_]+)_(?P<noise>[\d.]+)_(?P<run>\d+)$')

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
    incumbent = groups['incumbent']
    noise = groups['noise']
    print "Plotting", task, incumbent, noise
    if not resultGroups.has_key(incumbent):
        resultGroups[incumbent] = dict()
    if not resultGroups[incumbent].has_key(noise):
        resultGroups[incumbent][noise] = {'paths': [], 'incumbents': None, 'incumbent_vals': None}
    resultGroups[incumbent][noise]['paths'].append(resultPath)

task = Branin()

for incumbent, resultGroupN in resultGroups.iteritems():
    for noise, resultGroup in resultGroupN.iteritems():
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

        median = np.median(resultGroup['incumbent_vals'], axis=0)
        lower = median - np.percentile(resultGroup['incumbent_vals'], 5, axis=0)
        upper = median + np.percentile(resultGroup['incumbent_vals'], 95, axis=0)

        plt.figure()
        plt.title("Noise: %s, Inc: %s" % (incumbent, noise))
        #plt.plot(mean)
        plt.plot(median)
        plt.fill_between(list(range(len(median))), upper, lower)
        plt.yscale('log')
        plt.xlabel('iteration')
        plt.ylabel('regret (f(inc)-fopt)')
        plt.savefig('accum_%s_%s.svg' % (incumbent, noise))

plt.show(block=True)