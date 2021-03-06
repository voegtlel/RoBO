import sys
import os
#sys.path.insert(0, '../')
import unittest
import errno
import numpy as np
import random
import GPy
from robo.models.GPyModel import GPyModel
from robo.acquisition.PI import PI
from robo.util.visualization import Visualization
import matplotlib.pyplot as plt

from robo.recommendation.incumbent import compute_incumbent


here = os.path.abspath(os.path.dirname(__file__))


#@unittest.skip("skip first test\n")
class PITestCase1(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.62971589], [0.63273273], [0.17867868], [0.17447447], [1.88558559]])
        self.y = np.array([[-3.69925653], [-3.66221988], [-3.65560591], [-3.58907791], [-8.06925984]])
        self.kernel = GPy.kern.RBF(input_dim=1, variance=30.1646253727, lengthscale=0.435343653946)
        self.noise = 1e-20
        self.model = GPyModel(self.kernel, noise_variance=self.noise, optimize=False)
        self.model.train(self.x, self.y)

    def test(self):
        X_upper = np.array([2.1])
        X_lower = np.array([-2.1])

        x_test = np.array([[1.7], [2.0]])
        pi_estimator = PI(self.model, X_lower, X_upper)

        assert pi_estimator(x_test[0, np.newaxis], incumbent=np.array([1.88558559]))[0] > 0.0
        assert pi_estimator(x_test[1, np.newaxis], incumbent=np.array([1.88558559]))[0] > 0.0

        self.assertAlmostEqual(pi_estimator(self.x[-1, np.newaxis], incumbent=np.array([1.88558559]))[0], 0.0, delta=10E-5)

if __name__=="__main__":
    unittest.main()
