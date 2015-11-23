import setup_logger
import unittest

import numpy as np

import GPy

from  scipy.optimize import check_grad

from robo.models.gpy_model import GPyModel
from robo.acquisition.log_ei import LogEI
from robo.recommendation.incumbent import compute_incumbent


class LogEITestCase1(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[0.62971589], [0.63273273], [0.17867868],
                           [0.17447447], [1.88558559]])
        self.y = np.array([[-3.69925653], [-3.66221988], [-3.65560591],
                           [-3.58907791], [-8.06925984]])
        self.kernel = GPy.kern.RBF(input_dim=1, variance=30.1646253727,
                                   lengthscale=0.435343653946)
        self.noise = 1e-20
        self.model = GPyModel(self.kernel,
                              noise_variance=self.noise,
                              optimize=False)
        self.model.train(self.x, self.y)

        X_upper = np.array([2.1])
        X_lower = np.array([-2.1])

        self.log_ei = LogEI(self.model, X_lower, X_upper,
                                 compute_incumbent=compute_incumbent)

    def test(self):

        x_test = np.array([[1.7], [2.0]])

        assert self.log_ei(x_test[0, np.newaxis])[0] > -np.Infinity
        assert self.log_ei(x_test[1, np.newaxis])[0] > -np.Infinity

        #assert(log_ei_estimator(self.x[-1, np.newaxis])[0]) == -np.Infinity
# Not implemented yet!
#    def test_check_grads(self):
#        x_ = np.array([[np.random.rand()]])
#        assert check_grad(self.log_ei, lambda x: -self.log_ei(x, True)[1], x_) < 1e-5

if __name__ == "__main__":
    unittest.main()
