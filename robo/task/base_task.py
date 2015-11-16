'''
Created on 12.07.2015

@author: Aaron Klein
'''

import numpy as np


class BaseTask(object):
    '''
    classdocs
    '''

    def __init__(self, X_lower, X_upper,
                 opt=None, fopt=None,
                 types=None, do_scaling=True):
        '''
        Constructor
        '''
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.n_dims = self.X_lower.shape[0]

        assert self.n_dims == self.X_upper.shape[0]

        if types is None:
            self.types = np.zeros([self.n_dims])
        else:
            self.types = types

        self.opt = opt
        self.fopt = fopt
        self.do_scaling = do_scaling

        if do_scaling:
            self.original_X_lower = self.X_lower
            self.original_X_upper = self.X_upper
            self.original_opt = opt
            self.original_fopt = fopt

            self.X_lower = np.zeros(self.original_X_lower.shape)
            self.X_upper = np.ones(self.original_X_upper.shape)
            if self.opt is not None:
                self.opt = self.transform(self.opt)

    def objective_function(self, x):
        pass

    def objective_function_test(self, x):
        pass

    def transform(self, x):
        """
        Transforms from  original space to the space [0, 1]
        Parameters
        ----------
        X: np.ndarray (1, D)
            Data point in original space

        Returns
        ----------
        np.ndarray (1, D)
            Input point in [0, 1] input space
        """
        return np.true_divide((x - self.original_X_lower),
                    (self.original_X_upper - self.original_X_lower))

    def retransform(self, x):
        """
        Scales from [0, 1] back to original space
        Parameters
        ----------
        X: np.ndarray (1, D)
            Data point in [0, 1] space

        Returns
        ----------
        np.ndarray (1, D)
            Input point in original input space
        """
        return self.original_X_lower + (self.original_X_upper -
                                        self.original_X_lower) * x

    def evaluate(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_dims
        assert np.all(x >= self.X_lower)
        assert np.all(x <= self.X_upper)

        if self.do_scaling:
            x = self.retransform(x)
        return self.objective_function(x)

    def evaluate_test(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_dims
        assert np.all(x >= self.X_lower)
        assert np.all(x <= self.X_upper)

        if self.do_scaling:
            x = self.retransform(x)
        return self.objective_function_test(x)
