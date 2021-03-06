import numpy as np
import GPy
from robo import BayesianOptimization
from robo.models.GPyModel import GPyModel
from robo.acquisition.EntropyMC import EntropyMC
from robo.acquisition.LogEI import LogEI
from robo.acquisition.EI import EI
from robo.maximizers.maximize import grid_search
from robo.util.loss_functions import logLoss
from robo.util.visualization import Visualization
from robo.recommendation.incumbent import compute_incumbent
import matplotlib


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)
#
# Defining our object of interest
#
X_lower = np.array([0])
X_upper = np.array([6])
dims = X_lower.shape[0]


def objective_funktion(x):
    return np.exp(x) / ((x + 0.5) ** 2.0 * (np.sin(4.0 * x) + np.exp(1.0 / 3.0 * x))) + np.random.normal(0, 0.01, x.shape)

num_initial = 4
initial_X = np.empty((num_initial, 1))
initial_X[0, :] = np.array([0.2])
initial_X[1:num_initial, :] = np.random.rand(num_initial - 1, dims) * (X_upper - X_lower) + X_lower
initial_Y = objective_funktion(initial_X)
print initial_X, initial_Y

#
# Creating our RoBO Environment
#

kernel = GPy.kern.RBF(input_dim=dims)
kernel = GPy.kern.Matern52(input_dim=dims)
maximize_fkt = grid_search
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

# entropy = Entropy(model, X_upper= X_upper, X_lower=X_lower, sampling_acquisition= LogEI, Nb=10, Np=600, loss_function = logLoss)
entropy_mc = EntropyMC(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, sampling_acquisition=LogEI, Nb=10, Np=300, Nf=3500, loss_function=logLoss)
#ei = EI(model, X_upper=X_upper, X_lower=X_lower, par=0.3)
# pi = PI(model, X_upper= X_upper, X_lower=X_lower, par =0.3)

for acquisition_fkt in [entropy_mc]:
    bo = BayesianOptimization(acquisition_fkt=acquisition_fkt,
                              model=model,
                              maximize_fkt=maximize_fkt,
                              X_lower=X_lower,
                              X_upper=X_upper,
                              dims=dims,
                              objective_fkt=objective_funktion,
                              save_dir=None)
    next_x = bo.choose_next(initial_X, initial_Y)
    print model.m
    Visualization(bo,
                  next_x,
                  X=initial_X,
                  Y=initial_Y,
                  show_acq_method=False,
                  show_obj_method=True,
                  show_model_method=True,
                  resolution=1000,
                  dest_folder="./test_output")
