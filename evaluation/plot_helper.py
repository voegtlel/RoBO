"""
Contextual Bayesian optimization, example
"""

import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_2d_contour(ax_real, ax_predicted, ax_perf, task, bo):
    if task.do_scaling:
        bounds_x = (task.original_X_lower[0], task.original_X_upper[0])
        bounds_y = (task.original_X_lower[1], task.original_X_upper[1])
    else:
        bounds_x = (task.X_lower[0], task.X_upper[0])
        bounds_y = (task.X_lower[1], task.X_upper[1])

    # Create grid for contours
    x = np.linspace(bounds_x[0], bounds_x[1], num=200)
    y = np.linspace(bounds_y[0], bounds_y[1], num=200)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten(order="C")
    Y_flat = Y.flatten(order="C")
    Z_flat = task.objective_function(np.transpose(np.array((X_flat, Y_flat)))).flatten(order="C")
    Z = np.reshape(Z_flat, X.shape, order="C")

    # Create real contour plot
    CS = ax_real.contour(X, Y, Z, levels=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4],
                     norm=matplotlib.colors.LogNorm())
    ax_real.clabel(CS, inline=1, fontsize=10)
    ax_real.set_title('function (payoff)')
    ax_real.set_xlabel('x_1')
    ax_real.set_ylabel('x_2')
    ax_real.set_xlim(bounds_x)
    ax_real.set_ylim(bounds_y)
    imx, imy = np.mgrid[bounds_x[0]:bounds_x[1]:100j, bounds_y[0]:bounds_y[1]:100j]
    resampled = griddata((X_flat, Y_flat), Z_flat, (imx, imy))
    implt = ax_real.imshow(resampled.T, extent=(bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]),
                           interpolation='bicubic', origin='lower', cmap=plt.get_cmap('hot'),
                           norm=matplotlib.colors.LogNorm(), aspect='auto')
    resampled_limits = (resampled.min(), resampled.max())
    plt.colorbar(implt, ax=ax_real)

    # Create predicted contour plot (use the real bounds here)
    x = np.linspace(task.X_lower[0], task.X_upper[0], num=50)
    y = np.linspace(task.X_lower[1], task.X_upper[1], num=50)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten(order="C")
    Y_flat = Y.flatten(order="C")
    mean, var = np.zeros(len(X_flat)), np.zeros(len(X_flat))
    for i in range(len(X_flat)):
        mean[i], var[i] = bo.model.predict(np.array((X_flat[i], Y_flat[i]), ndmin=2))
    Z_flat = np.clip(mean, 1e-5, None)
    Z = np.reshape(Z_flat, X.shape, order="C")

    if task.do_scaling:
        # Use plotting bounds
        x = np.linspace(bounds_x[0], bounds_x[1], num=50)
        y = np.linspace(bounds_y[0], bounds_y[1], num=50)
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten(order="C")
        Y_flat = Y.flatten(order="C")

    CS = ax_predicted.contour(X, Y, np.maximum(Z, 1e-3), levels=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2,
                                                        1e3, 3e3, 1e4], norm=matplotlib.colors.LogNorm())
    ax_predicted.clabel(CS, inline=1, fontsize=10)
    ax_predicted.set_title('predicted function (payoff)')
    ax_predicted.set_xlabel('x_1')
    ax_predicted.set_ylabel('x_2')
    ax_predicted.set_xlim(bounds_x)
    ax_predicted.set_ylim(bounds_y)
    imx, imy = np.mgrid[bounds_x[0]:bounds_x[1]:100j, bounds_y[0]:bounds_y[1]:100j]
    resampled = griddata((X_flat, Y_flat), Z_flat, (imx, imy))
    implt = ax_predicted.imshow(resampled.T,
                                extent=(bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]),
                                interpolation='bicubic', origin='lower', cmap=plt.get_cmap('hot'),
                                norm=matplotlib.colors.LogNorm(), aspect='auto',
                                vmin=resampled_limits[0], vmax=resampled_limits[1])
    plt.colorbar(implt, ax=ax_predicted)
    if task.do_scaling:
        ax_predicted.plot(bo.X[:, 0] * (bounds_x[1] - bounds_x[0]) + bounds_x[0],
                          bo.X[:, 1] * (bounds_y[1] - bounds_y[0]) + bounds_y[0], 'x')
    else:
        ax_predicted.plot(bo.X[:, 0], bo.X[:, 1], 'x')

    if task.fopt is not None:
        perf = bo.Y.flatten() - task.fopt
        ax_perf.set_title("regret")
        ax_perf.set_xlabel("iteration")
        ax_perf.set_ylabel("regret")
        ax_perf.set_yscale('log')
        ax_perf.plot(np.arange(1, len(perf) + 1), perf, "o")
        ax_perf.set_xlim((0, len(perf) + 1))

        plt.axhline(task.fopt, axes=ax_perf)
