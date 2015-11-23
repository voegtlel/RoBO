"""
Contextual Bayesian optimization, example
"""

import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_2d_cont(task, eval_fn, pts_X=None, numpts=50, ax=None, clip_min=None, clip_max=None, plt_title="value",
                 vmin=None, vmax=None, logplot=True):
    """
    Plots 2d contours for some function with background coloring.

    Parameters
    ----------
    task: robo.task.base_task.BaseTask
        Task to evaluate the borders
    eval_fn: callable
        Function which takes 2D array with one 2D-point.
    pts_X: np.array
        If not none, contains additional points to be plotted
    numpts: number
        Number of points to be rendered
    ax:
        Target axis for the plot (if None, the axis will be created).
    clip_min: number
        Value for clipping the values
    clip_max: number
        Value for clipping the values
    plt_title: string
        Title of the plot
    vmin: number
        Minimal value for the target value range
    vmax: number
        Maximal value for the target value range
    logplot: bool
        If true, then the target value is plottet in log scale

    Returns
    -------
    (X, Y, Z):
        X, Y, Z contain the 2D Array used for plotting

    """
    if task.do_scaling:
        bounds_x = (task.original_X_lower[0], task.original_X_upper[0])
        bounds_y = (task.original_X_lower[1], task.original_X_upper[1])
    else:
        bounds_x = (task.X_lower[0], task.X_upper[0])
        bounds_y = (task.X_lower[1], task.X_upper[1])

    # Create predicted contour plot (use the real bounds here)
    x = np.linspace(task.X_lower[0], task.X_upper[0], num=numpts)
    y = np.linspace(task.X_lower[1], task.X_upper[1], num=numpts)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten(order="C")
    Y_flat = Y.flatten(order="C")
    Z_flat = np.zeros(len(X_flat))
    for i in range(len(X_flat)):
        Z_flat[i] = eval_fn(np.array((X_flat[i], Y_flat[i]), ndmin=2))
    if clip_min is not None or clip_max is not None:
        Z_flat = np.clip(Z_flat, clip_min, clip_max)
    Z = np.reshape(Z_flat, X.shape, order="C")

    if task.do_scaling:
        # Use plotting bounds
        x = np.linspace(bounds_x[0], bounds_x[1], num=50)
        y = np.linspace(bounds_y[0], bounds_y[1], num=50)
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten(order="C")
        Y_flat = Y.flatten(order="C")

    if logplot:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        levels = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4]
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        levels = None

    CS = ax.contour(X, Y, Z, levels=levels, norm=norm)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title(plt_title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(bounds_x)
    ax.set_ylim(bounds_y)
    imx, imy = np.mgrid[bounds_x[0]:bounds_x[1]:100j, bounds_y[0]:bounds_y[1]:100j]
    resampled = griddata((X_flat, Y_flat), Z_flat, (imx, imy))
    implt = ax.imshow(resampled.T,
                      extent=(bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]),
                      interpolation='bicubic', origin='lower', cmap=plt.get_cmap('hot'),
                      aspect='auto', norm=norm)
    cb = plt.colorbar(implt, ax=ax)
    if pts_X is not None:
        if task.do_scaling:
            ax.plot(pts_X[:, 0] * (bounds_x[1] - bounds_x[0]) + bounds_x[0],
                    pts_X[:, 1] * (bounds_y[1] - bounds_y[0]) + bounds_y[0], 'x')
        else:
            ax.plot(pts_X[:, 0], pts_X[:, 1], 'x')
    return ax, CS, implt, cb


def plot_2d_contour(task, model, fig=None, ax_real=None, ax_pred_mean=None, ax_pred_var=None, ax_perf=None):
    """
    Plots the 2d contours for the given task an the model.

    Parameters
    ----------
    task: robo.task.base_task.BaseTask
        Task to evaluate
    model: robo.models.base_model.BaseModel
    fig: matplotlib.figure.Figure|None
    ax_real: matplotlib.Axes|True|None
    ax_pred_mean: matplotlib.Axes|True|None
    ax_pred_var: matplotlib.Axes|True|None
    ax_perf: matplotlib.Axes|True|None

    Returns
    -------

    """

    # Build plot
    num_axes = 0
    ax = []
    if ax_real is True:
        num_axes += 1
    if ax_pred_mean is True:
        num_axes += 1
    if ax_pred_var is True:
        num_axes += 1
    if ax_perf is True:
        num_axes += 1
    if num_axes > 0:
        if not fig:
            fig = plt.figure(figsize=(15, 10))
        else:
            fig = plt.figure(fig.number)
            plt.clf()
        if num_axes == 1:
            ax = [plt.subplot2grid((1, 1), (0, 0))]
        elif num_axes == 2:
            ax = [plt.subplot2grid((1, 2), (0, 0)),
                  plt.subplot2grid((1, 2), (0, 1))]
        elif num_axes == 3:
            if ax_perf is True:
                ax = [plt.subplot2grid((2, 2), (0, 0)),
                      plt.subplot2grid((2, 2), (0, 1)),
                      plt.subplot2grid((2, 2), (1, 0), colspan=1)]
            else:
                ax = [plt.subplot2grid((2, 2), (0, 0)),
                      plt.subplot2grid((2, 2), (0, 1)),
                      plt.subplot2grid((2, 2), (1, 0))]
        elif num_axes == 4:
            ax = [plt.subplot2grid((2, 2), (0, 0)),
                  plt.subplot2grid((2, 2), (0, 1)),
                  plt.subplot2grid((2, 2), (1, 0)),
                  plt.subplot2grid((2, 2), (1, 1))]
    if ax_real is True:
        ax_real = ax.pop(0)
    if ax_pred_mean is True:
        ax_pred_mean = ax.pop(0)
    if ax_pred_var is True:
        ax_pred_var = ax.pop(0)
    if ax_perf is True:
        ax_perf = ax.pop(0)

    # Plot the graphs
    vmin = None
    vmax = None
    if ax_real:
        _, _, _, cb = plot_2d_cont(task, lambda x: task.evaluate_test(x), pts_X=model.X, ax=ax_real,
                                              plt_title="function")
        vmin, vmax = cb.get_clim()
    if ax_pred_mean:
        plot_2d_cont(task, lambda x: model.predict(x)[0], pts_X=model.X, ax=ax_pred_mean, plt_title="predicted mean",
                     vmin=vmin, vmax=vmax)
    if ax_pred_var:
        plot_2d_cont(task, lambda x: np.sqrt(model.predict(x)[1]), pts_X=model.X, ax=ax_pred_var,
                     plt_title="predicted variance", logplot=False)
    if ax_perf:
        perf = model.Y.flatten() - task.fopt
        ax_perf.set_title("regret")
        ax_perf.set_xlabel("iteration")
        ax_perf.set_ylabel("regret")
        ax_perf.set_yscale('log')
        ax_perf.plot(np.arange(1, len(perf) + 1), perf, "o")
        ax_perf.set_xlim((0, len(perf) + 1))

    return fig, ax_real, ax_pred_mean, ax_pred_var, ax_perf
