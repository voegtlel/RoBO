import os
import numpy as np


class Visualization(object):
    def __init__(self,
                 bayesian_opt,
                 new_x=None,
                 X=None,
                 Y=None,
                 dest_folder=None,
                 prefix="robo",
                 show_acq_method=False,
                 show_obj_method=False,
                 show_model_method=False,
                 show_incumbent_gap=False,
                 resolution=1000,
                 interactive=False,
                 incumbent_list_x=None,
                 incumbent_list_y=None):
        if dest_folder is not None:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            interactive = False
        if interactive:
            import matplotlib; matplotlib.use('GTKAgg')
            import matplotlib.pyplot as plt;
            plt.ion()

        if bayesian_opt.dims > 1 and show_acq_method:
            raise AttributeError("The acquisition function can only be visualized if the objective function has only one dimension")
        if bayesian_opt.dims > 1 and show_obj_method:
            raise AttributeError("The objective function can only be visualized if the objective function has only one dimension")
        if bayesian_opt.dims > 1 and show_model_method:
            raise AttributeError("The model can only be visualized if the objective function has only one dimension")

        self.prefix = prefix
        self.fig = plt.figure()

        self.new_x = new_x
        self.obj_plot_min_y = None
        self.obj_plot_max_y = None
        one_dim_min = bayesian_opt.X_lower[0]
        one_dim_max = bayesian_opt.X_upper[0]
        self.num_subplots = 0

        self.plotting_range = np.linspace(one_dim_min, one_dim_max, num=resolution)
        if show_acq_method:
            self.acquisition_fkt = bayesian_opt.acquisition_fkt
            self.acquisition_fkt.plot(self.fig, one_dim_min, one_dim_max)
        insert_last = False
        if show_obj_method:
            self.objective_fkt = bayesian_opt.objective_fkt
            self.plot_objective_fkt(self.fig, one_dim_min, one_dim_max)
            insert_last = True
        if show_model_method:
            self.model = bayesian_opt.model
            self.plot_model(self.fig, one_dim_min, one_dim_max, insert_last)

        if show_incumbent_gap:
            print show_incumbent_gap
            self.plot_incumbent_gap(self.fig, incumbent_list_x, incumbent_list_y, bayesian_opt.objective_fkt.X_stars, bayesian_opt.objective_fkt.Y_star)
        if not interactive:
            self.fig.savefig(os.path.join(dest_folder, prefix + ".png"), format='png')
            print "Save figure as %s " % (os.path.join(dest_folder, prefix + ".png"))
            self.fig.clf()
            plt.close()
        else:
            plt.show(block=True)

    def plot_model(self, fig, one_dim_min, one_dim_max, insert_last=False):
        ax = fig.axes[-1]
        if not insert_last:
            n = len(fig.axes)
            for i in range(n):
                fig.axes[i].change_geometry(n + 1, 1, i + 1)
            ax = fig.add_subplot(n + 1, 1, n + 1)
        if hasattr(self.model, "visualize"):
            self.model.visualize(ax, one_dim_min, one_dim_max)
        _min_y, _max_y = ax.get_ylim()
        mu, var = self.model.predict(self.new_x)
        ax.plot(self.new_x[0], mu[0], "r.", markeredgewidth=5.0)
        if self.obj_plot_min_y is not  None and self.obj_plot_max_y is not None:
            self.obj_plot_min_y = min(_min_y, self.obj_plot_min_y)
            self.obj_plot_max_y = max(_max_y, self.obj_plot_max_y)
            ax.set_ylim(self.obj_plot_min_y, self.obj_plot_max_y)
        else:
            self.obj_plot_min_y = _min_y
            self.obj_plot_max_y = _max_y
        return ax

    def plot_objective_fkt(self, fig, one_dim_min, one_dim_max):
        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n + 1, 1, i + 1)
        ax = fig.add_subplot(n + 1, 1, n + 1)
        ax.plot(self.plotting_range, self.objective_fkt(self.plotting_range[:, np.newaxis]), color='b', linestyle="--")
        ax.set_xlim(one_dim_min, one_dim_max)
        _min_y, _max_y = ax.get_ylim()
        if self.obj_plot_min_y is not  None and self.obj_plot_max_y is not None:
            self.obj_plot_min_y = min(_min_y, self.obj_plot_min_y)
            self.obj_plot_max_y = max(_max_y, self.obj_plot_max_y)
            ax.set_ylim(self.obj_plot_min_y, self.obj_plot_max_y)
        else:
            self.obj_plot_min_y = _min_y
            self.obj_plot_max_y = _max_y
            ax.set_ylim(self.obj_plot_min_y, self.obj_plot_max_y)

        return ax

    def plot_incumbent_gap(self, fig, inc_x, inc_y, best_x, best_y):
        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n + 1, 1, i + 1)
        ax = fig.add_subplot(n + 1, 1, n + 1)
        print range(inc_y.shape[0]), inc_y - best_y
        ax.plot(range(inc_y.shape[0]), inc_y - best_y, color='b', linestyle="--")
