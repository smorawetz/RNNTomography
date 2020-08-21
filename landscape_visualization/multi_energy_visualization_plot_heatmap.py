import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

NH = 100
LR = 0.001
EP = 2000
SEED = 1
DIM = 201

NICE_NAMES_DICT = {
    "xy_no_symm": "Standard XY Model",
    "xy_hard_symm": "Symmetry-enforced XY Model",
    "tfim_no_symm": "TFIM",
}

NUM_PARS = 6

SEED = 1
torch.manual_seed(SEED)

params = {
    "text.usetex": True,
    "font.family": "serif",
    "legend.fontsize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1,
    "patch.edgecolor": "black",
}


def multi_energy_plot_heatmap(physics_models, symm_types, Ns, max_changes):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))

    for i in range(len(physics_models)):

        physics_model = physics_models[i]
        symm_type = symm_types[i]
        N = Ns[i]
        max_change = max_changes[i]

        model_name = "{0}_{1}".format(physics_model, symm_type)
        results_folder = "results/{0}_results".format(model_name)
        study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(N, NH, LR, EP)

        # Define the intervals which alpha, beta will take
        alpha_min, alpha_max = -max_change, max_change
        beta_min, beta_max = -max_change, max_change

        alpha_vals = np.linspace(alpha_min, alpha_max, DIM)
        beta_vals = np.linspace(beta_min, beta_max, DIM)
        grid_alphas, grid_betas = np.meshgrid(alpha_vals, beta_vals)

        landscape_file = "energy_landscape_range{0}_dim{1}_{2}_{3}_seed{4}.txt".format(
            max_change, DIM, model_name, study_name, SEED
        )
        loss_data = np.loadtxt(
            "{0}/{1}/{2}".format(results_folder, study_name, landscape_file)
        )

        # Now make plots of parameters trajectory
        # First retrieve final values of params
        max_ep = 2000  # how long training occurs for
        opt_params = []
        for par_num in range(NUM_PARS):
            param_filename = "param{0}_{1}_epoch2000_N{2}_nh{3}_lr{4}_ep{5}.txt".format(
                par_num, model_name, N, NH, LR, EP
            )
            param = np.loadtxt(
                "{0}/{1}/param_vals/{2}".format(results_folder, study_name, param_filename)
            )
            opt_params.append(param)

        # Get recorded values from data files
        period = 5  # how often parameter info recorded
        epochs = range(period, max_ep + 1, period)

        epochs_dict = {}  # store epoch: listof params
        for epoch in epochs:
            params = []
            for par_num in range(NUM_PARS):
                param_filename = "param{0}_{1}_epoch{2}_N{3}_nh{4}_lr{5}_ep{6}.txt".format(
                    par_num, model_name, epoch, N, NH, LR, EP
                )
                param = np.loadtxt(
                    "{0}/{1}/param_vals/{2}".format(
                        results_folder, study_name, param_filename
                    )
                )
                param -= opt_params[par_num]  # centre at optimum
                params.append(param)
            epochs_dict[epoch] = params

        # Get deltas and etas using same seed
        deltas = []
        etas = []
        for par_num in range(NUM_PARS):
            delta = np.loadtxt(
                "{0}/{1}/delta{2}_{3}_N{4}_nh{5}_lr{6}_ep{7}.txt".format(
                    results_folder, study_name, par_num, model_name, N, NH, LR, EP
                )
            )
            eta = np.loadtxt(
                "{0}/{1}/eta{2}_{3}_N{4}_nh{5}_lr{6}_ep{7}.txt".format(
                    results_folder, study_name, par_num, model_name, N, NH, LR, EP
                )
            )
            deltas.append(delta)
            etas.append(eta)

        # Now compute alphas and betas for every epoch
        alphas = []
        betas = []
        for epoch in epochs:
            alpha = 0
            beta = 0
            for par_num in range(NUM_PARS):
                # Need to expand dims to use tensor dot
                curr_params = epochs_dict[epoch][par_num]
                delta = deltas[par_num]
                eta = etas[par_num]
                if len(curr_params.shape) == 1:
                    curr_params = np.expand_dims(curr_params, axis=0)
                    delta = np.expand_dims(delta, axis=0)
                    eta = np.expand_dims(eta, axis=0)
                alpha += np.tensordot(curr_params, delta)
                beta += np.tensordot(curr_params, eta)
            alphas.append(alpha)
            betas.append(beta)

        # norm = colors.LogNorm(vmin=loss_data.min(), vmax=loss_data.max())
        norm = colors.Normalize(vmin=loss_data.min(), vmax=loss_data.max())

        ax = axs[i]
        contour = ax.imshow(
            loss_data,
            cmap="YlGn",
            norm=norm,
            extent=[-max_change, max_change, -max_change, max_change],
        )
        ax.set_xlim(-max_change, max_change)
        ax.set_ylim(-max_change, max_change)
        fig.colorbar(contour, ax=ax)
        ax.plot(alphas, betas, color="C0", linestyle="-")
        ax.scatter(alphas[::40], betas[::40], color="C3")

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1 - x0) / (y1 - y0))

        xticks = np.arange(-10, 10, 0.5).tolist()
        yticks = np.arange(-10, 10, 0.5).tolist()
        xticks = [x for x in xticks if -max_change <= x <= max_change]
        yticks = [y for y in yticks if -max_change <= y <= max_change]
        xticks.append(-max_change) if -max_change not in xticks else None
        xticks.append(max_change) if max_change not in xticks else None
        yticks.append(-max_change) if -max_change not in yticks else None
        yticks.append(max_change) if max_change not in yticks else None

        title_index = physics_model + "_" + symm_type
        ax.set_title(r"{0}".format(NICE_NAMES_DICT[title_index]))
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")

    fig.tight_layout()

    compare_names = []
    for i in range(len(physics_models)):
        name = "{0}_{1}_N{2}_range{3}_".format(
            physics_models[i], symm_types[i], Ns[i], max_changes[i]
        )
        compare_names.append(name)

    compare_name = ""
    for name in compare_names:
        compare_name += name
    compare_name += "lr{0}_nh{1}_ep{2}".format(LR, NH, EP)

    if not os.path.exists("compare_results/{0}".format(compare_name)):
        os.makedirs("compare_results/{0}".format(compare_name))

    plt.savefig("compare_results/{0}/compare_energy_heatmap.pdf".format(compare_name))
