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
DIM = 101

NICE_NAMES_DICT = {"xy": "XY Model", "tfim": "TFIM"}

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


def energy_plot_heatmap(physics_model, symm_type, N, max_change):

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

    # NOTE: This stuff is for interpolating, which isn't good in this case
    # n_points = 10000
    # alpha_indices = np.random.choice(DIM, n_points)
    # beta_indices = np.random.choice(DIM, n_points)

    # alpha_points = alpha_vals[alpha_indices]
    # beta_points = beta_vals[beta_indices]
    # loss_points = loss_data[alpha_indices, beta_indices]
    # points_tuple = (alpha_points, beta_points)
    # grid_tuple = (grid_alphas, grid_betas)

    # interp_vals = griddata(
    # points_tuple, loss_points, grid_tuple, method="cubic"
    # )

    fig, ax = plt.subplots()
    # ax.contourf(grid_alphas, grid_betas, interp_vals)
    norm = colors.Normalize(vmin=loss_data.min(), vmax=loss_data.max())
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
    ax.set_title(
        r"{0} energy landscape projection".format(NICE_NAMES_DICT[physics_model])
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    # ax.set_zlabel(r"Loss")

    image_name = (
        "plot_energy_landscape_heatmap_range{0}_dim{1}_{2}_{3}_seed{4}.pdf".format(
            max_change, DIM, model_name, study_name, SEED
        )
    )
    plt.savefig("{0}/{1}/{2}".format(results_folder, study_name, image_name))
