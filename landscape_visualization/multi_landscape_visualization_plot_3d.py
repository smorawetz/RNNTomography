import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

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


def multi_loss_plot_3d(physics_models, symm_types, Ns, max_changes):

    fig = plt.figure(figsize=plt.figaspect(1 / len(physics_models) - 0.1))

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

        landscape_file = "loss_landscape_range{0}_dim{1}_{2}_{3}_seed{4}.txt".format(
            max_change, DIM, model_name, study_name, SEED
        )
        loss_data = np.loadtxt(
            "{0}/{1}/{2}".format(results_folder, study_name, landscape_file)
        )

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

        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        # ax.plot_surface(grid_alphas, grid_betas, interp_vals)  # interpolation
        ax.plot_surface(grid_alphas, grid_betas, loss_data)
        title_index = physics_model + "_" + symm_type
        ax.set_title(r"{0}".format(NICE_NAMES_DICT[title_index]))
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")
        ax.set_zlabel(r"Loss")

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

    plt.savefig("compare_results/{0}/compare_loss_3d.pdf".format(compare_name))
