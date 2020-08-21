import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch.nn as nn
import torch.optim as optim

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

NH = 100
LR = 0.001
EP = 2000
SEED = 1
DIM = 101

NICE_NAMES_DICT = {"xy": "XY Model", "tfim": "TFIM"}

def loss_plot_3d(physics_model, symm_type, N, max_change):

    model_name = "{0}_{1}".format(physics_model, symm_type)
    results_folder = "results/{0}_results".format(model_name)
    study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(N, NH, LR, EP)

    # Define the intervals which alpha, beta will take
    alpha_min, alpha_max = -max_change, max_change
    beta_min, beta_max = -max_change, max_change

    alpha_vals = np.linspace(alpha_min, alpha_max, DIM)
    beta_vals = np.linspace(beta_min, beta_max, DIM)
    grid_alphas, grid_betas = np.meshgrid(alpha_vals, beta_vals)

    landscape_file = (
        "loss_landscape_range{0}_dim{1}_{2}_{3}_seed{4}.txt".format(
            max_change, DIM, model_name, study_name, SEED
        )
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

    # ax.plot_surface(grid_alphas, grid_betas, interp_vals)  # interpolation
    ax.plot_surface(grid_alphas, grid_betas, loss_data)

    # ax.set_title(r"{0} loss landscape projection".format(NICE_NAMES_DICT[physics_model]))
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel(r"Loss")

    image_name = (
        "plot_loss_landscape_3d_range{0}_dim{1}_{2}_{3}_seed{4}.pdf".format(
            max_change, DIM, model_name, study_name, SEED
        )
    )
    plt.savefig(
        "{0}/{1}/{2}".format(results_folder, study_name, image_name)
    )
