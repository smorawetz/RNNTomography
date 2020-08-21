import torch
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

import models.rnn_model_no_symm as rnn_model_no_symm
import models.rnn_model_hard_symm as rnn_model_hard_symm
import models.rnn_model_soft_symm as rnn_model_soft_symm

PHYSICS_MODEL = "xy"
SYMM_TYPE = "no_symm"
MODEL_NAME = "{0}_{1}".format(PHYSICS_MODEL, SYMM_TYPE)
N = 10
NH = 100
LR = 0.001
EP = 2000
SEED = 1

models_dict = {"no_symm": rnn_model_no_symm, "hard_symm": rnn_model_hard_symm}
rnn_model = models_dict[SYMM_TYPE]

results_folder = "{0}_results".format(MODEL_NAME)
study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(N, NH, LR, EP)
rnn_state = "rnn_state_{0}_{1}_seed{2}.pt".format(MODEL_NAME, study_name, SEED)

# Define the intervals which alpha, beta will take
max_change = 0.25
alpha_min, alpha_max = -max_change, max_change
beta_min, beta_max = -max_change, max_change
num_vals = 101

alpha_vals = np.linspace(alpha_min, alpha_max, num_vals)
beta_vals = np.linspace(beta_min, beta_max, num_vals)
grid_alphas, grid_betas = np.meshgrid(alpha_vals, beta_vals)

landscape_file = "loss_landscape_dim{0}_{1}_{2}_seed{3}.pt".format(
    num_vals, MODEL_NAME, study_name, SEED
)
loss_data = np.loadtxt("{0}/{1}/{2}".format(results_folder, study_name, landscape_file))

norm = mpl.colors.LogNorm(vmin=loss_data.min(), vmax=loss_data.max())

fig, ax = plt.subplots()
im = ax.imshow(loss_data, norm=norm, extent=[-1, 1, -1, 1])

fig.colorbar(im)

image_name = "plot_loss_landscape_dim{0}_{1}_{2}_seed{3}.png".format(
    num_vals, MODEL_NAME, study_name, SEED
)
plt.savefig("{0}/{1}/{2}".format(results_folder, study_name, image_name))
