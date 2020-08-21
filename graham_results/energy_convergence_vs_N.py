import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.optim as optim

import models.rnn_model_no_symm as rnn_model_no_symm
import models.rnn_model_hard_symm as rnn_model_hard_symm
import models.rnn_model_soft_symm as rnn_model_soft_symm

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

model_names_dict = {  # assocating file naming with model type
    "xy_no_symm": rnn_model_no_symm,
    "xy_hard_symm": rnn_model_hard_symm,
    "xy_soft_symm": rnn_model_soft_symm,
}

NUM_SEEDS = 5  # the number of different seeds to average over
N_VALS = [2, 4, 6, 8, 10, 16, 20, 30, 40, 50]
NH = 100
LR = 0.001
EP = 2000
NUM_SAMPLES = 10000
MODEL_NAMES = ["xy_no_symm", "xy_hard_symm"]

TAKE_LAST = 5  # take this last number of energies to compute 'converged' energy
MARGIN = 0.10  # margin to come within converged error to count as that

# Want to compute the energy estimator of each parametrization for
# every value in N_VALS for each seed up to NUM_SEED by averaging
# over the energy of NUM_SAMPLES from the RNN

# Now make plot

colours = ["C0", "C3"]
formats = ["o", "^"]
# capstyles = ["round", "projecting"]
legend_dict = {
    "xy_no_symm": "No symmetry",
    "xy_hard_symm": "Symmetry imposed",
}

# First want to determine, the average energy of the last several samples
epochs_dict = {}
for model_name in MODEL_NAMES:
    results_folder = "{0}_results".format(model_name)
    N_epochs = []
    for N in N_VALS:
        study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(N, NH, LR, EP)
        vals_file = (
            "{0}/{1}/training_results_rnn_{2}_N{3}_nh{4}_lr{5}_ep{6}_seed1.txt".format(
                results_folder, study_name, model_name, N, NH, LR, EP
            )
        )
        epochs = np.loadtxt(vals_file)[:, 0]
        energies = np.loadtxt(vals_file)[:, 3]
        conv_energy = np.mean(energies[-TAKE_LAST:])
        first_index = np.argwhere(energies < (1 + MARGIN) * conv_energy)[0].item()
        N_epochs.append(epochs[first_index])
    epochs_dict[model_name] = N_epochs

fig, ax = plt.subplots(figsize=(5, 3.5))
for i in range(len(MODEL_NAMES)):
    model_name = MODEL_NAMES[i]
    epoch_data = epochs_dict[model_name]
    label = legend_dict[model_name]
    ax.plot(N_VALS, epoch_data, formats[i], color=colours[i], label=label)
ax.legend(frameon=False)
# ax.set_title(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$ for various system sizes")
# ax.set_ylabel(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$")
ax.set_ylabel(r"Convergence epoch")
ax.set_xlabel(r"$N$")

plt.tight_layout()

plt.savefig("compare_results/all_compare/energy_convergence_vs_N.pdf".format(LR))
