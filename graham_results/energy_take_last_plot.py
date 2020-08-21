import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.optim as optim

import models.rnn_model_no_symm as rnn_model_no_symm
import models.rnn_model_hard_symm as rnn_model_hard_symm
import models.rnn_model_soft_symm as rnn_model_soft_symm
import models.rnn_model_delay_hard_symm as rnn_model_delay_hard_symm
import models.rnn_model_delay_soft_symm as rnn_model_delay_soft_symm

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
    "xy_delay_hard_symm_symm_first": rnn_model_delay_hard_symm,
    "xy_delay_hard_symm_no_symm_first": rnn_model_delay_hard_symm,
    "xy_delay_soft_symm_symm_first": rnn_model_delay_soft_symm,
    "xy_delay_soft_symm_no_symm_first": rnn_model_delay_soft_symm,
}

NUM_SEEDS = 5  # the number of different seeds to average over
N_VALS = [2, 4, 6, 8, 10, 16, 20, 30, 40, 50]
NH = 100
LR = 0.001
EP = 2000
NUM_SAMPLES = 10000
MODEL_NAMES = ["xy_no_symm", "xy_hard_symm"]

# Want to compute the energy estimator of each parametrization for
# every value in N_VALS for each seed up to NUM_SEED by averaging
# over the energy of NUM_SAMPLES from the RNN

energy_avgs_dict = {}
energy_stdevs_dict = {}

for model_name in MODEL_NAMES:
    E_avgs = []
    E_stds = []
    for N in N_VALS:
        energy = 0
        stdev = 0
        for seed in range(1, NUM_SEEDS + 1):
            results_folder = "{0}_results".format(model_name)
            study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(N, NH, LR, EP)
            data_file = "training_results_rnn_{0}_{1}_seed{2}.txt".format(
                model_name, study_name, seed
            )
            data_path = "{0}/{1}/{2}".format(results_folder, study_name, data_file)
            data = np.loadtxt(data_path)
            seed_energy = data[-1, 3]
            seed_stdev = data[-1, 5]
            energy += seed_energy
            stdev += seed_stdev
        energy /= NUM_SEEDS
        stdev /= NUM_SEEDS
        E_avgs.append(energy)
        E_stds.append(stdev)
    energy_avgs_dict[model_name] = E_avgs
    energy_stdevs_dict[model_name] = E_stds

# Now make plot

colours = ["C0", "C3"]
formats = ["o", "^"]
# capstyles = ["round", "projecting"]
legend_dict = {
    "xy_no_symm": "No symmetry",
    "xy_hard_symm": "Symmetry imposed",
}

fig, ax = plt.subplots(figsize=(5, 3.5))
for i in range(len(MODEL_NAMES)):
    model_name = MODEL_NAMES[i]
    energy_avgs = energy_avgs_dict[model_name]
    energy_stdevs = energy_stdevs_dict[model_name]
    label = legend_dict[model_name]
    ax.errorbar(  # main plot with errorbars
        N_VALS,
        energy_avgs,
        yerr=energy_stdevs,
        fmt=formats[i],
        label=label,
        capsize=5,
        color=colours[i],
        markeredgecolor="black",
        # solid_capstyle=capstyles[i],
    )
    ax.ticklabel_format(style="sci", scilimits=(-3, 3), axis="y")
    # ax.plot(N_VALS, np.poly1d(np.polyfit(N_VALS, energy_avgs, 1))(N_VALS), linestyle="dashed", color=colour)
ax.legend(frameon=False)
# ax.set_title(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$ for various system sizes")
# ax.set_ylabel(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$")
ax.set_ylabel(r"$\varepsilon$")
ax.set_xlabel(r"$N$")

plt.tight_layout()

plt.savefig("compare_results/all_compare/energy_vs_N.pdf".format(LR))
