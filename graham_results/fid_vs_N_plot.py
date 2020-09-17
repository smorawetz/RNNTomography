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

model_names_dict = {  # assocating file naming with model type
    "no_symm": rnn_model_no_symm,
    "hard_symm": rnn_model_hard_symm,
    "soft_symm": rnn_model_soft_symm,
    "delay_hard_symm_symm_first": rnn_model_delay_hard_symm,
    "delay_hard_symm_no_symm_first": rnn_model_delay_hard_symm,
    "delay_soft_symm_symm_first": rnn_model_delay_soft_symm,
    "delay_soft_symm_no_symm_first": rnn_model_delay_soft_symm,
}

CALCULATE_AVGS = False  # whether or not to calculate or just make plots

NUM_SEEDS = 5  # the number of different seeds to average over
N_VALS = [2, 4, 6, 8, 10]
LR = 0.001
MODEL_NAMES = ["no_symm", "hard_symm", "soft_symm"]
NUM_SAMPLES = 1000

# Want to compute the energy estimator of each parametrization for
# every value in N_VALS for each seed up to NUM_SEED by averaging
# over the energy of NUM_SAMPLES from the RNN

fid_avgs_dict = {}
fid_stdevs_dict = {}

for model_name in MODEL_NAMES:
    fid_avgs = []
    fid_stdevs = []
    for N in N_VALS:
        seed_fids = []
        for seed in range(1, NUM_SEEDS + 1):
            results_folder = "xy_{0}_results".format(model_name)
            study_folder = "N{0}_nh100_lr{1}_ep1000".format(N, LR)
            data_file = "training_results_rnn_xy_{0}_{1}_seed{2}.txt".format(
                model_name, study_folder, seed
            )
            data = np.loadtxt(
                "{0}/{1}/{2}".format(results_folder, study_folder, data_file)
            )
            fid = data[-1, 1]
            seed_fids.append(fid)

        fid_avgs.append(np.mean(seed_fids))
        fid_stdevs.append(np.std(seed_fids))

    fid_avgs_dict[model_name] = fid_avgs.copy()
    fid_stdevs_dict[model_name] = fid_stdevs.copy()


# Get all the same info for fidelities


# Now make plot

colours = ["C0", "C2", "C3"]
legend_dict = {
    "no_symm": "No symmetry",
    "hard_symm": "'Hard' symmetry",
    "soft_symm": "'Soft' symmetry",
}

fig, ax = plt.subplots(figsize=(12, 5))
for i in range(len(MODEL_NAMES)):
    model_name = MODEL_NAMES[i]
    fid_avgs = fid_avgs_dict[model_name]
    fid_stdevs = fid_stdevs_dict[model_name]
    label = legend_dict[model_name]
    colour = colours[i]
    ax.errorbar(  # main plot with errorbars
        N_VALS,
        1 - np.array(fid_avgs),
        yerr=fid_stdevs,
        fmt="o",
        label=label,
        capsize=5,
        color=colour,
    )
    ax.set_yscale("log")
    # ax.plot(N_VALS, np.poly1d(np.polyfit(energy_avgs, fid_avgs, 1))(energy_avgs), linestyle="dashed", color=colour)
ax.legend()
ax.set_title(r"Fidelity vs average energy of samples generated at convergence")
ax.set_ylabel(r"Fidelity")
ax.set_xlabel(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$")

plt.tight_layout()

plt.savefig("compare_results/all_compare/fid_vs_N_lr{0}.png".format(LR))
