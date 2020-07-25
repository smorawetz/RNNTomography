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

CALCULATE_AVGS = True  # whether or not to calculate or just make plots

NUM_SEEDS = 5  # the number of different seeds to average over
N_VALS = [2, 4, 6, 8, 10, 16, 20, 30, 40, 50]
LR = 0.1
MODEL_NAMES = [
    "delay_hard_symm_symm_first",
    "delay_hard_symm_no_symm_first",
    "delay_soft_symm_symm_first",
    "delay_soft_symm_no_symm_first",
]
MODEL_NAME = MODEL_NAMES[0]
NUM_SAMPLES = 1000

SYMM_EPS = [50, 100, 250, 500]
SYMM_TYPES = ["no_symm_first", "symm_first"]

# Want to compute the energy estimator of each parametrization for
# every value in N_VALS for each seed up to NUM_SEED by averaging
# over the energy of NUM_SAMPLES from the RNN

energy_avgs_dict = {}
energy_stdevs_dict = {}

if CALCULATE_AVGS:
    for symm_ep in SYMM_EPS:
        energy_avgs = []
        energy_stdevs = []
        for N in N_VALS:
            seed_energies = []
            for seed in range(1, NUM_SEEDS + 1):
                model_type = model_names_dict[MODEL_NAME]
                model = model_type.PositiveWaveFunction(
                    N, num_hidden=100, num_layers=3, inc_bias=True, unit_cell=nn.GRUCell
                )
                results_folder = "xy_{0}_results".format(MODEL_NAME)
                study_name = "N{0}_nh100_lr{1}_ep1000_symm_ep{2}".format(N, LR, symm_ep)
                state_name = "rnn_state_xy_{0}_{1}_seed{2}.pt".format(
                    MODEL_NAME, study_name, seed
                )
                state_file = "{0}/{1}/{2}".format(
                    results_folder, study_name, state_name
                )
                if os.path.exists(state_file):
                    model_state = torch.load("{0}".format(state_file))
                    model.load_state_dict(model_state["model_state_dict"])
                    energy_val = np.loadtxt("energies/energy_N{0}.txt".format(N)).item()

                    impose_symm = "symm_no_symm" in MODEL_NAME
                    avg_energy = model_type.energy(
                        model, energy_val, "xy", NUM_SAMPLES, impose_symm
                    )
                    seed_energies.append(avg_energy)

            vals = []
            energy_avg = np.mean(seed_energies)
            energy_stdev = np.std(seed_energies)
            vals.append(energy_avg)
            vals.append(energy_stdev)
            energy_avgs.append(energy_avg)
            energy_stdevs.append(energy_stdev)

            np.savetxt(
                "energies/avg_energy_{0}_N{1}_lr{2}.txt".format(MODEL_NAME, N, LR), vals
            )

        energy_avgs_dict[symm_ep] = energy_avgs.copy()
        energy_stdevs_dict[symm_ep] = energy_stdevs.copy()


if not CALCULATE_AVGS:
    for symm_ep in SYMM_EPS:
        energy_avgs = []
        energy_stdevs = []
        for N in N_VALS:
            vals = np.loadtxt(
                "energies/avg_energy_{0}_N{1}_lr{2}.txt".format(model_name, N, LR)
            )
            energy_avgs.append(vals[0])  # avg is stored first
            energy_stdevs.append(vals[1])  # stdev is stored second
        energy_avgs_dict[symm_ep] = energy_avgs.copy()
        energy_stdevs_dict[symm_ep] = energy_stdevs.copy()

# Now make plot

colours = ["C0", "C1", "C2", "C3"]
legend_dict = {
    "delay_hard_symm_symm_first": "'Hard' symmetry first",
    "delay_hard_symm_no_symm_first": "'Hard' symmetry later'",
    "delay_soft_symm_symm_first": "'Soft' symmetry first",
    "delay_soft_symm_no_symm_first": "'Soft' symmetry later",
}

fig, ax = plt.subplots(figsize=(14, 5))
for i in range(len(SYMM_EPS)):
    symm_ep = SYMM_EPS[i]
    energy_avgs = energy_avgs_dict[symm_ep]
    energy_stdevs = energy_stdevs_dict[symm_ep]
    label = "Impose epoch: {0}".format(symm_ep)
    colour = colours[i]
    ax.errorbar(  # main plot with errorbars
        N_VALS,
        energy_avgs,
        yerr=energy_stdevs,
        fmt="o",
        label=label,
        capsize=5,
        color=colour,
    )
    # ax.plot(N_VALS, np.poly1d(np.polyfit(N_VALS, energy_avgs, 1))(N_VALS), linestyle="dashed", color=colour)
ax.legend()
title = legend_dict[MODEL_NAME]
ax.set_title(r"$\frac{{|E_{{RNN}} - E_{{DMRG}}|}}{{N}}$ for {0}".format(title))
ax.set_ylabel(r"$\frac{|E_{{RNN}} - E_{{DMRG}}|}{N}$")
ax.set_xlabel(r"$N$")

plt.tight_layout()

plt.savefig(
    "compare_results/all_compare/{0}_energy_vs_N_lr{1}.png".format(MODEL_NAME, LR)
)
