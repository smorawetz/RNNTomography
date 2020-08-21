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
# N_VALS = [2, 4, 6, 8, 10, 16, 20, 30, 40, 50]
N_VALS = [16, 20, 30, 40, 50]
LR = 0.001
NUM_SAMPLES = 10000
MODEL_NAMES = ["no_symm"]

NUM_AT_ONCE = 1000  # number of samples to draw at a time, must divide NUM_SAMPLES

# Want to compute the energy estimator of each parametrization for
# every value in N_VALS for each seed up to NUM_SEED by averaging
# over the energy of NUM_SAMPLES from the RNN

energy_avgs_dict = {}
energy_stdevs_dict = {}

if CALCULATE_AVGS:
    for model_name in MODEL_NAMES:
        energy_avgs = []
        energy_stdevs = []
        for N in N_VALS:
            seed_energies = []
            seed_stdevs = []
            for seed in range(1, NUM_SEEDS + 1):
                model_type = model_names_dict[model_name]
                model = model_type.PositiveWaveFunction(
                    N, num_hidden=100, num_layers=3, inc_bias=True, unit_cell=nn.GRUCell
                )
                results_folder = "xy_{0}_results".format(model_name)
                study_name = "N{0}_nh100_lr{1}_ep1000".format(N, LR)
                state_name = "rnn_state_xy_{0}_{1}_seed{2}.pt".format(
                    model_name, study_name, seed
                )
                state_file = "{0}/{1}/{2}".format(
                    results_folder, study_name, state_name
                )
                if os.path.exists(state_file):
                    model_state = torch.load("{0}".format(state_file))
                    model.load_state_dict(model_state["model_state_dict"])
                    energy_val = np.loadtxt("energies/energy_N{0}.txt".format(N)).item()

                    if False:
                    # if NUM_SAMPLES > NUM_AT_ONCE:
                        E_list = []
                        print(N)
                        for i in range(NUM_SAMPLES // NUM_AT_ONCE):
                            print(i)
                            energies = model_type.energy(
                                model, energy_val, "xy", NUM_SAMPLES, return_list=True
                            )
                            energies = energies.detach().numpy()
                            E_list.append(energies)
                        for i in range(NUM_SAMPLES // NUM_AT_ONCE - 1):
                            E_list[i + 1] = np.append(E_list[i], E_list[i + 1])
                        avg = abs(np.mean(E_list[-1]))
                        std = np.std(E_list[-1])
                        std /= np.sqrt(NUM_SAMPLES - 1)

                    else:
                        avg, std = model_type.energy(
                            model, energy_val, "xy", NUM_SAMPLES
                        )

                    seed_energies.append(avg)
                    seed_stdevs.append(std)

            vals = []
            energy_avg = np.mean(seed_energies)
            energy_stdev = np.mean(seed_stdevs)
            vals.append(energy_avg)
            vals.append(energy_stdev)
            energy_avgs.append(energy_avg)
            energy_stdevs.append(energy_stdev)

            np.savetxt(
                "energies/avg_energy_{0}_N{1}_lr{2}.txt".format(model_name, N, LR), vals
            )

        energy_avgs_dict[model_name] = energy_avgs.copy()
        energy_stdevs_dict[model_name] = energy_stdevs.copy()


if not CALCULATE_AVGS:
    for model_name in MODEL_NAMES:
        energy_avgs = []
        energy_stdevs = []
        for N in N_VALS:
            vals = np.loadtxt(
                "energies/avg_energy_{0}_N{1}_lr{2}.txt".format(model_name, N, LR)
            )
            energy_avgs.append(vals[0])  # avg is stored first
            energy_stdevs.append(vals[1])  # stdev is stored second
        energy_avgs_dict[model_name] = energy_avgs.copy()
        energy_stdevs_dict[model_name] = energy_stdevs.copy()


# Now make plot

colours = ["C0", "C3"]
formats = ["o", "^"]
legend_dict = {
    "no_symm": "No symmetry",
    "hard_symm": "Symmetry imposed",
}

fig, ax = plt.subplots(figsize=(14, 5))
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
    )
    # ax.plot(N_VALS, np.poly1d(np.polyfit(N_VALS, energy_avgs, 1))(N_VALS), linestyle="dashed", color=colour)
ax.legend()
# ax.set_title(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$ for various system sizes")
ax.set_ylabel(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$")
ax.set_xlabel(r"$N$")

plt.tight_layout()

plt.savefig("compare_results/all_compare/energy_vs_N_lr{0}.png".format(LR))
