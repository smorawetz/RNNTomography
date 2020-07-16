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

NUM_SEEDS = 5  # the number of different seeds to average over
N_VALS = [2, 4, 6, 8, 10, 16, 20, 30, 40, 50]
LR = 0.001
MODEL_NAME = "soft_symm"
NUM_SAMPLES = 1000

# Want to compute the energy estimator of each parametrization for
# every value in N_VALS for each seed up to NUM_SEED by averaging
# over the energy of NUM_SAMPLES from the RNN

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
        study_name = "N{0}_nh100_lr{1}_ep1000".format(N, LR)
        state_name = "rnn_state_xy_{0}_{1}_seed{2}.pt".format(MODEL_NAME, study_name, seed)
        state_file = "{0}/{1}/{2}".format(results_folder, study_name, state_name)
        if os.path.exists(state_file):
            model_state = torch.load("{0}".format(state_file))
            model.load_state_dict(model_state["model_state_dict"])
            energy_val = np.loadtxt("energies/energy_N{0}.txt".format(N)).item()

            avg_energy = model_type.energy(model, energy_val, "xy", NUM_SAMPLES)
            seed_energies.append(avg_energy)

    vals = []
    vals.append(np.mean(seed_energies)) 
    vals.append(np.std(seed_energies))

    np.savetxt("energies/avg_energy_{0}_N{1}.txt".format(MODEL_NAME, N), vals)
