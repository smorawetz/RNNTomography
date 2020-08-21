import torch
import numpy as np
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

optimizer = optim.SGD

models_dict = {"no_symm": rnn_model_no_symm, "hard_symm": rnn_model_hard_symm}
rnn_model = models_dict[SYMM_TYPE]

results_folder = "{0}_results".format(MODEL_NAME)
study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(N, NH, LR, EP)
rnn_state = "rnn_state_{0}_{1}_seed{2}.pt".format(MODEL_NAME, study_name, SEED)

rnn_state_path = "{0}/{1}/{2}".format(results_folder, study_name, rnn_state)

kwargs_dict = {
    "input_dim": 2,
    "num_hidden": NH,
    "num_layers": 1,
    "inc_bias": True,
    "batch_size": 20000,
    "unit_cell": nn.GRUCell,
    "manual_param_init": True,
}

if SYMM_TYPE == "hard_symm":
    kwargs_dict["fixed_mag"] = 0

model = rnn_model.PositiveWaveFunction(N, **kwargs_dict)
optimizer = optimizer(model.parameters(), lr=LR)

checkpoint = torch.load(rnn_state_path)
epoch = checkpoint["epoch"]
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optim_state_dict"])

curr_params = []
deltas = []
etas = []

for param in model.parameters():
    curr_params.append(param.clone().detach())
    deltas.append(torch.randn(param.shape))
    etas.append(torch.randn(param.shape))


# Now define the intervals which alpha, beta will take
max_change = 0.25
alpha_min, alpha_max = -max_change, max_change
beta_min, beta_max = -max_change, max_change
num_vals = 101

alpha_vals = np.linspace(alpha_min, alpha_max, num_vals)
beta_vals = np.linspace(beta_min, beta_max, num_vals)

loss_vals = np.zeros((num_vals, num_vals))

data_names = {"tfim": "TFIM", "xy": "XYModel"}
data_path = "../Datasets/{0}/samples_N{1}.txt".format(data_names[PHYSICS_MODEL], N)

samples = torch.Tensor(np.loadtxt(data_path))

# Apply one-hot encoding to data
samples_hot = samples.unsqueeze(2).repeat(1, 1, 2)  # expand dim
for i in range(len(samples)):
    samples_hot[i, :, :] = rnn_model.one_hot(samples[i, :].long())  # encode

# Current dimension is batch x N x input, swap first two for RNN input
samples = samples.permute(1, 0)
samples_hot = samples_hot.permute(1, 0, 2)

prev_nn_outputs = torch.zeros(samples_hot.shape)

for i in range(num_vals):  # alpha
    alpha = alpha_vals[i]
    for j in range(num_vals):  # beta
        beta = beta_vals[j]
        print(r"alpha = {0}, beta = {1}".format(alpha, beta))
        for param_num, param in enumerate(model.parameters()):
            curr_param = curr_params[param_num]
            delta = deltas[param_num]
            eta = etas[param_num]
            # if param_num == 5:
            # print(param)
            param.data = curr_param + alpha * delta + beta * eta
            # if param_num == 5:
            # print(param)
        nn_outputs = model(samples_hot)
        # if torch.all(torch.eq(nn_outputs, prev_nn_outputs)):
        # print("same as before!")
        # prev_nn_outputs = nn_outputs.clone()
        loss = rnn_model.log_prob(nn_outputs, samples)
        loss_vals[i, j] = loss

landscape_file = "loss_landscape_dim{0}_{1}_{2}_seed{3}.pt".format(
    num_vals, MODEL_NAME, study_name, SEED
)
np.savetxt("{0}/{1}/{2}".format(results_folder, study_name, landscape_file), loss_vals)
