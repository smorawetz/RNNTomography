import sys
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim

import models.rnn_model_no_symm as rnn_model_no_symm
import models.rnn_model_hard_symm as rnn_model_hard_symm

PHYSICS_MODEL = "xy"
SYMM_TYPE = "no_symm"
MODEL_NAME = "{0}_{1}".format(PHYSICS_MODEL, SYMM_TYPE)
N = 10

NH = 100
LR = 0.001
EP = 2000
SEED = 1

torch.manual_seed(SEED)

optimizer = optim.SGD

models_dict = {"no_symm": rnn_model_no_symm, "hard_symm": rnn_model_hard_symm}
rnn_model = models_dict[SYMM_TYPE]

results_folder = "results/{0}_results".format(MODEL_NAME)
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

# checkpoint = torch.load(rnn_state_path)
# epoch = checkpoint["epoch"]
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optim_state_dict"])

curr_params = []
deltas = []
etas = []

for param in model.parameters():
    curr_params.append(param.clone().detach())
    print(param.shape)
    deltas.append(torch.randn(param.shape))
    etas.append(torch.randn(param.shape))

print(deltas[0])
print(etas[0])
