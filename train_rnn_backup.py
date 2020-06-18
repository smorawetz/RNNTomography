# ------------------- IN THIS VERSION -------------------
# ---- OPTION TO MANUAL INITIALIZE WAVEFUNC PARAMS ------
# ---- CHECKING HOW MAGNETIZATION AFFECTS GRADIENTS -----

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import rnn_model  # local, has relevant functions
from make_tree import build_tree  # locally defined to make tree of probs.

import torch.nn as nn
import torch.optim as optim

from matplotlib.ticker import MaxNLocator

# Define physical parameters
num_spins = 10  # number of spins/qubits
input_dim = 2  # values inputs can take, e.g. 2 for spin-1/2

# Define NN parameters

num_hidden = 100  # size of the RNN hidden unit vector
num_layers = 3  # number of stacked unit cells
inc_bias = True  # include bias in activation function
unit_cell = nn.GRUCell  # basic cell of NN (e.g. RNN, LSTM, etc.)
wide_init = False  # whether or not to initialize parameters from N(0,1)

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)

# Define training parameters

batch_size = 50  # size of mini_batches of data
num_epochs = 1000  # number of epochs of training to perform
optimizer = optim.SGD  # what optimizer to use
# optimizer = optim.Adadelta
lr = 0.001  # learning rate
# lr = 1


# Information about where to find data
data_name = "xy"
samples_path = "../../../generating_data/data_generated/N{0}_{1}_samples.txt".format(
    num_spins, data_name
)
state_path = "../../../generating_data/data_generated/N{0}_{1}_groundstate.txt".format(
    num_spins, data_name
)

# If data is for one spin, add second dimension to array
samples = torch.Tensor(np.loadtxt(samples_path))
if len(samples.size()) == 1:
    samples = samples.unsqueeze(1)

# The first column is the real part of the ground_state
true_state = torch.Tensor(np.loadtxt(state_path)[:, 0])

# Name chosen for this model to store data under
wide_suffix = "_wide_init" if wide_init else ""
model_name = "{0}".format(data_name, wide_suffix)


# Apply one-hot encoding to sample data
samples_hot = samples.unsqueeze(2).repeat(1, 1, 2)  # additional encoding dimension
for i in range(len(samples)):
    for j in range(num_spins):
        samples_hot[i, j, :] = rnn_model.one_hot(int(samples[i, j]))


# Current dimension is batch x N x input, swap first two for RNN input
samples_hot = samples_hot.permute(1, 0, 2)
samples = samples.permute(1, 0)


# Instantiate stuff
hilb_space = rnn_model.prep_hilb_space(num_spins, input_dim)
model = rnn_model.PositiveWaveFunction(
    num_spins,
    input_dim=input_dim,
    num_hidden=num_hidden,
    num_layers=num_layers,
    inc_bias=inc_bias,
    batch_size=batch_size,
    unit_cell=unit_cell,
    manual_param_init=True,
    large_init_stdev=wide_init,
)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optimizer(model.parameters(), lr=lr)

num_pars = sum([1 for _ in model.parameters()])

# Information used to evaluate training, respectively:
# Wavefunction coefficients at every epoch,
# Differences between reconstruction and true basis measurement probabilities,
# Fidelities, KL divergences, and average loss function value per epoch
coeffs_array = torch.zeros((num_epochs + 1, input_dim ** num_spins))
diffs_array = torch.zeros((num_epochs + 1, input_dim ** num_spins))
fidelity_array = np.zeros(num_epochs + 1)
div_array = np.zeros(num_epochs + 1)
loss_array = np.zeros(num_epochs)
avg_grads_array = torch.zeros((num_epochs, num_pars))
max_grads_array = torch.zeros((num_epochs, num_pars))

# These track how the gradients depend on the magnetization of data samples
mags_array = torch.zeros((num_epochs, samples.size(1)))
mag_grads_array = torch.zeros((num_epochs, samples.size(1), num_pars))

# Fill first row of relevant arrays with initial values
nn_probs = rnn_model.probability(model, hilb_space)
fidelity_array[0] = rnn_model.fidelity(true_state, model, nn_probs)
div_array[0] = rnn_model.KL_div(true_state, model, nn_probs)
coeffs_array[0, :] = torch.sqrt(nn_probs)
diffs_array[0, :] = rnn_model.prob_diff(true_state, model, nn_probs)


# Do training
for epoch in range(1, num_epochs + 1):

    # NOTE: These are used to keep track of time
    # forward_time = 0
    # backward_time = 0
    # fid_time = 0
    # div_time = 0
    # probdiff_time = 0
    # coeffs_time = 0

    # Split data into batches and then loop over all batches
    permutation = torch.randperm(samples.size(1))
    samples_hot = samples_hot[:, permutation, :]
    samples = samples[:, permutation]

    avg_loss = 0  # used to track average value of loss function
    avg_grads = torch.zeros(num_pars)
    max_grads = torch.zeros(num_pars)

    for batch_start in range(0, samples.size(1), batch_size):
        batch_hot = samples_hot[:, batch_start : batch_start + batch_size, :]
        batch = samples[:, batch_start : batch_start + batch_size]

        optimizer.zero_grad()  # clear gradients

        # pre_forward_time = time.time()

        nn_outputs = model(batch_hot)  # forward pass

        # post_forward_time = time.time()
        # forward_time += post_forward_time - pre_forward_time

        # Compute log-probability to use as cost function
        log_prob = rnn_model.log_prob(nn_outputs, batch)

        # Looking at this for training evaluation purposes
        avg_loss += log_prob

        log_prob.backward()  # backward pass

        # post_backward_time = time.time()
        # backward_time += post_backward_time - post_forward_time

        # This exists to track the gradients
        for par_num, par in enumerate(model.parameters()):
            avg_grads[par_num] += par.grad.abs().mean()
            max_grads[par_num] += par.grad.abs().mean()

        # For tracking magnetizations
        mags_array[epoch - 1, batch_start : batch_start + batch_size] = torch.sum(
            1 - 2 * batch, dim=0
        )
        for par_num, par in enumerate(model.parameters()):
            mag_grads_array[
                epoch - 1, batch_start : batch_start + batch_size, par_num
            ] = par.grad.abs().mean()

        optimizer.step()  # update gradients

    # pre_fid_time = time.time()

    nn_probs = rnn_model.probability(model, hilb_space)
    fid = rnn_model.fidelity(true_state, model, nn_probs)

    # post_fid_time = time.time()
    # fid_time += post_fid_time - pre_fid_time

    div = rnn_model.KL_div(true_state, model, nn_probs)

    # post_div_time = time.time()
    # div_time += post_div_time - post_fid_time

    print("Epoch: ", epoch)
    print("Fidelity: ", fid)
    print("KL div: ", div)

    # Keep track of relevant training information
    fidelity_array[epoch] = fid
    div_array[epoch] = div
    # pre_probdiff_time = time.time()
    diffs_array[epoch, :] = rnn_model.prob_diff(true_state, model, nn_probs)
    # post_probdiff_time = time.time()
    coeffs_array[epoch, :] = torch.sqrt(nn_probs)
    # post_coeffs_time = time.time()
    loss_array[epoch - 1] = avg_loss / (samples.size(1) // batch_size)
    avg_grads_array[epoch - 1, :] = avg_grads / (samples.size(1) // batch_size)
    max_grads_array[epoch - 1, :] = max_grads / (samples.size(1) // batch_size)

    # probdiff_time += post_probdiff_time - pre_probdiff_time 
    # coeffs_time += post_coeffs_time - post_probdiff_time

    # Print out timing
    # print(
    #     "This epoch, we have\nForward time: ",
    #     forward_time,
    #     "\nBackward time: ",
    #     backward_time,
    #     "\nFidelity time: ",
    #     fid_time,
    #     "\nKL div time: ",
    #     div_time,
    #     "\nProb. diffs time: ",
    #     probdiff_time,
    #     "\nCoefficients time: ",
    #     coeffs_time,
    # )


# Make folder to store outputs if it does not already exist
if not os.path.exists("../results/{0}_results".format(model_name)):
    os.makedirs("../results/{0}_results".format(model_name))

np.savetxt(
    "../results/{0}_results/training_results_rnn_{0}_N{1}_nh{2}_lr{3}_ep{4}.txt".format(
        model_name, num_spins, num_hidden, lr, num_epochs
    ),
    fidelity_array,
)


# Create array to use in plotting
epochs = np.arange(1, num_epochs + 1)

# Make two subplots of fidelity and KL div. vs epoch
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 3))

ax = axs[0]
ax.plot(
    np.arange(num_epochs + 1), fidelity_array, "o", color="C0", markeredgecolor="black"
)
ax.set_title(r"Reconstruction Fidelity for {0}".format(data_name.upper()))
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Fidelity")
# ax.set_ylim(0.0, 1.0)

ax = axs[1]
ax.plot(np.arange(num_epochs + 1), div_array, "o", color="C1", markeredgecolor="black")
ax.set_title(r"Reconstruction KL divergence for {0}".format(data_name.upper()))
ax.set_ylabel(r"KL Divergence")
ax.set_xlabel(r"Epoch")

plt.tight_layout()

plt.savefig(
    "../results/{0}_results/fid_KL_rnn_{0}_N{1}_nh{2}_lr{3}_ep{4}.png".format(
        model_name, num_spins, num_hidden, lr, num_epochs
    )
)


period = num_epochs // 5  # record coeffs and prob diffs at every *period* epochs

# Reconstructed and target measurement probabilities difference

fig, ax = plt.subplots()
for epoch in range(0, num_epochs + 1, period):
    ax.plot(
        np.arange(input_dim ** num_spins),
        diffs_array[epoch, :],
        "o",
        markersize=2.5,
        label="Epoch: {0}".format(epoch),
    )
ax.legend()
ax.set_title(
    r"Reconstruction measurement probabilities comparison for {0}".format(
        data_name.upper()
    )
)
ax.set_xlabel(r"Basis state b")
ax.set_ylabel(r"$P_{model}(b) - P_{true}(b)$")

plt.savefig(
    "../results/{0}_results/prob_diffs_{0}_N{1}_nh{2}_lr{3}_ep{4}.png".format(
        model_name, num_spins, num_hidden, lr, num_epochs
    )
)

# Coefficients of reconstructed state

fig, ax = plt.subplots()
for epoch in range(0, num_epochs + 1, period):
    ax.plot(
        np.arange(input_dim ** num_spins),
        coeffs_array[epoch, :],
        "o",
        markersize=2.5,
        label="Epoch: {0}".format(epoch),
    )
ax.legend()
ax.set_title(
    r"Reconstruction basis state coefficieints for {0}".format(data_name.upper())
)
ax.set_xlabel(r"Basis state b")
ax.set_ylabel(r"$b_{model}$")

plt.savefig(
    "../results/{0}_results/coeffs_{0}_N{1}_nh{2}_lr{3}_ep{4}.png".format(
        model_name, num_spins, num_hidden, lr, num_epochs
    )
)


# Plot of average loss function value per epoch

fig, ax = plt.subplots()
ax.plot(epochs, loss_array, "o")
ax.set_title(r"Loss function for training {0}".format(data_name.upper()))
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Loss")

plt.savefig(
    "../results/{0}_results/loss_{0}_N{1}_nh{2}_lr{3}_ep{4}.png".format(
        model_name, num_spins, num_hidden, lr, num_epochs
    )
)


param_names = [
    "input-hidden_weights",
    "hidden-hidden_weights",
    "input-hidden_bias",
    "hidden-hidden_bias",
    "output_weights",
    "output_bias",
]

# Make plots of average gradients for each parameter

fig, ax = plt.subplots()
for par_num in range(num_pars):
    ax.plot(
        epochs, avg_grads_array[:, par_num], label="{0}".format(param_names[par_num])
    )
ax.legend()
ax.set_title(r"Average absolute gradients per parameter every epoch")
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Average absolute parameter gradient")

plt.savefig(
    "../results/{0}_results/avg_grad_{0}_N{1}_nh{2}_lr{3}_ep{4}.png".format(
        model_name, num_spins, num_hidden, lr, num_epochs
    )
)


# Make plots of max gradients for each parameter

fig, ax = plt.subplots()
for par_num in range(num_pars):
    ax.plot(
        epochs, max_grads_array[:, par_num], label="{0}".format(param_names[par_num]),
    )
ax.legend()
ax.set_title(r"Max absolute gradients per parameter every epoch")
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Max absolute parameter gradient")

plt.savefig(
    "../results/{0}_results/max_grad_{0}_N{1}_nh{2}_lr{3}_ep{4}.png".format(
        model_name, num_spins, num_hidden, lr, num_epochs
    )
)


# Make scatterplot of mean gradient vs magnetization for each parameter at different epochs

# Want to sort the grads by magnetization

unique_mags = torch.unique(mags_array, dim=1)
unique_mags_grads = torch.zeros((num_epochs, unique_mags.size(1), num_pars))
zeros = torch.zeros(unique_mags_grads.size())

for epoch in range(num_epochs):
    for par_num in range(num_pars):
        for mag_num in range(unique_mags.size(1)):
            temp_tens = torch.where(
                mags_array[epoch, :] == unique_mags[epoch, mag_num],
                mag_grads_array[epoch, :, par_num],
                zeros[epoch, :, par_num],
            )
            unique_mags_grads[epoch, mag_num, par_num] = torch.sum(
                temp_tens
            ) / torch.nonzero(temp_tens).size(0)


for par_num in range(num_pars):  # loop over parameters
    fig, ax = plt.subplots()
    for epoch in range(0, num_epochs + 1, period):  # loop over epochs
        epoch = 1 if epoch == 0 else epoch
        ax.plot(
            unique_mags[epoch - 1, :],
            unique_mags_grads[epoch - 1, :, par_num],
            "o",
            markersize=3,
            label="Epoch: {0}".format(epoch),
        )
    ax.legend()
    ax.set_title(
        r"Avg. abs. grads for {0} vs magnetization".format(param_names[par_num])
    )
    ax.set_xlabel(r"Magnetization")
    ax.set_ylabel(r"Average absolute parameter gradient")

    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    plt.savefig(
        "../results/{0}_results/mag_grad_param_{1}_{0}_N{2}_nh{3}_lr{4}_ep{5}.png".format(
            model_name, param_names[par_num], num_spins, num_hidden, lr, num_epochs
        )
    )


# Make probability tree

for epoch in range(0, num_epochs + 1, period):
    wavefunc = coeffs_array[epoch, :].numpy()
    build_tree(
        num_spins,
        wavefunc,
        "../results/{0}_results/tree_probs_{0}_ep{1}_N{2}_nh{3}_lr{4}".format(
            model_name, epoch, num_spins, num_hidden, lr
        ),
    )
