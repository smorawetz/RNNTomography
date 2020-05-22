import numpy as np
import torch
import os
import matplotlib.pyplot as plt

import rnn_model  # local, has relevant functions

import torch.nn as nn
import torch.optim as optim

# Define physical parameters
num_spins = 10  # number of spins/qubits
input_dim = 2  # values inputs can take, e.g. 2 for spin-1/2

# Define NN parameters

num_hidden = 100  # size of the RNN hidden unit vector
num_layers = 3  # number of stacked unit cells
inc_bias = True  # include bias in activation function
unit_cell = nn.GRU  # basic cell of NN (e.g. RNN, LSTM, etc.)
optimizer = optim.SGD  # what optimizer to use

torch.set_default_tensor_type(torch.DoubleTensor)

# Define training parameters

batch_size = 50  # size of mini_batches of data
num_epochs = 100  # number of epochs of training to perform
lr = 0.001  # learning rate


# Information about where to find data
data_name = "data_source_placeholder"
samples_path = "../../generating_data/data_generated/N{0}_{1}_samples.txt".format(
    num_spins, data_name
)
state_path = "../../generating_data/data_generated/N{0}_{1}_groundstate.txt".format(
    num_spins, data_name
)

# If data is for one spin, add second dimension to array
samples = torch.Tensor(np.loadtxt(samples_path))
if len(samples.size()) == 1:
    samples = samples.unsqueeze(1)

# The first column is the real part of the ground_state
true_state = torch.Tensor(np.loadtxt(state_path)[:, 0])

# Name chosen for this model to store data under
model_name = "output_name_placeholder"


# Apply one-hot encoding to sample data
samples_hot = samples.unsqueeze(2).repeat(1, 1, 2)  # additional encoding dimension
for i in range(len(samples)):
    for j in range(num_spins):
        samples_hot[i, j, :] = rnn_model.one_hot(int(samples[i, j]))


# Current dimension is batch x N x input, swap first two for RNN input
samples_hot = samples_hot.permute(1, 0, 2)
samples = samples.permute(1, 0)


# Instantiate stuff
model = rnn_model.PositiveWaveFunction(
    num_spins,
    input_dim=input_dim,
    num_hidden=num_hidden,
    num_layers=num_layers,
    inc_bias=inc_bias,
    batch_size=batch_size,
    unit_cell=unit_cell,
)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optimizer(model.parameters(), lr=lr)


# Information used to evaluate training, respectively:
# Wavefunction coefficients at every epoch
# Differences between reconstruction and true basis measurement probabilities
# Fidelities, KL divergences, and average loss function value per epoch
coeffs_array = torch.zeros((num_epochs + 1, input_dim ** num_spins))
diffs_array = torch.zeros((num_epochs + 1, input_dim ** num_spins))
fidelity_array = np.zeros(num_epochs + 1)
div_array = np.zeros(num_epochs + 1)
loss_array = np.zeros(num_epochs)

# Fill first row of these with initial values
fidelity_array[0] = rnn_model.fidelity(true_state, model)
div_array[0] = rnn_model.KL_div(true_state, model)
coeffs_array[0, :] = torch.sqrt(rnn_model.probability(model))
diffs_array[0, :] = rnn_model.prob_diff(true_state, model)


# Do training
for epoch in range(1, num_epochs + 1):
    # Split data into batches and then loop over all batches
    permutation = torch.randperm(samples.size(1))
    samples_hot = samples_hot[:, permutation, :]
    samples = samples[:, permutation]

    avg_loss = 0  # used to track average value of loss function

    for batch_start in range(0, samples.size(1), batch_size):
        batch_hot = samples_hot[:, batch_start : batch_start + batch_size, :]
        batch = samples[:, batch_start : batch_start + batch_size]

        optimizer.zero_grad()  # clear gradients

        batch_probs = model(batch_hot)  # forward pass

        # reorder tensors since CrossEntropyLoss(a,b) takes
        # a: batch_size x output_size x num_spins
        # b: batch_size x num_spins
        batch_probs = batch_probs.permute(1, 2, 0)
        batch = batch.permute(1, 0).long()

        # NOTE: The subsequent comment is only relevant for the torch entropy
        # Index this way since want cross-entropy between prob and next spin
        # loss = loss_function(batch_probs[:, :, : num_spins - 1], batch[:, 1:])
        cross_ent = 0
        for sample_num in range(batch_size):
            # loss += loss_function(batch_probs[:, :, spin], batch[:, spin + 1])
            for spin in range(num_spins - 1):
                cross_ent -= torch.log(
                    batch_probs[sample_num, batch[sample_num, spin + 1], spin]
                )

        cross_ent /= batch_size

        # Looking at this for training evaluation purposes
        avg_loss += cross_ent

        # loss.backward()  # backward pass
        cross_ent.backward()
        optimizer.step()  # update gradients

    fid = rnn_model.fidelity(true_state, model)
    div = rnn_model.KL_div(true_state, model)
    print("Epoch: ", epoch)
    print("Fidelity: ", fid)
    print("KL div: ", div)

    # Keep track of relevant training information
    fidelity_array[epoch] = fid
    div_array[epoch] = div
    diffs_array[epoch, :] = rnn_model.prob_diff(true_state, model)
    coeffs_array[epoch, :] = torch.sqrt(rnn_model.probability(model))
    loss_array[epoch - 1] = avg_loss / samples.size(1) // batch_size


# Make folder to store outputs if it does not already exist
if not os.path.exists("../results/{0}_results".format(model_name)):
    os.makedirs("../results/{0}_results".format(model_name)) 

np.savetxt(
    "../results/{0}_results/training_results_rnn_{0}_N{1}_nh{2}.txt".format(
        model_name, num_spins, num_hidden
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
    "../results/{0}_results/fid_KL_rnn_{0}_N{1}_nh{2}.pdf".format(
        model_name, num_spins, num_hidden
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
    "../results/{0}_results/prob_diffs_{0}_N{1}_nh{2}.pdf".format(
        model_name, num_spins, num_hidden
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
    "../results/{0}_results/coeffs_{0}_N{1}_nh{2}.pdf".format(
        model_name, num_spins, num_hidden
    )
)


# Plot of average loss function value per epoch

fig, ax = plt.subplots()
ax.plot(epochs, loss_array, "o")
ax.set_title(r"Loss function for training {0}".format(data_name.upper()))
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Loss")

plt.savefig(
    "../results/{0}_results/loss_{0}_N{1}_nh{2}.pdf".format(
        model_name, num_spins, num_hidden
    )
)
