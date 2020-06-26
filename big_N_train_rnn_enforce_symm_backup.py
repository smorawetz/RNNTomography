import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from matplotlib.ticker import MaxNLocator

import rnn_model_enforce_symm as rnn_model  # local, has relevant functions

# Define physical parameters
# num_spins = 4  # number of spins/qubits -- NOTE: Outdated
input_dim = 2  # values inputs can take, e.g. 2 for spin-1/2

# Define NN parameters
# num_hidden = 100  # size of the RNN hidden unit vector -- NOTE: Outdated
num_layers = 3  # number of stacked unit cells
inc_bias = True  # include bias in activation function
unit_cell = nn.GRUCell  # basic cell of NN (e.g. RNN, LSTM, etc.)

# Define training parameters
batch_size = 50  # size of mini_batches of data
# num_epochs = 250  # number of epochs of training to perform -- NOTE: Outdated
# optimizer = optim.SGD  # what optimizer to use -- NOTE: Parameter
# lr = 0.001  # learning rate -- NOTE: Outdated

# Define training evaluation parameters
num_samples = 100  # number of samples to average energy over

# Define numerical parameters
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)

# Information about where to find data
data_folder = "replace_with_path_to_data"


def run_training(data_name, num_spins, num_hidden, lr, num_epochs, optimizer):
    """
        data_name:  str
                    name under which data is stored, can be 'tfim' or 'xy'
        num_spins:  int
                    the number of spins in the system
        num_hidden: int
                    the number of hidden units in the RNN parametrization
        lr:         float
                    the learning rate
        num_epochs: int
                    the number of epochs to train for
        optimizer:  torch.optim
                    the type of optimizer to use in training
    """
    # Find data according to file naming structure
    samples_path = "{0}/samples_name.txt".format(data_folder, num_spins, data_name)
    energy_path = "{0}/energy_name.txt".format(data_folder, num_spins, data_name)

    # Load in samples
    samples = torch.Tensor(np.loadtxt(samples_path))
    true_energy = np.loadtxt(energy_path).item()

    # Name chosen for this model to store data under
    model_name = "{0}".format(data_name)

    # Apply one-hot encoding to sample data
    samples_hot = samples.unsqueeze(2).repeat(1, 1, 2)  # encoding dimension
    for i in range(len(samples)):
        samples_hot[i, :, :] = rnn_model.one_hot(samples[i, :].long())

    # Current dimension is batch x N x input, swap first two for RNN input
    samples_hot = samples_hot.permute(1, 0, 2)
    samples = samples.permute(1, 0)

    # Create Hilbert space, instantiate model and optimizer
    model = rnn_model.PositiveWaveFunction(
        num_spins,
        input_dim=input_dim,
        num_hidden=num_hidden,
        num_layers=num_layers,
        inc_bias=inc_bias,
        batch_size=batch_size,
        unit_cell=unit_cell,
        manual_param_init=True,
    )
    optimizer = optimizer(model.parameters(), lr=lr)

    num_pars = sum([1 for _ in model.parameters()])  # number of model param types
    period = 1  # evaluate training every period

    # Arrays to store energy estimator and avg. loss function values
    energy_array = np.zeros(num_epochs // period + 1)
    loss_array = np.zeros(num_epochs // period)

    # Add initial value of energy estimator to array
    energy_array[0] = rnn_model.energy(model, true_energy, data_name, num_samples)

    # Do training
    for epoch in range(1, num_epochs + 1):

        # Split data into batches and then loop over all batches
        permutation = torch.randperm(samples.size(1))
        samples_hot = samples_hot[:, permutation, :]
        samples = samples[:, permutation]

        avg_loss = 0  # for tracking loss function
        avg_grads = torch.zeros(num_pars)
        max_grads = torch.zeros(num_pars)

        for batch_start in range(0, samples.size(1), batch_size):
            batch_hot = samples_hot[:, batch_start : batch_start + batch_size, :]
            batch = samples[:, batch_start : batch_start + batch_size]

            optimizer.zero_grad()  # clear gradients

            nn_outputs = model(batch_hot)  # forward pass

            # Compute log-probability to use as cost function
            log_prob = rnn_model.log_prob(nn_outputs, batch)

            log_prob.backward()  # backward pass
            optimizer.step()  # update parameters

            avg_loss += log_prob.detach()

        if epoch % period == 0:
            energy = rnn_model.energy(model, true_energy, data_name, num_samples)

            print("Epoch: ", epoch)
            print("Abs. energy diff: ", energy)

            index = epoch // period
            samples_per_batch = samples.size(1) // batch_size
            energy_array[index] = energy
            loss_array[index - 1] = avg_loss / samples_per_batch

    # ---- Saving important training info and parametrized NN state ----

    # Make folder to store outputs if it does not already exist
    results_path = "replace_with_path_to_results"
    study_path = "{0}/N{1}_nh{2}_lr{3}_ep{4}".format(
        results_path, num_spins, num_hidden, lr, num_epochs
    )

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(study_path):
        os.makedirs(study_path)

    store_array = np.zeros((num_epochs // period + 1, 3))
    store_array[:, 0] = np.arange(0, num_epochs + 1, period)
    store_array[:, 1] = energy_array
    store_array[1:, 2] = loss_array

    np.savetxt(
        "{0}/training_results_rnn_{1}_N{2}_nh{3}_lr{4}_ep{5}.txt".format(
            study_path, model_name, num_spins, num_hidden, lr, num_epochs
        ),
        store_array,
    )

    # Save NN state
    model_data = model.state_dict()
    torch.save(
        model_data,
        "{0}/rnn_state_{1}_N{2}_nh{3}_lr{4}_ep{5}.pt".format(
            study_path, model_name, num_spins, num_hidden, lr, num_epochs
        ),
    )
