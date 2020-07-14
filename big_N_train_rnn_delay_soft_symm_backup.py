import os
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

import rnn_model_delay_soft_symm as rnn_model  # local, has relevant functions

# Define physical parameters
input_dim = 2  # values inputs can take, e.g. 2 for spin-1/2
fixed_mag = 0  # value of total magnetization to enforce during training

# Define NN parameters
num_layers = 3  # number of stacked unit cells
inc_bias = True  # include bias in activation function
unit_cell = nn.GRUCell  # basic cell of NN (e.g. RNN, LSTM, etc.)

# Define training parameters
batch_size = 50  # size of mini_batches of data

# Define training evaluation parameters
num_samples = 100  # number of samples to average energy over

# Define numerical parameters
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)

# Information about where to find data
data_folder = "replace_with_path_to_data"


def run_training(
    data_name,
    num_spins,
    num_hidden,
    lr,
    num_epochs,
    optimizer,
    impose_symm_ep,
    impose_symm_type,
):
    """
        data_name:          str
                            name under which data is stored, can be 'tfim' or 'xy'
        num_spins:          int
                            the number of spins in the system
        num_hidden:         int
                            the number of hidden units in the RNN parametrization
        lr:                 float
                            the learning rate
        num_epochs:         int
                            the number of epochs to train for
        optimizer:          torch.optim
                            the type of optimizer to use in training
        impose_symm_ep:     int
                            the epoch at which to begin imposing symmetry
        impose_symm_type:   str
                            one of "no_symm_first" or "symm_first"
    """
    # Find data according to file naming structure
    samples_path = "{0}/samples_name.txt".format(data_folder, num_spins)
    energy_path = "{0}/energy_name.txt".format(data_folder, num_spins)

    # Load in samples
    samples = torch.Tensor(np.loadtxt(samples_path))
    true_energy = np.loadtxt(energy_path).item()

    # Name chosen for this model to store data under
    model_name = "{0}_delay_soft_symm_{1}".format(data_name, impose_symm_type)

    # Make folder to store outputs if it does not already exist
    results_path = "../results/{0}_results".format(model_name)
    study_path = "{0}/N{1}_nh{2}_lr{3}_ep{4}_symm_ep{5}".format(
        results_path, num_spins, num_hidden, lr, num_epochs, impose_symm_ep
    )

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(study_path):
        os.makedirs(study_path)

    # Define names for training data and results
    training_results_name = "{0}/training_results_rnn_{1}_N{2}_nh{3}_lr{4}_ep{5}.txt".format(
        study_path, model_name, num_spins, num_hidden, lr, num_epochs
    )
    training_model_name = "{0}/rnn_state_{1}_N{2}_nh{3}_lr{4}_ep{5}.pt".format(
        study_path, model_name, num_spins, num_hidden, lr, num_epochs
    )

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
        fixed_mag=fixed_mag,
        input_dim=input_dim,
        num_hidden=num_hidden,
        num_layers=num_layers,
        inc_bias=inc_bias,
        batch_size=batch_size,
        unit_cell=unit_cell,
        manual_param_init=True,
    )
    optimizer = optimizer(model.parameters(), lr=lr)

    if os.path.isfile(training_model_name):
        checkpoint = torch.load(training_model_name)
        init_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
    else:
        init_epoch = 1

    period = 25  # evaluate training every period

    # Add initial value of energy estimator to file
    if init_epoch == 1:
        init_energy = rnn_model.energy(
            model, true_energy, data_name, num_samples, False
        )
        training_file = open(training_results_name, "w")
        training_file.write("{0} {1} {2}".format(0, init_energy, 0))
        training_file.write("\n")
        training_file.close()

    # Do training
    for epoch in range(init_epoch, num_epochs + 1):

        # Split data into batches and then loop over all batches
        permutation = torch.randperm(samples.size(1))
        samples_hot = samples_hot[:, permutation, :]
        samples = samples[:, permutation]

        avg_loss = 0  # for tracking loss function

        if impose_symm_type == "no_symm_first":
            impose_symm = epoch >= impose_symm_ep
        elif impose_symm_type == "symm_first":
            impose_symm = epoch < impose_symm_ep

        for batch_start in range(0, samples.size(1), batch_size):
            batch_hot = samples_hot[:, batch_start : batch_start + batch_size, :]
            batch = samples[:, batch_start : batch_start + batch_size]

            optimizer.zero_grad()  # clear gradients

            nn_outputs = model(batch_hot, impose_symm=impose_symm)  # forward pass

            # Compute log-probability to use as cost function
            log_prob = rnn_model.log_prob(nn_outputs, batch)

            log_prob.backward()  # backward pass
            optimizer.step()  # update parameters

            avg_loss += log_prob.detach().item()

        # print("Epoch: ", epoch)
        if epoch % period == 0:
            energy = rnn_model.energy(
                model, true_energy, data_name, num_samples, impose_symm
            )
            samples_per_batch = samples.size(1) // batch_size
            avg_loss /= samples_per_batch

            # print("Abs. energy diff: ", energy)
            # print("Loss function value: ", avg_loss)

            # Write training info and data to files
            training_file = open(training_results_name, "a")
            training_file.write("{0} {1} {2}".format(epoch, energy, avg_loss))
            training_file.write("\n")
            training_file.close()

        # Save the relevant data at every epoch for checkpointing
        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }
        torch.save(save_dict, training_model_name)
