import os
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

import rnn_model_hard_symm as rnn_model  # local, has relevant functions

# Define physical parameters
input_dim = 2  # values inputs can take, e.g. 2 for spin-1/2
fixed_mag = 0  # enforced total magnetization of samples

# Define NN parameters
num_layers = 3  # number of stacked unit cells
inc_bias = True  # include bias in activation function
unit_cell = nn.GRUCell  # basic cell of NN (e.g. RNN, LSTM, etc.)

# Define training parameters
batch_size = 50  # size of mini_batches of data
period1 = 5  # evaluate training every period
period2 = 25  # period after some time as passed
period_crossover = 50  # epoch after which to use period2

# Define training evaluation parameters
num_samples = 1000  # number of samples to average energy over

# Define numerical parameters
torch.set_default_tensor_type(torch.DoubleTensor)

# Information about where to find data
data_folder = "replace_with_path_to_data"


def run_training(
    data_name, num_spins, num_hidden, lr, num_epochs, optimizer, track_fid, seed
):
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
    track_fid:  bool
                whether or not to keep track of fidelity and KL divergence
    seed:       int
                seed to use for RNG (for reproducibility)
    """
    torch.manual_seed(seed)

    # Find data according to file naming structure
    samples_path = "{0}/samples_N{1}.txt".format(data_folder, num_spins)
    energy_path = "{0}/energy_N{1}.txt".format(data_folder, num_spins)
    if track_fid:
        state_path = "{0}/psi_N{1}.txt".format(data_folder, num_spins)

    # Load in samples
    samples = torch.Tensor(np.loadtxt(samples_path))
    true_energy = np.loadtxt(energy_path).item()
    if track_fid:
        true_state = torch.Tensor(np.loadtxt(state_path)[:, 0])

    # Name chosen for this model to store data under
    model_name = "{0}_long_hard_symm".format(data_name)

    # Make folder to store outputs if it does not already exist
    results_path = "../results/{0}_results".format(model_name)
    study_path = "{0}/N{1}_nh{2}_lr{3}_ep{4}".format(
        results_path, num_spins, num_hidden, lr, num_epochs
    )

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(study_path):
        os.makedirs(study_path)

    # Define names for training data and results
    training_results_name = (
        "{0}/training_results_rnn_{1}_N{2}_nh{3}_lr{4}_ep{5}_seed{6}.txt".format(
            study_path, model_name, num_spins, num_hidden, lr, num_epochs, seed
        )
    )
    training_model_name = "{0}/rnn_state_{1}_N{2}_nh{3}_lr{4}_ep{5}_seed{6}.pt".format(
        study_path, model_name, num_spins, num_hidden, lr, num_epochs, seed
    )

    # Apply one-hot encoding to sample data
    samples_hot = samples.unsqueeze(2).repeat(1, 1, 2)  # encoding dimension
    for i in range(len(samples)):
        samples_hot[i, :, :] = rnn_model.one_hot(samples[i, :].long())

    # Current dimension is batch x N x input, swap first two for RNN input
    samples_hot = samples_hot.permute(1, 0, 2)
    samples = samples.permute(1, 0)

    # Create Hilbert space, instantiate model and optimizer
    if track_fid:
        hilb_space = rnn_model.prep_hilb_space(num_spins, input_dim)

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

    # Add initial value of evaluators to file
    if init_epoch == 1:
        if track_fid:
            init_nn_probs = rnn_model.probability(model, hilb_space)
            init_fid = rnn_model.fidelity(true_state, init_nn_probs)
            init_div = rnn_model.KL_div(true_state, init_nn_probs)
        init_energy, energy_errors = rnn_model.energy(
            model, true_energy, data_name, num_samples
        )
        training_file = open(training_results_name, "w")

        if track_fid:
            training_file.write(
                "{0} {1} {2} {3} {4} {5}".format(
                    0, init_fid, init_div, init_energy, 0, energy_errors
                )
            )
        else:
            training_file.write(
                "{0} {1} {2} {3} {4} {5}".format(0, 0, 0, init_energy, 0, energy_errors)
            )
        training_file.write("\n")
        training_file.close()

    # Do training
    for epoch in range(init_epoch, num_epochs + 1):

        # Split data into batches and then loop over all batches
        permutation = torch.randperm(samples.size(1))
        samples_hot = samples_hot[:, permutation, :]
        samples = samples[:, permutation]

        avg_loss = 0  # for tracking loss function

        for batch_start in range(0, samples.size(1), batch_size):
            batch_hot = samples_hot[:, batch_start : batch_start + batch_size, :]
            batch = samples[:, batch_start : batch_start + batch_size]

            optimizer.zero_grad()  # clear gradients

            nn_outputs = model(batch_hot)  # forward pass

            # Compute log-probability to use as cost function
            log_prob = rnn_model.log_prob(nn_outputs, batch)

            log_prob.backward()  # backward pass
            optimizer.step()  # update parameters

            avg_loss += log_prob.detach().item()

        print("Epoch: ", epoch)
        if (epoch % period1 == 0 and epoch <= period_crossover) or epoch % period2 == 0:
            if track_fid:
                nn_probs = rnn_model.probability(model, hilb_space)
                fid = rnn_model.fidelity(true_state, nn_probs)
                div = rnn_model.KL_div(true_state, nn_probs)

            energy, energy_errors = rnn_model.energy(
                model, true_energy, data_name, num_samples
            )
            samples_per_batch = samples.size(1) // batch_size
            avg_loss /= samples_per_batch

            if track_fid:
                print("Fidelity: ", fid)
                print("KL div: ", div)
            print("Abs. energy diff: ", energy)
            print("Loss function value: ", avg_loss)

            # Write training info and data to files
            training_file = open(training_results_name, "a")
            if track_fid:
                training_file.write(
                    "{0} {1} {2} {3} {4} {5}".format(
                        epoch, fid, div, energy, avg_loss, energy_errors
                    )
                )
            else:
                training_file.write(
                    "{0} {1} {2} {3} {4} {5}".format(
                        epoch, 0, 0, energy, avg_loss, energy_errors
                    )
                )
            training_file.write("\n")
            training_file.close()

        # Save the relevant data at every epoch for checkpointing
        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }
        torch.save(save_dict, training_model_name)
