import os
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

import rnn_model  # local, has relevant functions

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
num_samples = 1000  # number of samples to average energy over

# Indicate whether fidelity and KL divergence will be tracked
track_fid = True

# Flag whether to make certain plots
plot_coeffs = False  # plot wavefunction coefficients at different epochs
plot_avg_grad_per_param = False  # plot avg. grad. per parameter type
plot_max_grad_per_param = False  # plot max grad. per parameter type
plot_avg_grad_vs_mag = False  # plot avg. grad. vs. sample mag. per param type
plot_sample_tree = False  # make binary tree of probs. showing sampling probs

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
    samples_path = "{0}/samples_name.txt".format(data_folder, num_spins)
    energy_path = "{0}/energy_name.txt".format(data_folder, num_spins)

    # Load in samples
    samples = torch.Tensor(np.loadtxt(samples_path))
    true_energy = np.loadtxt(energy_path).item()

    # Name chosen for this model to store data under
    model_name = "{0}_no_symm".format(data_name)

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
    )
    optimizer = optimizer(model.parameters(), lr=lr)

    num_pars = sum([1 for _ in model.parameters()])  # number of model param types
    period = max(1, num_epochs // 250)  # evaluate training every period

    # Information used to evaluate training, respectively:
    # Wavefunction coefficients at every epoch,
    # Differences between reconstruction and true basis measurement probabilities,
    # Fidelities, KL divergences, and average loss function value per epoch
    # Average and maximum gradients per parameter type at every epoch
    coeffs_array = torch.zeros((num_epochs // period + 1, input_dim ** num_spins))
    diffs_array = torch.zeros((num_epochs // period + 1, input_dim ** num_spins))
    fidelity_array = np.zeros(num_epochs // period + 1)
    div_array = np.zeros(num_epochs // period + 1)
    energy_array = np.zeros(num_epochs // period + 1)
    loss_array = np.zeros(num_epochs // period)
    if plot_avg_grad_per_param:
        avg_grads_array = torch.zeros((num_epochs // period, num_pars))
    if plot_max_grad_per_param:
        max_grads_array = torch.zeros((num_epochs // period, num_pars))

    # These track how the gradients depend on the magnetization of data samples
    if plot_avg_grad_vs_mag:
        mags_array = torch.zeros((num_epochs // period, samples.size(1)))
        mag_grads_array = torch.zeros((num_epochs // period, samples.size(1), num_pars))

    # Fill first row of relevant arrays with initial values
    nn_probs = rnn_model.probability(model, hilb_space)

    fidelity_array[0] = rnn_model.fidelity(true_state, nn_probs)
    div_array[0] = rnn_model.KL_div(true_state, nn_probs)
    energy_array[0] = rnn_model.energy(model, true_energy, data_name, num_samples)
    coeffs_array[0, :] = torch.sqrt(nn_probs)
    diffs_array[0, :] = rnn_model.prob_diff(true_state, nn_probs)

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

            avg_loss += log_prob

            log_prob.backward()  # backward pass

            # Track the gradients
            for par_num, par in enumerate(model.parameters()):
                avg_grads[par_num] += par.grad.abs().mean()
                max_grads[par_num] += par.grad.abs().mean()

            # Tracking magnetizations effect on gradients
            if plot_avg_grad_vs_mag:
                mags_array[
                    epoch - 1, batch_start : batch_start + batch_size
                ] = torch.sum(1 - 2 * batch, dim=0)
                for par_num, par in enumerate(model.parameters()):
                    mag_grads_array[
                        epoch - 1, batch_start : batch_start + batch_size, par_num
                    ] = par.grad.abs().mean()

            optimizer.step()  # update parameters

        if epoch % period == 0:
            nn_probs = rnn_model.probability(model, hilb_space)
            fid = rnn_model.fidelity(true_state, nn_probs)
            div = rnn_model.KL_div(true_state, nn_probs)
            energy = rnn_model.energy(model, true_energy, data_name, num_samples)

            print("Epoch: ", epoch)
            print("Fidelity: ", fid)
            print("KL div: ", div)
            print("Abs. energy diff: ", energy)

            # Keep track of relevant training information
            index = epoch // period
            samples_per_batch = samples.size(1) // batch_size
            fidelity_array[index] = fid
            div_array[index] = div
            energy_array[index] = energy
            diffs_array[index, :] = rnn_model.prob_diff(true_state, nn_probs)
            coeffs_array[index, :] = torch.sqrt(nn_probs)
            loss_array[index - 1] = avg_loss / samples_per_batch
            if plot_avg_grad_per_param:
                avg_grads_array[index - 1, :] = avg_grads / samples_per_batch
            if plot_max_grad_per_param:
                max_grads_array[index - 1, :] = max_grads / samples_per_batch

    # ---- Saving important training info and parametrized NN state ----

    # Make folder to store outputs if it does not already exist
    results_path = "replace_with_path_to_results"
    study_path = "{0}/N{1}_nh{2}_lr{3}_ep{4}".format(
        results_path, model_name, num_spins, num_hidden, lr, num_epochs
    )

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(study_path):
        os.makedirs(study_path)

    store_array = np.zeros((num_epochs // period + 1, 5))
    store_array[:, 0] = np.arange(0, num_epochs + 1, period)
    store_array[:, 1] = fidelity_array
    store_array[:, 2] = div_array
    store_array[:, 3] = energy_array
    store_array[1:, 4] = loss_array

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

    # ---- All of the following plots are done here to avoid saving huge data ----

    epochs = np.arange(num_epochs // period)  # for use in plotting
    num_epochs_shown = 5  # in plots where multiple epochs plotted, how many to plot

    # Coefficients of reconstructed state

    if plot_coeffs:
        fig, ax = plt.subplots()
        for epoch in range(0, num_epochs + 1, num_epochs // num_epochs_shown):
            ax.plot(
                np.arange(input_dim ** num_spins),
                coeffs_array[epoch // period, :],
                "o",
                markersize=2.5,
                label="Epoch: {0}".format(epoch),
            )
        ax.legend()
        ax.set_title(
            r"Reconstruction basis state coefficieints for {0}".format(
                data_name.upper()
            )
        )
        ax.set_xlabel(r"Basis state b")
        ax.set_ylabel(r"$b_{model}$")

        plt.savefig(
            "{0}/coeffs_{1}_N{2}_nh{3}_lr{4}_ep{5}.png".format(
                study_path, model_name, num_spins, num_hidden, lr, num_epochs
            )
        )

    # ---- These plots look at different types of model parameters

    # Define parameter names for use in parameter-dependent plots
    param_names = [
        "input-hidden_weights",
        "hidden-hidden_weights",
        "input-hidden_bias",
        "hidden-hidden_bias",
        "output_weights",
        "output_bias",
    ]

    # Make plots of average gradients for each parameter

    if plot_avg_grad_per_param:
        fig, ax = plt.subplots()
        for par_num in range(num_pars):
            ax.plot(
                epochs,
                avg_grads_array[:, par_num],
                label="{0}".format(param_names[par_num]),
            )
        ax.legend()
        ax.set_title(r"Average absolute gradients per parameter every epoch")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Average absolute parameter gradient")

        plt.savefig(
            "{0}/avg_grad_{1}_N{2}_nh{3}_lr{4}_ep{5}.png".format(
                study_path, model_name, num_spins, num_hidden, lr, num_epochs
            )
        )

    # Make plots of max gradients for each parameter

    if plot_max_grad_per_param:
        fig, ax = plt.subplots()
        for par_num in range(num_pars):
            ax.plot(
                epochs,
                max_grads_array[:, par_num],
                label="{0}".format(param_names[par_num]),
            )
        ax.legend()
        ax.set_title(r"Max absolute gradients per parameter every epoch")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Max absolute parameter gradient")

        plt.savefig(
            "{0}/max_grad_{1}_N{2}_nh{3}_lr{4}_ep{5}.png".format(
                study_path, model_name, num_spins, num_hidden, lr, num_epochs
            )
        )

    # Scatterplot: mean gradient vs magnetization for each parameter, varying epoch

    # First sort the gradients by magnetization
    if plot_avg_grad_vs_mag:
        unique_mags = torch.unique(mags_array, dim=1)
        unique_mags_grads = torch.zeros((num_epochs, unique_mags.size(1), num_pars))
        zeros = torch.zeros(unique_mags_grads.size())

    # Average over all samples of a particular magnetization
    if plot_avg_grad_vs_mag:
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

    if plot_avg_grad_vs_mag:
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
                study_path
                + "/mag_grad_param_{0}_{1}_N{2}_nh{3}_lr{4}_ep{5}.png".format(
                    model_name,
                    num_spins,
                    num_hidden,
                    lr,
                    num_epochs,
                    param_names[par_num],
                    param_names[par_num],
                )
            )

    # Makes a binary tree showing the probabilities during autoregressive sampling
    if plot_sample_tree:
        for epoch in range(0, num_epochs + 1, num_epochs // num_epochs_shown):
            wavefunc = coeffs_array[epoch // period, :].numpy()
            build_tree(
                num_spins,
                wavefunc,
                "{0}/tree_probs_{1}_epoch{2}_N{3}_nh{4}_lr{5}_ep{6}".format(
                    study_path, model_name, epoch, num_spins, num_hidden, lr, num_epochs
                ),
            )
