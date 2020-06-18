import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.utils import probs_to_logits


torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(777)  # for reproducibility

# For this parametrization, fix the initial input to be one-hot encoded zero
init_state = torch.Tensor([1, 0])


# What distribution to initialize NN parameters according to
param_init_dist = torch.nn.init.normal_


# ---- These functions are all for evaluating training ----

# Create tensor representing Hilbert space


def prep_hilb_space(num_spins, input_dim):
    """
        num_spins:  int
                    number of spins in system
        input_dim:  int
                    number of values spins can take (2 for spin-1/2)

        returns:    torch.Tensor
                    Hilbert space dimension N x input_dim ** N x input_dim
    """
    hilb_dim = input_dim ** num_spins

    # Hilbert space holding encoded basis, num_spins x hilb_dim x input_dim
    hilb_space = torch.zeros(num_spins, hilb_dim, input_dim)

    for i in range(hilb_dim):
        # Convert int to spin bitstring
        bit_list = list(format(i, "b").zfill(num_spins))
        binary_state = torch.Tensor(np.array(bit_list).astype(int))

        for j in range(len(binary_state)):
            hilb_space[j, i, :] = one_hot(int(binary_state[j].item()))

    return hilb_space


# Calculate probability of each basis state


def probability(wavefunction):
    """
        wavefunction:   PositiveWaveFunction
                        a RNN trained to represent some state

        returns:        torch.Tensor
                        entries are probabilites of computational basis states
    """
    hilb_space = prep_hilb_space(wavefunction.num_spins, wavefunction.input_dim)
    hilb_dim = hilb_space.size(1)

    # Outputs are N x hilb_dim x input_dim
    nn_outputs = (
        wavefunction(hilb_space.clone(), mod_batch_size=hilb_dim).clone().detach()
    )

    # Taking dot product between one-hot encoding, then mulitply over spins
    nn_probs = torch.sum(nn_outputs * hilb_space, dim=2)
    nn_probs = torch.prod(nn_probs, 0)

    return nn_probs


# Calculate the fidelity of a reconstructed state


def fidelity(target_state, wavefunction):
    """
        target_state:   torch.Tensor
                        know state coefficients ordered in computational basis
        wavefunction:   PositiveWaveFunction
                        a RNN trained to represent some state

        returns:        float
                        the fidelity of reconstruction
    """
    nn_probs = probability(wavefunction)
    fid = torch.dot(torch.sqrt(nn_probs), torch.abs(target_state))

    return fid.item()


# Return abs. difference between target and RNN coefficients for entire basis


def prob_diff(target_state, wavefunction):
    """
        target_state:   torch.Tensor
                        know state coefficients ordered in computational basis
        wavefunction:   PositiveWaveFunction
                        a RNN trained to represent some state

        returns:        torch.Tensor
                        each entry is diff. squared between target and RNN coeff
    """
    nn_probs = probability(wavefunction)
    probability_diff = nn_probs - target_state ** 2

    return probability_diff


# Calculate the KL divergence of reconstructed state


def KL_div(target_state, wavefunction):
    """
        target_state:   torch.Tensor
                        the state to be reconstructed
        wavefunction:   PositiveWaveFunction 
                        RNN reconstruction of wavefunction

        returns:        float
                        the KL divergence of distributions
    """
    hilb_space = prep_hilb_space(wavefunction.num_spins, wavefunction.input_dim)
    hilb_dim = hilb_space.size(1)

    nn_outputs = (
        wavefunction(hilb_space.clone(), mod_batch_size=hilb_dim).clone().detach()
    )

    nn_probs = torch.sum(nn_outputs * hilb_space, dim=2)
    nn_probs = torch.prod(nn_probs, 0)

    targ_probs = torch.pow(target_state, 2)  # get probs by squaring coeffs

    div = torch.sum(targ_probs * probs_to_logits(targ_probs)) - torch.sum(
        targ_probs * probs_to_logits(nn_probs)
    )

    return div.item()


# ---- End functions for evaluating training --

# ---- The following functions are for other tasks ----


# Perform one-hot encoding of 0 and 1 in pytorch tensors


def one_hot(x, input_dim=2):
    """
        x:          int or torch.Tensor
                    integers between 0 and input_dim-1 or tensor thereof
        input_dim:  int
                    input dimension of system (2 for spin-1/2)

        returns:    torch.Tensor
                    a one-hot encoding of input integer x or tensor of integers
    """
    if type(x) == int:
        encoded = torch.zeros(input_dim)
        encoded[x] = 1
    else:
        encoded = torch.zeros((len(x), input_dim))
        for i in range(len(x)):
            encoded[i, x[i]] = 1
    return encoded


# Compute the log-probability of samples over a data set


def log_prob(nn_outputs, data):
    """
        nn_outputs:     torch.Tensor
                        forward pass, num_spins x batch_size x input_dim
        data:           torch.Tensor
                        training dataset, num_spins x batch_size

        returns:        float
                        log-probability of getting some configuration
    """
    # Index the relevant probabilties
    probs = nn_outputs.gather(2, data.unsqueeze(2).long()).squeeze(2)
    # Multiply across input dimension
    log_probs = torch.sum(torch.log(probs), dim=0)
    log_prob = -torch.mean(log_probs)
    return log_prob


# Create a class for the model


class PositiveWaveFunction(nn.Module):
    def __init__(
        self,
        num_spins,
        fixed_mag=0,
        input_dim=2,
        num_hidden=None,
        num_layers=1,
        inc_bias=True,
        batch_size=10,
        unit_cell=nn.RNNCell,
        manual_param_init=False,
    ):
        """
            num_spins:          int
                                the number of spins/qubits in the system
            fixed_mag           int
                                fixed magnetization of system (from symmetry)
            input_dim:          int
                                the number of values the spins can take on
            num_hidden:         int
                                number of hidden units
            num_layers:         int
                                number of cell layers
            inc_bias:           bool
                                whether to use bias parameters
            batch_size:         int
                                the number of samples in each training batch
            unit_cell:          torch.nn
                                building block cell, e.g. RNN or LSTM
            manual_param_init:  bool
                                whether to change default param initialization
        """
        super(PositiveWaveFunction, self).__init__()

        num_hidden = num_hidden if num_hidden else num_spins

        self.num_spins = num_spins
        self.fixed_mag = fixed_mag
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cell = unit_cell(input_dim, num_hidden, bias=inc_bias)

        self.lin_trans = nn.Linear(num_hidden, input_dim)

        self.initialize_parameters() if manual_param_init else None

    def initialize_parameters(self):
        """Initializes the NN parameters to specified distribution"""
        param_init_dist(self.cell.weight_ih, std=1 / self.num_hidden)
        param_init_dist(self.cell.weight_hh, std=1 / self.num_hidden)
        param_init_dist(self.cell.bias_ih, std=1 / self.num_hidden)
        param_init_dist(self.cell.bias_hh, std=1 / self.num_hidden)

    def init_hidden(self, mod_batch_size=None):
        """Initializes hidden units num_layers x batch_size x num_hidden"""
        batch_size = mod_batch_size if mod_batch_size else self.batch_size
        return torch.zeros((batch_size, self.num_hidden))

    # def autoreg_sample(self, num_samples):
    #     """
    #         num_samples:    int
    #                         the number of samples to generate
    #
    #         Returns:        torch.Tensor
    #                         autoregressive, num_spins x num_samples x input_dim
    #     """
    #     # Initialize initial input and hidden state
    #     inputs = init_state.repeat(num_samples, 1)
    #     self.hidden = self.init_hidden(mod_batch_size=num_samples)
    #
    #     # Initialize tensor to hold generated samples
    #     samples = torch.zeros((self.num_spins, num_samples, self.input_dim))
    #
    #     for spin_num in range(self.num_spins):
    #         for _ in range(self.num_layers):
    #             # Output is num_samples x num_hidden
    #             self.hidden = self.cell(inputs, self.hidden)
    #
    #         # Linear transformation, then softmax to output
    #         probs = F.softmax(self.lin_trans(self.hidden), dim=1)
    #         sample_dist = torch.distributions.Categorical(probs)
    #         gen_samples = sample_dist.sample()
    #         # Add samples to tensor and feed these as next inputs
    #         inputs = one_hot(gen_samples)
    #         samples[spin_num, :, :] = inputs.clone()
    #
    #     return samples

    def forward(self, data_in, mod_batch_size=None):
        """
            data_in:        torch.tensor
                            input data, shape num_spins x batch_size x input size
            mod_batch_size: int
                            used in fidelity calculation to have batch size 1

            Returns:        torch.Tensor
                            forward pass, num_spins x batch_size x input_dim
        """
        # Hack to enable fidelities and samples to be computed with forward pass
        batch_size = mod_batch_size if mod_batch_size else self.batch_size
        # Initialize hidden units
        self.hidden = self.init_hidden(mod_batch_size=mod_batch_size)

        # Keep track of the cumulative spins to enforce magnetization
        half_spins = data_in.size(0) // 2
        plus1_if_odd = data_in.size(0) % 2  # +1 to up spin threshold if odd total
        thresh_up_spins = half_spins + plus1_if_odd + self.fixed_mag // 2
        thresh_down_spins = half_spins - self.fixed_mag // 2

        cum_up_spins = torch.cumsum(data_in[:, :, 0], dim=0)
        cum_up_spins = cum_up_spins.unsqueeze(2).repeat(1, 1, 2)
        cum_down_spins = torch.cumsum(data_in[:, :, 1], dim=0)
        cum_down_spins = cum_down_spins.unsqueeze(2).repeat(1, 1, 2)

        up_prob_guaranteed = one_hot(0).unsqueeze(0).repeat(batch_size, 1)
        down_prob_guaranteed = one_hot(1).unsqueeze(0).repeat(batch_size, 1)

        # Replace first spin with fixed state, shift others "right" so that
        # networks makes predictions s_0 -> s_1, s_1 -> s_2, ... s_N-1 -> s_N
        data = data_in.clone()
        temp_data = data_in[: self.num_spins - 1, :, :].clone()
        data[0, :, :] = init_state.repeat(batch_size, 1)
        data[1:, :, :] = temp_data

        # Initialize tensor to hold NN outputs

        state_probs = torch.zeros((self.num_spins, batch_size, self.input_dim))
        corr_state_probs = torch.zeros((self.num_spins, batch_size, self.input_dim))

        for spin_num in range(self.num_spins):
            for _ in range(self.num_layers):
                # Output is batch_size x num_hidden
                self.hidden = self.cell(data[spin_num, :, :], self.hidden)

            # Linear transformation, then softmax to output
            state_probs[spin_num, :, :] = F.softmax(self.lin_trans(self.hidden), dim=1)

            if spin_num != 0:
                corr_state_probs[spin_num, :, :] = torch.where(
                    cum_up_spins[spin_num - 1, :, :] >= thresh_up_spins,
                    down_prob_guaranteed,
                    state_probs[spin_num, :, :],
                )
                corr_state_probs[spin_num, :, :] = torch.where(
                    cum_down_spins[spin_num - 1, :, :] >= thresh_down_spins,
                    up_prob_guaranteed,
                    corr_state_probs[spin_num, :, :],
                )

        corr_state_probs[0, :, :] = state_probs[0, :, :]  # first spin unchanged

        return corr_state_probs
