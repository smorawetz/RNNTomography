import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.utils import probs_to_logits


torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(777)  # for reproducibility

# For this parametrization, fix the initial input to be one-hot encoded zero
init_state = torch.Tensor([1, 0])


# Perform one-hot encoding of 0 and 1 in pytorch tensors


def one_hot(x, input_dim=2):
    """
        x:          int
                    between 0 and input_dim-1
        input_dim:  int
                    input dimension of system (2 for spin-1/2)

        returns:    torch.Tensor
                    a one-hot encoding of input integer x
    """
    encoded = torch.zeros(input_dim)
    encoded[x] = 1
    return encoded


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

    # print("nn_outputs is", nn_outputs)
    # print("hilb_space is", hilb_space)
    # print("hilb_space size is ", hilb_space.size())

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


# Create a class for the model


class PositiveWaveFunction(nn.Module):
    def __init__(
        self,
        num_spins,
        input_dim=2,
        num_hidden=None,
        num_layers=1,
        inc_bias=True,
        batch_size=10,
        unit_cell=nn.RNN,
    ):
        """
            num_spins:  int
                        the number of spins/qubits in the system
            input_dim:  int
                        the number of values the spins can take on
            num_hidden: int
                        number of hidden units
            num_layers: int
                        number of cell layers
            inc_bias:   bool
                        whether to use bias parameters
            batch_size: int
                        the number of samples in each training batch
            unit_cell:  torch.nn
                        building block cell, e.g. RNN or LSTM
        """
        super(PositiveWaveFunction, self).__init__()

        num_hidden = num_hidden if num_hidden else num_spins

        self.num_spins = num_spins
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cell = unit_cell(
            input_dim, num_hidden, num_layers=num_layers, bias=inc_bias
        )

        self.lin_trans = nn.Linear(num_hidden, input_dim)

    def init_hidden(self, mod_batch_size=None):
        """Initializes hidden units num_layers x batch_size x num_hidden"""
        batch_size = mod_batch_size if mod_batch_size else self.batch_size
        return torch.zeros((self.num_layers, batch_size, self.num_hidden))

    def forward(self, data, mod_batch_size=None):
        """
            data:           torch.tensor
                            input data, shape num_spins x batch_size x input size
            mod_batch_size: int
                            used in fidelity calculation to have batch size 1

            Returns:        torch.Tensor
                            forward pass, num_spins x batch_size x input_dim
        """
        batch_size = mod_batch_size if mod_batch_size else self.batch_size
        self.hidden = self.init_hidden(mod_batch_size=mod_batch_size)

        # Replace first spin with fixed state, shift others "right" so that
        # networks makes predictions s_0 -> s_1, s_1 -> s_2, ... s_N-1 -> s_N
        temp_data = data[: self.num_spins - 1, :, :].clone()
        data[0, :, :] = init_state.repeat(batch_size, 1)
        data[1:, :, :] = temp_data

        # Outputs is dimension num_spins x batch_size x hidden_size
        outputs, self.hidden = self.cell(data, self.hidden)
        # Linear transformation, then softmax to output
        state_probs = F.softmax(self.lin_trans(outputs), dim=2)

        return state_probs
