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


# Compute average energy of RNN samples


def energy(wavefunction, true_energy, model_name, num_samples=100, J=1, B=1):
    """
        wavefunction:   PositiveWaveFunction
                        trained RNN parametrization of state
        true_energy:    float
                        the true energy of the ground state
        model_name:     str
                        name of model, can be "tfim" or "xy"
        num_samples:    int
                        the number of samples over which to average energy
        J:              float
                        coupling coefficient from Hamiltonian
        B:              float
                        magnetic field strength from Hamiltonian

        returns:        float
                        the average energy of samples generated by the RNN
    """
    # One-hot autoregressive samples, num_spins x num_samples x input_dim
    samples_hot = wavefunction.autoreg_sample(num_samples)
    samples = samples_hot[:, :, 1]

    tfim_kwargs = {"J": J, "B": B}
    xy_kwargs = {"J": J}
    kwarg_dict = {"tfim": tfim_kwargs, "xy": xy_kwargs}
    model_dict = {"tfim": tfim_energy, "xy": xy_energy}

    energy_function = model_dict[model_name]
    kwargs = kwarg_dict[model_name]
    E = energy_function(wavefunction, samples_hot, samples, num_samples, **kwargs)

    return abs(E / wavefunction.num_spins - true_energy).item()


# Average energy of RNN samples for TFIM


def tfim_energy(wavefunction, samples_hot, samples, num_samples, J=1, B=1):
    """
        wavefunction:   PositiveWaveFunction
                        trained RNN parametrization of state
        samples_hot:    torch.Tensor
                        autoregressive samples generated, with shape
                        num_spins x num_samples x input_dim
        samples:        torch.Tensor
                        autoregressive samples, num_spins x num_samples
        num_samples:    int
                        the number of samples over which to average energy
        J:              float
                        coupling coefficient from Hamiltonian
        B:              float
                        magnetic field strength from Hamiltonian

        returns:        float
                        the average energy of TFIM ground state parametrization
    """
    E = 0

    samples = 1 - 2 * samples  # map 0, 1 spins to 1, -1
    # Loop over all neighbour pairs of spins (diagonal observable)
    for spin_num in range(wavefunction.num_spins - 1):
        E -= J * torch.mean(samples[spin_num, :] * samples[spin_num + 1, :])

    # Loop over all spins to calculate ratio of coeffs with that spin flipped
    for spin_num in range(wavefunction.num_spins):
        inputs = samples_hot.clone()
        inputs[spin_num, :, :] = 1 - inputs[spin_num, :, :]  # flip spin
        flipped_outs = wavefunction(inputs, mod_batch_size=num_samples)
        unflipped_outs = wavefunction(samples_hot, mod_batch_size=num_samples)

        flipped_probs = torch.sum(flipped_outs * inputs, dim=2)
        flipped_probs = torch.prod(flipped_probs, dim=0).detach()
        unflipped_probs = torch.sum(unflipped_outs * samples_hot, dim=2)
        unflipped_probs = torch.prod(unflipped_probs, dim=0).detach()

        coeff_ratios = np.sqrt(flipped_probs / unflipped_probs)
        E -= torch.mean(B * coeff_ratios)

    return E


# Average energy of RNN samples for XY


def xy_energy(wavefunction, samples_hot, samples, num_samples, J=1):
    """
        wavefunction:   PositiveWaveFunction
                        trained RNN parametrization of state
        samples_hot:    torch.Tensor
                        autoregressive samples generated, with shape
                        num_spins x num_samples x input_dim
        samples:        torch.Tensor
                        autoregressive samples, num_spins x num_samples
        num_samples:    int
                        the number of samples over which to average energy
        J:              float
                        coupling coefficient from Hamiltonian

        returns:        float
                        the average energy of TFIM ground state parametrization
    """
    E = 0

    # Loop over all spins to calculate ratio of coeffs with that spin flipped
    for spin_num in range(wavefunction.num_spins - 1):
        inputs = samples_hot.clone()
        inputs[spin_num, :, :] = 1 - inputs[spin_num, :, :]  # flip spin
        inputs[spin_num + 1, :, :] = 1 - inputs[spin_num + 1, :, :]
        flipped_outs = wavefunction(inputs, mod_batch_size=num_samples)
        unflipped_outs = wavefunction(samples_hot, mod_batch_size=num_samples)

        flipped_probs = torch.sum(flipped_outs * inputs, dim=2)
        flipped_probs = torch.prod(flipped_probs, dim=0).detach()
        unflipped_probs = torch.sum(unflipped_outs * samples_hot, dim=2)
        unflipped_probs = torch.prod(unflipped_probs, dim=0).detach()

        # If spins are same S_y S_y contribution is -1, if different it's +1
        # The S_x S_x contribution is always 1, they can add or cancel
        factor = 1 + (-1) ** (1 + samples[spin_num] + samples[spin_num + 1])

        coeff_ratios = np.sqrt(flipped_probs / unflipped_probs)
        E -= torch.mean(J * factor * coeff_ratios)

    return E


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
        hilb_space[:, i, :] = one_hot(binary_state.long())

    return hilb_space


# Calculate probability of each basis state


def probability(wavefunction, hilb_space):
    """
        wavefunction:   PositiveWaveFunction
                        a RNN trained to represent some state
        hilb_space:     torch.Tensor
                        Hilbert space, dimension N x input_dim ** N x input_dim

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


def fidelity(target_state, nn_probs):
    """
        target_state:   torch.Tensor
                        know state coefficients ordered in computational basis
        wavefunction:   PositiveWaveFunction
                        a RNN trained to represent some state
        nn_probs:       torch.Tensor
                        probabilities of each basis state, predicted by NN

        returns:        float
                        the fidelity of reconstruction
    """
    fid = torch.dot(torch.sqrt(nn_probs), torch.abs(target_state))

    return fid.item()


# Return abs. difference between target and RNN coefficients for entire basis


def prob_diff(target_state, nn_probs):
    """
        target_state:   torch.Tensor
                        know state coefficients ordered in computational basis
        wavefunction:   PositiveWaveFunction
                        a RNN trained to represent some state
        nn_probs:       torch.Tensor
                        probabilities of each basis state, predicted by NN

        returns:        torch.Tensor
                        each entry is difference between target and RNN probs
    """
    targ_probs = torch.pow(torch.abs(target_state), 2)
    probability_diff = nn_probs - targ_probs

    return probability_diff


# Calculate the KL divergence of reconstructed state


def KL_div(target_state, nn_probs):
    """
        target_state:   torch.Tensor
                        the state to be reconstructed
        wavefunction:   PositiveWaveFunction
                        RNN reconstruction of wavefunction
        nn_probs:       torch.Tensor
                        probabilities of each basis state, predicted by NN

        returns:        float
                        the KL divergence of distributions
    """
    targ_probs = torch.pow(torch.abs(target_state), 2)

    div = torch.sum(targ_probs * probs_to_logits(targ_probs)) - torch.sum(
        targ_probs * probs_to_logits(nn_probs)
    )

    return div.item()


# ---- End functions for evaluating training --

# ---- The following functions are for other tasks ----


# Perform one-hot encoding of 0 and 1 in pytorch tensors


def one_hot(inputs, input_dim=2):
    """
        inputs:     int or torch.Tensor
                    integers between 0 and input_dim-1 or tensor thereof
        input_dim:  int
                    input dimension of system (2 for spin-1/2)

        returns:    torch.Tensor
                    a one-hot encoding of inputs
    """
    if type(inputs) == int:
        encoded = torch.zeros(input_dim)
        encoded[inputs] = 1
    else:
        encoded = torch.zeros((len(inputs), input_dim))
        for i in range(len(inputs)):
            encoded[i, inputs[i]] = 1
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


# Compute n permute k for an entire array, element-wise

def n_perm_k(n, k_tensor):
    """
        n:          int
                    value to use for n in n permute k
        k_tensor:   torch.Tensor
                    tensor of values to use for k in n permute k 

        returns:    torch.Tensor
                    tensor where i-th element is n permute k_tensor[i]
    """
    perm_tensor = torch.zeros(k_tensor.size())
    for i in range(k_tensor.size(0)):
        n_fact = np.math.factorial(n)
        k_fact = np.math.factorial(max(0, k_tensor[i]))
        perm_tensor[i] = (n_fact // k_fact) * np.heaviside(k_tensor[i].item(), 1)

    return perm_tensor


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

        # Initialize attributes
        self.num_spins = num_spins
        self.fixed_mag = fixed_mag
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cell = unit_cell(input_dim, num_hidden, bias=inc_bias)

        # Keep track of the cumulative spins to enforce magnetization
        half_spins = num_spins // 2
        plus1_if_odd = num_spins % 2  # +1 to up spin threshold if odd total
        self.thresh_up = half_spins + plus1_if_odd + self.fixed_mag // 2
        self.thresh_dn = half_spins - self.fixed_mag // 2

        # Initialize FC layer
        self.lin_trans = nn.Linear(num_hidden, input_dim)

        # Parameter intialization
        self.initialize_parameters() if manual_param_init else None

    def initialize_parameters(self):
        """Initializes the NN parameters to specified distribution"""
        stdev = 1 / np.sqrt(self.num_hidden)
        param_init_dist(self.cell.weight_ih, std=stdev)
        param_init_dist(self.cell.weight_hh, std=stdev)
        param_init_dist(self.cell.bias_ih, std=stdev)
        param_init_dist(self.cell.bias_hh, std=stdev)

    def init_hidden(self, mod_batch_size=None):
        """Initializes hidden units num_layers x batch_size x num_hidden"""
        batch_size = mod_batch_size if mod_batch_size else self.batch_size
        return torch.zeros((batch_size, self.num_hidden))

    def autoreg_sample(self, num_samples):
        """
            num_samples:    int
                            the number of samples to generate
    
            Returns:        torch.Tensor
                            autoregressive samples, num_spins x num_samples x input_dim
        """
        # Initialize tensor to hold generated samples
        samples = torch.zeros((self.num_spins, num_samples, self.input_dim))

        # Initialize initial input and hidden state
        inputs = init_state.repeat(num_samples, 1)
        self.hidden = self.init_hidden(mod_batch_size=num_samples)

        # Define tensors to keep track of up and down spins
        cum_up_spins = torch.zeros(num_samples)
        cum_dn_spins = torch.zeros(num_samples)

        # These represent NN outputs which guarantee up and down spins resp.
        up_certain = one_hot(0).unsqueeze(0).repeat(num_samples, 1)
        dn_certain = one_hot(1).unsqueeze(0).repeat(num_samples, 1)

        for spin_num in range(self.num_spins):
            for _ in range(self.num_layers):
                # Output is num_samples x num_hidden
                self.hidden = self.cell(inputs, self.hidden)

            # Linear transformation, then softmax to output
            outputs = F.softmax(self.lin_trans(self.hidden), dim=1)

            # Kill probabilities if quote of up/down spins reached
            up_part = up_certain * outputs
            up_part *= np.heaviside(self.thresh_up - cum_up_spins, 0).unsqueeze(1)
            dn_part = dn_certain * outputs
            dn_part *= np.heaviside(self.thresh_dn - cum_dn_spins, 0).unsqueeze(1)
            probs = up_part + dn_part

            probs /= torch.sum(probs, dim=1).unsqueeze(1)  # renormalize

            sample_dist = torch.distributions.Categorical(probs)
            gen_samples = sample_dist.sample()

            # Add samples to tensor and feed them as next inputs
            inputs = one_hot(gen_samples)
            samples[spin_num, :, :] = inputs

            # Add generated samples to cumulative up and down spins
            cum_up_spins += 1 - gen_samples
            cum_dn_spins += gen_samples

        return samples

    def forward(self, data_in, mod_batch_size=None):
        """
            data_in:        torch.tensor
                            data, shape num_spins x batch_size x input size
            mod_batch_size: int
                            used in fidelity calculation to have batch size 1

            Returns:        torch.Tensor
                            forward pass, num_spins x batch_size x input_dim
        """
        # Hack to get fidelities and samples with forward pass
        batch_size = mod_batch_size if mod_batch_size else self.batch_size

        # Initialize hidden units
        self.hidden = self.init_hidden(mod_batch_size=mod_batch_size)

        # Define tensors to keep track of up and down spins
        cum_up_spins = torch.zeros(batch_size)
        cum_dn_spins = torch.zeros(batch_size)

        # Replace first spin with fixed state, shift others "right" so that
        # networks makes predictions s_0 -> s_1, s_1 -> s_2, ... s_N-1 -> s_N
        data = data_in.clone()
        temp_data = data_in[: self.num_spins - 1, :, :].clone()
        data[0, :, :] = init_state.repeat(batch_size, 1)
        data[1:, :, :] = temp_data

        # Initialize tensors to hold NN outputs
        probs = torch.zeros((self.num_spins, batch_size, self.input_dim))

        for spin_num in range(self.num_spins):
            for _ in range(self.num_layers):
                # Output is batch_size x num_hidden
                self.hidden = self.cell(data[spin_num, :, :], self.hidden)

            # Linear transformation, then softmax to output
            probs[spin_num, :, :] = F.softmax(self.lin_trans(self.hidden), dim=1)

            if spin_num != 0:
                # Index current spin since data already shifted "right"
                cum_up_spins += data[spin_num, :, 0]
                cum_dn_spins += data[spin_num, :, 1]
                to_alloc = self.num_spins - spin_num  # number allocated so far
                up_alloc = self.num_spins // 2 - cum_up_spins  # up spins remaining
                dn_alloc = self.num_spins // 2 - cum_dn_spins  # down spins remaining
                up_weight = torch.Tensor(n_perm_k(to_alloc, up_alloc))
                dn_weight = torch.Tensor(n_perm_k(to_alloc, dn_alloc))
                eps = 1e-12  # to handle when both weights are 0 (fidelity)
                up_weight = (up_weight + eps).unsqueeze(1)
                dn_weight = (dn_weight + eps).unsqueeze(1)
                weights = torch.cat((up_weight, dn_weight), dim=1)
                probs[spin_num, :, :] *= weights

        # Renormalize after step function kills impossible spins
        probs /= torch.sum(probs, dim=2).unsqueeze(2)

        return probs
