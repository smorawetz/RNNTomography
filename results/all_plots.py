import os
import numpy as np
import itertools

from plotting.plot_energy import plot_energy
from plotting.plot_energy_log import plot_energy_log
from plotting.plot_energy_vs_fid import plot_energy_vs_fid
from plotting.plot_log_energy_vs_fid import plot_log_energy_vs_fid
from plotting.plot_fid import plot_fid
from plotting.plot_loss import plot_loss
from plotting.plot_fid_error_log import plot_fid_error_log

# ---- Define parameters related to which studies to make plots for ----
# ---- where each is a list of possible values that some parameter -----
# ---- can take on, which will be looped over when producing plots ----- 


studies = ["xy", "xy_enforce_symm", "xy_late_symm", "xy_soft_symm"]
spin_nums = [2, 4, 6, 8, 10, 12]  # spin numbers to make plots for
num_hiddens = [100]  # number of hidden units to make plots for
lrs = [0.1, 0.01, 0.001]  # number of learning rates to make plots for
epoch_nums = [250, 1000]  # total epochs of studies to make plots for

# Define a list of which of the above specified plots to actually make
plots_to_make = ["energy", "energy_log", "log energy vs. fidelity", "energy vs. fidelity", "fidelity", "log fidelity error", "loss"]


# Define a dictionary to relate keywords to plotting functions to call
plots_dict = {"energy": plot_energy, "energy_log": plot_energy_log, "energy vs. fidelity": plot_energy_vs_fid, "log energy vs. fidelity": plot_log_energy_vs_fid, "fidelity": plot_fid, "log fidelity error": plot_fid_error_log, "loss": plot_loss}


for vals in itertools.product(studies, spin_nums, num_hiddens, lrs, epoch_nums):
    if os.path.exists("{0}_results/N{1}_nh{2}_lr{3}_ep{4}".format(*vals)):
        for plot_type in plots_to_make:
            plots_dict[plot_type](*vals)
