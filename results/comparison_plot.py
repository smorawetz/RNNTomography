import os
import numpy as np
import itertools

from compare_plotting.plot_compare_energy import plot_compare_energy
from compare_plotting.plot_compare_log_energy import plot_compare_log_energy
from compare_plotting.plot_compare_fid_energy import plot_compare_fid_energy
from compare_plotting.plot_compare_log_fid_energy import plot_compare_log_fid_energy

# ---- Define parameters related to which studies to make plots for ----
# ---- where each is a list of possible values that some parameter -----
# ---- can take on, which will be looped over when producing plots -----


studies = ["tfim_track_symm", "xy_track_symm"]
spin_numss = [10, 10]  # spin numbers to make plots for
num_hidden = 100  # number of hidden units to make plots for
lrs = [0.001, 0.001, 0.001]  # number of learning rates to make plots for
epoch_num = 2000  # total epochs of studies to make plots for

# Define a list of which of the above specified plots to actually make
plots_to_make = ["compare log fid energy"]


# Define a dictionary to relate keywords to plotting functions to call
plots_dict = {
    "compare energy": plot_compare_energy,
    "compare log energy": plot_compare_log_energy,
    "compare fid energy": plot_compare_fid_energy,
    "compare log fid energy": plot_compare_log_fid_energy,
}

params = [studies, spin_numss, num_hidden, lrs, epoch_num]

for plot_type in plots_to_make:
    plots_dict[plot_type](*params)
