import os
import numpy as np
import itertools

from plotting.plot_late_symm_fid import plot_late_symm_fid

# ---- Define parameters related to which studies to make plots for ----
# ---- where each is a list of possible values that some parameter -----
# ---- can take on, which will be looped over when producing plots -----


studies = ["xy_late_symm"]
spin_nums = [10]  # spin numbers to make plots for
num_hiddens = [100]  # number of hidden units to make plots for
lrs = [0.1, 0.001]  # number of learning rates to make plots for
epoch_nums = [1000]  # total epochs of studies to make plots for

# Define a list of which of the above specified plots to actually make
plots_to_make = ["late symm fidelity"]


# Define a dictionary to relate keywords to plotting functions to call
plots_dict = {"late symm fidelity": plot_late_symm_fid}


for vals in itertools.product(studies, spin_nums, num_hiddens, lrs, epoch_nums):
    if os.path.exists("{0}_results/N{1}_nh{2}_lr{3}_ep{4}".format(*vals)):
        for plot_type in plots_to_make:
            plots_dict[plot_type](*vals)
