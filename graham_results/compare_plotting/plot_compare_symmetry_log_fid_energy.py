import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

params = {
    "text.usetex": True,
    "font.family": "serif",
    "legend.fontsize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1,
    "patch.edgecolor": "black",
}


def plot_compare_symmetry_log_fid_energy(
    model_names, num_spinss, num_hidden, lrs, num_epochs
):
    """
    model_names:    listof str
                    name of models for which making plot, e.g.
                    tfim, xy, or xy_enforce_symm
    num_spinss:     listof int
                    list of number of spins of each plot
    num_hidden:     int
                    number of hidden units in RNN
    lrs:            listof float
                    learning rate of each study being compared
    num_epochs:     int
                    number of epochs of training

    returns:        None
                    saves, does not return anything
    """

    num_studies = len(model_names)

    # --------- Getting data for each study -------------

    data_list = []

    for i in range(num_studies):
        model_name = model_names[i]
        num_spins = num_spinss[i]
        lr = lrs[i]
        study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(
            num_spins, num_hidden, lr, num_epochs
        )
        study_path = "{0}_results/{1}".format(model_name, study_name)
        data_file = "training_results_rnn_{0}_{1}_seed3.txt".format(
            model_name, study_name
        )
        data_path = "{0}_results/{1}/{2}".format(model_name, study_name, data_file)

        rbm_energy_data = "rbm_results/energies

        data_list.append(np.loadtxt(data_path))

    # -------- Make the plot ----------

    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex="col", sharey="row", figsize=(5, 7),
    )

    nice_names_dict = {
        "xy_no_symm": "No symmetry",
        "xy_hard_symm": "Symmetry imposed",
        "xy_soft_symm": "Soft symmetry",
    }

    colours = ["C0", "C3"]
    symbols = ["o", "^"]

    all_spins_same = num_spinss.count(num_spinss[0]) == len(num_spinss)

    for i in range(num_studies):
        data = data_list[i]
        epochs = data[:, 0]
        fids = data[:, 1]
        energies = data[:, 3]
        energy_errors = data[:, -1]

        ax = axs[0]
        ax.plot(
            epochs,
            energies,
            symbols[i],
            color=colours[i],
            markeredgecolor="black",
            markersize=5,
            label=nice_names_dict[model_names[i]],
        )
        ax.set_yscale("log")
        # if all_spins_same:
        # ax.set_title(r"{0}".format(nice_names_dict[model_names[i]]))
        # else:
        # ax.set_title(
        # r"{0}, N = {1}".format(nice_names_dict[model_names[i]], num_spinss[i])
        # )
        ax.set_ylim(6e-7, 2)
        if i == 0:
            # ax.set_ylabel(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$")
            ax.set_ylabel(r"$\epsilon$")

        ax = axs[1]
        ax.set_yscale("log")
        if num_spinss[i] <= 12:
            ax.plot(
                epochs,
                (1 - fids),
                symbols[i],
                color=colours[i],
                markeredgecolor="black",
                markersize=5,
                # label=nice_names_dict[model_names[i]],
            )
            if i == 0:
                ax.set_ylabel(r"$1 - \mathcal{F}$")
                ax.set_xlabel(r"Epoch")

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.legend(loc=(0.49, 0.89), frameon=False)

    fig.tight_layout()

    compare_names = []
    for i in range(num_studies):
        name = "{0}_N{1}_lr{2}_".format(model_names[i], num_spinss[i], lrs[i])
        compare_names.append(name)

    compare_name = ""
    for name in compare_names:
        compare_name += name
    compare_name += "nh{0}_ep{1}".format(num_hidden, num_epochs)

    if not os.path.exists("compare_results/{0}".format(compare_name)):
        os.makedirs("compare_results/{0}".format(compare_name))

    plt.savefig(
        "compare_results/{0}/compare_symmetry_log_fid_energy.pdf".format(compare_name),
        dpi=1000,
    )
