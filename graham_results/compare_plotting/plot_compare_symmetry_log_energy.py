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

seed_dict = { 30: "1234", 40: "1357", 50: "2222" }


def plot_compare_symmetry_log_energy(
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

    rbm_epochs_list = []
    rbm_energies_list = []

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

        data_list.append(np.loadtxt(data_path))

        model_name = model_names[i].replace("no_symm", "hard_symm")
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

        data_list.append(np.loadtxt(data_path))

        rbm_hidden = num_spins  # NOTE: Change this with new data
        rbm_seed = seed_dict[num_spins]

        rbm_energy_file = "rbm_results/energies/energies_errors_N={0}_nh={1}_seed={2}_SGD_Gibbs.txt".format(
            num_spins, rbm_hidden, rbm_seed
        )
        rbm_energy_data = np.loadtxt(rbm_energy_file)

        rbm_period = 100
        rbm_max_ep = 2000
        rbm_last_index = rbm_max_ep // rbm_period

        rbm_epochs = np.arange(rbm_period, rbm_max_ep + 1, rbm_period)
        rbm_energies = rbm_energy_data[:rbm_last_index, 1]

        rbm_epochs_list.append(rbm_epochs)
        rbm_energies_list.append(rbm_energies)

    # -------- Make the plot ----------

    fig, axs = plt.subplots(nrows=1, ncols=num_studies, sharey=True, figsize=(13, 3.5))

    nice_names_dict = {
        "xy_no_symm": "Symmetry-free RNN",
        "xy_hard_symm": "Symmetry-imposed RNN",
        "xy_soft_symm": "Soft symmetry",
    }

    for i in range(2 * num_studies):
        data = data_list[i]
        epochs = data[:, 0]
        energies = data[:, 3]

        colours = ["C0", "C3"]
        symbols = ["o", "^"]
        labels = ["No symmetry", "Symmetry imposed"]

        ax = axs[i // 2]
        ax.plot(
            epochs,
            energies,
            symbols[i % 2],
            color=colours[i % 2],
            markeredgecolor="black",
            label=labels[i % 2] if i // 2 == 0 else "",
        )
        ax.set_yscale("log")
        ax.set_title(r"N = {0}".format(num_spinss[i // 2]), fontsize=10)
        if i == 0:
            # ax.set_ylabel(r"$\frac{\left\vert E_{RNN} - E_{DMRG}\right\vert}{N}$")
            ax.set_ylabel(r"$\epsilon$")
        if i == num_studies:
            ax.set_xlabel(r"Epoch")

        if i % 2 == 0:  # only plot one RBM per size
            rbm_epochs = rbm_epochs_list[i // 2]
            rbm_energies = rbm_energies_list[i // 2]
            ax.plot(
                rbm_epochs,
                rbm_energies,
                "s",
                color="C2",
                markeredgecolor="black",
                label="Symmetry-free RBM" if i // 2 == 0 else "",
            )

    plt.subplots_adjust(wspace=0)
    fig.legend(loc=(0.79, 0.52), frameon=False)

    plt.tight_layout(rect=[0, 0.02, 1, 1])

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
        "compare_results/{0}/compare_symmetry_log_energy.pdf".format(compare_name),
        dpi=1000,
    )
