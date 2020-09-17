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

seed_dict = {4: "1234", 6: "1234", 10: "1234", 30: "1234", 40: "1357", 50: "2222"}


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

    for i in range(len(num_spinss)):
        for j in range(num_studies):
            model_name = model_names[j]
            num_spins = num_spinss[i]
            lr = lrs[j]
            study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(
                num_spins, num_hidden, lr, num_epochs
            )
            study_path = "{0}_results/{1}".format(model_name, study_name)
            data_file = "training_results_rnn_{0}_{1}_seed3.txt".format(
                model_name, study_name
            )
            data_path = "{0}_results/{1}/{2}".format(model_name, study_name, data_file)

            data_list.append(np.loadtxt(data_path))

    rbm_fid_data = []
    rbm_energy_data = []

    for i in range(len(num_spinss)):
        rbm_spins = num_spinss[i]
        rbm_hidden = rbm_spins
        rbm_seed = seed_dict[rbm_spins]

        rbm_energy_file = "rbm_results/energies/energies_errors_N={0}_nh={1}_seed={2}_SGD_Gibbs.txt".format(
            rbm_spins, rbm_hidden, rbm_seed
        )
        rbm_fidelity_file = (
            "rbm_results/fidelities/fidelity_N={0}_nh={1}_seed=1234_SGD_Gibbs.txt".format(
                rbm_spins, rbm_hidden
            )
        )
        rbm_E_data = np.loadtxt(rbm_energy_file)
        rbm_fidelity_data = np.loadtxt(rbm_fidelity_file)

        rbm_period = 100
        rbm_max_ep = 2000
        rbm_last_index = rbm_max_ep // rbm_period

        rbm_epochs = np.arange(rbm_period, rbm_max_ep + 1, rbm_period)

        rbm_energies = rbm_E_data[:rbm_last_index, 1]
        rbm_fids = rbm_fidelity_data[:rbm_last_index, 1]

        rbm_fid_data.append(rbm_fids)
        rbm_energy_data.append(rbm_energies)

    # -------- Make the plot ----------

    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(num_spinss),
        sharex="col",
        sharey="row",
        figsize=(9, 7),
    )

    nice_names_dict = {
        "xy_no_symm": "Symetry-free RNN",
        "xy_hard_symm": "Symmetry-imposed RNN",
        "xy_soft_symm": "Soft symmetry",
    }

    colours = ["C0", "C3"]
    symbols = ["o", "^"]

    all_spins_same = num_spinss.count(num_spinss[0]) == len(num_spinss)

    for i in range(len(num_spinss)):
        for j in range(num_studies):
            data = data_list[2 * i + j]
            epochs = data[:, 0]
            fids = data[:, 1]
            energies = data[:, 3]
            energy_errors = data[:, -1]

            ax = axs[0, i]
            ax.plot(
                epochs,
                energies,
                symbols[j],
                color=colours[j],
                markeredgecolor="black",
                markersize=5,
                label=nice_names_dict[model_names[j]] if i == 0 else "",
            )
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(r"N = {0}".format(num_spinss[i]))
            ax.set_ylim(6e-7, 2)
            if i == 0:
                # ax.set_ylabel(r"$\frac{|E_{RNN} - E_{DMRG}|}{N}$")
                ax.set_ylabel(r"$\epsilon$")

            ax = axs[1, i]
            ax.set_yscale("log")
            ax.plot(
                epochs,
                (1 - fids),
                symbols[j],
                color=colours[j],
                markeredgecolor="black",
                markersize=5,
                # label=nice_names_dict[model_names[i]],
            )
            ax.set_ylim(6e-5, 1)
            if i == 0 and j == 0:
                ax.set_ylabel(r"$1 - \mathcal{F}$")
                ax.set_xlabel(r"Epoch")

    for i in range(len(num_spinss)):
        rbm_energies = rbm_energy_data[i]
        rbm_fids = rbm_fid_data[i]

        ax = axs[0, i]
        ax.plot(
            rbm_epochs,
            rbm_energies,
            "s",
            color="C2",
            markeredgecolor="black",
            markersize=5,
            label="Symmetry-free RBM" if i == 0 else "",
        )

        ax = axs[1, i]
        ax.plot(
            rbm_epochs,
            (1 - rbm_fids),
            "s",
            color="C2",
            markeredgecolor="black",
            markersize=5,
        )

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.legend(loc=(0.54, 0.55), frameon=False)

    fig.tight_layout()

    compare_names = []
    for i in range(len(num_spinss)):
        for j in range(num_studies):
            name = "{0}_N{1}_lr{2}_".format(model_names[j], num_spinss[i], lrs[j])
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
