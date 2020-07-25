import os
import numpy as np
import matplotlib.pyplot as plt

plot_lrs = [0.001, 0.1]


def plot_compare_log_fid_energy(model_names, num_spinss, num_hidden, lrs, num_epochs):
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

    lr_data_list = []
    for lr in plot_lrs:
        data_list = []
        for i in range(num_studies):
            model_name = model_names[i]
            num_spins = num_spinss[i]
            study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(
                num_spins, num_hidden, lr, num_epochs
            )
            study_path = "{0}_results/{1}".format(model_name, study_name)
            data_file = "training_results_rnn_{0}_{1}_seed1.txt".format(
                model_name, study_name
            )
            data_path = "{0}_results/{1}/{2}".format(model_name, study_name, data_file)

            data_list.append(np.loadtxt(data_path))
        lr_data_list.append(data_list.copy())

    # -------- Make the plot ----------

    fig, axs = plt.subplots(
        nrows=2,
        ncols=num_studies,
        sharex="col",
        sharey="row",
        figsize=(4 * num_studies, 6),
    )

    for i in range(num_studies):
        small_lr_data = lr_data_list[0][i]
        small_lr_epochs = small_lr_data[:, 0]
        small_lr_fids = small_lr_data[:, 1]
        small_lr_energies = small_lr_data[:, 3]
        big_lr_data = lr_data_list[1][i]
        big_lr_epochs = big_lr_data[:, 0]
        big_lr_fids = big_lr_data[:, 1]
        big_lr_energies = big_lr_data[:, 3]

        ax = axs[0, i]
        ax.plot(
            small_lr_epochs,
            small_lr_energies,
            "o",
            label="SGD" if i == 0 else "",
            color="C0",
            markeredgecolor="black",
        )
        ax.plot(
            big_lr_epochs,
            big_lr_energies,
            "o",
            label="Adadelta" if i == 0 else "",
            color="C3",
            markeredgecolor="black",
        )
        ax.set_yscale("log")
        ax.set_title(r"{0}, N = {1}".format(model_names[i].upper(), num_spinss[i]))
        if i == 0:
            ax.set_ylabel(r"$\frac{|E_{RNN} - E_{targ}|}{N}$")

        ax = axs[1, i]
        ax.set_yscale("log")
        if num_spinss[i] <= 12:
            ax.plot(
                small_lr_epochs,
                (1 - small_lr_fids),
                "o",
                color="C0",
                markeredgecolor="black",
            )
            ax.plot(
                big_lr_epochs,
                (1 - big_lr_fids),
                "o",
                color="C3",
                markeredgecolor="black",
            )
            if i == 0:
                ax.set_ylabel(r"$\log|1 - \mathcal{F}|$")

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.text(0.5, 0.02, r"Epoch", ha="center")
    fig.legend()

    compare_names = []
    for i in range(num_studies):
        name = "{0}_N{1}_".format(model_names[i], num_spinss[i],)
        compare_names.append(name)

    compare_name = ""
    for name in compare_names:
        compare_name += name
    compare_name += "nh{0}_ep{1}".format(num_hidden, num_epochs)

    if not os.path.exists("compare_results/{0}".format(compare_name)):
        os.makedirs("compare_results/{0}".format(compare_name))

    plt.savefig("compare_results/{0}/compare_log_fid_energy.png".format(compare_name))
