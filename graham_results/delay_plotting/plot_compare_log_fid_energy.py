import os
import numpy as np
import matplotlib.pyplot as plt


def plot_compare_log_fid_energy(
    model_names, num_spinss, num_hidden, lrs, num_epochs, delay_types, delay_eps
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
    delay_types:    listof str
                    one of 'symm_first' or 'no_symm_first'
    delay_eps:      listof int
                    epoch when symmetry is impose/lifted by delay_type

    returns:        None
                    saves, does not return anything
    """

    num_studies = len(model_names)

    # --------- Getting data for each study -------------

    data_list = []

    for i in range(num_studies):
        delay_type = delay_types[i]
        model_names[i] += "_{0}".format(delay_type)
        model_name = model_names[i]
        num_spins = num_spinss[i]
        lr = lrs[i]
        delay_ep = delay_eps[i]
        study_name = "N{0}_nh{1}_lr{2}_ep{3}_symm_ep{4}".format(
            num_spins, num_hidden, lr, num_epochs, delay_ep
        )
        study_path = "{0}_results/{1}".format(model_name, study_name)
        data_file = "training_results_rnn_{0}_{1}_seed1.txt".format(
            model_name, study_name
        )
        data_path = "{0}_results/{1}/{2}".format(model_name, study_name, data_file)

        data_list.append(np.loadtxt(data_path))

    # -------- Make the plot ----------

    fig, axs = plt.subplots(
        nrows=2,
        ncols=num_studies,
        sharex="col",
        sharey="row",
        figsize=(4 * num_studies, 6),
    )

    nice_names_dict = {
        "xy_delay_hard_symm_symm_first": "'Hard' symmetry first",
        "xy_delay_hard_symm_no_symm_first": "'Hard' symmetry later",
        "xy_delay_soft_symm_symm_first": "'Soft' symmetry first",
        "xy_delay_soft_symm_no_symm_first": "'Soft' symmetry later",
    }

    for i in range(num_studies):
        data = data_list[i]
        epochs = data[:, 0]
        fids = data[:, 1]
        energies = data[:, 3]

        ax = axs[0, i]
        ax.plot(
            epochs,
            energies,
            "o",
            color="C0",
            markeredgecolor="black",
        )
        ax.axvline(x=delay_eps[i], color="C3", linestyle="--")
        ax.set_yscale("log")
        ax.set_title(
            r"{0}, N = {1}".format(nice_names_dict[model_names[i]], num_spinss[i])
        )
        ax.set_ylim(6e-7, 2)
        if i == 0:
            ax.set_ylabel(r"$\frac{|E_{RNN} - E_{targ}|}{N}$")

        ax = axs[1, i]
        ax.set_yscale("log")
        if num_spinss[i] <= 12:
            ax.plot(epochs, (1 - fids), "o", color="C0", markeredgecolor="black")
            ax.axvline(x=delay_eps[i], color="C3", linestyle="--")
            if i == 0:
                ax.set_ylabel(r"$\log|1 - Fidelity|$")

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.text(0.5, 0.02, r"Epoch", ha="center")

    compare_names = []
    for i in range(num_studies):
        name = "{0}_N{1}_lr{2}_symm_ep{3}_".format(
            model_names[i], num_spinss[i], lrs[i], delay_eps[i]
        )
        compare_names.append(name)

    compare_name = ""
    for name in compare_names:
        compare_name += name
    compare_name += "nh{0}_ep{1}".format(num_hidden, num_epochs)

    if not os.path.exists("compare_results/{0}".format(compare_name)):
        os.makedirs("compare_results/{0}".format(compare_name))

    print(compare_name)

    plt.savefig("compare_results/{0}/compare_log_fid_energy.png".format(compare_name))
