import os
import numpy as np
import matplotlib.pyplot as plt


def plot_compare_log_energy(model_names, num_spinss, num_hidden, lrs, num_epochs):
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
        data_file = "training_results_rnn_{0}_{1}_seed1.txt".format(
            model_name, study_name
        )
        data_path = "{0}_results/{1}/{2}".format(model_name, study_name, data_file)

        data_list.append(np.loadtxt(data_path))

    # -------- Make the plot ----------

    fig, axs = plt.subplots(nrows=1, ncols=num_studies, sharey=True)

    for i in range(num_studies):
        data = data_list[i]
        epochs = data[:, 0]
        energies = data[:, 3]

        ax = axs[i]
        ax.plot(
            epochs, energies, "o", color="C0", markeredgecolor="black",
        )
        ax.set_yscale("log")
        ax.set_title(r"{0}, N = {1}".format(model_names[i].upper(), num_spinss[i]))
        if i == 0:
            ax.set_ylabel(r"$\frac{|E_{RNN} - E_{targ}|}{N}$")

    plt.subplots_adjust(wspace=0)
    fig.text(0.5, 0.02, r"Epoch", ha="center")

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

    plt.savefig("compare_results/{0}/compare_log_energy.pdf".format(compare_name))
