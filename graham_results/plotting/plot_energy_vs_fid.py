import numpy as np
import matplotlib.pyplot as plt


def plot_energy_vs_fid(model_name, num_spins, nh, lr, num_epochs):
    """
        model_name: str
                    name of model for which making plot, e.g.
                    tfim, xy, or xy_enforce_symm
        num_spins:  int
                    number of spins in system
        num_hidden: int
                    number of hidden units in RNN
        lr:         float
                    learning rate
        num_epochs: int
                    number of epochs of training

        returns:    None
                    saves, does not return anything
    """

    # ------- Getting data -------------

    study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(num_spins, nh, lr, num_epochs)
    data_file = "training_results_rnn_{0}_{1}_seed1.txt".format(model_name, study_name)

    study_path = "{0}_results/{1}".format(model_name, study_name)
    data_path = "{0}_results/{1}/{2}".format(model_name, study_name, data_file)

    data = np.loadtxt(data_path)

    fids = data[:, 1]
    energies = data[:, 3]

    # ------- Making the plot ------------

    fig, ax = plt.subplots()

    ax.plot(fids, energies, "o", color="C0", markeredgecolor="black")
    ax.set_title(
        r"Abs. energy diff. vs fidelity for {0}, N = {1}".format(
            model_name.upper(), num_spins
        )
    )
    ax.set_ylabel(r"$\frac{|E_{RNN} - E_0|}{N}$")
    ax.set_xlabel(r"Fidelity")

    plt.tight_layout()

    plt.savefig("{0}/energy_vs_fid_{1}.png".format(study_path, study_name))
