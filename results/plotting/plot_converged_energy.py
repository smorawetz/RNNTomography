import os
import numpy as np
import matplotlib.pyplot as plt


avg_over = 10  # number of final energies to average over

def plot_converged_energy(model_name, num_spins, num_hidden, lr, num_epochs, N_list):
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

    # -------- Getting data from all spins

    final_E_list = []
    final_fid_list = []

    study_name = "allN_nh{0}_lr{1}_ep{2}".format(num_hidden, lr, num_epochs)
    study_path = "{0}_results/{1}".format(model_name, study_name)

    if not os.path.exists(study_path):
        os.makedirs(study_path)


    for N in N_list:
        temp_study = "N{0}_nh{1}_lr{2}_ep{3}".format(N, num_hidden, lr, num_epochs)
        data_file = "training_results_rnn_{0}_{1}.txt".format(model_name, temp_study)

        data_path = "{0}_results/{1}/{2}".format(model_name, temp_study, data_file)

        data = np.loadtxt(data_path)

        avg_fid = np.mean(data[-avg_over:, 1])
        final_fid_list.append(avg_fid)
        avg_energy = np.mean(data[-avg_over:, 3])
        final_E_list.append(avg_energy)

    fig, ax = plt.subplots()

    ax.scatter(N_list, final_E_list, c=final_fid_list)
    fig.colorbar(mappable=ax.collections[0], ax=ax)
    ax.set_title(r"Avg. energy value, final {0} epochs vs N".format(avg_over))
    ax.set_ylabel(r"Average energy")
    ax.set_xlabel(r"$N$")

    plt.tight_layout()

    plt.savefig("{0}/conv_energy_{1}.png".format(study_path, study_name))
