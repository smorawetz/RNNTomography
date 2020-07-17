import numpy as np
import matplotlib.pyplot as plt


def plot_loss(model_name, num_spins, num_hidden, lr, num_epochs):
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

    study_name = "N{0}_nh{1}_lr{2}_ep{3}".format(num_spins, num_hidden, lr, num_epochs)
    data_file = "training_results_rnn_{0}_{1}.txt".format(model_name, study_name)

    study_path = "{0}_results/{1}".format(model_name, study_name)
    data_path = "{0}_results/{1}/{2}".format(model_name, study_name, data_file)

    data = np.loadtxt(data_path)

    epochs = data[1:, 0]
    loss = data[1:, 4]

    # ------- Making the plot ------------

    fig, ax = plt.subplots()
    ax.plot(epochs, loss, "o", color="C0", markeredgecolor="black")
    ax.set_title(r"Loss function for training {0}".format(model_name.upper()))
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"Loss")

    plt.savefig("{0}/loss_{1}.png".format(study_path, study_name))
