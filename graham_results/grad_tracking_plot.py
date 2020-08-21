import numpy as np
import matplotlib.pyplot as plt

# First plot average values of gradients

FILENAME = "xy_track_symm"
N = 10
NH = 100
LR = 0.001
EP = 2000

PERIOD = 5

INPUT_DIM = 2  # 0 and 1

STUDY_NAME = "N{0}_nh{1}_lr{2}_ep{3}".format(N, NH, LR, EP)

avg_grad_filename = "{0}_results/{1}/avg_grads_rnn_{0}_{1}.txt".format(
    FILENAME, STUDY_NAME
)
max_grad_filename = "{0}_results/{1}/max_grads_rnn_{0}_{1}.txt".format(
    FILENAME, STUDY_NAME
)

avg_grad_data = np.loadtxt(avg_grad_filename)
max_grad_data = np.loadtxt(max_grad_filename)

num_pars = avg_grad_data.shape[1]

avg_param_names = [
    r"$\langle \vert W_{ij} \vert \rangle$",
    r"$\langle \vert U_{ij} \vert \rangle$",
    r"$\langle \vert b_i \vert \rangle$",
    r"PLACEHOLDER SHOULDN'T BE USED",
    r"$\langle \vert V_{ij} \vert \rangle$",
    r"$\langle \vert c_i \vert \rangle$",
]

max_param_names = [
    r"$\max_{ij} \vert W_{ij} \vert$",
    r"$\max_{ij} \vert U_{ij} \vert$",
    r"$\max_{i} \vert b_i \vert$",
    r"PLACEHOLDER SHOULDN'T BE USED",
    r"$\max_{ij} \vert V_{ij} \vert$",
    r"$\max_{i} \vert c_i \vert$",
]


# First make plot for average grad data

epochs = PERIOD * np.arange(1, avg_grad_data.shape[0] + 1)

fig, ax = plt.subplots()
for par_num in range(num_pars):
    if par_num == 2:  # special hack for input/hidden biases
        new_data = (avg_grad_data[:, 2] + avg_grad_data[:, 3]) / 2
        ax.plot(epochs, new_data, label=avg_param_names[par_num])
    elif par_num == 3:
        continue
    else:
        ax.plot(epochs, avg_grad_data[:, par_num], label=avg_param_names[par_num])

ax.legend(loc=(0.8, 0.40), frameon=False)
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Average abs. gradient value")

plt.savefig(
    "{0}_results/{1}/avg_grads_plot_{0}_{1}.png".format(FILENAME, STUDY_NAME), dpi=1000,
)


# Then make plot for max grad data

fig, ax = plt.subplots()
for par_num in range(num_pars):
    if par_num == 2:  # special hack for input/hidden biases
        new_data = (max_grad_data[:, 2] + max_grad_data[:, 3]) / 2
        ax.plot(epochs, new_data, label=max_param_names[par_num])
    elif par_num == 3:
        continue
    else:
        ax.plot(epochs, max_grad_data[:, par_num], label=max_param_names[par_num])

ax.legend(loc=(0.77, 0.38), frameon=False)
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Average abs. gradient value")

plt.savefig(
    "{0}_results/{1}/max_grads_plot_{0}_{1}.png".format(FILENAME, STUDY_NAME), dpi=1000,
)


# Make plot for average grad of ALL parameters

num_pars_dict = {
    0: 3 * NH * INPUT_DIM,
    1: 3 * NH * NH,
    2: 3 * NH,
    3: 3 * NH,
    4: INPUT_DIM * NH,
    5: INPUT_DIM,
}

avgs_per_ep = np.zeros(avg_grad_data.shape[0])

fig, ax = plt.subplots()
for par_num in range(num_pars):
    # Want to compute WEIGHTED average of parameter gradients
    weight = num_pars_dict[par_num]
    avgs_per_ep += weight * avg_grad_data[:, par_num]
    
norm = 0
for weight in num_pars_dict.values():
    norm += weight

avgs_per_ep /= norm

ax.plot(epochs, avgs_per_ep)
ax.set_xlabel(r"Epoch")
ax.set_ylabel(r"Average parameter gradient")

plt.savefig(
    "{0}_results/{1}/total_avg_grads_plot_{0}_{1}.png".format(FILENAME, STUDY_NAME), dpi=1000,
)
