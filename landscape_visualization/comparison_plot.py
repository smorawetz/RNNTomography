from multi_landscape_visualization_plot_3d import multi_loss_plot_3d
from multi_landscape_visualization_plot_heatmap import multi_loss_plot_heatmap
from multi_energy_visualization_plot_3d import multi_energy_plot_3d
from multi_energy_visualization_plot_heatmap import multi_energy_plot_heatmap

# PLOTS_TO_MAKE = ["loss_3d", "loss_heatmap", "energy_3d", "energy_heatmap"]
PLOTS_TO_MAKE = ["loss_heatmap", "energy_heatmap"]

PLOTS_DICT = {
    # "loss_3d": multi_loss_plot_3d,
    "loss_heatmap": multi_loss_plot_heatmap,
    # "energy_3d": multi_energy_plot_3d,
    "energy_heatmap": multi_energy_plot_heatmap,
}

LIST_OF_PHYSICS_MODELS = [["xy", "xy"]]
LIST_OF_SYMM_TYPES = [["no_symm", "hard_symm"]]
N_VALS = [10, 10]
# LIST_OF_RANGES = [[0.25, 0.25], [1.0, 1.0], [5.0, 5.0]]
LIST_OF_RANGES = [[0.25, 0.25]]

for i in range(len(LIST_OF_PHYSICS_MODELS)):
    physics_models = LIST_OF_PHYSICS_MODELS[i]
    symm_types = LIST_OF_SYMM_TYPES[i]
    for ranges in LIST_OF_RANGES:
        for plot_type in PLOTS_TO_MAKE:
            PLOTS_DICT[plot_type](physics_models, symm_types, N_VALS, ranges)
