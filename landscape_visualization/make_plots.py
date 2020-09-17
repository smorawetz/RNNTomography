from landscape_visualization_plot_3d import loss_plot_3d
from landscape_visualization_plot_heatmap import loss_plot_heatmap
from energy_visualization_plot_3d import energy_plot_3d
from energy_visualization_plot_heatmap import energy_plot_heatmap

PLOTS_TO_MAKE = ["loss_3d", "loss_heatmap", "energy_3d", "energy_heatmap"]
# PLOTS_TO_MAKE = ["loss_heatmap", "energy_heatmap"]
# PLOTS_TO_MAKE = ["loss_heatmap"]
# PLOTS_TO_MAKE = ["energy_heatmap"]

PLOTS_DICT = {
    "loss_3d": loss_plot_3d,
    "loss_heatmap": loss_plot_heatmap,
    "energy_3d": energy_plot_3d,
    "energy_heatmap": energy_plot_heatmap,
}

PHYSICS_MODELS = ["xy", "tfim"]
# PHYSICS_MODELS = ["xy"]
SYMM_TYPES_DICT = {"xy": ["no_symm", "hard_symm"], "tfim": ["no_symm"]}
N_VALS_DICT = {"xy": [10], "tfim": [10]}
# RANGES = [0.1, 0.25, 1.0, 5.0, 10.0]
RANGES = [0.25]

for plot_type in PLOTS_TO_MAKE:
    for physics_model in PHYSICS_MODELS:
        for symm_type in SYMM_TYPES_DICT[physics_model]:
            for N in N_VALS_DICT[physics_model]:
                for max_change in RANGES:
                    plot_func = PLOTS_DICT[plot_type]
                    plot_func(physics_model, symm_type, N, max_change)
