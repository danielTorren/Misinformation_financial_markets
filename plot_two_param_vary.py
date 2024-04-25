"""Plot multiple simulations varying two parameters
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from utility import (
    load_object
)
import numpy as np

def double_phase_diagram(
    fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, levels = 10):
    
    fig, ax = plt.subplots(constrained_layout=True)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(col_dict["title"])
    ax.set_ylabel(row_dict["title"])

    X, Y = np.meshgrid(col_dict["vals"], row_dict["vals"])

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5, levels = levels)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_%s" % (Y_param)
    #fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


def main(
    fileName = "results/two_param_sweep_average_small_world_18_32_05_26_03_2024",
    SW = "results/two_param_sweep_average_small_world_18_32_05_26_03_2024",
    SBM = "results/two_param_sweep_average_SBM_18_34_51_26_03_2024",
    SF_misinfo = "results/two_param_sweep_average_scale_free_18_29_04_26_03_2024",
    SF_info = "results/two_param_sweep_average_scale_free_18_22_46_26_03_2024",
    levels = 10,#this varies the number of contour lines colours
):

    fontsize= 24
    ticksize = 24
    figsize = (12, 9)
    params = {'font.family':'serif',
        "figure.figsize":figsize, 
        'figure.dpi': 80,
        'figure.edgecolor': 'k',
        'font.size': fontsize, 
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': ticksize,
        'ytick.labelsize': ticksize
    }
    plt.rcParams.update(params)

    # Import current plot data
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    dev_price_mean = load_object(fileName + "/Data", "dev_price_mean")
    excess_var_mean = load_object(fileName + "/Data", "excess_var_mean")
    excess_autocorr_mean = load_object(fileName + "/Data", "excess_autocorr_mean")
    kurtosis_mean = load_object(fileName + "/Data", "kurtosis_mean")

    double_phase_diagram(fileName, dev_price_mean, r"dev_price", "dev_price",variable_parameters_dict, get_cmap("Reds"), levels)  
    double_phase_diagram(fileName, excess_var_mean, r"excess_var", "excess_var",variable_parameters_dict, get_cmap("Blues"),levels)
    double_phase_diagram(fileName, excess_autocorr_mean, r"excess_autocorr", "excess_autocorr",variable_parameters_dict, get_cmap("Greens"), levels)
    double_phase_diagram(fileName, kurtosis_mean, r"kurtosis", "kurtosis",variable_parameters_dict, get_cmap("Oranges"), levels)

    plt.show()

if __name__ == '__main__':

    main(
    fileName =   "results/two_param_sweep_average_SBM_22_20_30_10_04_2024",
    )


