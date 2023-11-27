"""Plot multiple simulations varying two parameters
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from utility import (
    load_object
)
import numpy as np

fontsize= 18
ticksize = 16
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

def double_phase_diagram(
    fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, dpi_save, levels):
    
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
    #cbar.set_label(Y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_%s" % (Y_param)
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def double_phase_diagram_subplot(
    ax, Z, Y_title, Y_param, variable_parameters_dict, cmap, levels):
    
    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(col_dict["title"])
    ax.set_ylabel(row_dict["title"])
    ax.set_title(Y_title)

    X, Y = np.meshgrid(col_dict["vals"], row_dict["vals"])

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5, levels=levels)
    cbar = plt.colorbar(
        cp,
        ax=ax,
    )


def main(
    fileName = "results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023",
    dpi_save = 1200,
    levels = 10,#this varies the number of contour lines colours
) -> None:
        
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")

    dev_price_mean = load_object(fileName + "/Data", "dev_price_mean")
    excess_var_mean = load_object(fileName + "/Data", "excess_var_mean")
    excess_autocorr_mean = load_object(fileName + "/Data", "excess_autocorr_mean")
    kurtosis_mean = load_object(fileName + "/Data", "kurtosis_mean")

    # double_phase_diagram(fileName, dev_price_mean, r"dev_price", "dev_price",variable_parameters_dict, get_cmap("Reds"),dpi_save, levels)  
    # double_phase_diagram(fileName, excess_var_mean, r"excess_var", "excess_var",variable_parameters_dict, get_cmap("Blues"),dpi_save, levels)
    # double_phase_diagram(fileName, excess_autocorr_mean, r"excess_autocorr", "excess_autocorr",variable_parameters_dict, get_cmap("Greens"),dpi_save, levels)
    # double_phase_diagram(fileName, kurtosis_mean, r"kurtosis", "kurtosis",variable_parameters_dict, get_cmap("Oranges"),dpi_save, levels)

    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    double_phase_diagram_subplot(
        axs[0, 0], dev_price_mean, r"Price Deviation", "dev_price", variable_parameters_dict, get_cmap("Reds"), levels)
    double_phase_diagram_subplot(
        axs[0, 1], excess_var_mean, r"Excess Variance", "excess_var", variable_parameters_dict, get_cmap("Blues"), levels)
    double_phase_diagram_subplot(
        axs[1, 0], excess_autocorr_mean, r"Excess Autocorrelation", "excess_autocorr", variable_parameters_dict, get_cmap("Greens"), levels)
    double_phase_diagram_subplot(
        axs[1, 1], kurtosis_mean, r"Kurtosis", "kurtosis", variable_parameters_dict, get_cmap("Oranges"), levels)

    plotName = fileName + "/Plots"
    f = plotName + "/double_phase_diagram_%s" 
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
    plt.show()

if __name__ == '__main__':
    main(
        fileName = "results/two_param_sweep_average_small-world_09_53_19_13_11_2023"#CHANGE THIS TO WHATEVER THE FILENAME IS
    )
