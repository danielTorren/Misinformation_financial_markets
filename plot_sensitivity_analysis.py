"""Performs sobol sensitivity analysis on the model. 

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from SALib.analyze import sobol
import numpy.typing as npt
from utility import (
    load_object
)


def multi_scatter_seperate_total_sensitivity_analysis_plot(
    fileName, data_dict, dict_list, names, dpi_save, N_samples, order
):
    """
    Create scatter chart of results.
    """

    fig, axes = plt.subplots(ncols=len(dict_list), nrows=1, constrained_layout=True , sharey=True,figsize=(12, 6))#,#sharex=True# figsize=(14, 7) # len(list(data_dict.keys())))
    
    plt.rc('ytick', labelsize=4) 

    for i, ax in enumerate(axes.flat):
        if order == "First":
            ax.errorbar(
                data_dict[dict_list[i]]["data"]["S1"].tolist(),
                names,
                xerr=data_dict[dict_list[i]]["yerr"]["S1"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"],
            )
        else:
            ax.errorbar(
                data_dict[dict_list[i]]["data"]["ST"].tolist(),
                names,
                xerr=data_dict[dict_list[i]]["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"],
            )
        ax.legend()
        ax.set_xlim(left=0)

    fig.supxlabel(r"%s order Sobol index" % (order))

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    f_png = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot.png"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")
    fig.savefig(f_png, dpi=dpi_save, format="png")


def get_plot_data(
    problem: dict,
    price_mean: npt.NDArray,
    price_var: npt.NDArray,
    price_autocorr: npt.NDArray,
    price_skew: npt.NDArray,
    calc_second_order: bool,
) -> tuple[dict, dict]:
    """
    Take the input results data from the sensitivity analysis  experiments for the four variables measures and now preform the analysis to give
    the total, first (and second order) sobol index values for each parameter varied. Then get this into a nice format that can easily be plotted
    with error bars.
    Parameters
    ----------
    problem: dict
        Outlines the number of variables to be varied, the names of these variables and the bounds that they take
    price_mean: npt.NDArray
        values for the price_mean = total network price_mean/(N*M) at the end of the simulation run time. One entry for each
        parameter set tested
    price_var: npt.NDArray
         values for the mean Individual identity normalized by N*M ie mu/(N*M) at the end of the simulation run time.
         One entry for each parameter set tested
    price_autocorr: npt.NDArray
         values for the variance of Individual identity in the network at the end of the simulation run time. One entry
         for each parameter set tested
    price_skew: npt.NDArray
         values for the coefficient of variance of Individual identity normalized by N*M ie (sigma/mu)*(N*M) in the network
         at the end of the simulation run time. One entry for each parameter set tested
    calc_second_order: bool
        Whether or not to conduct second order sobol sensitivity analysis, if set to False then only first and total order results will be
        available. Setting to True increases the total number of runs for the sensitivity analysis but allows for the study of interdependancies
        between parameters
    Returns
    -------
    data_sa_dict_total: dict[dict]
        dictionary containing dictionaries each with data regarding the total order sobol analysis results for each output measure
    data_sa_dict_first: dict[dict]
        dictionary containing dictionaries each with data regarding the first order sobol analysis results for each output measure
    """

    Si_price_mean , Si_mu , Si_var , Si_price_skew = analyze_results(problem,price_mean,price_var,price_autocorr,price_skew,calc_second_order) 

    total_price_mean, first_price_mean = Si_price_mean.to_df()
    total_price_var, first_mu = Si_mu.to_df()
    total_price_autocorr, first_price_autocorr = Si_var.to_df()
    total_price_skew,first_price_skew= Si_price_skew.to_df()

    total_data_sa_price_mean, total_yerr_price_mean = get_data_bar_chart(total_price_mean)
    total_data_sa_price_var, total_yerr_price_var = get_data_bar_chart(total_price_var)
    total_data_sa_price_autocorr, total_yerr_price_autocorr = get_data_bar_chart(total_price_autocorr)
    total_data_sa_price_skew, total_yerr_price_skew = get_data_bar_chart(total_price_skew)

    first_data_sa_price_mean, first_yerr_price_mean = get_data_bar_chart(first_price_mean)
    first_yerr_sa_price_var, first_yerr_price_var = get_data_bar_chart(first_mu)
    first_data_sa_price_autocorr, first_yerr_price_autocorr = get_data_bar_chart(first_price_autocorr)
    first_data_sa_price_skew,first_yerr_price_skew= get_data_bar_chart(first_price_skew)

    data_sa_dict_total = {
        "price_mean": {
            "data": total_data_sa_price_mean,
            "yerr": total_yerr_price_mean,
        },
        "price_var": {
            "data": total_data_sa_price_var,
            "yerr": total_yerr_price_var,
        },
        "price_autocorr": {
            "data": total_data_sa_price_autocorr,
            "yerr": total_yerr_price_autocorr,
        },
        "price_skew": {
            "data": total_data_sa_price_skew,
            "yerr": total_yerr_price_skew,
        },
    }
    data_sa_dict_first = {
        "price_mean": {
            "data": first_data_sa_price_mean,
            "yerr": first_yerr_price_mean,
        },
        "price_var": {
            "data": first_yerr_sa_price_var,
            "yerr": first_yerr_price_var,
        },
        "price_autocorr": {
            "data": first_data_sa_price_autocorr,
            "yerr": first_yerr_price_autocorr,
        },
        "price_skew": {
            "data": first_data_sa_price_skew,
            "yerr": first_yerr_price_skew,
        },
    }

    return data_sa_dict_total, data_sa_dict_first

def get_data_bar_chart(Si_df):
    """
    Taken from: https://salib.readthedocs.io/en/latest/_modules/SALib/plotting/bar.html
    Reduce the sobol index dataframe down to just the bits I want for easy plotting of sobol index and its error

    Parameters
    ----------
    Si_df: pd.DataFrame,
        Dataframe of sensitivity results.
    Returns
    -------
    Sis: pd.Series
        the value of the index
    confs: pd.Series
        the associated error with index
    """

    # magic string indicating DF columns holding conf bound values
    conf_cols = Si_df.columns.str.contains("_conf")
    confs = Si_df.loc[:, conf_cols]  # select all those that ARE in conf_cols!
    confs.columns = [c.replace("_conf", "") for c in confs.columns]
    Sis = Si_df.loc[:, ~conf_cols]  # select all those that ARENT in conf_cols!

    return Sis, confs

def Merge_dict_SA(data_sa_dict: dict, plot_dict: dict) -> dict:
    """
    Merge the dictionaries used to create the data with the plotting dictionaries for easy of plotting later on so that its drawing from
    just one dictionary. This way I seperate the plotting elements from the data generation allowing easier re-plotting. I think this can be
    done with some form of join but I have not worked out how to so far
    Parameters
    ----------
    data_sa_dict: dict
        Dictionary of dictionaries of data associated with each output measure from the sensitivity analysis for a specific sobol index
    plot_dict: dict
        data structure that contains specifics about how a plot should look for each output measure from the sensitivity analysis

    Returns
    -------
    data_sa_dict: dict
        the joined dictionary of dictionaries
    """
    #print("data_sa_dict",data_sa_dict)
    #print("plot_dict",plot_dict)
    for i in data_sa_dict.keys():
        for v in plot_dict[i].keys():
            #if v in data_sa_dict:
            data_sa_dict[i][v] = plot_dict[i][v]
            #else:
            #    pass
    return data_sa_dict

def analyze_results(
    problem: dict,
    price_mean: npt.NDArray,
    price_var: npt.NDArray,
    price_autocorr: npt.NDArray,
    price_skew: npt.NDArray,

    calc_second_order: bool,
) -> tuple:
    """
    Perform sobol analysis on simulation results
    """
    
    Si_price_mean = sobol.analyze(
        problem,
        price_mean,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )

    Si_mu = sobol.analyze(
        problem, price_var, calc_second_order=calc_second_order, print_to_console=False
    )
    Si_var = sobol.analyze(
        problem, price_autocorr, calc_second_order=calc_second_order, print_to_console=False
    )
    Si_price_skew = sobol.analyze(
        problem,
        price_skew,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )


    return Si_price_mean , Si_mu , Si_var , Si_price_skew

def main(
    fileName = " results/sensitivity_analysis_16_29_30__27_07_2023",
    plot_outputs = ["price_mean","price_var","price_autocorr","price_skew"],
    dpi_save = 1200,
    latex_bool = 0
    ) -> None: 
    
    plot_dict = {
        "price_mean": {"title": r"price mean", "colour": "red", "linestyle": "--"},
        "price_var": {"title": r"price var", "colour": "blue", "linestyle": "-"},
        "price_autocorr": {"title": r"price autocorrelation", "colour": "green", "linestyle": "*"},
        "price_skew": {"title": r"price_skew","colour": "orange","linestyle": "-.",},
    }

    titles = [
        r'network density', 
        r'$\Theta$', 
        r'$\sigma_\Theta$', 
        r'$\Gamma$', 
        r'$\sigma_\Gamma$', 
        r'$\sigma_\epsilon$', 
        r'$\% \mathrm{dogmatic}_\Theta$', 
        r'$\% \mathrm{dogmatic}_\Gamma$', 
        r'$\mathrm{AR}(1)$'
    ]


    
    price_mean = load_object(fileName + "/Data", "price_mean")
    price_var = load_object(fileName + "/Data", "price_var")
    price_autocorr = load_object(fileName + "/Data", "price_autocorr")
    price_skew = load_object(fileName + "/Data", "price_skew")
    N_samples = load_object(fileName + "/Data","N_samples" )
    problem = load_object(fileName + "/Data", "problem")
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")

    data_sa_dict_total, data_sa_dict_first = get_plot_data(problem, price_mean, price_var, price_autocorr, price_skew, calc_second_order)

    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)
    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    
    ###############################

    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first,plot_outputs, titles, dpi_save, N_samples, "First")
    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_total,plot_outputs, titles, dpi_save, N_samples, "Total")

    plt.show()

if __name__ == '__main__':
    fileName_Figure_6 = main(
    fileName = "results/sensitivity_analysis_17_50_43_12_10_2023",
    plot_outputs = ["price_mean","price_var","price_autocorr","price_skew"],
    dpi_save = 1200,
    latex_bool = 0
    )

