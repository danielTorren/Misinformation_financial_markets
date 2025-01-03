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

fontsize= 18
ticksize = 16
figsize = (16, 9)
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

def multi_scatter_seperate_total_sensitivity_analysis_plot(
    fileName,data_dict, dict_list, names, N_samples, order,
    plotName = "SW"
):
    """
    Create scatter chart of results.
    """

    fig, axes = plt.subplots(ncols=len(dict_list), nrows=1, constrained_layout=True , sharey=True)#,#sharex=True# figsize=(14, 7) # len(list(data_dict.keys())))
    
    #plt.rc('ytick', labelsize=4) 

    for i, ax in enumerate(axes.flat):
        if order == "First":
            ax.errorbar(
                data_dict[dict_list[i]]["data"]["S1"].tolist(),
                names,
                xerr=data_dict[dict_list[i]]["yerr"]["S1"].tolist(),
                fmt="o",
                ecolor="k",
                markersize=8, 
                capsize=5,
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"]
            )
        else:
            ax.errorbar(
                data_dict[dict_list[i]]["data"]["ST"].tolist(),
                names,
                xerr=data_dict[dict_list[i]]["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                markersize=8, 
                capsize=5,
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"]
            )
        ax.set_title(data_dict[dict_list[i]]["title"])
        #ax.legend(loc='lower right')
        ax.set_xlim(left=0)

    fig.supxlabel(r"%s order Sobol index" % (order))

    plt.tight_layout()

    
    #get the netowrk name from the base_params.pkl file in the Data folder
    networkName = load_object(fileName + "/Data", "base_params")["network_type"]
    misinfo = ["misinfo" if load_object(fileName + "/Data", "base_params")["misinformed_central"] else "info"][0]
    f_png = (
        plotName
        + "/"
        + "%s_sensitivity_analysis_plot.png"
        % (order)
    )
    fig.savefig(f_png, dpi=300, format="png")


def get_plot_data(
    problem: dict,
    price_var: npt.NDArray,
    price_skew: npt.NDArray,
    price_kurtosis: npt.NDArray,
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
    price_skew: npt.NDArray
         values for the variance of Individual identity in the network at the end of the simulation run time. One entry
         for each parameter set tested
    price_kurtosis: npt.NDArray
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

    Si_mu , Si_var , Si_price_kurtosis = analyze_results(problem,price_var,price_skew,price_kurtosis,calc_second_order) 

    total_price_var, first_mu = Si_mu.to_df()
    total_price_skew, first_price_skew = Si_var.to_df()
    total_price_kurtosis,first_price_kurtosis= Si_price_kurtosis.to_df()

    total_data_sa_price_var, total_yerr_price_var = get_data_bar_chart(total_price_var)
    total_data_sa_price_skew, total_yerr_price_skew = get_data_bar_chart(total_price_skew)
    total_data_sa_price_kurtosis, total_yerr_price_kurtosis = get_data_bar_chart(total_price_kurtosis)

    first_yerr_sa_price_var, first_yerr_price_var = get_data_bar_chart(first_mu)
    first_data_sa_price_skew, first_yerr_price_skew = get_data_bar_chart(first_price_skew)
    first_data_sa_price_kurtosis,first_yerr_price_kurtosis= get_data_bar_chart(first_price_kurtosis)

    data_sa_dict_total = {
        "price_var": {
            "data": total_data_sa_price_var,
            "yerr": total_yerr_price_var,
        },
        "price_skew": {
            "data": total_data_sa_price_skew,
            "yerr": total_yerr_price_skew,
        },
        "price_kurtosis": {
            "data": total_data_sa_price_kurtosis,
            "yerr": total_yerr_price_kurtosis,
        },
    }
    data_sa_dict_first = {
        "price_var": {
            "data": first_yerr_sa_price_var,
            "yerr": first_yerr_price_var,
        },
        "price_skew": {
            "data": first_data_sa_price_skew,
            "yerr": first_yerr_price_skew,
        },
        "price_kurtosis": {
            "data": first_data_sa_price_kurtosis,
            "yerr": first_yerr_price_kurtosis,
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
    price_var: npt.NDArray,
    price_skew: npt.NDArray,
    price_kurtosis: npt.NDArray,

    calc_second_order: bool,
) -> tuple:
    """
    Perform sobol analysis on simulation results
    """

    Si_mu = sobol.analyze(
        problem, price_var, calc_second_order=calc_second_order, print_to_console=False
    )
    Si_var = sobol.analyze(
        problem, price_skew, calc_second_order=calc_second_order, print_to_console=False
    )
    Si_price_kurtosis = sobol.analyze(
        problem,
        price_kurtosis,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )


    return  Si_mu , Si_var , Si_price_kurtosis

def main(
    fileName = "results/sensitivity_analysis_10_57_34_10_04_2024",
    plot_outputs = ["price_var","price_skew","price_kurtosis"],
    plotName = "SW"
    ) -> None: 
    
    plot_dict = {
        
        "price_var": {"title": r"Excess Variance", "colour": "blue", "linestyle": "-"},
        "price_skew": {"title": r"Skeweness", "colour": "green", "linestyle": "*"},
        "price_kurtosis": {"title": r"Kurtosis","colour": "orange","linestyle": "-.",},
    }

    titles = [
        r'$\beta$',
        r'$\sigma_{\eta}$', 
        r'$\sigma_{\gamma}$', 
        r'$\sigma_{\epsilon}$', 
        r'$\lambda$', 
        r'$\xi$', 
        r'$w$'
    ]
    
    
    price_var = load_object(fileName + "/Data", "price_var")
    price_skew = load_object(fileName + "/Data", "price_skew")
    price_kurtosis = load_object(fileName + "/Data", "price_kurtosis")
    N_samples = load_object(fileName + "/Data","N_samples" )
    problem = load_object(fileName + "/Data", "problem")
    calc_second_order = load_object(fileName + "/Data", "calc_second_order")

    data_sa_dict_total, data_sa_dict_first = get_plot_data(problem, price_var, price_skew, price_kurtosis, calc_second_order)

    data_sa_dict_first = Merge_dict_SA(data_sa_dict_first, plot_dict)
    data_sa_dict_total = Merge_dict_SA(data_sa_dict_total, plot_dict)
    
    ###############################

    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_first,plot_outputs, titles, N_samples, "First", plotName=plotName)
    multi_scatter_seperate_total_sensitivity_analysis_plot(fileName, data_sa_dict_total,plot_outputs, titles, N_samples, "Total", plotName=plotName)

    plt.show()

if __name__ == '__main__':
    fileName_Figure_6 = main(
    fileName = "results/sensitivity_analysis_10_57_34_10_04_2024",
    plot_outputs = ["price_var","price_skew","price_kurtosis"]
    )

