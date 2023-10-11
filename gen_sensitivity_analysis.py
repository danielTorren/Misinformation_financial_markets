"""Generate data for sensitivity analysis



Created: 10/10/2022
"""

# imports
import json
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from SALib.sample import sobol
import numpy.typing as npt
from joblib import Parallel, delayed
import multiprocessing
from utility import (
    createFolder,
    save_object,
    produce_name_datetime,
)
from generate_data import generate_data_single
import pyperclip

# modules
def generate_sensitivity_output(params: dict):
    """
    Generate data from a set of parameter contained in a dictionary. Average results over multiple stochastic seeds contained in params["seed_list"]

    """

    print_simu = 0

    price_mean_list = []
    price_var_list = []
    price_autocorr_list = []
    price_skew_list = []

    for v in params["seed_list"]:
        params["set_seed"] = v
        data = generate_data_single(params, print_simu)
        
        price_mean = np.mean(data.history_p_t)
        price_var = np.var(data.history_p_t)
        price_autocorr = np.corrcoef(data.history_p_t,data.history_p_t1)
        price_skew = kurtosis(data.history_p_t)

        price_mean_list.append(price_mean)
        price_var_list.append(price_var)
        price_autocorr_list.append(price_autocorr)
        price_skew_list.append(price_skew)

    stochastic_norm_price_mean = np.mean(price_mean)
    stochastic_norm_price_var = np.mean(price_var)
    stochastic_norm_price_autocorr = np.mean(price_autocorr)
    stochastic_norm_price_skew = np.mean(price_skew)

    return (
    stochastic_norm_price_mean,
    stochastic_norm_price_var,
    stochastic_norm_price_autocorr,
    stochastic_norm_price_skew,
    )

def parallel_run_sa(
    params_dict: dict[dict],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate data for sensitivity analysis for model varying lots of parameters dictated by params_dict, producing output
    measures emissions,mean,variance and coefficient of variance. Results averaged over multiple runs with different stochastic seed

    """

    res = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(generate_sensitivity_output)(i) for i in params_dict
    )
    #res = [generate_sensitivity_output(i) for i in params_dict]

    results_price_mean, results_price_var, results_price_autocorr, results_price_skew = zip(
        *res
    )

    return (
        np.asarray(results_price_mean),
        np.asarray(results_price_var),
        np.asarray(results_price_autocorr),
        np.asarray(results_price_skew)
    )

def generate_problem(
    variable_parameters_dict: dict[dict],
    N_samples: int,
    AV_reps: int,
    calc_second_order: bool,
) -> tuple[dict, str, npt.NDArray]:
    """
    Generate the saltelli.sample given an input set of base and variable parameters, generate filename and folder. Satelli sample used
    is 'a popular quasi-random low-discrepancy sequence used to generate uniform samples of parameter space.' - see the SALib documentation

    Parameters
    ----------
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. See sa_run for example data structure.
    N_samples: int
        Number of samples taken per parameter, If calc_second_order is False, the Satelli sample give N * (D + 2), (where D is the number of parameter) parameter sets to run the model
        .There are then extra runs per parameter set to account for stochastic variation. If calc_second_order is True, then this is N * (2D + 2) parameter sets.
    AV_reps: int
        number of repetitions performed to average over stochastic effects
    calc_second_order: bool
        Whether or not to conduct second order sobol sensitivity analysis, if set to False then only first and total order results will be
        available. Setting to True increases the total number of runs for the sensitivity analysis but allows for the study of interdependancies
        between parameters
    Returns
    -------
    problem: dict
        Outlines the number of variables to be varied, the names of these variables and the bounds that they take
    fileName: str
        name of file where results may be found
    param_values: npt.NDArray
        the set of parameter values which are tested in the sensitivity analysis, generated using saltelli.sample - see:
        https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.saltelli.sample for more details
    """
    D_vars = len(variable_parameters_dict)
    
    if calc_second_order:
        samples = N_samples * (2*D_vars + 2)
    else:
        samples = N_samples * (D_vars + 2)

    print("samples: ", samples)
    print("Total runs: ",samples*AV_reps)

    names_list = [x["property"] for x in variable_parameters_dict.values()]
    bounds_list = [[x["min"], x["max"]] for x in variable_parameters_dict.values()]
    

    problem = {
        "num_vars": D_vars,
        "names": names_list,
        "bounds": bounds_list,
    }

    return problem

def produce_param_list_SA(
    param_values: npt.NDArray, base_params: dict, variable_parameters_dict: dict[dict]
) -> list:
    """
    Generate the list of dictionaries containing informaton for each experiment. We combine the base_params with the specific variation for
    that experiment from param_values and we just use variable_parameters_dict for the property

    Parameters
    ----------
    param_values: npt.NDArray
        the set of parameter values which are tested in the sensitivity analysis, generated using saltelli.sample - see:
        https://salib.readthedocs.io/en/latest/api/SALib.sample.html#SALib.sample.saltelli.sample for more details
    base_params: dict
        This is the set of base parameters which act as the default if a given variable is not tested in the sensitivity analysis.
        See sa_run for example data structure
    variable_parameters_dict: dict[dict]
        These are the parameters to be varied. The data is structured as follows: The key is the name of the property, the value is composed
        of another dictionary which contains itself several properties; "property": the parameter name, "min": the minimum value to be used in
        the sensitivity analysis, "max": the maximum value to be used in the sensitivity analysis, "title": the name to be used when plotting
        sensitivity analysis results. See sa_run for example data structure.

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []
    for i, X in enumerate(param_values):
        base_params_copy = (
            base_params.copy()
        )  # copy it as we dont want the changes from one experiment influencing another
        variable_parameters_dict_toList = list(
            variable_parameters_dict.values()
        )  # turn it too a list so we can loop through it as X is just an array not a dict
        for v in range(len(X)):  # loop through the properties to be changed
            base_params_copy[variable_parameters_dict_toList[v]["property"]] = X[
                v
            ]  # replace the base variable value with the new value for that experiment
        params_list.append(base_params_copy)
    return params_list

def main(
        BASE_PARAMS_LOAD = "package/constants/base_params.json",
        VARIABLE_PARAMS_LOAD = "package/constants/variable_parameters_dict_SA.json"
         ) -> str: 
    
    calc_second_order = False

    # load base params
    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)
    N_samples = base_params["N_samples"]

    # load variable params
    f_variable_parameters = open(VARIABLE_PARAMS_LOAD)
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    ##AVERAGE RUNS
    AV_reps = len(base_params["seed_list"])
    print("Average reps: ", AV_reps)

    problem = generate_problem(
        variable_parameters_dict, N_samples, AV_reps, calc_second_order
    )

        ########################################

    # GENERATE PARAMETER VALUES
    print("problem, N_samples", problem, N_samples, type( N_samples))
    param_values = sobol.sample(
        problem, N_samples, calc_second_order=calc_second_order
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    #DEAL WITH ROUNDED VARIABLES
    round_variable_list = [x["property"] for x in variable_parameters_dict.values() if x["round"]]
    print("round_variable_list:",round_variable_list)
    for i in round_variable_list:
        index_round = problem["names"].index(i)
        param_values[:,index_round] = np.round(param_values[:,index_round])


    params_list_sa = produce_param_list_SA(
        param_values, base_params, variable_parameters_dict
    )

    price_mean, price_var, price_autocorr, price_skew = parallel_run_sa(
        params_list_sa
    )

    ###################################################

    root = "sensitivity_analysis"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)
    #pyperclip.copy(fileName)

    createFolder(fileName)

    save_object(base_params, fileName + "/Data", "base_params")
    save_object(params_list_sa, fileName + "/Data", "params_list_sa")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(problem, fileName + "/Data", "problem")

    save_object(price_mean, fileName + "/Data", "price_mean")
    save_object(price_var, fileName + "/Data", "price_var")
    save_object(price_autocorr, fileName + "/Data", "price_autocorr")
    save_object(price_skew, fileName + "/Data", "price_skew")

    save_object(N_samples , fileName + "/Data","N_samples")
    save_object(calc_second_order, fileName + "/Data","calc_second_order")

    return fileName

if __name__ == '__main__':
    fileName_Figure_6 = main(
    BASE_PARAMS_LOAD = "constants/base_params_SA.json",
    VARIABLE_PARAMS_LOAD = "constants/variable_parameters_dict_SA.json"
)