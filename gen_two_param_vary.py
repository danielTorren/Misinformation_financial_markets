"""Run multiple simulations varying two parameters
A module that use input data to generate data from multiple social networks varying two properties
between simulations so that the differences may be compared. Useful for comparing the influences of these parameters
on each other to generate phase diagrams.



Created: 10/10/2022
"""
# imports
import json
import numpy as np
from logging import raiseExceptions
from matplotlib.colors import Normalize, LogNorm
import numpy as np
from utility import (
    createFolder, 
    save_object, 
    produce_name_datetime,
)
from generate_data import (
    generate_data_parallel_single_explore
)

# module

def produce_param_list_n_double(
    params_dict: dict, variable_parameters_dict: dict[dict]
) -> list[dict]:
    """Creates a list of the param dictionaries. This only varies both parameters at the same time in a grid like fashion.

    Parameters
    ----------
    params_dict: dict,
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
    variable_parameters_dict: dict[dict]
        dictionary of dictionaries containing details for range of parameters to vary.

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []

    for i in variable_parameters_dict["row"]["vals"]:
        for j in variable_parameters_dict["col"]["vals"]:
            params_dict[variable_parameters_dict["row"]["property"]] = i
            params_dict[variable_parameters_dict["col"]["property"]] = j
            for v in range(params_dict["seed_reps"]):
                params_dict["set_seed"] = int(v+1)#seed for 0 and 1 is the same in python so start from 1
                params_list.append(params_dict.copy())  

    return params_list

def generate_vals_variable_parameters_and_norms(variable_parameters_dict):
    """using minimum and maximum values for the variation of a parameter generate a list of
     data and what type of distribution it uses

     Parameters
    ----------
    variable_parameters_dict: dict[dict]
        dictionary of dictionaries  with parameters used to generate attributes, dict used for readability instead of super
        long list of input parameters. Each key in this out dictionary gives the names of the parameter to be varied with details
        of the range and type of distribution of these values found in the value dictionary of each entry.

    Returns
    -------
    variable_parameters_dict: dict[dict]
        Same dictionary but now with extra entries of "vals" and "norm" in the subset dictionaries

    """
    for i in variable_parameters_dict.values():
        if i["divisions"] == "linear":
            i["vals"] = np.linspace(i["min"], i["max"], i["reps"])
            i["norm"] = Normalize()
        elif i["divisions"] == "log":
            i["vals"] = np.logspace(i["min"], i["max"], i["reps"])
            i["norm"] = LogNorm()
        else:
            raiseExceptions("Invalid divisions, try linear or log")
    return variable_parameters_dict

def unpack_and_mean(Data_list, variable_parameters, base_params, axis_mean):
    Data_array = np.asarray(Data_list)
    Data_matrix =  Data_array.reshape((variable_parameters["row"]["reps"], variable_parameters["col"]["reps"], base_params["seed_reps"]))
    Data_matrix_mean = Data_matrix.mean(axis = axis_mean)
    return Data_matrix_mean

def main(
        base_params_load = "constants/base_params_2D.json",
        varied_param_load = "constants/variable_parameters_dict_2D.json", 
        print_simu = 1,
        ):
    
    # load base params
    f_base_params = open(base_params_load)
    base_params = json.load(f_base_params)
    f_base_params.close()

    # load variable params
    f_variable_parameters = open(varied_param_load)
    variable_parameters_dict = json.load(f_variable_parameters)
    f_variable_parameters.close()

    # AVERAGE OVER MULTIPLE RUNS
    variable_parameters_dict = generate_vals_variable_parameters_and_norms(
        variable_parameters_dict
    )

    rootName = "two_param_sweep_average_" + base_params["network_type"]
    fileName = produce_name_datetime(rootName)
    

    params_list = produce_param_list_n_double(base_params, variable_parameters_dict)
    print("TOTAL REPS", len(params_list))

    #Data_list = generate_data_parallel(params_list,print_simu) 
    Data_list = generate_data_parallel_single_explore(params_list, print_simu)

    dev_price_list = [d["dev_price"] for d in Data_list]
    excess_var_list = [d["excess_var"] for d in Data_list]
    excess_autocorr_list = [d["excess_autocorr"] for d in Data_list]
    kurtosis_list = [d["kurtosis"] for d in Data_list]

    dev_price_mean = unpack_and_mean(dev_price_list, variable_parameters_dict,base_params, 2)
    excess_var_mean = unpack_and_mean(excess_var_list, variable_parameters_dict, base_params, 2)
    excess_autocorr_mean = unpack_and_mean(excess_autocorr_list, variable_parameters_dict, base_params, 2)
    kurtosis_mean = unpack_and_mean(kurtosis_list, variable_parameters_dict, base_params, 2)

    # run the simulation
    createFolder(fileName)
    save_object(Data_list, fileName + "/Data", "financial_market_list")
    save_object(dev_price_mean, fileName + "/Data", "dev_price_mean")
    save_object(excess_var_mean, fileName + "/Data", "excess_var_mean")
    save_object(excess_autocorr_mean, fileName + "/Data", "excess_autocorr_mean")
    save_object(kurtosis_mean, fileName + "/Data", "kurtosis_mean")

    save_object(base_params, fileName + "/Data", "base_params")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    print("fileName:", fileName)

    return fileName

if __name__ == "__main__":
    fileName = main(
        base_params_load = "constants/base_params_2D.json",
        varied_param_load = "constants/variable_parameters_dict_2D.json", 
        print_simu = 1,
    )
