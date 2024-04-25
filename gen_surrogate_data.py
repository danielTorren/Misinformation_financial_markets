"""Generate data for surrogate model
This version: 9/4/2024
"""

# imports
import json
import numpy as np
from SALib.sample import  saltelli
from utility import (
    createFolder,
    save_object,
    produce_name_datetime,
)
import numpy as np
from SALib.sample import saltelli
from gen_sensitivity_analysis import generate_problem, produce_param_list_SA
from generate_data import generate_data_surrogate
# modules


def main(
        base_params_load = "package/constants/base_params.json",
        varied_param_load = "package/constants/variable_parameters_dict_SA.json"
         ) -> str: 
    
    calc_second_order = False

    # load base params
    f = open(base_params_load)
    base_params = json.load(f)
    N_samples = base_params["N_samples"]

    # load variable params
    f_variable_parameters = open(varied_param_load)
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

    #param_values = sobol.sample(
    #    problem, N_samples, calc_second_order=calc_second_order
    #)  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    param_values = saltelli.sample(
        problem, N_samples, calc_second_order=calc_second_order
    )  # NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters

    #DEAL WITH ROUNDED VARIABLES
    round_variable_list = [x["property"] for x in variable_parameters_dict.values() if x["round"]]

    for i in round_variable_list:
        index_round = problem["names"].index(i)
        param_values[:,index_round] = np.round(param_values[:,index_round])


    params_list_sa = produce_param_list_SA(
        param_values, base_params, variable_parameters_dict
    )

    price_mean, price_var, price_autocorr, price_kurtosis = generate_data_surrogate(
        params_list_sa
    )

    ###################################################

    root = "surrogate_model"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    createFolder(fileName)

    save_object(base_params, fileName + "/Data", "base_params")
    save_object(params_list_sa, fileName + "/Data", "params_list_sa")
    save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
    save_object(problem, fileName + "/Data", "problem")

    save_object(price_mean, fileName + "/Data", "price_mean")
    save_object(price_var, fileName + "/Data", "price_var")
    save_object(price_autocorr, fileName + "/Data", "price_autocorr")
    save_object(price_kurtosis, fileName + "/Data", "price_kurtosis")

    save_object(N_samples , fileName + "/Data","N_samples")
    save_object(calc_second_order, fileName + "/Data","calc_second_order")


    return fileName

if __name__ == '__main__':
    fileName_Figure_14 = main(
    base_params_load = "constants/base_params_surrogate.json",
    varied_param_load = "constants/variable_parameters_dict_SA.json"
)