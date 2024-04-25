"""Generate and save model data whilst varying a single parameter

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 1/12/2022
"""
# imports
import json
import time
import pyperclip

import numpy as np
from matrix_model import Market
from utility import (
    createFolder, 
    save_object, 
    produce_name_datetime,
)
from generate_data import (

    generate_data_parallel
)

def generate_data(params, print_simu = False):

    #generate the inital data, move forward in time and return
    if print_simu:
        start_time = time.time()

    financial_market = Market(params)

    #### RUN TIME STEPS
    time_counter = 0
    while time_counter < params["total_steps"]:
        financial_market.next_step()
        time_counter += 1

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return financial_market

def gen_param_list(params: dict, property_list: list, property_varied: str) -> list[dict]:

    """
    Produce a list of the dicts for each experiment

    Parameters
    ----------
    params: dict
        base parameters
    porperty_list: list
        list of values for the property to be varied
    property_varied: str
        property to be varied

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []
    for i in property_list:
        params[property_varied] = i
        params_list.append(
            params.copy()
        )  # have to make a copy so that it actually appends a new dict and not just the location of the params dict
    #print(params_list)
    return params_list

def main(
        base_params_load = "constants/base_params.json",
        varied_param_load = "constants/varied_params.json"
        ):
    
    #load in exogenous parameters
    f = open(base_params_load)
    params = json.load(f)

    f_varied = open(varied_param_load)
    var_param = json.load(f_varied)

    property_varied = var_param["property"]
    property_list = list(range(var_param["min"], var_param["max"]))

    rootName = params["network_type"] + "single_vary_" + property_varied
    fileName = produce_name_datetime(rootName)
    
    #copy the filename variable to the clipboard
    pyperclip.copy(fileName)

    params_list = gen_param_list(params, property_list, property_varied)

    Data_list = generate_data_parallel(params_list) 
    print(dir(Data_list[0]))
    # run the simulation
    createFolder(fileName)

    save_object(Data_list, fileName + "/Data", "financial_market_list")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_list, fileName + "/Data", "property_list")
    save_object(params, fileName + "/Data", "base_params")
    print("FILENAME:", fileName) 

    return fileName

if __name__ == "__main__":
    fileName = main(
        base_params_load = "constants/base_params.json",
        varied_param_load = "constants/varied_params.json"
    )