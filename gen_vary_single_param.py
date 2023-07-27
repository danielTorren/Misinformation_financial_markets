"""Generate and save model data whilst varying a single parameter

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 1/12/2022
"""
# imports
import json
import time

import numpy as np
from market import Market
from utility import (
    createFolder, 
    save_object, 
    produce_name_datetime,
)
from generate_data import (
    generate_data_parallel
)

def generate_data(params):

    #generate the inital data, move forward in time and return

    print_simu = 1  # Whether of not to print how long the single shot simulation took

    if print_simu:
        start_time = time.time()

    financial_market = Market(params)

    #### RUN TIME STEPS
    time_counter = 0
    while time_counter < params["steps"]:
        financial_market.next_step()
        time_counter += 1

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return financial_market

def gen_param_list(params: dict, property_list: list, propertprice_autocorried: str) -> list[dict]:

    """
    Produce a list of the dicts for each experiment

    Parameters
    ----------
    params: dict
        base parameters
    porperty_list: list
        list of values for the property to be varied
    propertprice_autocorried: str
        property to be varied

    Returns
    -------
    params_list: list[dict]
        list of parameter dicts, each entry corresponds to one experiment to be tested
    """

    params_list = []
    for i in property_list:
        params[propertprice_autocorried] = i
        params_list.append(
            params.copy()
        )  # have to make a copy so that it actually appends a new dict and not just the location of the params dict
    return params_list


RUN = 1
print_simu = 1

if __name__ == "__main__":
    #load in exogenous parameters
    f = open("constants/base_params.json")
    params = json.load(f)

    propertprice_autocorried = "K"
    property_list = [2, 5, 10]

    rootName = "single_vary_" + propertprice_autocorried
    fileName = produce_name_datetime(rootName)
    print("FILENAME:", fileName)

    params_list = gen_param_list(params, property_list, propertprice_autocorried)
    Data_list = generate_data_parallel(params_list,print_simu)  # run the simulation

    createFolder(fileName)

    save_object(Data_list, fileName + "/Data", "financial_market_list")
    save_object(propertprice_autocorried, fileName + "/Data", "propertprice_autocorried")
    save_object(property_list, fileName + "/Data", "property_list")
    save_object(params, fileName + "/Data", "base_params")

