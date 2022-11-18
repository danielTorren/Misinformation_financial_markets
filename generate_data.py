"""Generate and save model data

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
# imports
import json
import time

import numpy as np
from market import Market
from utility import (
    createFolder, 
    save_object, 
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

RUN = 1

if __name__ == "__main__":
    #load in exogenous parameters
    f = open("constants/base_params.json")
    params = json.load(f)

    fileName = "results/test"
    print("FILENAME:", fileName)

    Data = generate_data(params)  # run the simulation

    createFolder(fileName)

    save_object(Data, fileName + "/Data", "financial_market")

