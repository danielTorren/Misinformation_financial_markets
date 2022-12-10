import time
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from market import Market
import multiprocessing

def generate_data_single(params):

    #generate the inital data, move forward in time and return

    print_simu = 1  # Whether of not to print how long the single shot simulation took

    if print_simu:
        start_time = time.time()

    financial_market = Market(params)

    #### RUN TIME STEPS
    while financial_market.step_count < params["total_steps"]:
        financial_market.next_step()

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return financial_market

def generate_data_parallel(params_list):
    """
    Generate data from a list of parameter dictionaries, parallelize the execution of each single shot simulation

    Parameters
    ----------
    params_list: list[dict],
        list of dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
        Each entry corresponds to a different society.

    Returns
    -------
    data_parallel: list[list[Network]]
        serialized list of networks, each generated with a different set of parameters
    """

    num_cores = multiprocessing.cpu_count()
    #data_parallel = [generate_data(i) for i in params_dict]
    data_parallel = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_data_single)(i) for i in params_list
    )
    
    return data_parallel
