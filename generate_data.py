import time
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from market import Market
import multiprocessing

def generate_data_single(params,print_simu):

    #generate the inital data, move forward in time and return

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

def generate_data_parallel(params_list,print_simu):
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
    print(params_list)
    data_parallel = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_data_single)(i, print_simu) for i in params_list
    )
    
    return data_parallel
