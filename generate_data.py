import time
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from matrix_model import Market
from scipy.stats import kurtosis
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

def generate_data_single_explore(params,print_simu = 0):

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
    rational_price =(financial_market.d/ (financial_market.R - 1) + (financial_market.history_theta_t)/ (financial_market.R - financial_market.ar_1_coefficient))
    price_mean = np.mean(financial_market.history_p_t - rational_price)
    price_var = np.var(financial_market.history_p_t) - np.var(rational_price)
    price_autocorr = np.corrcoef(financial_market.history_p_t[:-1],financial_market.history_p_t[1:])[0,1] - financial_market.ar_1_coefficient
    return_kurt = kurtosis(np.asarray(financial_market.history_p_t[1:])/np.asarray(financial_market.history_p_t[:-1]) - 1)
    target_outputs = {"dev_price" : price_mean, 
                      "excess_var": price_var, 
                      "excess_autocorr": price_autocorr,
                      "kurtosis": return_kurt
                      }
    return target_outputs

def generate_data_parallel(params_list,print_simu = 0):
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
    #print(params_list)
    data_parallel = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_data_single)(i, print_simu) for i in params_list
    )
    
    return data_parallel

def generate_data_parallel_single_explore(params_list):
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
    #print(params_list)
    data_parallel = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(generate_data_single_explore)(i) for i in params_list
    )
    
    return data_parallel


def gen_output_surrogate(params):
    
    financial_market = Market(params)

    #### RUN TIME STEPS
    while financial_market.step_count < params["total_steps"]:
        financial_market.next_step()

    returns_timeseries = np.asarray(financial_market.history_p_t[1:])/np.asarray(financial_market.history_p_t[:-1]) - 1

    return returns_timeseries

def generate_data_surrogate(params_list):
    num_cores = multiprocessing.cpu_count()
    returns_timeseries_list = Parallel(n_jobs=num_cores, verbose=10)(
        delayed(gen_output_surrogate)(i) for i in params_list
    )
    
    return returns_timeseries_list

