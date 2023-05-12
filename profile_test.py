from market import Market

if __name__ == "_main_":

    params = {"save_timeseries_data": 1, 
    "network_structure": "barabasi_albert_graph",
    "degroot_aggregation": 1,
    "heterogenous_priors": 0,
    "heterogenous_wealth":0,
    "endogenous_c_switching": 1,
    "broadcast_quality":0.0,
    "T_h_prop": 0.0,
    "c_prop": 0.4,
    "total_steps": 300,
    "compression_factor": 1,
    "I": 100,
    "K": 10,
    "k_new_node": 5,
    "prob_rewire": 0.1,
    "set_seed": 0,
    "R": 1.01,
    "a": 5.5,
    "d": 1.1,
    "mu_0": 0,
    "theta_mean" : 0.0,
    "gamma_mean": 0.0, 
    "theta_sigma" :0.31622776601,
    "epsilon_sigma": 0.31622776601,
    "gamma_sigma": 0.31622776601,
    "var_0": 1,
    "phi_theta": 0.33,
    "phi_omega": 0.33,
    "W_0": 100,
    "c_info": 0.1,
    "beta": 1,
    "switch_s": 1.0,
    "zeta_threshold": 0.63245553202,
    "error_tolerance" : 0.01,
    "delta":1e-5,
    "non_c_W_0": 80,
    "priors_beta_a_no_c": 2,
    "priors_beta_b_no_c": 3,
    "priors_beta_a_c": 3,
    "priors_beta_b_c": 2,
    "network_alpha" : 0.4,
    "network_beta": 0.4,
    "network_gamma" : 0.2,
    "network_delta_in" : 0.1,
    "network_delta_out" : 0.1
}
    #start_time = time.time()
    financial_market = Market(params)

    #### RUN TIME STEPS
    while financial_market.step_count < params["total_steps"]:
        financial_market.next_step()
    
    #print(
    #    "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
    #    "or %s s" % ((time.time() - start_time)),
    #)