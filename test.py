import yaml
import io

# Define data
data = {
    "save_data": 1, 
    "network_structure": "barabasi_albert_graph",
    "degroot_aggregation": 1,
    "c_fountain": 0,
    "update_c": 0,
    "steps": 1000,
    "I": 100,
    "K": 10,
    "prob_rewire": 0.1,
    "set_seed": 0,
    "R": 1.01,
    "a": 5.5,
    "d": 1.1,
    "mu_0": 0,
    "theta_sigma" : 0.31622776601,
    "epsilon_sigma": 0.31622776601,
    "gamma_sigma": 0.31622776601,
    "var_0": 0.1,
    "phi_tau": 0.33,
    "phi_omega": 0.33,
    "phi_rho": 0.34,
    "W_0": 100,
    "c_info": 0.1,
    "beta": 1,
    "zeta_threshold": 0.63245553202,
    "delta":1e-5,
    "T_h_prop": 0.05,
    "c_prop": 0.3,
    "k_new_node": 6,
    "non_c_W_0_prop": 0.8
}




# Write YAML file
with io.open('data.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

# Read YAML file
with open("data.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)

print(data == data_loaded)