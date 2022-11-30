from utility import (
    createFolder, 
    load_object, 
)
import numpy as np
fileName = "results/single_shot_steps_3000_I_200_network_structure_small_world_degroot_aggregation_1"
createFolder(fileName)
Data = load_object(fileName + "/Data", "financial_market")

print(Data.__dict__.keys())

weighting_zeta_mean_t_c = [np.mean([i.history_weighting_vector[v][1] for i in Data.agent_list if not i.c_bool]) for v in range(len(Data.history_time))]
print(weighting_zeta_mean_t_c)
weighting_zeta_mean_c = np.mean(weighting_zeta_mean_t_c)
print(weighting_zeta_mean_c)


print([i.history_weighting_vector[1] for i in Data.agent_list if not i.c_bool])
