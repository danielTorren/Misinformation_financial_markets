# (Mis)Information Diffusion and the Financial Market
This repository contains the code to reproduce the results in the paper (Mis)Information Diffusion and the Financial Market, Di Francesco & Torren-Peraire (2023). 

## Content of the folder

'consumer.py' defines the class of Consumer. A consumer is the agent of the market and we define here all the variables and updating mechanism that are agent specific.
'market.py' defined the class of Financial Market. The financial market regulates the interaction among the agents. All the aggregated variables such as the market price are computed here.
'gen_explore_single_param.py' provides the code to simulate the model for a specific set of the parameters.
'gen_sensitivty_analysis.py' provides the code to generate the Sobol sensitivity analysis.
'gen_two_param_vary.py', provides the code to simulate multiple instances of the model by varying two parameters at the time. Each parameter combination is simulated n times with n different stochastic seeds.
'gen_vary_single_param.py', provides the code to simulate multiple instances of the model by varying one parameter at the time, keeping fixed the stochastic seed.
'generate_data.py', defines all the functions to simulate the model.

