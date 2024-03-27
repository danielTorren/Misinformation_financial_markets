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


## Description of matrix_model
- self.category vector: stores the category of each agents. 1 is for informed, 0 is for uninformed, -1 is for misinformed;
- if misinformed agents are central we flip this vector, otherwise the order is informed first and misinformed last. This makes so that when using the SBM network the two end up in separate blocks.
When the networkf is scale free then one group is in the central positions. For the Small Worlds network it does not matter, so this is the most convenient configuration to acheive the desired positions.
-self.prior_variance_matrix: matrix containing the own subjective beliefs regarding the variance of theta. It will be 0 for informed and misinformed agents and sigma_eta for the others. If agent i is not connected to agent j then position (i,j) is nan.
- self.prior_mean_vector: contains the prior means. it is updated every time step and then passed to the prior mean matrix.
- self.prior_mean_matrix: matrix containing the prior beliefs of all agents. Position (i,j) is nan if agent i is not connected to agent j.
- self.source_error_matrix: contains the error of all sources computed on the realized payoff y(t+1). We initialize it with the variance of epsilon
- self.implied_variance: we compute it by multiplying the error matrix with the prior_variance_matrix, then divide with the diagonal. In this way row i is divided by the source error associated with agent i.
- self.weighting_matrix: contains the weight implied by the bayesian updating. To calculate first we take the implied variance matrix and fill the NANs with 1. Then we set every element of a row to be the product of the whole row. Finally we divide by the respective element. In this way poistion (i,j) will contain the product of all variances in row i except for the own variance (i,i). Using 1 here is fine since it is the multiplicative identity. The result is the numerator of the weighting matrix. In the same function we compute also the numerator of the posterior variance which is simply given by the product over rows.
Then we create another matrix by filling the elemtns that would have to be NANs in the numerator to 0. In this way when we sum over all the combinations we obtain the right result since 0 is invariant under sum. CAVEAT: when computing the weights we might have some denominator which is very small, if this happens than the weight will possibly overflow. to overcome this we set all values smaller the 1e-16 to be equal to 1e-16.
- self.posterior_mean_vector: now we can just get it by multiply the prior matrix with the weight matrix. Also here we fill the NANs in the prior matrix with zeros, since is does not change the result.

- caveat: we add 1e-16 to the prior variance matrix to avoid divisions by 0. In this way the prior variance of dogmatic agent is 1e-16 and the posterior mean will be almost exaclty equal to the prior mean anyway.
