# (Mis)Information Diffusion and the Financial Market
This repository contains the code to reproduce the results in the paper (Mis)Information Diffusion and the Financial Market, Di Francesco & Torren-Peraire (2023). 


## Sensitivity analysis
To run the sensittivity analysis one can:
- Define the base parameters in the file 'base_params.json'. These are the parameters that will be kept constant over the different simulations. Apart from the base parameters described in the paper, one can choose two extra one. 'N_samples' defines the number of samples taken per parameter.
- Define the dictionary of variable parameters in the file 'variable_parameters_dict_SA.json'. This is done by defining the parameter name, the lower and upper bound of the range to be explored and the label of the parameters which will be used for plotting.
- At this point one can run the file 'gen_sensitivty_analysis.py'. This will generate a subfolder in the result folder and provide the user with a filename.
- To plot the results you can run the file 'plot_sensitivity_analysis.py' after changing the filename in the main function. This script will display the first and total order Sobol index and save the picture in the same subfolder in the result folder.

## Contour plot for two parameters varying simultaneously
- One can choose the base parameters in the 'base_params_2D.json" file. In addition to the usual parameters the user can select the number of repetitions with different stochastic seed for each parameter combination by using the parameter "seed_reps".
- The two parameters to be varied and their range can be chosen in the file 'variable_parameters_dict_2D.json' in whcih the user can also choose the grid size.
- Then one can simply run the file 'gen_two_param_vary.py' to generate data in a subfolder in the result folder and the realitve filename.
- The filename can be copied into the main function into the 'plot_two_param_vary.py' file to generate the contour plots. The scritp will display the figures and save them in the same subfolder.

## Description of the main functions 'matrix_model.py'
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

