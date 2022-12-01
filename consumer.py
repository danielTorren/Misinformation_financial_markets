"""Create consumers

Author: Tommaso Di Francesco and Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt

class Consumer:
    "Class of consumer"
    def __init__(self, parameters: dict, c_bool):
        "Construct initial data of Consumer class"
        self.save_data = parameters["save_data"]
        self.W_0 = parameters["W_0"]
        self.expectation_theta_mean = parameters["mu_0"]
        self.expectation_theta_variance = parameters["var_0"]
        self.weighting_vector = parameters["weighting_vector_0"]
        self.R = parameters["R"]
        self.a = parameters["a"]
        self.d = parameters["d"]
        self.epsilon_sigma = parameters["epsilon_sigma"]


        self.beta = parameters["beta"]
        self.delta = parameters["delta"]


        self.c_0 = parameters["c_info"]
        #cost 
        self.c_bool = c_bool
        if self.c_bool:
            self.c_list = [self.c_0 ,0,0]#for calculating profits from each signal need to know how much each one costs, will only be used if paying as else signal will be np.nan
        else:
            self.c_list = [0,0,0]
            self.weighting_vector[0] = 0#np.nan

        self.signal_variances = self.compute_signal_variances()

        if self.save_data:
            self.history_theoretical_X_list = [[0,0,0]]#no demand at start?
            self.history_theoretical_profit_list = [[0,0,0]]#no demand at start?
            self.history_weighting_vector = [list(self.weighting_vector)] 
            self.history_profit = [self.W_0]#if in doubt 0!, just so everyhting is the same length!
            self.history_expectation_theta_mean = [self.expectation_theta_mean]
            self.history_expectation_theta_variance = [self.expectation_theta_variance] 
            self.history_lambda_t = [0]#if in doubt 0!
            self.history_c_bool = [self.c_bool]

    def compute_c_status(self,d_t,p_t):

        c_effect = self.compute_profit(d_t,p_t,self.theoretical_X_list[0],self.c_list[0]) - self.R*self.W_0
        if c_effect <= 0.0:#turn it off if negative profit
            self.c_bool = 0
            self.c_list = [0,0,0]

    def compute_profit(self,d_t,p_t,X_t,c):

        profit = self.R*(self.W_0 - p_t*X_t) + d_t*X_t - c

        return profit

    def compute_X(self,S):
        E = self.d + S
        V = self.epsilon_sigma

        #print("EEEE", E, "self.R*self.p_t",self.R*self.p_t)
        X_t_source = (E - self.R*self.p_t)/(self.a*V)

        return X_t_source        

    def compute_weighting_vector(self):

        #weighting_vector = self.weighting_vector#copy it 
        non_nan_index_list = []
        nan_index_list = []

        for v in range(len(self.S_list)):
            if np.isnan(self.S_list[v]):
                nan_index_list.append(v)
            else:
                non_nan_index_list.append(v)

        weighting_vector = self.weighting_vector

        if not non_nan_index_list:
            #empty list therefore all signal are nan, keep weighting static - DO NOTHING
            theoretical_X_list = [np.nan,np.nan,np.nan] #if weighting is all nan then dont buy anything
            theoretical_profit_list = [np.nan,np.nan,np.nan]
            pass
        else:
            theoretical_X_list = [self.compute_X(self.S_list[v]) for v in non_nan_index_list]

            #print("X LIst: THESE ARE ALL NEGATIVE", theoretical_X_list)

            #calc the theoretical profit
            theoretical_profit_list = [self.compute_profit(self.d_t, self.p_t, theoretical_X_list[v], self.c_list[v]) for v in range(len(theoretical_X_list))]

            #print("profit list", profit_list)

            #plug those into the equation
            denominator_weighting = sum(np.exp(self.beta*profit) for profit in theoretical_profit_list)

            #print("denominator_weighting",denominator_weighting)

            if not nan_index_list:#if all values are present
                weighting_vector = [np.exp(self.beta*profit)/denominator_weighting for profit in  theoretical_profit_list]
            else:
                #at least 1 nan value
                weighting_vector_short = np.asarray([np.exp(self.beta*profit)/denominator_weighting for profit in  theoretical_profit_list])
                weighting_vector = weighting_vector_short*(1-sum(weighting_vector[v] for v in nan_index_list))#1 - the effect of others?
                for v in nan_index_list:
                    weighting_vector = np.insert(weighting_vector, v ,self.weighting_vector[v])# DOES THIS PUT IT IN THE RIGHT PLACE?
            
            for v in nan_index_list:
                theoretical_X_list = np.insert(theoretical_X_list, v , 0)# DOES THIS PUT IT IN THE RIGHT PLACE?
                theoretical_profit_list = np.insert(theoretical_profit_list, v , np.nan)

        #print("theoretical_X_list",theoretical_X_list)

        return np.asarray(theoretical_X_list), np.asarray(weighting_vector), np.asarray(theoretical_profit_list)
        

    def compute_signal_variances(self):
        signal_variances = -1/(10*np.log(1 + self.delta - self.weighting_vector))
        return signal_variances

    def compute_posterior_mean_variance(self,S_list):
        prior_theta_variance = self.expectation_theta_variance
        prior_theta_mean = self.expectation_theta_mean

        #add priors for cycling
        full_signal_variances_dirty = np.append(self.signal_variances, prior_theta_variance)
        full_signal_means_dirty = np.append(S_list, prior_theta_mean)

        #Only want to use signals available in the posterior calculation
        nan_mask = ~np.isnan(full_signal_means_dirty)
        full_signal_means = full_signal_means_dirty[nan_mask]
        full_signal_variances = full_signal_variances_dirty[nan_mask]

        denominator = sum(np.product(np.delete(full_signal_variances, v)) for v in range(len(full_signal_variances)))
        posterior_theta_variance = np.prod(full_signal_variances)/denominator

        #mean
        numerator_mean =  sum(np.product(np.append(np.delete(full_signal_variances, v),full_signal_means[v])) for v in range(len(full_signal_variances)))
        #print("numerator_mean",numerator_mean)
        posterior_theta_mean = numerator_mean/denominator

        #print("posterior_mean,posterior_variance:",posterior_mean, posterior_variance)

        return posterior_theta_mean,posterior_theta_variance 

    def append_data(self):
        self.history_theoretical_X_list.append(list(self.theoretical_X_list))
        self.history_theoretical_profit_list.append(list(self.theoretical_profit_list))
        self.history_weighting_vector.append(list(self.weighting_vector))#convert it for plotting
        self.history_profit.append(self.profit)
        self.history_expectation_theta_mean.append(self.expectation_theta_mean) 
        self.history_expectation_theta_variance.append(self.expectation_theta_variance) 
        self.history_lambda_t.append(self.S_list[2])
        self.history_c_bool.append(self.c_bool)

    def next_step(self,d_t,p_t,X_t, S_tau, S_omega, S_rho):
        #recieve dividend and demand
        #compute profit
        self.d_t = d_t
        self.p_t = p_t

        self.S_list = [S_tau,S_omega,S_rho]

        #print("self.S_list",self.S_list)

        self.profit = self.compute_profit(self.d_t,self.p_t,X_t,self.c_list[0])

        #update weighting
        #print("BEFORE self.weighting_vector", self.weighting_vector)
        self.theoretical_X_list, self.weighting_vector, self.theoretical_profit_list = self.compute_weighting_vector()

        #print("AFTER self.weighting_vector", self.weighting_vector)
    
        self.signal_variances = self.compute_signal_variances()

        # #update cost
        # if self.c_bool:
        #     self.compute_c_status(self.d_t,self.p_t)
        
        #compute posterior expectations
        self.expectation_theta_mean, self.expectation_theta_variance = self.compute_posterior_mean_variance(self.S_list)

        if self.save_data:  
            self.append_data()