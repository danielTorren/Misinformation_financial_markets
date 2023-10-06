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
    def __init__(self, parameters: dict, weighting_vector ,dogmatic_state, baseline_mean, baseline_var, adj_vector, id):
        "Construct initial data of Consumer class"
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]
        self.baseline_mean = baseline_mean
        self.baseline_var = baseline_var

        self.prior_mean =  self.baseline_mean
        self.prior_variance = self.baseline_var
        self.theta_expectation = self.prior_mean
        self.theta_expectation_t1 = 0
        self.theta_variance = self.prior_variance

        self.id = id

        self.dogmatic_state = dogmatic_state
        self.weighting_vector = weighting_vector

        self.R = parameters["R"]
        self.d = parameters["d"]
        self.theta_prior_variance = parameters["theta_variance"]
        self.ar_1_coefficient = parameters["ar_1_coefficient"]
        self.adj_vector = np.where(adj_vector == 0, np.nan, adj_vector)
        self.source_variance = 1
        self.own_sample_variance = 1
        self.avg_source_variance = 1
        self.avg_sample_variance = 1

        self.cumulative_profit = 0

        if self.save_timeseries_data:
            self.history_profit = [0]#if in doubt 0!, just so everyhting is the same length!
            self.history_cumulative_profit = [0]
            self.history_prior_mean = [self.prior_mean]
            self.history_prior_variance = [self.prior_variance] 
            self.history_source_variance = [self.source_variance]
            self.history_X_t = [0]
        else:
            self.history_cumulative_profit = [0]

    def compute_profit(self,d_t, p_t, p_t1, X_t1):
        profit = (d_t + p_t - self.R * p_t1)*X_t1
        return profit        


    def compute_source_variance(self, d_t, p_t, S_k, steps):
        prediction = (self.d * self.R)/(self.R -1) + (self.R * S_k)/(self.R - self.ar_1_coefficient)
        
        current_error = (d_t + p_t - prediction)**2
        
        if steps < 2:
            source_variance = current_error
        else:
            source_variance = current_error/(steps - 1) + self.source_variance *((steps - 2)/(steps - 1))
        return current_error, source_variance
    
    def compute_own_sample_variance(self, d_t, p_t, steps):
        prediction = (self.d * self.R)/(self.R -1) + (self.R * self.theta_expectation_t1)/(self.R - self.ar_1_coefficient)  
        current_error = (d_t + p_t - prediction)**2
        if steps < 2:
            variance = current_error
        else:
            variance = current_error/(steps - 1) + self.own_sample_variance *((steps - 2)/(steps - 1))
        return current_error, variance

    def compute_posterior_mean_variance(self,S_array):
        prior_variance = self.prior_variance
        prior_mean = self.prior_mean
        #add priors for cycling, tour de france
        # here we need to force self.theta_variance, so that we do not get an errro in the case of dogmatic agents
        converted_variance = self.theta_prior_variance * self.source_variance[~np.isnan(self.source_variance)]/self.own_sample_variance
        full_signal_variances= np.append(converted_variance, prior_variance)#np.append(self.source_variance[~np.isnan(self.source_variance)], prior_variance)
        full_signal_means = np.append(S_array[~np.isnan(S_array)], prior_mean) 
        #print("length of mean vector is: ", len(full_signal_means), "length of var vector is: ", len(full_signal_variances))
        #for both mean and variance
        denominator = sum(np.product(np.delete(full_signal_variances, v)) for v in range(len(full_signal_variances)))
        #mean
        numerator_mean =  sum(np.product(np.append(np.delete(full_signal_variances, v),full_signal_means[v])) for v in range(len(full_signal_variances)))
        posterior_mean = (numerator_mean/denominator)        
        posterior_variance = (np.prod(full_signal_variances)/denominator)
        
        return posterior_mean,posterior_variance 

    def append_data(self, X):
        if self.save_timeseries_data:
            self.history_profit.append(self.profit)
            self.history_cumulative_profit.append(self.cumulative_profit)
            self.history_prior_mean.append(self.prior_mean) 
            self.history_prior_variance.append(self.prior_variance) 
            self.history_X_t.append(X)
        else:
            self.history_cumulative_profit.append(self.cumulative_profit)

    def next_step(self,d_t,p_t, p_t1,X_t,X_t1, S, S_t1, steps, informed_expectation = None):
        
        if informed_expectation is None:
            
            self.prior_variance = self.theta_prior_variance
            
        else:
            #First we reset the expectations and variance
            self.prior_mean = informed_expectation# corresponds to the dogmatic case
            self.prior_variance = self.baseline_var

        #recieve dividend and demand
        #compute profit
        self.d_t = d_t
        self.p_t = p_t
        self.p_t1 = p_t1
        self.S_array = S
        self.S_array_t1 = S_t1
        self.profit = self.compute_profit(self.d_t,self.p_t, self.p_t1, X_t1)
        self.cumulative_profit += self.profit

        #update weighting
        self.source_variance, self.avg_source_variance = self.compute_source_variance(self.d_t, self.p_t, self.S_array_t1, steps)
        self.own_sample_variance, self.avg_sample_variance = self.compute_own_sample_variance(self.d_t, self.p_t, steps)
        #compute posterior expectations
        self.theta_expectation_t1 = self.theta_expectation
        self.theta_expectation, self.theta_variance = self.compute_posterior_mean_variance(self.S_array)

        if  (steps % self.compression_factor == 0):  
            self.append_data(X_t)
        