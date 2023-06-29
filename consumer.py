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
    def __init__(self, parameters: dict, W_0, weighting_vector,dogmatic_state, baseline_theta_mean, baseline_theta_var, adj_vector):
        "Construct initial data of Consumer class"
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]
        self.W_0 = W_0
        self.baseline_theta_mean = baseline_theta_mean
        self.baseline_theta_var = baseline_theta_var
        self.expectation_theta_mean =  self.baseline_theta_mean
        self.expectation_theta_variance = self.baseline_theta_var
        self.dogmatic_state = dogmatic_state
        self.weighting_vector = weighting_vector
        self.R = parameters["R"]
        self.a = parameters["a"]
        self.d = parameters["d"]
        self.beta = parameters["beta"]
        self.delta = parameters["delta"]
        self.ar_1_coefficient = parameters["ar_1_coefficient"]
        self.adj_vector = np.where(adj_vector == 0, np.nan, adj_vector)
        self.signal_variances = self.compute_signal_variances()

        if self.save_timeseries_data:
            #self.history_weighting_vector = [list(self.weighting_vector)] 
            self.history_profit = [0]#if in doubt 0!, just so everyhting is the same length!
            self.history_expectation_theta_mean = [self.expectation_theta_mean]
            self.history_expectation_theta_variance = [self.expectation_theta_variance] 

    def compute_profit(self,d_t,p_t,X_t):
        profit = self.R*(self.W_0 - p_t*X_t) + d_t*X_t - self.R*self.W_0

        return profit        

    def calc_squared_error_norm(self, d_t, d, S_k):
        full_errors = (((d_t - (d + S_k)))**2)*self.adj_vector
        #full_errors = np.abs((d_t - (d + S_k)))*self.adj_vector
        #or in percentage term
        #full_errors = np.abs((d_t - (d + S_k))/d_t)*self.adj_vector
        return full_errors #full_errors[~np.isnan(full_errors)]

    def calc_weighting_vector(self,squared_error_array): 
        denominator_weighting = np.nansum(np.exp(-self.beta*squared_error_array))

        weighting_vector = (np.exp(-self.beta*squared_error_array))/denominator_weighting

        #weighting_vector = [np.exp(-self.beta*squared_error)/denominator_weighting for squared_error in squared_error_array]
        return weighting_vector#np.asarray(weighting_vector)

    def compute_weighting_vector(self):
        squared_error_array = self.calc_squared_error_norm(self.d_t, self.d, self.S_array)
        weighting_vector = self.calc_weighting_vector(squared_error_array)
        return weighting_vector 
        
    def compute_signal_variances(self):
        signal_variances = 1/(self.delta + self.weighting_vector) - 1#delta as some of the weightings may be zero? but this is unlikley? delta should be very very very small
        return signal_variances

    def compute_posterior_mean_variance(self,S_array):
        prior_theta_variance = self.expectation_theta_variance
        prior_theta_mean = self.expectation_theta_mean
        #add priors for cycling, tour de france
        full_signal_variances= np.append(self.signal_variances[~np.isnan(self.signal_variances)], prior_theta_variance)
        full_signal_means = np.append(S_array[~np.isnan(S_array)], prior_theta_mean) 
        #print("length of mean vector is: ", len(full_signal_means), "length of var vector is: ", len(full_signal_variances))
        #for both mean and variance
        denominator = sum(np.product(np.delete(full_signal_variances, v)) for v in range(len(full_signal_variances)))
        #mean
        numerator_mean =  sum(np.product(np.append(np.delete(full_signal_variances, v),full_signal_means[v])) for v in range(len(full_signal_variances)))
        posterior_theta_mean = numerator_mean/denominator
        #print(numerator_mean)
        #variance
        posterior_theta_variance = np.prod(full_signal_variances)/denominator
        return posterior_theta_mean,posterior_theta_variance 

    def append_data(self):
        #self.history_weighting_vector.append(list(self.weighting_vector))#convert it for plotting
        self.history_profit.append(self.profit)
        self.history_expectation_theta_mean.append(self.expectation_theta_mean) 
        self.history_expectation_theta_variance.append(self.expectation_theta_variance) 

    def next_step(self,d_t,p_t,X_t,S,steps, expectation_theta_mean = None):
        
        if expectation_theta_mean is None:
            #pass
            self.expectation_theta_mean = self.expectation_theta_mean * self.ar_1_coefficient
            self.expectation_theta_variance = self.baseline_theta_var#Normal case
        else:
            #First we reset the expectations and variance
            self.expectation_theta_mean = expectation_theta_mean# corresponds to the dogmatic case
            self.expectation_theta_variance = 0#same for all time!

        #recieve dividend and demand
        #compute profit
        self.d_t = d_t
        self.p_t = p_t
        self.S_array = S
        self.profit = self.compute_profit(self.d_t,self.p_t,X_t)

        #update weighting
        self.weighting_vector = self.compute_weighting_vector()
        self.signal_variances = self.compute_signal_variances()
    
        #compute posterior expectations
        self.expectation_theta_mean, self.expectation_theta_variance = self.compute_posterior_mean_variance(self.S_array)

        if  (steps % self.compression_factor == 0) and (self.save_timeseries_data):  
            self.append_data()
            
            