"""Create consumers

Author: Tommaso Di Francesco and Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
import warnings

class Consumer:
    "Class of consumer"
    def __init__(self, parameters: dict,dogmatic_state, baseline_mean, baseline_var):
        "Construct initial data of Consumer class"
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]
        self.baseline_mean = baseline_mean
        self.baseline_var = baseline_var

        self.prior_mean =  self.baseline_mean
        self.prior_variance = self.baseline_var
        self.theta_expectation = self.prior_mean
        self.theta_expectation_t = self.prior_mean
        self.theta_expectation_tplus1 = self.prior_mean
        self.theta_variance = self.prior_variance
        self.payoff_expectation = parameters["d"] / (parameters["R"] -1)
        self.payoff_variance = parameters["epsilon_variance"]

        self.dogmatic_state = dogmatic_state
        #self.weighting_vector = weighting_vector

        self.R = parameters["R"]
        self.d = parameters["d"]
        self.theta_prior_variance = parameters["theta_variance"]
        self.epsilon_variance = parameters["epsilon_variance"]
        self.gamma_variance = parameters["gamma_variance"]
        self.ar_1_coefficient = parameters["ar_1_coefficient"]
        self.w = parameters["w"]
        self.source_variance = 1
        self.own_sample_variance = 1
        self.avg_source_variance = 1
        self.avg_sample_variance = 1

        if self.save_timeseries_data:
            self.history_theta_expectation = []
            self.history_theta_variance = [] 
            self.history_source_variance = []
            
        else:
            pass


    def compute_source_variance(self, steps):
        prediction = (self.d * self.R)/(self.R -1) + (self.R * self.S_tminus1)/(self.R - self.ar_1_coefficient)
        current_error = (self.d_tminus1 + self.p_tminus1 - prediction)**2
        exp_moving_avg = self.w * current_error + (1 - self.w) * self.source_variance
        if steps < 2:
            source_variance = current_error
        else:
            source_variance = current_error/(steps - 1) + self.source_variance *((steps - 2)/(steps - 1))
        return exp_moving_avg, source_variance
    
    def compute_own_sample_variance(self, steps):
        prediction = (self.d * self.R)/(self.R -1) + (self.R * self.theta_expectation_tminus1)/(self.R - self.ar_1_coefficient)  
        current_error = (self.d_tminus1 + self.p_tminus1 - prediction)**2
        exp_moving_avg = self.w * current_error + (1 - self.w) * self.own_sample_variance
        if steps < 2:
            variance = current_error
        else:
            variance = current_error/(steps - 1) + self.own_sample_variance *((steps - 2)/(steps - 1))
        return exp_moving_avg, variance

    def compute_posterior_mean_variance(self):
        with warnings.catch_warnings(record=True) as w:
            prior_variance = self.prior_variance
            prior_mean = self.prior_mean
            #add priors for cycling, tour de france
            # here we need to force self.theta_variance, so that we do not get an errro in the case of dogmatic agents
            converted_variance = self.theta_prior_variance * self.source_variance[~np.isnan(self.source_variance)]/self.own_sample_variance
            full_signal_variances= np.append(converted_variance, prior_variance)#np.append(self.source_variance[~np.isnan(self.source_variance)], prior_variance)
            full_signal_means = np.append(self.S_tplus1[~np.isnan(self.S_tplus1)], prior_mean) 
            #print("length of mean vector is: ", len(full_signal_means), "length of var vector is: ", len(full_signal_variances))
            #for both mean and variance
            denominator = sum(np.product(np.delete(full_signal_variances, v)) for v in range(len(full_signal_variances)))
            #mean
            numerator_mean =  sum(np.product(np.append(np.delete(full_signal_variances, v),full_signal_means[v])) for v in range(len(full_signal_variances)))
            posterior_mean = (numerator_mean/denominator)        
            posterior_variance = (np.prod(full_signal_variances)/denominator)
        if w:
        # Handle the warning, print it, or take action as needed
            print(f"Warning: {w[0].message}, Keeping prior beliefs")   
            posterior_mean = self.prior_mean
            posterior_variance = self.prior_variance
        return posterior_mean,posterior_variance 

    def compute_payoff_beliefs(self):
        payoff_expectation = (self.d * self.R)/(self.R -1) + (self.R * self.theta_expectation_tplus1)/(self.R - self.ar_1_coefficient)
        if self.dogmatic_state == "theta":
            payoff_variance = self.epsilon_variance + self.theta_prior_variance/(self.R - self.ar_1_coefficient)**2
        elif self.dogmatic_state == "gamma":
            payoff_variance = self.epsilon_variance + (self.theta_prior_variance + self.gamma_variance)/(self.R - self.ar_1_coefficient)**2
        else:
            payoff_variance = self.epsilon_variance + self.theta_variance_tplus1
        return payoff_expectation, payoff_variance

    def append_data(self, X):
        if self.save_timeseries_data:
            self.history_theta_expectation.append(self.payoff_expectation) 
            self.history_theta_variance.append(self.payoff_variance) 
        else:
            pass

    def next_step(self,d_tminus1,p_tminus1, S_tplus1, S_tminus1, steps, informed_expectation = None):
        
        if informed_expectation is None:
            
            self.prior_variance = self.theta_prior_variance
            
        else:
            #First we reset the expectations and variance
            self.prior_mean = informed_expectation# corresponds to the dogmatic case
            self.prior_variance = self.baseline_var

        #recieve dividend and demand
        #compute profit
        self.d_tminus1 = d_tminus1
        self.p_tminus1 = p_tminus1
        self.S_tplus1 = S_tplus1
        self.S_tminus1 = S_tminus1
        self.theta_expectation_tminus1 = self.theta_expectation_t
        self.theta_expectation_t = self.theta_expectation_tplus1
        #update weighting
        self.source_variance, self.avg_source_variance = self.compute_source_variance(steps)
        self.own_sample_variance, self.avg_sample_variance = self.compute_own_sample_variance(steps)
        #compute posterior expectations
        self.theta_expectation_tplus1, self.theta_variance_tplus1 = self.compute_posterior_mean_variance()
        #compute posterior payoff
        self.payoff_expectation, self.payoff_variance = self.compute_payoff_beliefs()
        