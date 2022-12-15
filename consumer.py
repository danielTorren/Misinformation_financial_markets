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
    def __init__(self, parameters: dict, c_bool, W_0, weighting_vector):
        "Construct initial data of Consumer class"
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]

        self.W_0 = W_0
        self.expectation_theta_mean = parameters["mu_0"]
        self.expectation_theta_variance = parameters["var_0"]
        
        self.weighting_vector = weighting_vector
        self.R = parameters["R"]
        self.a = parameters["a"]
        self.d = parameters["d"]
        self.epsilon_sigma = parameters["epsilon_sigma"]
        self.T_h = parameters["T_h"]
        self.beta = parameters["beta"]
        self.delta = parameters["delta"]
        self.c_0 = parameters["c_info"]

        #cost 
        self.c_bool = c_bool

        if self.c_bool:
            pass
        else:
            self.weighting_vector[0] = 0#np.nan

        self.signal_variances = self.compute_signal_variances()

        if self.save_timeseries_data:
            self.history_weighting_vector = [list(self.weighting_vector)] 
            self.history_profit = [0]#if in doubt 0!, just so everyhting is the same length!
            self.history_expectation_theta_mean = [self.expectation_theta_mean]
            self.history_expectation_theta_variance = [self.expectation_theta_variance] 
            #self.history_lambda_t = [0]#if in doubt 0!
            self.history_c_bool = [self.c_bool]

    def compute_profit(self,d_t,p_t,X_t):
        if self.c_bool:
            profit = self.R*(self.W_0 - p_t*X_t) + d_t*X_t - self.c_0 - self.R*self.W_0
        else:
            profit = self.R*(self.W_0 - p_t*X_t) + d_t*X_t - self.R*self.W_0

        return profit

    def compute_X(self,S):
        E = self.d + S
        V = self.epsilon_sigma

        #print("EEEE", E, "self.R*self.p_t",self.R*self.p_t)
        X_t_source = (E - self.R*self.p_t)/(self.a*V)

        return X_t_source            

    
    def calc_weighting_vector(self,non_nan_index_list,nan_index_list,weighting_vector,theoretical_profit_list_mean_history):
        if not non_nan_index_list:
            #empty list therefore all signal are nan, keep weighting static - DO NOTHING
            #print("DO NOTHERING LIST EMPTY, ALL NANs")
            pass
        else:
            #print("AT LEAST ONE NON NAN")
            #plug those into the equation
            denominator_weighting = sum(np.exp(self.beta*theoretical_profit_list_mean_history [i]) for i in non_nan_index_list)

            #print("denominator_weighting",denominator_weighting)

            if not nan_index_list:#if all values are present
                #print("ALL VALUES PRESENT")
                weighting_vector = [np.exp(self.beta*profit)/denominator_weighting for profit in theoretical_profit_list_mean_history]
            else:
                #print("AT LEAST ONE NAN")
                #at least 1 nan value
                #print("non nan profit", [self.beta*theoretical_profit_list_mean_history[i] for i in non_nan_index_list],[np.exp(self.beta*theoretical_profit_list_mean_history[i]) for i in non_nan_index_list],[np.exp(self.beta*theoretical_profit_list_mean_history[i])/denominator_weighting for i in non_nan_index_list])
                weighting_vector_short = np.asarray([np.exp(self.beta*theoretical_profit_list_mean_history[i])/denominator_weighting for i in non_nan_index_list])
                #print("weighting_vector_short",weighting_vector_short)
                weighting_vector = weighting_vector_short*(1-sum(weighting_vector[v] for v in nan_index_list))#1 - the effect of others?
                #print("before weighting_vector",weighting_vector)
                for v in nan_index_list:
                    weighting_vector = np.insert(weighting_vector, v ,self.weighting_vector[v])# DOES THIS PUT IT IN THE RIGHT PLACE?
        
        #print("weighting_vector: c",weighting_vector,self.c_bool)
        return weighting_vector

    def calc_squared_error_norm(self, d_t, d, S_k):
        #print("(d_t - (d + S_k))/d_t", (d_t - (d + S_k))/d_t)
        return np.abs((d_t - (d + S_k)))/d_t

    def calc_weighting_vector_accuracy(self,non_nan_index_list,nan_index_list,weighting_vector,squared_error_list):
        
        if not non_nan_index_list:#empty list therefore all signal are nan, keep weighting static - DO NOTHING
            #print("DO NOTHERING LIST EMPTY, ALL NANs")
            pass
        else:#at least one non nan in signal
            #plug those into the equation
            denominator_weighting = sum(np.exp(-self.beta*squared_error_list[i]) for i in non_nan_index_list)

            #print("denominator_weighting",denominator_weighting)

            if not nan_index_list:#if all values are present
                #print("ALL VALUES PRESENT")
                weighting_vector = [np.exp(-self.beta*squared_error)/denominator_weighting for squared_error in squared_error_list]
            else:
                #print("AT LEAST ONE NAN")
                #at least 1 nan value
                #print("non nan profit", [self.beta*theoretical_profit_list_mean_history[i] for i in non_nan_index_list],[np.exp(self.beta*theoretical_profit_list_mean_history[i]) for i in non_nan_index_list],[np.exp(self.beta*theoretical_profit_list_mean_history[i])/denominator_weighting for i in non_nan_index_list])
                
                """
                weighting_vector_short = np.asarray([np.exp(-self.beta*squared_error_list[i])/denominator_weighting for i in non_nan_index_list])
                print("weighting_vector_short",weighting_vector_short)

                #THIS IS THE BIT THAT IS MESSING THINGS UP AT THE MOMENT, SEE WHAT hAPPEnS IF I REMVOE
                weighting_vector = weighting_vector_short*(1-sum(weighting_vector[v] for v in nan_index_list))#1 - the effect of others?
                print("before weighting_vector",weighting_vector)
                """
                weighting_vector = np.asarray([np.exp(-self.beta*squared_error_list[i])/denominator_weighting for i in non_nan_index_list])
                #print("weighting_vector_short",weighting_vector)

                for v in nan_index_list:
                    weighting_vector = np.insert(weighting_vector, v ,self.weighting_vector[v])# DOES THIS PUT IT IN THE RIGHT PLACE?
        
        #print("weighting_vector: c",weighting_vector,self.c_bool)
        return weighting_vector

    def compute_weighting_vector_accuracy(self):
        non_nan_index_list = []
        nan_index_list = []

        for v in range(len(self.S_list)):
            if np.isnan(self.S_list[v]):
                nan_index_list.append(v)
            else:
                non_nan_index_list.append(v)

        #print("non_nan_index_list ",non_nan_index_list )
        #print("nan_index_list ",nan_index_list )

        weighting_vector = self.weighting_vector

        squared_error_list = self.calc_squared_error_norm(self.d_t, self.d, np.asarray(self.S_list))
        weighting_vector = self.calc_weighting_vector_accuracy(non_nan_index_list,nan_index_list,weighting_vector,squared_error_list)     

        return np.asarray(weighting_vector)   
        

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

        posterior_theta_mean = numerator_mean/denominator

        return posterior_theta_mean,posterior_theta_variance 

    def append_data(self):
        self.history_weighting_vector.append(list(self.weighting_vector))#convert it for plotting
        self.history_profit.append(self.profit)
        self.history_expectation_theta_mean.append(self.expectation_theta_mean) 
        self.history_expectation_theta_variance.append(self.expectation_theta_variance) 
        #self.history_lambda_t.append(self.S_list[2])
        self.history_c_bool.append(self.c_bool)

    def next_step(self,d_t,p_t,X_t, S_theta, S_omega, S_lamba,steps):
        #print("NEXT")
        #recieve dividend and demand
        #compute profit
        self.d_t = d_t
        self.p_t = p_t

        #self.S_list = [S_theta,S_omega,S_lamba]
        self.S_list = [S_theta,S_omega] + list(S_lamba)
        #print("S_list", self.S_list)
        #quit()

        #print("self.d_t,self.p_t,X_t", self.d_t,self.p_t,X_t)

        self.profit = self.compute_profit(self.d_t,self.p_t,X_t)
        #print("self.profit",self.profit)

        #update weighting
        self.weighting_vector = self.compute_weighting_vector_accuracy()
        #print("self.weighting_vector ", self.weighting_vector )

        self.signal_variances = self.compute_signal_variances()
        #print("self.signal_variances", self.signal_variances)
    
        #compute posterior expectations
        self.expectation_theta_mean, self.expectation_theta_variance = self.compute_posterior_mean_variance(self.S_list)
        #print("self.expectation_theta_mean",self.expectation_theta_mean)
        #print("self.expectation_theta_variance",self.expectation_theta_variance)
        #quit()

        if  (steps % self.compression_factor == 0) and (self.save_timeseries_data):  
            self.append_data()