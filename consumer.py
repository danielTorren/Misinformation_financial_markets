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
    def __init__(self, parameters: dict, c_0):
        "Construct initial data of Consumer class"
        self.W_0 = parameters["W_0"]
        self.expectation_mean = parameters["mu_0"]
        self.expectation_variance = parameters["var_0"]
        self.weighting_vector = parameters["weighting_vector_0"]
        self.R = parameters["R"]
        self.a = parameters["a"]
        self.d = parameters["d"]
        self.epsilon_sigma = parameters["epsilon_sigma"]

        self.beta = parameters["beta"]
        self.delta = parameters["delta"]

        #cost related nonsense
        
        self.c_0 = c_0
        self.c_list = [self.c_0,0,0]

        self.signal_variances = self.compute_signal_variances()
        

    def compute_c_status(self):
        if ~np.isnan(self.weighting_vector[0]):#check if weighting is nan of purchasing signal
            c = self.c_0
        else:
            c = 0
        
        return c

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
            pass
        else:
            X_list = [self.compute_X(self.S_list[v]) for v in non_nan_index_list]

            print("X LIst: THESE ARE ALL NEGATIVE", X_list)

            #calc the theoretical profit
            profit_list = [self.compute_profit(self.d_t, self.p_t, X_list[v], self.c_list[v]) for v in range(len(X_list))]

            #print("profit list", profit_list)

            #plug those into the equation
            denominator_weighting = sum(np.exp(self.beta*profit) for profit in profit_list)

            #print("denominator_weighting",denominator_weighting)

            if not nan_index_list:
                #if all values are present
                #print("THESE ARE A PROBLEM!",np.exp(self.beta*profit_list[2]),denominator_weighting )
                weighting_vector = [np.exp(self.beta*profit)/denominator_weighting for profit in  profit_list]
                #print("after",weighting_vector)
            else:
                #at least 1 nan value
                weighting_vector_short = np.asarray([np.exp(self.beta*profit)/denominator_weighting for profit in  profit_list])
                weighting_vector = weighting_vector_short*(1-sum(self.weighting_vector[v] for v in nan_index_list))
                for v in nan_index_list:
                    weighting_vector = np.insert(weighting_vector, v ,self.weighting_vector[v])#(array, position, what to put)

        return np.asarray(weighting_vector)

    def compute_signal_variances(self):
        signal_variances = -1/(10*np.log(1 + self.delta - self.weighting_vector))
        return signal_variances

    def compute_posterior_mean_variance(self,S_list):
        prior_variance = self.expectation_variance
        prior_mean = self.expectation_mean

        full_signal_variances_dirty = np.append(self.signal_variances, prior_variance)
        full_signal_means_dirty = np.append(S_list, prior_mean)

        nan_mask = ~np.isnan(full_signal_means_dirty)
        full_signal_means = full_signal_means_dirty[nan_mask]
        full_signal_variances = full_signal_variances_dirty[nan_mask]

        denominator = sum(np.product(np.delete(full_signal_variances, v)) for v in range(len(full_signal_variances)))
        posterior_variance = np.prod(full_signal_variances)/denominator

        #mean
        numerator_mean =  sum(np.product(np.append(np.delete(full_signal_variances, v),full_signal_means[v])) for v in range(len(full_signal_variances)))
        #print("numerator_mean",numerator_mean)
        posterior_mean = numerator_mean/denominator

        #print("posterior_mean,posterior_variance:",posterior_mean, posterior_variance)

        return posterior_mean,posterior_variance 

    def next_step(self,d_t,p_t,X_t, S_tau, S_omega, S_rho):
        #recieve dividend and demand
        #compute profit
        self.d_t = d_t
        self.p_t = p_t

        self.S_list = [S_tau,S_omega,S_rho]

        #print("self.S_list",self.S_list)

        self.profit = self.compute_profit(d_t,p_t,X_t,self.c_list[0])

        #update weighting
        #print("BEFORE self.weighting_vector", self.weighting_vector)
        self.weighting_vector = self.compute_weighting_vector()

        #print("AFTER self.weighting_vector", self.weighting_vector)
    
        self.signal_variances = self.compute_signal_variances()

        #update cost
        self.c_list[0] = self.compute_c_status()
        
        #compute posterior expectations
        self.expectation_mean, self.expectation_variance = self.compute_posterior_mean_variance(self.S_list)