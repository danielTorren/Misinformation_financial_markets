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
    def __init__(self, parameters: dict, c_bool, W_0):
        "Construct initial data of Consumer class"
        self.save_data = parameters["save_data"]
        self.update_c = parameters["update_c"]
        self.c_fountain = parameters["c_fountain"]

        self.W_0 = W_0
        self.expectation_theta_mean = parameters["mu_0"]
        self.expectation_theta_variance = parameters["var_0"]
        self.weighting_vector = parameters["weighting_vector_0"]
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
            self.c_list = [self.c_0 ,0,0]#for calculating profits from each signal need to know how much each one costs, will only be used if paying as else signal will be np.nan
        else:
            self.c_list = [0,0,0]
            self.weighting_vector[0] = 0#np.nan


        self.signal_variances = self.compute_signal_variances()

        self.theoretical_profit_list_history_array = np.asarray([[0,0,0]])#SHOULD BE THE SAME BUT ARRAY AS self.history_theoretical_profit_list

        if self.save_data:
            self.history_theoretical_X_list = [[0,0,0]]#no demand at start?
            self.history_theoretical_profit_list = [[0,0,0]]#no demand at start?
            self.history_weighting_vector = [list(self.weighting_vector)] 
            self.history_profit = [0]#if in doubt 0!, just so everyhting is the same length!
            self.history_expectation_theta_mean = [self.expectation_theta_mean]
            self.history_expectation_theta_variance = [self.expectation_theta_variance] 
            self.history_lambda_t = [0]#if in doubt 0!
            self.history_c_bool = [self.c_bool]

    def compute_c_status(self,d_t,p_t):

        c_effect = self.compute_profit(d_t,p_t,self.theoretical_X_list[0],self.c_list[0])
        if c_effect <= 0.0:#turn it off if negative profit
            self.c_bool = 0
            self.c_list = [0,0,0]

    def compute_profit(self,d_t,p_t,X_t,c):

        profit = self.R*(self.W_0 - p_t*X_t) + d_t*X_t - c - self.R*self.W_0

        return profit

    def compute_X(self,S):
        E = self.d + S
        V = self.epsilon_sigma

        #print("EEEE", E, "self.R*self.p_t",self.R*self.p_t)
        X_t_source = (E - self.R*self.p_t)/(self.a*V)

        return X_t_source        


    def calc_theoretical_X(self,non_nan_index_list,nan_index_list):
        #Calc the weighting vector
        if not non_nan_index_list:
            #empty list therefore all signal are nan, keep weighting static - DO NOTHING
            theoretical_X_list = [np.nan,np.nan,np.nan] #if weighting is all nan then dont buy anything
        else:
            theoretical_X_list = [self.compute_X(self.S_list[v]) for v in non_nan_index_list]
            
            for v in nan_index_list:
                theoretical_X_list = np.insert(theoretical_X_list, v , 0)# DOES THIS PUT IT IN THE RIGHT PLACE?
        return theoretical_X_list        

    def calc_theoretical_profit_list_history_array(self,theoretical_profit_list,theoretical_profit_list_history_array):
        #print("BEFORE ",theoretical_profit_list_history_array)
        #print("double array",np.asarray([theoretical_profit_list]))

        #new_theoretical_profit_list_history_array = np.insert(arr = theoretical_profit_list_history_array, obj = 0, values = np.asarray([theoretical_profit_list]),   axis=0)
        new_theoretical_profit_list_history_array = np.vstack([theoretical_profit_list_history_array, np.asarray([theoretical_profit_list])])
        #print("test",test)


        #print("BF", new_theoretical_profit_list_history_array)
        if new_theoretical_profit_list_history_array.shape[0] > self.T_h + 1: #when T_h = 0 ie only want current value, want to delete if len is 2, so forth. When T_h = 1, that means present and 1 step back, so want to delete for 3
            new_theoretical_profit_list_history_array = new_theoretical_profit_list_history_array[1:]#select all except first


        #theoretical_profit_list_history_array = np.append(np.asarray([theoretical_profit_list]),theoretical_profit_list_history_array, axis = 0)
        #print("After ",theoretical_profit_list_history_array)
        
        return new_theoretical_profit_list_history_array
    
    def calc_theoretical_profit_list_mean_history(self,non_nan_index_list,nan_index_list,theoretical_X_list):
        # add theoretical profit list to historical theoretical profit list
        # sum over history of each element in historical theoretical profit list

        if not non_nan_index_list:
            #print("PROFIT DO NOTHERING LIST EMPTY, ALL NANs")
            #empty list therefore all signal are nan, keep weighting static - DO NOTHING
            theoretical_profit_list = [np.nan,np.nan,np.nan]
            self.theoretical_profit_list_history_array = self.calc_theoretical_profit_list_history_array(np.asarray(theoretical_profit_list),self.theoretical_profit_list_history_array)
            theoretical_profit_list_mean_history = np.nanmean(self.theoretical_profit_list_history_array, axis=0)#sums over the coloumns
        else:
            #print("PROFIT AT LEAST ONE NON NAN")
            #calc the theoretical profit
            theoretical_profit_list = [self.compute_profit(self.d_t, self.p_t, theoretical_X_list[v], self.c_list[v]) for v in range(len(theoretical_X_list))]
            #print("theoretical_profit_list ",theoretical_profit_list )

            self.theoretical_profit_list_history_array = self.calc_theoretical_profit_list_history_array(np.asarray(theoretical_profit_list),self.theoretical_profit_list_history_array)
            #print("self.theoretical_profit_list_history_array", self.theoretical_profit_list_history_array)

            theoretical_profit_list_mean_history = np.nanmean(self.theoretical_profit_list_history_array, axis=0)# over the coloumns, I THOUTH THE AXIS WOULD BE 1 BUT ITS ZERO??
            #print("vals",theoretical_profit_list_mean_history)
            
            for v in nan_index_list:
                #theoretical_profit_list_mean_history = np.insert(theoretical_profit_list_mean_history, v , np.nan)
                theoretical_profit_list_mean_history[v] = np.nan

        return theoretical_profit_list_mean_history

    def calc_theoretical_profit_list_mean_history_alt(self,non_nan_index_list,nan_index_list,theoretical_X_list):
        # add theoretical profit list to historical theoretical profit list
        # sum over history of each element in historical theoretical profit list

        if not non_nan_index_list:
            #print("PROFIT DO NOTHERING LIST EMPTY, ALL NANs")
            #empty list therefore all signal are nan, keep weighting static - DO NOTHING
            theoretical_profit_list = [np.nan,np.nan,np.nan]
        else:
            #print("PROFIT AT LEAST ONE NON NAN")
            #calc the theoretical profit
            theoretical_profit_list = [self.compute_profit(self.d_t, self.p_t, theoretical_X_list[v], self.c_list[v]) for v in range(len(theoretical_X_list))]

            for v in nan_index_list:
                #theoretical_profit_list_mean_history = np.insert(theoretical_profit_list_mean_history, v , np.nan)
                theoretical_profit_list[v] = np.nan

        return theoretical_profit_list
    
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


    def compute_weighting_vector_demand_profit(self):
        """SPLIT PROCESSS OVER THREE FUNCTIONS, CALC DEMAND, PROFIT THEN WEIGHTING"""
        #weighting_vector = self.weighting_vector#copy it 
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

        #Calc the theoretical X 
        theoretical_X_list = self.calc_theoretical_X(non_nan_index_list,nan_index_list)
        #print(" theoretical_X_list", theoretical_X_list)

        #calc theoretical_profit_list_mean_history
        theoretical_profit_list_mean_history = self.calc_theoretical_profit_list_mean_history(non_nan_index_list,nan_index_list,theoretical_X_list)
        #print("compare",theoretical_profit_list_mean_history,self.calc_theoretical_profit_list_mean_history_alt(non_nan_index_list,nan_index_list,theoretical_X_list))
        #print("theoretical_profit_list_mean_history",theoretical_profit_list_mean_history)

        #calc_weighting
        weighting_vector = self.calc_weighting_vector(non_nan_index_list,nan_index_list,weighting_vector,theoretical_profit_list_mean_history)
        #print("weighting_vector",weighting_vector)
        return np.asarray(theoretical_X_list), np.asarray(theoretical_profit_list_mean_history), np.asarray(weighting_vector) 
        

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
        self.history_theoretical_X_list.append(list(self.theoretical_X_list))
        self.history_theoretical_profit_list.append(list(self.theoretical_profit_list))
        self.history_weighting_vector.append(list(self.weighting_vector))#convert it for plotting
        self.history_profit.append(self.profit)
        self.history_expectation_theta_mean.append(self.expectation_theta_mean) 
        self.history_expectation_theta_variance.append(self.expectation_theta_variance) 
        self.history_lambda_t.append(self.S_list[2])
        self.history_c_bool.append(self.c_bool)

    def next_step(self,d_t,p_t,X_t, S_theta, S_omega, S_lamba):
        #print("NEXT")
        #recieve dividend and demand
        #compute profit
        self.d_t = d_t
        self.p_t = p_t

        self.S_list = [S_theta,S_omega,S_lamba]

        if self.c_bool and self.c_fountain:

            self.profit = self.compute_profit(self.d_t,self.p_t,X_t,self.c_list[0])

            #update weighting
            self.theoretical_X_list = np.array([self.compute_X(self.S_list[0]), 0, 0])
            self.theoretical_profit_list = np.array([self.profit, 0, 0])
            self.weighting_vector = np.array([1, 0, 0])

            self.signal_variances = np.array([0, np.nan, np.nan])
            self.expectation_theta_mean = S_theta
            self.expectation_theta_variance = 0
        else:
            self.profit = self.compute_profit(self.d_t,self.p_t,X_t,self.c_list[0])

            #update weighting
            self.theoretical_X_list, self.theoretical_profit_list, self.weighting_vector = self.compute_weighting_vector_demand_profit()

            self.signal_variances = self.compute_signal_variances()

            #update cost
            if self.c_bool and self.update_c:
                 self.compute_c_status(self.d_t,self.p_t)
        
            #compute posterior expectations
            self.expectation_theta_mean, self.expectation_theta_variance = self.compute_posterior_mean_variance(self.S_list)

        if self.save_data:  
            self.append_data()