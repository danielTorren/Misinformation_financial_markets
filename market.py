"""Create social network with individuals
A module that use input data to

Author: Tommaso Di Francesco and Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import numpy as np
import networkx as nx
import numpy.typing as npt
from consumer import Consumer

# modules
class Market:
    """
    Class to represent social network of simulation which is composed of individuals each with identities and behaviours

    ...

    Parameters
    ----------
    parameters : dict
        Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

    Attributes
    ----------
    set_seed : int
        stochastic seed of simulation for reproducibility

    Methods
    -------
    normlize_matrix(matrix: npt.NDArray) ->  npt.NDArray:
        Row normalize an array

    """

    def __init__(self, parameters: dict):
        """
        Constructs all the necessary attributes for the Network object.

        Parameters
        ----------
        parameters : dict
            Dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters

        """

        #######################################################################################
        #Model hyperparameters, dictate what type of run it will be
        self.degroot_aggregation = parameters["degroot_aggregation"]
        self.c_fountain = parameters["c_fountain"]
        self.heterogenous_priors = parameters["heterogenous_priors"]
        self.heterogenous_wealth = parameters["heterogenous_wealth"]
        self.accuracy_weighting = parameters["accuracy_weighting"]
        self.dynamic_weighting_matrix = parameters["dynamic_weighting_matrix"]
        self.endogenous_c_switching = parameters["endogenous_c_switching"]

        #########################################################################################
        #model parameters
        self.set_seed = parameters["set_seed"]
        np.random.seed(self.set_seed)

        self.save_timeseries_data = parameters["save_timeseries_data"]

        self.compression_factor = parameters["compression_factor"]
        self.I = parameters["I"]
        self.I_array = np.arange(self.I)
        self.K = parameters["K"]
        self.R = parameters["R"]
        self.a = parameters["a"]
        self.d = parameters["d"]
        self.theta_sigma = parameters["theta_sigma"]
        self.epsilon_sigma = parameters["epsilon_sigma"]
        self.phi_sigma = parameters["gamma_sigma"] 
        self.zeta_threshold = parameters["zeta_threshold"]
        self.switch_s = parameters["switch_s"]  #elasticty of the switch probability to the accuracy
        if self.dynamic_weighting_matrix:
            self.psi = parameters["psi"]

        self.W_0 = parameters["W_0"]
        
        if self.heterogenous_wealth:
            
            self.non_c_W_0 = parameters["non_c_W_0"]
            self.W_0_list = np.asarray([self.W_0 if i else self.non_c_W_0 for i in self.c_bool_list]) + np.random.normal(0, 5, self.I)
        else:
            self.W_0_list = np.asarray([self.W_0]*self.I)
        
        self.var_0 =parameters["var_0"]
        self.c_info = parameters["c_info"]
        self.beta = parameters["beta"]
        self.delta = parameters["delta"]
        self.theta_mean = parameters["theta_mean"]
        self.gamma_mean = parameters["gamma_mean"]
        self.step_count = 0
        self.total_steps = parameters["total_steps"]
        
        if self.endogenous_c_switching:
            self.switch_vals = np.random.uniform(size = (self.total_steps,self.I)) #use numpy for the sake of seed setting

        self.T_h_prop = parameters["T_h_prop"]
        self.T_h = int(round(self.T_h_prop*self.total_steps))
        #print("self.T_h", self.T_h)

        self.epsilon_t = np.random.normal(0, self.epsilon_sigma, self.total_steps+1)
        self.theta_t = np.random.normal(self.theta_mean, self.theta_sigma, self.total_steps+1) #+1 is for the zeroth step update of the signal
        self.gamma_t = np.random.normal(self.gamma_mean, self.theta_sigma, self.total_steps+1)#np.sin(np.linspace(-4*np.pi, 4*np.pi, self.total_steps + 1)) + 
        self.zeta_t = self.compute_zeta_t()
        
        self.num_c = int(round(parameters["c_prop"]*self.I))
        self.num_notc = self.I - self.num_c
        self.c_bool_list = np.concatenate((np.zeros(self.num_notc), np.ones(self.num_c)), axis = None)
        #print("BEFORE c list",self.c_bool_list )
        
        randomize = np.arange(self.I)
        np.random.shuffle(randomize)# just shuffle, does not create a new array, allows us to set multiple diffenret array with the same shuffle

        #print("randomize", randomize)
        #shuffle order
        self.c_bool_list = self.c_bool_list[randomize]#np.random.shuffle(self.c_bool_list) 
        #print("After c list",self.c_bool_list )

        if self.heterogenous_priors:

            self.priors_beta_a_no_c = parameters["priors_beta_a_no_c"]
            self.priors_beta_b_no_c = parameters["priors_beta_b_no_c"]
            self.mu_0_i_no_c = np.random.beta(self.priors_beta_a_no_c, self.priors_beta_b_no_c, size=self.num_notc) - 0.5

            self.priors_beta_a_c = parameters["priors_beta_a_c"]
            self.priors_beta_b_c = parameters["priors_beta_b_c"]
            self.mu_0_i_c = np.random.beta(self.priors_beta_a_c, self.priors_beta_b_c, size=self.num_c) - 0.5

            self.mu_0_i = np.concatenate((self.mu_0_i_no_c,self.mu_0_i_c), axis = None)
            #print("BEFORE mu_0_i list",self.mu_0_i )
            self.mu_0_i = self.mu_0_i[randomize]
            #print("After mu_0_i list",self.mu_0_i )
        else:
            self.mu_0 = parameters["mu_0"]


        

        #self.c_bool_list = np.random.choice(a = [0,1], size = self.I)
        #print("self.c_bool_list",self.c_bool_list)


        #create network
        self.network_structure = parameters["network_structure"]
        if self.network_structure == "small_world":
            self.K = int(round(parameters["K"]))  # round due to the sampling method producing floats in the Sobol Sensitivity Analysis (SA)
            self.prob_rewire = parameters["prob_rewire"]
        elif self.network_structure == "barabasi_albert_graph":
            self.k_new_node = parameters["k_new_node"]
        elif self.network_structure == "scale_free_directed":
            self.network_alpha = parameters["network_alpha"] # Probability for adding a new node connected to an existing node chosen randomly according to the in-degree distribution.
            self.network_beta = parameters["network_beta"]
            self.network_gamma = parameters["network_gamma"]
            self.network_delta_in = parameters["network_delta_in"]
            self.network_delta_out = parameters["network_delta_in"]

        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        self.weighting_vector_0 = np.array([parameters["phi_tau"], parameters["phi_omega"],parameters["phi_rho"]])

        self.agent_list = self.create_agent_list()


        #update_expectations of agents based on their network and initial signals#WHY ARE WE DOING THIS????
        #for i in self.I_array:
        #    if self.agent_list[i].c_bool:
        #        theta_init = self.theta_t[0]
        #    else:
        #        theta_init = np.nan
        #    self.agent_list[i].expectation_theta_mean, self.agent_list[i].expectation_theta_variance = self.agent_list[i].compute_posterior_mean_variance([theta_init, self.zeta_t[0], self.mu_0])

        self.d_t = self.d #uninformed expectation
        self.p_t = self.d / self.R #uninformed price
        self.X_it = [0]*self.I
        self.lambda_i = self.compute_network_signal()

        if self.save_timeseries_data:
            self.history_p_t = [0]
            self.history_d_t= [0]
            self.history_time = [self.step_count]
            self.history_X_it = [[0]*self.I]
            self.history_weighting_matrix = [self.weighting_matrix]
            self.history_informed_proportion = [parameters["c_prop"]]
        
        #self.update_vector_v = np.vectorize(self.update_vector)

    def compute_zeta_t(self):
        
        """
        DECAYING BROADCAST
        
        zeta_list = []
        zeta_tracker = 0
        decay_factor = 0.8

        for i in range(0,self.total_steps+1):
            if (np.abs(self.theta_t[i]) + np.abs(gamma_t[i]) > self.zeta_threshold):
                zeta = (self.theta_t[i] + gamma_t[i])
            else:
                zeta = zeta_tracker*decay_factor
            zeta_tracker = zeta

            zeta_list.append(zeta)

        return np.asarray(zeta_list)
        """
        """
        INSTANT BROADCAST
        
        """

        #zeta = [(self.theta_t[i] + self.gamma_t[i]) if (np.abs(self.theta_t[i]) + np.abs(self.gamma_t[i]) > self.zeta_threshold) else 0 for i in range(0,self.total_steps+1)]#+1 is for the zeroth step update of the signal
        zeta = self.gamma_t #-self.theta_t #try with self.gamma_t
        return zeta
        

    def normlize_matrix(self, matrix: npt.NDArray) -> npt.NDArray:
        """
        Row normalize an array

        Parameters
        ----------
        matrix: npt.NDArray
            array to be row normalized

        Returns
        -------
        norm_matrix: npt.NDArray
            row normalized array
        """

        row_sums = matrix.sum(axis=1)

        if np.any(row_sums):
            """THIS IS VERY VERY CLUMBSY"""
            mask = row_sums == 0# check which rows are negative
            reduced_array = matrix[~mask]#create new array with only the non zero rows
            reduced_sums = row_sums[~mask]#create list 
            location_zero_elements = np.nonzero(mask)[0]#get location of where the zeros are
            reduced_norm_matrix = reduced_array/reduced_sums[:, np.newaxis]#norm the matrix
            norm_matrix = np.insert(reduced_norm_matrix, location_zero_elements - np.arange(len(location_zero_elements)), matrix[mask], 0)#put the zeros back in, ()
        else:
            norm_matrix = matrix/row_sums[:, np.newaxis]#norm the matrix

        return norm_matrix

    def create_weighting_matrix(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create graph using Networkx library

        Parameters
        ----------
        None

        Returns
        -------
        weighting_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        norm_weighting_matrix: npt.NDArray[float]
            an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
            opinion of agent j as much as j does i's opinion
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """

        if self.network_structure == "small_world":
            G = nx.watts_strogatz_graph(n=self.I, k=self.K, p=self.prob_rewire, seed=self.set_seed)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        if self.network_structure == "barabasi_albert_graph":
            G = nx.barabasi_albert_graph(n=self.I, m=self.k_new_node, seed=self.set_seed)
        if self.network_structure == "scale_free_directed":
            G = nx.scale_free_graph(n=self.I, alpha=self.network_alpha, beta=self.network_beta, gamma=self.network_gamma, delta_in=self.network_delta_in, delta_out=self.network_delta_out)

        weighting_matrix = nx.to_numpy_array(G)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

        print("Network density:", nx.density(G))

        return (
            weighting_matrix,
            norm_weighting_matrix,
            G,
        )
    
    def create_agent_list(self) -> list[Consumer]:
        """
        Create list of consumer objects

        Parameters
        ----------
        None

        Returns
        -------
        agent_list: list[Consumer]
        """ 
        consumer_params = {
            "save_timeseries_data": self.save_timeseries_data,
            "compression_factor": self.compression_factor,
            "var_0":self.var_0,
            "weighting_vector_0": self.weighting_vector_0,
            "R":self.R,
            "a": self.a,
            "d": self.d,
            "epsilon_sigma": self.epsilon_sigma,
            "beta": self.beta,
            "delta":self.delta,
            "c_info": self.c_info,
            "T_h": self.T_h, 
            "c_fountain": self.c_fountain,
            "accuracy_weighting": self.accuracy_weighting,
        }

        if self.heterogenous_priors:
            agent_list = []
            for i in self.I_array:
                consumer_params["mu_0"] = self.mu_0_i[i]
                agent_list.append(Consumer(consumer_params,self.c_bool_list[i],self.W_0_list[i]))
        else:
            consumer_params["mu_0"] = self.mu_0
            agent_list = [
                Consumer(
                    consumer_params,self.c_bool_list[i],self.W_0_list[i]
                )
                for i in self.I_array
            ]

        return agent_list

    def get_consumers_dt_mean_variance(self):

        expectations_theta_mean_vector = np.asarray([i.expectation_theta_mean for i in self.agent_list])
        expectations_theta_variance_vector = np.asarray([i.expectation_theta_variance for i in self.agent_list])

        dt_expectations_mean = self.d + expectations_theta_mean_vector
        dt_expectations_variance = self.epsilon_sigma**2 + expectations_theta_variance_vector

        return dt_expectations_mean, dt_expectations_variance,expectations_theta_mean_vector

    def compute_price(self):

        term_1 = sum((self.dt_expectations_mean)/(self.dt_expectations_variance ))
        term_2 = 1/(sum(1/self.dt_expectations_variance ))
        #print("compute price term 1", term_1)
        #print("compute price term 2", term_2)

        aggregate_price = term_1*term_2/self.R

        #print("aggregate price", aggregate_price)

        return aggregate_price

    def compute_demand(self):
        demand_numerator = self.dt_expectations_mean - self.R*self.p_t
        demand_denominator = self.a*self.dt_expectations_variance
        
        #print("demand_numerator" , demand_numerator)
        #print("demand_denominator",demand_denominator)

        demand_vector = demand_numerator/demand_denominator

        #print("demand_vector",demand_vector, "sum demand_vector SHOULD BE ZERO :", sum(demand_vector))

        return demand_vector

    def compute_dividends(self):
        
        d_t = self.d + self.theta_t[self.step_count] + self.epsilon_t[self.step_count]

        #print("dividend at t", d_t)

        return d_t

    def compute_network_signal(self):

        if self.degroot_aggregation:
            behavioural_attitude_matrix = np.array([(n.expectation_theta_mean ) for n in self.agent_list])
            neighbour_influence = np.matmul(self.weighting_matrix, behavioural_attitude_matrix)
            if np.any(neighbour_influence):#DEAL WITH INDIVIDUALS WHO HAVE NO MATES
                neighbour_influence[neighbour_influence == 0] = np.nan#replace signal for those with no neighbours with nan
        else:
            #mask = self.weighting_matrix > 0
            if np.any([np.any(i) for i in self.weighting_matrix], axis=0):#first any gives list of true or false as to whether the row is all zeros i believe?
                
                mask = [np.any(i) for i in self.weighting_matrix]
                reduced_array = self.weighting_matrix[~mask]#create new array with only the non zero rows
                location_zero_elements = np.nonzero(mask)[0]#get location of where the zeros are

                k_list = [np.random.choice(self.I_array, 1, p=i)[0] for i in reduced_array]#get list of chosen neighbours to imitate
                neighbour_influence = [self.agent_list[k].expectation_theta_mean for k in k_list]#get the influence of each neighbour
                
                neighbour_influence = np.insert(neighbour_influence, location_zero_elements- np.arange(len(location_zero_elements)) , np.nan)#put nan at the location
            else:
                k_list = [
                    np.random.choice(self.I_array, 1, p=self.weighting_matrix[i])[0]
                    for i in self.I_array
                ]  # for each individual select a neighbour using the row of the weighting matrix as the probability
                neighbour_influence = [self.agent_list[k].expectation_theta_mean for k in k_list]

        return neighbour_influence

    def update_vector(self, i,d_t,p_t,X, theta, zeta, lambda_val):
        self.agent_list[i].next_step(d_t,p_t,X, theta, zeta, lambda_val)

    def update_consumers(self):

        theta_v = [self.theta_t[self.step_count] if i.c_bool else np.nan for i in self.agent_list]#those with c recieve signal else they dont?
        #print(theta_v)
        zeta = self.zeta_t[self.step_count]

        for i in self.I_array:
            self.agent_list[i].next_step(self.d_t,self.p_t,self.X_it[i], theta_v[i], zeta, self.lambda_i[i],self.step_count)

        #d_t_v, p_t_v, zeta_v = np.full((self.I),self.d_t), np.full((self.I),self.p_t), np.full((self.I), zeta)
        #self.update_vector_v(self.I_array, d_t_v, p_t_v,self.X_it, theta_v, zeta_v, self.lambda_i)
    
    def update_weighting_matrix(self):
        #asdasda
        norm_ME_array = np.abs((self.d_t -  (self.d + self.expectations_theta_mean_vector))/self.d_t)

        #print("MSE_array",MSE_array, MSE_array.shape)
        alpha_numerator = np.exp(-np.multiply(self.psi, norm_ME_array))
        #print("alpha_numerator",alpha_numerator , alpha_numerator.shape)

        tile_alpha_numerator = np.tile(alpha_numerator, (self.I, 1))
        #print("tile_alpha_numerator", tile_alpha_numerator, tile_alpha_numerator.shape)

        connections_weighting_matrix = (self.adjacency_matrix * tile_alpha_numerator)  # We want only those values that have network connections
        #print("connections_weighting_matrix", connections_weighting_matrix, connections_weighting_matrix.shape)

        norm_weighting_matrix = self.normlize_matrix(
            connections_weighting_matrix
        )  # normalize the matrix row wise

        #print("norm_weighting_matrix", norm_weighting_matrix, norm_weighting_matrix.shape)
        #quit()
        return norm_weighting_matrix

    def update_c_bools(self):
        #repetition FIX
        #norm_ME_array = np.abs((self.d_t -  (self.d + self.expectations_theta_mean_vector))/self.d_t)
        #print(" - norm_ME_array",- norm_ME_array)
        #print("np.exp(-norm_ME_array)",np.exp(-norm_ME_array))
        #P_switch = (1/(1+ self.switch_s*np.exp(-norm_ME_array))) - 0.5
        profit_vector = np.asarray([(self.d_t  - self.p_t/self.R)*self.X_it[i] - self.c_info  if self.agent_list[i].c_bool else (self.d_t  - self.p_t/self.R)*self.X_it[i] for i in self.I_array])
        P_switch = [1/(1+ self.switch_s*np.exp(profit_vector[i])) - 0.5 if profit_vector[i] < 0.0 else 0.0 for i in range(len(profit_vector))]
        #print("P_switch",P_switch)
        #print("self.switch_vals[self.step_count]",self.switch_vals[self.step_count])
        mask = self.switch_vals[self.step_count] <= P_switch
        #print("mask",mask)
        for i in range(len(self.agent_list)):
            if mask[i]:
                self.agent_list[i].c_bool = not self.agent_list[i].c_bool
                if not self.agent_list[i].c_bool:
                    self.agent_list[i].weighting_vector[0] = 0.0
                
        return np.sum([self.agent_list[i].c_bool for i in range(len(self.agent_list))])/len(self.agent_list)           


    def append_data(self):
        self.history_p_t.append(self.p_t)
        self.history_d_t.append(self.d_t)
        self.history_time.append(self.step_count)
        self.history_X_it.append(self.X_it)
        self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_informed_proportion.append(self.informed_proportion)

    def next_step(self):
        """
        Push the simulation forwards one time step.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        

        #compute and return profits
        self.update_consumers()


        #Recieve expectations of mean and variances
        self.dt_expectations_mean, self.dt_expectations_variance, self.expectations_theta_mean_vector = self.get_consumers_dt_mean_variance()

        #Compute aggregate price
        self.p_t = self.compute_price()

        #compute indivdiual demands
        self.X_it = self.compute_demand()

        #simulate dividend
        self.d_t = self.compute_dividends()

        #update network signal
        self.lambda_i = self.compute_network_signal()

        if self.dynamic_weighting_matrix:
           self.weighting_matrix = self.update_weighting_matrix()

        if self.endogenous_c_switching:
            self.informed_proportion = self.update_c_bools()

        self.step_count +=1  
       
        if (self.step_count % self.compression_factor == 0) and (self.save_timeseries_data):
            self.append_data()

        
