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
        self.heterogenous_priors = parameters["heterogenous_priors"]
        self.heterogenous_wealth = parameters["heterogenous_wealth"]
        self.endogenous_c_switching = parameters["endogenous_c_switching"]
        self.broadcast_quality = parameters["broadcast_quality"]
        self.tol_err = parameters["error_tolerance"]

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
            self.switch_vals = np.random.uniform(size = (self.total_steps,self.I)) #need to determine if switching or not, based on draw <= p(switch)

        self.epsilon_t = np.random.normal(0, self.epsilon_sigma, self.total_steps+1)
        self.theta_t = np.random.normal(self.theta_mean, self.theta_sigma, self.total_steps+1) #+1 is for the zeroth step update of the signal
        self.gamma_t = np.random.normal(self.gamma_mean, self.theta_sigma, self.total_steps+1)#np.sin(np.linspace(-4*np.pi, 4*np.pi, self.total_steps + 1)) + 
        self.zeta_t = self.compute_zeta_t()
        
        self.num_c = int(round(parameters["c_prop"]*self.I))
        self.num_notc = self.I - self.num_c
        self.c_bool_list = np.concatenate((np.zeros(self.num_notc), np.ones(self.num_c)), axis = None)
        np.random.shuffle(self.c_bool_list)# just shuffle, does not create a new array, so that informed people are spread
        self.mu_0 = parameters["mu_0"]

        #create network
        self.K = int(round(parameters["K"]))  # round due to the sampling method producing floats in the Sobol Sensitivity Analysis (SA)
        self.prob_rewire = parameters["prob_rewire"]
        
        (
            self.adjacency_matrix,
            self.nan_adjacency_matrix,
            self.network,
        ) = self.create_adjacency_matrix()

        self.expectations_theta_mean_vector = np.asarray([self.mu_0]*self.I)
        self.S_lambda_matrix = self.get_s_lambda() # get the influence of neighbors

        #### here have to create a weighting matrix that include all of the people and the signal?
        self.weighting_vector_0_matrix = self.gen_init_weighing_matrix(parameters["phi_theta"], parameters["phi_omega"])
        self.agent_list = self.create_agent_list()
        self.d_t = self.d #uninformed expectation
        self.p_t = self.d / self.R #uninformed price
        self.X_it = [0]*self.I
        self.weighting_matrix = self.get_weighting_matrix() # get it back beacuse if c_bool the weighting vector changes. might do it beforehand

        if self.save_timeseries_data:
            self.history_p_t = [0]
            self.history_d_t= [0]
            self.history_time = [self.step_count]
            self.history_X_it = [[0]*self.I]
            #self.history_weighting_matrix = [self.weighting_matrix]
            self.history_informed_proportion = [parameters["c_prop"]]
            self.history_weighting_matrix = [self.weighting_matrix]

    def compute_zeta_t(self):

        zeta = (1-self.broadcast_quality)*self.gamma_t + self.broadcast_quality*self.theta_t #-self.theta_t #try with self.gamma_t
        return zeta

    def create_adjacency_matrix(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create graph using Networkx library

        Parameters
        ----------
        None

        Returns
        -------
        adjacency_matrix: npt.NDArray[bool]
            adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
        ws: nx.Graph
            a networkx watts strogatz small world graph
        """


        G = nx.watts_strogatz_graph(n=self.I, k=self.K, p=self.prob_rewire, seed=self.set_seed)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        adjacency_matrix = nx.to_numpy_array(G)
        print("Network density:", nx.density(G))
        nan_adjacency_matrix = np.copy(adjacency_matrix)
        nan_adjacency_matrix[nan_adjacency_matrix == 0] = np.nan
        return (
            adjacency_matrix,
            nan_adjacency_matrix,
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
            "R":self.R,
            "a": self.a,
            "d": self.d,
            "epsilon_sigma": self.epsilon_sigma,
            "beta": self.beta,
            "delta":self.delta,
            "c_info": self.c_info,
            "T_h": self.T_h,
            "error_tolerance": self.tol_err 
        }


        consumer_params["mu_0"] = self.mu_0
        agent_list = [
            Consumer(
                consumer_params,self.c_bool_list[i],self.W_0_list[i], self.weighting_vector_0_matrix[i]
            )
            for i in self.I_array
        ]

        return agent_list

    def get_s_lambda(self):
        reshape_expectations_theta = self.expectations_theta_mean_vector[:, np.newaxis]
        neighbour_influence = self.nan_adjacency_matrix*reshape_expectations_theta

        return neighbour_influence

    def gen_init_weighing_matrix(self,phi_theta,phi_omega):
        neighbour_count_vector = np.sum(self.adjacency_matrix, axis = 1)        
        inverted_neighbours = (1/neighbour_count_vector)        
        network_weighting_vector = (1- (phi_theta + phi_omega))*inverted_neighbours #initial confidence per neighbor
        reshape_network_weighting_vector = network_weighting_vector[:, np.newaxis]
        network_weighting_matrix = self.nan_adjacency_matrix*reshape_network_weighting_vector #confidence in you adjecency matrix
        S_theta_omega = np.tile(np.array([phi_theta, phi_omega]), (self.I,1)) 
        S_matrix = np.concatenate((S_theta_omega, network_weighting_matrix), axis=1) # generate the full signal matrix, by adding the confidence in the information and misinformaiton to each row
        return S_matrix

    def get_consumers_dt_mean_variance(self):
        expectations_theta_mean_vector = np.asarray([i.expectation_theta_mean for i in self.agent_list])
        expectations_theta_variance_vector = np.asarray([i.expectation_theta_variance for i in self.agent_list])
        dt_expectations_mean = self.d + expectations_theta_mean_vector
        dt_expectations_variance = self.epsilon_sigma**2 + expectations_theta_variance_vector
        return dt_expectations_mean, dt_expectations_variance,expectations_theta_mean_vector

    def compute_price(self):
        term_1 = sum((self.dt_expectations_mean)/(self.dt_expectations_variance))
        term_2 = 1/(sum(1/self.dt_expectations_variance ))
        aggregate_price = (term_1*term_2)/self.R
        return aggregate_price

    def compute_demand(self):
        demand_numerator = self.dt_expectations_mean - self.R*self.p_t
        demand_denominator = self.a*self.dt_expectations_variance
        demand_vector = demand_numerator/demand_denominator
        return demand_vector

    def compute_dividends(self):
        d_t = self.d + self.theta_t[self.step_count] + self.epsilon_t[self.step_count]
        return d_t

    def update_c_bools(self):
        # Excess proifts: return minus RW_0
        profit_vector = np.asarray([(self.d_t  - self.p_t/self.R)*self.X_it[i] - self.c_info  if self.agent_list[i].c_bool else (self.d_t  - self.p_t/self.R)*self.X_it[i] for i in self.I_array])
        P_switch = [(np.tanh(-profit_vector[i])) if profit_vector[i] < 0.0 else 0.0 for i in range(len(profit_vector))] 
        mask = self.switch_vals[self.step_count] <= P_switch
        for i in range(len(self.agent_list)):
            if mask[i]:
                self.agent_list[i].c_bool = not self.agent_list[i].c_bool
                if not self.agent_list[i].c_bool:
                    self.agent_list[i].weighting_vector[0] = 0.0     
        return np.sum([self.agent_list[i].c_bool for i in range(len(self.agent_list))])/len(self.agent_list)           
    
    def update_consumers(self):
        theta_v = [self.theta_t[self.step_count] if i.c_bool else np.nan for i in self.agent_list]#those with c recieve signal else they dont?
        zeta = self.zeta_t[self.step_count]        
        for i in self.I_array:
            self.agent_list[i].next_step(self.d_t,self.p_t,self.X_it[i], theta_v[i], zeta, self.S_lambda_matrix[i],self.step_count)

    def get_weighting_matrix(self):
        return np.asarray([v.weighting_vector for v in self.agent_list])

    def append_data(self):
        self.history_p_t.append(self.p_t)
        self.history_d_t.append(self.d_t)
        self.history_time.append(self.step_count)
        self.history_X_it.append(self.X_it)
        #self.history_weighting_matrix.append(self.weighting_matrix)
        self.history_informed_proportion.append(self.informed_proportion)
        self.history_weighting_matrix.append(self.weighting_matrix)

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
        self.S_lambda_matrix = self.get_s_lambda()

        if self.endogenous_c_switching:
            self.informed_proportion = self.update_c_bools()
        
        self.weighting_matrix = self.get_weighting_matrix()

        self.step_count +=1  
       
        if (self.step_count % self.compression_factor == 0) and (self.save_timeseries_data):
            self.append_data()
        if self.step_count % 10 == 0:
            print(self.step_count)

        
