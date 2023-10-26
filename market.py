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
import random

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

        #########################################################################################
        #model parameters
        self.set_seed = parameters["set_seed"]
        np.random.seed(self.set_seed)
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]
        self.network_type = parameters["network_type"]
        self.network_density = parameters["network_density"]
        self.I = int(round(parameters["I"]))
        self.K = int(round((self.I - 1)*self.network_density)) #reverse engineer the links per person using the density  d = 2m/n(n-1) where n is nodes and m number of edges
      # round due to the sampling method producing floats in the Sobol Sensitivity Analysis (SA)

        self.R = parameters["R"]
        self.a = parameters["a"]
        self.d = parameters["d"]
        self.theta_mean = parameters["theta_mean"]
        self.theta_sigma = parameters["theta_sigma"]
        self.gamma_mean = parameters["gamma_mean"]
        self.gamma_sigma = parameters["gamma_sigma"]
        self.epsilon_sigma = parameters["epsilon_sigma"]
        self.ar_1_coefficient = parameters["ar_1_coefficient"]  #elasticty of the switch probability to the accurac
        self.prob_rewire = parameters["prob_rewire"]

        self.step_count = 1 #we start at t=1 so that the previous time step is t = 0
        self.total_steps = parameters["total_steps"]

        self.epsilon_t = np.random.normal(0, self.epsilon_sigma, self.total_steps+2)
        #We change both theta and gamma to be random walks, that is ar(1) processes with coefficient = 1. shocks never dissipate
        self.theta_t = self.generate_ar1(0,self.ar_1_coefficient, self.theta_mean, self.theta_sigma, self.total_steps+2) #np.cumsum(np.random.normal(self.theta_mean, self.theta_sigma, self.total_steps+1)) #+1 is for the zeroth step update of the signal
        self.gamma_t = self.theta_t + np.random.normal(self.gamma_mean, self.gamma_sigma, self.total_steps+2)
        self.dividend_vector = self.d + self.theta_t + self.epsilon_t
        
        self.num_dogmatic_theta = int(np.floor(self.I*parameters["proportion_dogmatic_theta"])) #number of dogmatic theta
        #print("self.num_dogmatic_theta", self.num_dogmatic_theta, self.I)
        self.num_dogmatic_gamma = int(np.floor(self.I*parameters["proportion_dogmatic_gamma"]))#number of dogmatic gamma

        #create network
        (
            self.adjacency_matrix,
            self.network,
        ) = self.create_adjacency_matrix()
        
        #self.expectations_theta_mean_vector = np.asarray([self.mu_0]*self.I)
        #self.S_matrix = #self.calc_s() # get the influence of neighbors
        #### here have to create a weighting matrix that include all of the people and the signal?

        self.weighting_matrix = self.adjacency_matrix


        #self.dogmatic_state_theta_mean_var_vector = [("theta",(self.d * self.R)/(self.R -1) + (self.R * self.theta_t[0])/(self.R - self.ar_1_coefficient), self.epsilon_sigma**2)]*self.num_dogmatic_theta + [("gamma",(self.d * self.R)/(self.R -1) + (self.R * self.gamma_t[0])/(self.R - self.ar_1_coefficient),self.epsilon_sigma**2)]*self.num_dogmatic_gamma + [("normal",(self.d*self.R)/(self.R - 1) ,self.epsilon_sigma**2 + self.theta_sigma**2 * (1 + self.ar_1_coefficient/(self.R - self.ar_1_coefficient))**2)]*(self.I - self.num_dogmatic_theta - self.num_dogmatic_gamma)
        #self.dogmatic_state_theta_mean_var_vector = [("theta",self.theta_t[3], 0)]*self.num_dogmatic_theta + [("gamma", self.gamma_t[3],0)]*self.num_dogmatic_gamma + [("normal", 0, self.theta_sigma**2)]*(self.I - self.num_dogmatic_theta - self.num_dogmatic_gamma)
        self.dogmatic_state_theta_mean_var_vector = [("theta",self.theta_t[self.step_count+1], 0)]*self.num_dogmatic_theta + [("normal", 0, self.theta_sigma**2)]*(self.I - self.num_dogmatic_theta - self.num_dogmatic_gamma) + [("gamma", self.gamma_t[self.step_count+1],0)]*self.num_dogmatic_gamma 
        self.type = [agent[0] for agent in self.dogmatic_state_theta_mean_var_vector]

        if self.network_type == "small-world":
            # this is to make sure that even if we change the seed, we always keep the network in position.
            # In this way we can compere different agents across different seeds
            #np.random.shuffle(self.dogmatic_state_theta_mean_var_vector)
            rng_specific = np.random.default_rng(17)
            rng_specific.shuffle(self.dogmatic_state_theta_mean_var_vector)
            
        elif self.network_type == "scale_free":
            # Calculate node degrees
            self.dogmatic_state_theta_mean_var_vector = self.dogmatic_state_theta_mean_var_vector[::-1]
            # Sort nodes by degree in descending order
            # sorted_nodes = sorted(degrees, key=lambda x: degrees[x], reverse=False)
            # self.dogmatic_state_theta_mean_var_vector = [self.dogmatic_state_theta_mean_var_vector[node] for node in sorted_nodes]
            # print([agent[0] for agent in self.dogmatic_state_theta_mean_var_vector])
            # print(degrees)
        self.agent_list = self.create_agent_list()


        self.S_previous_matrix = self.init_calc_S(np.asarray([0 for v in self.agent_list])) #to avoid caluclations we assume everone is uninformed in the 0 step
        self.S_current_matrix = self.init_calc_S(np.asarray([agent[1] for agent in self.dogmatic_state_theta_mean_var_vector]))
        self.S_future_matrix = self.init_calc_S(np.asarray([agent[1] for agent in self.dogmatic_state_theta_mean_var_vector]))
        self.d_t1 = self.d #future dividends
        self.p_t = self.d / (self.R - 1) #uninformed price, THIS IS JUST FOR THE FIRST TIME STEP THIS IS P0 instead of P1
        #self.previous_pt = self.d / (self.R - 1) #previous price
        self.X_it = [0]*self.I
        #self.previous_X_it = [0]*self.I #previous demand

        self.create_history()

    def create_history(self):
        """
        if self.save_timeseries_data:
            self.history_p_t = [self.p_t]
            self.history_p_t1 = [self.previous_pt]
            self.history_d_t1 = [self.d]
            self.history_time = [self.step_count]
            self.history_X_it = [[0]*self.I]
            self.history_X_it1 = [[0]*self.I]
            self.history_weighting_matrix = [self.weighting_matrix]
        else:
            self.history_p_t = [self.p_t]
        """
        if self.save_timeseries_data:
            self.history_p_t = []
            self.history_p_t1 = []
            self.history_d_t1 = []
            self.history_time = []
            self.history_X_it = []
            self.history_X_it1 = []
            self.history_weighting_matrix = []
        else:
            self.history_p_t = []

    def generate_ar1(self, mean, acf, mu, sigma, N):
        data = [mean]
        for i in range(1,N):
            noise = np.random.normal(mu,sigma)
            data.append(mean + acf * (data[-1] - mean) + noise)
        return np.array(data)

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
        if self.network_type == "scale_free":
            G = nx.scale_free_graph(self.I)
        elif self.network_type == "random":
            G = nx.erdos_renyi_graph(self.I, 0.2)
        elif self.network_type == "small-world":
            G = nx.watts_strogatz_graph(n=self.I, k=self.K, p=self.prob_rewire, seed=self.set_seed)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        elif self.network_type == "SBM":
            block_sizes = [int(self.I/2), int(self.I/2)]  # Adjust the sizes as needed
            num_blocks = len(block_sizes)
            # Create the stochastic block model
            block_probs = np.asarray([[0.1, 0.001],[0.001, 0.1]])  # Make the matrix symmetric
            G = nx.stochastic_block_model(block_sizes, block_probs, seed=self.set_seed)
        adjacency_matrix = nx.to_numpy_array(G)
        #remove self loops, for the scale free network 
        np.fill_diagonal(adjacency_matrix, 0)
        #print("Network density:", nx.density(G))
        return (
            adjacency_matrix,
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
            "R":self.R,
            "d": self.d,
            "theta_variance": self.theta_sigma**2,
            "ar_1_coefficient":self.ar_1_coefficient,
            "epsilon_variance": self.epsilon_sigma**2,
            "gamma_variance": self.gamma_sigma**2
        }

        agent_list = [
            Consumer(
                consumer_params, 
                #self.weighting_matrix[i],
                self.dogmatic_state_theta_mean_var_vector[i][0], 
                self.dogmatic_state_theta_mean_var_vector[i][1], 
                self.dogmatic_state_theta_mean_var_vector[i][2]
            )
            for i in range(self.I)
            # we need to initialize consumers to form expectations on theta_2 and gamma_2
        ]

        return agent_list

    def init_calc_S(self, reshape_payoff_expectations):
        neighbour_influence =  np.where(self.adjacency_matrix == 0, np.nan, self.adjacency_matrix)*reshape_payoff_expectations 
        #print(neighbour_influence)
        return neighbour_influence

    def calc_S(self):
        adj_matrix_nan = np.where(self.adjacency_matrix == 0, np.nan, self.adjacency_matrix)
        theta_vector = np.where(self.type == "theta", self.theta_t[self.step_count +2], self.theta_expectations)
        theta_and_gamma_vector = np.where(self.type == "gamma", self.gamma_t[self.step_count +2], theta_vector)
        return adj_matrix_nan*theta_and_gamma_vector

    def get_consumers_theta_expectations(self):
        theta_expectations = np.asarray([i.theta_expectation_tplus1 for i in self.agent_list])
        theta_variances = np.asarray([i.theta_variance_tplus1 for i in self.agent_list])
        return theta_expectations, theta_variances
    
    def get_consumers_payoff_expectations(self):
        payoff_expectations = np.asarray([i.payoff_expectation for i in self.agent_list])
        payoff_variances = np.asarray([i.payoff_variance for i in self.agent_list])
        return payoff_expectations, payoff_variances


    def compute_price(self):
        term_1 = sum((self.payoff_expectations)/(self.payoff_variances))
        term_2 = 1/(sum(1/self.payoff_variances))
        aggregate_price = (term_1*term_2)/self.R
        return aggregate_price

    def compute_demand(self):
        demand_numerator = self.payoff_expectations - self.R*self.p_t
        demand_denominator = self.a*self.payoff_variances
        demand_vector = demand_numerator/demand_denominator
        return demand_vector
 
    
    def update_consumers(self):
        for i,agent in enumerate(self.agent_list):
            if agent.dogmatic_state =="theta":
                agent.next_step(self.d_t, self.previous_pt, self.S_future_matrix[i], self.S_previous_matrix[i], self.step_count, self.theta_t[self.step_count+1])
            elif agent.dogmatic_state =="gamma": #dogmatic gamma
                agent.next_step(self.d_t, self.previous_pt, self.S_future_matrix[i], self.S_previous_matrix[i], self.step_count, self.gamma_t[self.step_count+1])
            else:
                agent.next_step(self.d_t, self.previous_pt, self.S_future_matrix[i],self.S_previous_matrix[i], self.step_count)
            #                                 d_t,     p_t,     X_t,     S,               steps,         expectation_theta_mean = None, expectation_theta_var = None

    #def get_weighting_matrix(self):
    #    return np.asarray([v.weighting_vector for v in self.agent_list])

    def append_data(self):
        if self.save_timeseries_data:
            self.history_p_t.append(self.p_t)
            self.history_p_t1.append(self.previous_pt)
            self.history_d_t1.append(self.d_t)
            self.history_time.append(self.step_count)
            self.history_X_it.append(self.X_it)
            self.history_X_it1.append(self.previous_X_it)
            #self.history_informed_proportion.append(self.informed_proportion)
            self.history_weighting_matrix.append(self.weighting_matrix)
        else:
            self.history_p_t.append(self.p_t)

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
        #Update these for the new time step
        self.previous_pt = self.p_t#previous price
        self.previous_X_it = self.X_it#previousmatrix of demands 
         #previous matrix of signals
        #simulate dividend
        self.d_t = self.dividend_vector[self.step_count-1]
        #update consumers
        self.update_consumers()

        #Recieve expectations of mean and variances
        self.theta_expectations, self.theta_variances = self.get_consumers_theta_expectations()
        self.payoff_expectations, self.payoff_variances = self.get_consumers_payoff_expectations()

        #Compute aggregate price
        self.p_t = self.compute_price()#ON the first time is p_t =1

        #compute indivdiual demands
        self.X_it = self.compute_demand()

        #compute profit
        self.profit = np.asarray(self.previous_X_it) * (self.p_t + self.dividend_vector[self.step_count] - self.R*np.array(self.previous_pt))

        #update network signal
        self.S_previous_matrix = self.S_current_matrix 
        self.S_current_matrix = self.S_future_matrix
        self.S_future_matrix = self.calc_S()
        #self.weighting_matrix = self.get_weighting_matrix()
       
        if (self.step_count % self.compression_factor == 0):
            self.append_data()

        self.step_count +=1 
        if self.step_count % 50 == 0:
            print("step count: ", self.step_count)

        
