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

        #########################################################################################
        #model parameters
        self.set_seed = parameters["set_seed"]
        np.random.seed(self.set_seed)
        self.save_timeseries_data = parameters["save_timeseries_data"]
        self.compression_factor = parameters["compression_factor"]
        self.I = parameters["I"]
        self.K = parameters["K"]
        self.R = parameters["R"]
        self.a = parameters["a"]
        self.d = parameters["d"]
        self.delta = parameters["delta"]
        self.W_0 = parameters["W_0"]
        self.W_0_list = np.asarray([self.W_0]*self.I)
        self.beta = parameters["beta"]
        self.step_count = 0
        self.total_steps = parameters["total_steps"]
        self.theta_mean = parameters["theta_mean"]
        self.gamma_mean = parameters["gamma_mean"]
        self.theta_sigma = parameters["theta_sigma"]
        self.epsilon_sigma = parameters["epsilon_sigma"]
        self.gamma_sigma = parameters["gamma_sigma"] 
        self.switch_s = parameters["switch_s"]
        self.ar_1_coefficient = parameters["ar_1_coefficient"]  #elasticty of the switch probability to the accurac
        self.epsilon_t = np.random.normal(0, self.epsilon_sigma, self.total_steps+1)
        #We change both theta and gamma to be random walks, that is ar(1) processes with coefficient = 1. shocks never dissipate
        self.theta_t = self.generate_ar1(0,self.ar_1_coefficient, self.theta_mean, self.theta_sigma, self.total_steps+1) #np.cumsum(np.random.normal(self.theta_mean, self.theta_sigma, self.total_steps+1)) #+1 is for the zeroth step update of the signal
        self.gamma_t = -self.theta_t #+ np.random.normal(self.gamma_mean, self.gamma_sigma, self.total_steps+1)

        #create network
        self.K = int(round(parameters["K"]))  # round due to the sampling method producing floats in the Sobol Sensitivity Analysis (SA)
        self.prob_rewire = parameters["prob_rewire"]
        
        (
            self.adjacency_matrix,
            self.network,
        ) = self.create_adjacency_matrix()
        
        #self.expectations_theta_mean_vector = np.asarray([self.mu_0]*self.I)
        #self.S_matrix = #self.calc_s() # get the influence of neighbors
        #### here have to create a weighting matrix that include all of the people and the signal?

        self.weighting_matrix = self.gen_init_weighing_matrix()

        self.num_dogmatic_theta = int(np.floor(self.I*parameters["proportion_dogmatic_theta"])) #number of dogmatic theta
        #print("self.num_dogmatic_theta", self.num_dogmatic_theta, self.I)
        self.num_dogmatic_gamma = int(np.floor(self.I*parameters["proportion_dogmatic_gamma"]))#number of dogmatic gamma

        self.dogmatic_state_theta_mean_var_vector = [("theta",self.theta_t[0],0)]*self.num_dogmatic_theta + [("gamma",self.gamma_t[0],0)]*self.num_dogmatic_gamma + [("normal",0,self.theta_sigma**2)]*(self.I - self.num_dogmatic_theta - self.num_dogmatic_gamma)
        np.random.shuffle(self.dogmatic_state_theta_mean_var_vector)
        self.agent_list = self.create_agent_list()
        self.S_matrix = self.init_calc_S(np.asarray([v.expectation_theta_mean for v in self.agent_list]))
        self.d_t = self.d #uninformed expectation
        self.p_t = self.d / self.R #uninformed price
        self.X_it = [0]*self.I

        if self.save_timeseries_data:
            self.history_p_t = [self.d/self.R]
            self.history_d_t= [0]
            self.history_time = [self.step_count]
            self.history_X_it = [[0]*self.I]
            self.history_weighting_matrix = [self.weighting_matrix]

    def generate_ar1(self, mean, acf, mu, sigma, N):
        data = [mean]
        for i in range(1,N):
            noise = np.random.normal(mu,sigma)
            data.append(acf * data[-1] + noise)
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


        G = nx.watts_strogatz_graph(n=self.I, k=self.K, p=self.prob_rewire, seed=self.set_seed)  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
        adjacency_matrix = nx.to_numpy_array(G)
        print("Network density:", nx.density(G))
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
            "a": self.a,
            "d": self.d,
            "beta": self.beta,
            "delta":self.delta,
            "ar_1_coefficient":self.ar_1_coefficient
        }

        agent_list = [
            Consumer(
                consumer_params,self.W_0_list[i], self.weighting_matrix[i], 
                self.dogmatic_state_theta_mean_var_vector[i][0], self.dogmatic_state_theta_mean_var_vector[i][1], 
                self.dogmatic_state_theta_mean_var_vector[i][2], self.adjacency_matrix[i]
            )
            for i in range(self.I)
        ]

        return agent_list

    def init_calc_S(self, expectations_theta_mean_vector):
        reshape_expectations_theta = expectations_theta_mean_vector#[:, np.newaxis]
        neighbour_influence =  np.where(self.adjacency_matrix == 0, np.nan, self.adjacency_matrix)*reshape_expectations_theta
        #print(neighbour_influence)
        return neighbour_influence

    def calc_S(self):
        reshape_expectations_theta = self.expectations_theta_mean_vector#[:, np.newaxis]
        neighbour_influence = np.where(self.adjacency_matrix == 0, np.nan, self.adjacency_matrix)*reshape_expectations_theta

        return neighbour_influence

    def gen_init_weighing_matrix(self):
        """
        equal weighting amongst all neighbours at first step?
        """
        neighbour_count_vector = np.sum(self.adjacency_matrix, axis = 1)        
        network_weighting_vector = (1/neighbour_count_vector)        

        return network_weighting_vector

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
    
    def update_consumers(self):
        for i,agent in enumerate(self.agent_list):
            if agent.dogmatic_state =="theta":
                agent.next_step(self.d_t,self.p_t,self.X_it[i], self.S_matrix[i],self.step_count, self.theta_t[self.step_count])
            elif agent.dogmatic_state =="gamma": #dogmatic gamma
                agent.next_step(self.d_t,self.p_t,self.X_it[i], self.S_matrix[i],self.step_count, self.gamma_t[self.step_count])
            else:
                agent.next_step(self.d_t,self.p_t,self.X_it[i], self.S_matrix[i],self.step_count)
            #                                 d_t,     p_t,     X_t,     S,               steps,         expectation_theta_mean = None, expectation_theta_var = None

    def get_weighting_matrix(self):
        return np.asarray([v.weighting_vector for v in self.agent_list])

    def append_data(self):
        self.history_p_t.append(self.p_t)
        self.history_d_t.append(self.d_t)
        self.history_time.append(self.step_count)
        self.history_X_it.append(self.X_it)
        #self.history_informed_proportion.append(self.informed_proportion)
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
        self.S_matrix = self.calc_S()
        
        self.weighting_matrix = self.get_weighting_matrix()

        self.step_count +=1  
       
        if (self.step_count % self.compression_factor == 0) and (self.save_timeseries_data):
            self.append_data()
        if self.step_count % 10 == 0:
            print(self.step_count)
            

        
