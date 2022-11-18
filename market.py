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

        self.set_seed = parameters["set_seed"]
        np.random.seed(self.set_seed)
        self.save_data = parameters["save_data"]

        self.I = parameters["I"]
        self.K = parameters["K"]
        self.R = parameters["R"]
        self.a = parameters["a"]
        self.d = parameters["d"]
        self.theta_sigma = parameters["theta_sigma"]
        self.epsilon_sigma = parameters["epsilon_sigma"]
        self.phi_sigma = parameters["gamma_sigma"] 
        self.zeta_threshold = parameters["zeta_threshold"]

        self.W_0 = parameters["W_0"]
        self.mu_0 = parameters["mu_0"]
        self.var_0 =parameters["var_0"]
        self.c_info = parameters["c_info"]
        self.beta = parameters["beta"]
        self.delta = parameters["delta"]

        self.step_count = 0
        self.steps = parameters["steps"]

        self.S_tau_t = self.compute_S_tau_t()
        self.S_omega_t = self.compute_S_omega_t()

        self.c_0_list = np.random.choice(a = [0,1], size = self.I)
        #print("self.c_0_list",self.c_0_list)


        #create network
        self.network_structure = parameters["network_structure"]
        if self.network_structure == "small_world":
            self.K = int(round(parameters["K"]))  # round due to the sampling method producing floats in the Sobol Sensitivity Analysis (SA)
            self.prob_rewire = parameters["prob_rewire"]
        (
            self.adjacency_matrix,
            self.weighting_matrix,
            self.network,
        ) = self.create_weighting_matrix()

        self.weighting_vector_0 = np.array([parameters["tau"], parameters["omega"],parameters["rho"]])

        self.agent_list = self.create_agent_list()

        #update_expectations of agents based on their netork and initial signals
        S_list_init = [self.S_tau_t[0],self.S_omega_t[0], self.mu_0]
        for i in range(self.I):
            self.agent_list[i].expectation_mean, self.agent_list[i].expectation_variance = self.agent_list[i].compute_posterior_mean_variance(S_list_init)

        if self.save_data:
            self.history_p_t = [0]
            self.history_d_t= [0]
            self.history_time = [self.step_count]


    def compute_S_tau_t(self):
        return np.random.normal(0, self.theta_sigma, self.steps+1) #+1 is for the zeroth step update of the signal
        
    def compute_S_omega_t(self):
        gamma_t = np.random.normal(0, self.theta_sigma, self.steps+1) #+1 is for the zeroth step update of the signal
        
        zeta = [(self.S_tau_t[i] + gamma_t[i]) if (np.abs(self.S_tau_t[i]) + np.abs(gamma_t[i]) > self.zeta_threshold) else 0 for i in range(0,self.steps+1)]#+1 is for the zeroth step update of the signal

        return np.asarray(zeta)

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
        norm_matrix = matrix / row_sums[:, np.newaxis]

        return norm_matrix

    def create_weighting_matrix(self) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
        """
        Create watts-strogatz small world graph using Networkx library

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

        #nx.draw(G)

        weighting_matrix = nx.to_numpy_array(G)

        #print(self.network_structure, weighting_matrix)

        norm_weighting_matrix = self.normlize_matrix(weighting_matrix)

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
            "save_data": self.save_data,
            "W_0":self.W_0,
            "mu_0":self.mu_0,
            "var_0":self.var_0,
            "weighting_vector_0": self.weighting_vector_0,
            "R":self.R,
            "a": self.a,
            "d": self.d,
            "epsilon_sigma": self.epsilon_sigma,
            "beta": self.beta,
            "delta":self.delta,
        }

        agent_list = [
            Consumer(
                consumer_params,self.c_0_list[i]
            )
            for i in range(self.I)
        ]

        return agent_list

    def get_consumers_dt_mean_variance(self):

        expectations_mean_vector = [i.expectation_mean for i in self.agent_list]
        expectations_variance_vector = [i.expectation_variance for i in self.agent_list]

        dt_expectations_mean = self.d + np.asarray(expectations_mean_vector)
        dt_expectations_variance = self.epsilon_sigma**2 + np.asarray(expectations_variance_vector)

        return dt_expectations_mean, dt_expectations_variance

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
        
        d_t = self.d + self.S_tau_t[self.step_count] + np.random.normal(0, self.epsilon_sigma, 1)

        #print("dividend at t", d_t)

        return d_t

    def compute_network_signal(self):

        k_list = [np.random.choice(range(self.I), 1, p=self.weighting_matrix[i])[0] for i in range(self.I)]

        neighbour_influence = [self.agent_list[k].expectation_mean for k in k_list]
        return neighbour_influence

    def update_consumers(self):

        S_tau = self.S_tau_t[self.step_count] 
        S_omega = self.S_omega_t[self.step_count]

        for i in range(self.I):
            self.agent_list[i].next_step(self.d_t,self.p_t,self.X_it[i], S_tau, S_omega, self.S_rho_i[i])

    def append_data(self):
        self.history_p_t.append(self.p_t)
        self.history_d_t.append(self.d_t)
        self.history_time.append(self.step_count)

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
        self.step_count +=1  

        #Recieve expectations of mean and variances
        self.dt_expectations_mean, self.dt_expectations_variance = self.get_consumers_dt_mean_variance()

        #Compute aggregate price
        self.p_t = self.compute_price()

        #compute indivdiual demands
        self.X_it = self.compute_demand()

        #simulate dividend
        self.d_t = self.compute_dividends()

        #update network signal
        self.S_rho_i = self.compute_network_signal()

        #compute and return profits
        self.update_consumers()

        if self.save_data:
            self.append_data()
