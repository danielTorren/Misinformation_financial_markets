# Version of the model model with reduced parameters to produce return timeseries that can be used to train a surrogate model

# imports
import numpy as np
import networkx as nx
import warnings

class Market:
    def __init__(self, parameters: dict):
        #### model parameters
        self.save_timeseries_data = 0
        self.network_density = 0.1
        self.I = 200
        self.K = int(round((self.I - 1) * self.network_density))
        self.R = 1.0001
        self.a = 5.5
        self.d = 0.021
        self.theta_mean = 0.0
        self.theta_sigma = parameters["theta_sigma"]
        self.gamma_mean = 0.0
        self.gamma_sigma = parameters["gamma_sigma"]
        self.epsilon_sigma = 1
        self.ar_1_coefficient = 0.5# parameters["ar_1_coefficient"]
        self.prob_rewire = 0.01
        self.w = 0.9
        self.total_steps = 1500
        self.num_info = int(np.floor(self.I*0.05))
        self.num_misinfo = int(np.floor(self.I*0.05))

        self.p_0 = np.zeros(self.I)
        self.X_0 = np.zeros(self.I)
        #### Invariant objects

        # Network
        self.adjacency_matrix, self.network = self.create_adjacency_matrix()

        # Category vector
        self.category_vector = self.create_category_vector()


        #prior variance matrix
        self.prior_variance_matrix = self.create_prior_variance_matrix()

        #### Initial objects
        self.step_count = 0
        self.theta_0, self.gamma_0, self.epsilon_0 = self.theta_mean, self.gamma_mean, np.random.normal(0, self.epsilon_sigma)
        self.theta_1 = self.ar_1_coefficient * self.theta_0 + np.random.normal(0, self.theta_sigma)
        self.gamma_1 = self.ar_1_coefficient * self.gamma_0 + np.random.normal(0, self.gamma_sigma) #self.theta_1 + np.random.normal(self.gamma_mean, self.gamma_sigma)
        self.epsilon_1 = np.random.normal(0, self.epsilon_sigma)
        # we need a vector 0 to compute the source errors matrix based on p0 and theta_0
        self.prior_mean_vector_0 = np.where(self.category_vector == 1, self.theta_0, np.where(self.category_vector == -1, self.gamma_0, 0))
        self.prior_mean_vector = np.where(self.category_vector == 1, self.theta_1, np.where(self.category_vector == -1, self.gamma_1, 0))
        self.prior_mean_matrix_0 = self.compute_prior_mean_matrix(self.prior_mean_vector_0)
        self.prior_mean_matrix = self.compute_prior_mean_matrix(self.prior_mean_vector)
        self.source_errors_matrix = np.ones((self.I, self.I))*self.epsilon_sigma**2
        self.implied_variance_matrix = self.compute_implied_variance_matrix()
        self.weighting_matrix, self.posterior_variance_vector = self.compute_weighting_matrix_and_posterior_variance_vector()
        self.posterior_mean_vector = self.compute_posterior_mean_vector()
        self.payoff_expectation, self.payoff_variance = self.compute_payoff_beliefs()

        # history vectors
        self.create_history()
        
    #### functions

    def next_step(self):    
        # Generate the variables for step 1
        self.theta_1 = self.ar_1_coefficient * self.theta_0 + np.random.normal(0, self.theta_sigma)
        self.gamma_1 = self.ar_1_coefficient * self.gamma_0 + np.random.normal(0, self.gamma_sigma)#np.random.normal(self.gamma_mean, self.gamma_sigma) #+ self.theta_1 
        self.epsilon_1 = np.random.normal(0, self.epsilon_sigma)

        # update the prior mean vector
        self.prior_mean_vector = np.where(self.category_vector == 1, self.theta_1, np.where(self.category_vector == -1, self.gamma_1, self.ar_1_coefficient*self.posterior_mean_vector))
        self.prior_mean_matrix = self.compute_prior_mean_matrix(self.prior_mean_vector)

        #compute implied variance matrix
        self.implied_variance_matrix = self.compute_implied_variance_matrix()

        #compute weighting matrix and posterior 
        self.weighting_matrix, self.posterior_variance_vector = self.compute_weighting_matrix_and_posterior_variance_vector()
        self.posterior_mean_vector = self.compute_posterior_mean_vector()

        #compute payoff beliefs
        self.payoff_expectation, self.payoff_variance = self.compute_payoff_beliefs()

        # Compute the price p0 and the demand X0
        self.past_price = self.p_0#TO CALCULATE THE PROFITS
        self.past_demand = self.X_0
        self.p_0 = self.compute_price()
        self.X_0 = self.compute_demand()
        
        self.profits = self.calc_profits()

        # Update the source errors matrix
        self.update_source_variance_matrix()

        # Update the variables for the next step
        self.theta_0, self.gamma_0, self.epsilon_0 = self.theta_1, self.gamma_1, self.epsilon_1
        #self.prior_mean_vector_0 = self.prior_mean_vector
        self.prior_mean_vector_0 = self.posterior_mean_vector
        #self.prior_mean_matrix_0 = self.prior_mean_matrix
        self.prior_mean_matrix_0 = self.compute_prior_mean_matrix(self.prior_mean_vector_0)


        #Only in the last step we copy the own forecast error, for plotting
        if self.step_count == self.total_steps -1 :
            self.forecast_errors = np.diag(self.source_errors_matrix)
        
        # Append the data
        self.append_data()

        self.step_count +=1 
        # if self.step_count % 100 == 0:
        #      print("step count: ", self.step_count)
    
    def calc_profits(self):
        return (self.d + self.theta_1 +self.epsilon_1 - self.R*self.past_price)*self.past_demand
    
    def create_adjacency_matrix(self):
        G = nx.watts_strogatz_graph(n=self.I, k=self.K, p=self.prob_rewire, seed = 17)  # Watts–Strogatz small_world graph,watts_strogatz_graph( n, k, p[, seed])
        adjacency_matrix = nx.to_numpy_array(G)
        #fill the diagonal with 1s
        np.fill_diagonal(adjacency_matrix, 1)
        # remove parallel edges
        adjacency_matrix[adjacency_matrix > 1] = 1
        #transform zeros to np.nans
        adjacency_matrix[adjacency_matrix == 0] = np.nan
        return adjacency_matrix, G 
    
    def create_category_vector(self):
        # 0 indicates uninformed agents, 1 indicates informed agents, -1 indicates misinformed agents
        # intialize vector of zeros of length I
        category_vector = np.zeros(self.I)
        # select the first num_info indices to be 1
        category_vector[: self.num_info] = 1
        # select the last num_misinfo indices to be -1
        #first check that the number of misinformed is not 0
        if self.num_misinfo > 0:
            category_vector[-self.num_misinfo :] = -1
        else:
            pass
        return category_vector
    
    def create_prior_variance_matrix(self):
        # initialize the matrix with ones
        prior_variance_matrix = np.ones((self.I, self.I))
        #multiply each element by variance of theta
        prior_variance_matrix = self.theta_sigma**2 * prior_variance_matrix
        # get the diagonal and change the first num_info elements and the last num_misinfo elements to 0 if informed are cetnral
        #in practice we select two block matrices with the proper size and fill the diagonal with zeros
        np.fill_diagonal(prior_variance_matrix[:self.num_info, :self.num_info], 0)
        np.fill_diagonal(prior_variance_matrix[-self.num_misinfo:, -self.num_misinfo:], 0)
        # Finally multiply it elementwise by the adjacency matrix to have nans where there are no connections
        prior_variance_matrix = prior_variance_matrix * self.adjacency_matrix
        # add a small number to avoid division by zero
        return prior_variance_matrix + 1e-128
    
    def compute_prior_mean_matrix(self, prior_vector):
        # mutiply every row of the adjacency matrix by the prior mean vector, elementwise
        prior_mean_matrix = self.adjacency_matrix * prior_vector
        return prior_mean_matrix
    
    def compute_implied_variance_matrix(self):
        #multiply the prior matrix with the source errors matrix
        implied_variance_matrix = self.prior_variance_matrix * self.source_errors_matrix
        # then divide first row by first diagonal element, second row by second diagonal element, etc
        # we transpose the matrix to do this operation on the columns and then transpose it back
        implied_variance_matrix = np.transpose(np.transpose(implied_variance_matrix) / np.diag(self.source_errors_matrix))
        return implied_variance_matrix
    
    def compute_weighting_matrix_and_posterior_variance_vector(self):
        #We do the two operations at the same time to avoid repeating the same operations
        # first fill the nan to one since they are invariant to multiplication
        filled_with_one = np.copy(self.implied_variance_matrix)
        filled_with_one[np.isnan(self.implied_variance_matrix)] = 1
        numerator = np.transpose(np.prod(filled_with_one, axis=1) * np.ones((self.I, self.I)))/ filled_with_one
        # then get the product of the fillex matrix over rows
        product = np.prod(filled_with_one, axis=1)
        #then replace the position in the numerator with 0 where x was nan,as it is invariant to sum
        filled_with_zeros = np.copy(numerator)
        filled_with_zeros[np.isnan(self.implied_variance_matrix)] = 0
        #if some rows are below 1e-64, convert them to 1e-64
        if np.any(np.sum(filled_with_zeros, axis=1) < 1e-128):
            filled_with_zeros[np.sum(filled_with_zeros, axis=1) < 1e-128] = 1e-128
        # #if some element is np.inf, convert it to 1e128
        # if np.any(np.isinf(filled_with_zeros)):
        #     filled_with_zeros[np.isinf(filled_with_zeros)] = 1e128
        # then divide the numerator by the sum of filled_with_zeros over rows 
        a = np.transpose(np.transpose(filled_with_zeros)/np.sum(filled_with_zeros, axis=1))
        b = product/np.sum(filled_with_zeros, axis=1)
        return a, b
    
    def compute_posterior_mean_vector(self):
        # multiply the weighting matrix by the prior mean matrix then sum over rows
        #first we fill the nans with zeros since they are invariant to sum
        self.prior_mean_matrix[np.isnan(self.prior_mean_matrix)] = 0
        return np.sum(self.prior_mean_matrix*self.weighting_matrix, axis = 1)
    
    def compute_payoff_beliefs(self):
        payoff_expectation = (self.d * self.R)/(self.R -1) + (self.R * self.posterior_mean_vector)/(self.R - self.ar_1_coefficient)
        payoff_variance = self.epsilon_sigma**2 + self.posterior_variance_vector
        # add sigma_gamma and sigma theta where the category vector is -1
        payoff_variance = np.where(self.category_vector == -1, payoff_variance + (self.gamma_sigma**2)/(self.R - self.ar_1_coefficient)**2, payoff_variance)
        # add sigma_theta where the category vector is 1
        payoff_variance = np.where(self.category_vector == 1, payoff_variance + (self.theta_sigma**2)/(self.R - self.ar_1_coefficient)**2, payoff_variance)
        return payoff_expectation, payoff_variance

    def compute_price(self):
       term_1 = sum((self.payoff_expectation)/(self.payoff_variance))
       term_2 = 1/(sum(1/self.payoff_variance))
       aggregate_price = (term_1*term_2)/self.R
       return aggregate_price
    
    def compute_demand(self):
        demand_numerator = self.payoff_expectation - self.R*self.p_0
        demand_denominator = self.a*self.payoff_variance
        demand_vector = demand_numerator/demand_denominator
        return demand_vector
    
    def update_source_variance_matrix(self):
        #we update as soon as the price p0 is computed
        prediction = (self.d * self.R)/(self.R -1) + (self.R * self.prior_mean_matrix_0)/(self.R - self.ar_1_coefficient)
        current_error = (self.d + self.theta_0 + self.epsilon_0 + self.p_0 - prediction)**2
        self.source_errors_matrix = self.w * current_error + (1 - self.w) * self.source_errors_matrix

    def create_history(self):
        self.history_p_t = []

    def append_data(self):
        self.history_p_t.append(self.p_0)

        
