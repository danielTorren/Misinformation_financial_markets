import numpy as np

def compute_posterior_mean_variance(m_0, v_0 , mean_vector, sigma_vector):
    
    prior_variance = v_0
    prior_mean = m_0
    #add priors for cycling, tour de france
    # here we need to force self.theta_variance, so that we do not get an errro in the case of dogmatic agents
    converted_variance = v_0 * sigma_vector
    full_signal_variances= np.append(converted_variance, prior_variance)#np.append(self.source_variance[~np.isnan(self.source_variance)], prior_variance)
    full_signal_means = np.append(mean_vector, prior_mean) 
    #print("length of mean vector is: ", len(full_signal_means), "length of var vector is: ", len(full_signal_variances))
    #for both mean and variance
    denominator = sum(np.product(np.delete(full_signal_variances, v)) for v in range(len(full_signal_variances)))
    #mean
    numerator_mean =  sum(np.product(np.append(np.delete(full_signal_variances, v),full_signal_means[v])) for v in range(len(full_signal_variances)))
    posterior_mean = (numerator_mean/denominator)        
    posterior_variance = (np.prod(full_signal_variances)/denominator)
    return posterior_mean,posterior_variance 

def generate_ar1(mean, acf, mu, sigma, N):
        data = [mean]
        for i in range(1,N):
            noise = np.random.normal(mu,sigma)
            data.append(mean + acf * (data[-1] - mean) + noise)
        return np.array(data)

# ar_1_coefficient = 0.6
# theta_mean = 0
# theta_sigma = .50
# epsilon_sigma = 1
# total_steps = 10**5
# d = 1.1

# print("here")


# epsilon_t = np.random.normal(0, epsilon_sigma, total_steps+1).reshape(-1, 1)
# theta_t = generate_ar1(0,ar_1_coefficient, theta_mean, theta_sigma, total_steps+1).reshape(-1, 1) #np.cumsum(np.random.normal(self.theta_mean, self.theta_sigma, self.total_steps+1)) #+1 is for the zeroth step update of the signal

# d_t = theta_t + epsilon_t + d
# #estimate linear regression of d_t on theta_t   
# reg = LinearRegression().fit(theta_t, d_t)
# print(reg.score(theta_t, d_t))

# #now we regress d_t on its lagged value
# d_t_lagged = d_t[:-1]
# d_t = d_t[1:]
# reg = LinearRegression().fit(d_t_lagged, d_t)
# print(reg.score(d_t_lagged, d_t))


x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.zeros(len(x))

print([(index, element) for index, element in enumerate(x)])

