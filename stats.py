from utility import (
    createFolder, 
    load_object, 
)
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

f = open("constants/base_params.json")
params = json.load(f)
theta_sigma = params["theta_sigma"]
gamma_sigma = params["gamma_sigma"]
epsilon_sigma = params["epsilon_sigma"]
theta_mean = params["theta_mean"]
gamma_mean = params["gamma_mean"]
d = params["d"]

# fileName = "results/single_shot_steps_3000_I_200_network_structure_small_world_degroot_aggregation_1"
# createFolder(fileName)
# Data = load_object(fileName + "/Data", "financial_market")

# print(Data.__dict__.keys())

# weighting_zeta_mean_t_c = [np.mean([i.history_weighting_vector[v][1] for i in Data.agent_list if not i.c_bool]) for v in range(len(Data.history_time))]
# print(weighting_zeta_mean_t_c)
# weighting_zeta_mean_c = np.mean(weighting_zeta_mean_t_c)
# print(weighting_zeta_mean_c)


# print([i.history_weighting_vector[1] for i in Data.agent_list if not i.c_bool])


def prob_loosing_money(c, N = 10**5, a = 1.1, window = 1):
    k = window
    theta = np.random.normal(theta_mean, theta_sigma, (N,k))
    epsilon = np.random.normal(0.0, epsilon_sigma, (N,k))
    profit = theta**2 + epsilon*theta
    return np.mean(profit.mean(axis = 1) < a*c*epsilon_sigma**2)

def plot_prob_loosing_money(c_start = 0.0, c_end = 1.0, w_start = 1, w_end = 2, w_step = 1, N = 10**5):
    fig, ax = plt.subplots()
    x = np.linspace(c_start,c_end,100)
    for w in range(w_start,w_end,w_step):
        y = [prob_loosing_money(i, N = N, window=w) for i in x]
        ax.plot(x,y)   
    ax.set_xlabel('cost of information')
    ax.set_ylabel('p(loss)')    
    plt.savefig('results\p_loss.png', dpi = 300)
    plt.show()

plot_prob_loosing_money(N = 10**6)

def prob_smaller_theta_error(N = 10**5, window = 1):
    k = window
    theta = np.random.normal(theta_mean, theta_sigma, (N,k))
    epsilon = np.random.normal(0.0, epsilon_sigma, (N,k))
    gamma = np.random.normal(gamma_mean, gamma_sigma, (N,k))
    return np.mean(np.abs(epsilon) < np.abs(theta -gamma + epsilon))

def confidence_theta(theta_mean, gamma_mean, beta, d = 1.1, N = 10**5):
    theta = np.random.normal(theta_mean, theta_sigma, N)
    epsilon = np.random.normal(0.0, epsilon_sigma, N)
    gamma = np.random.normal(gamma_mean, gamma_sigma, N)
    APE_theta = np.abs(epsilon)/(d+theta+epsilon)
    APE_gamma = np.abs(theta - gamma + epsilon)/(d+theta+epsilon)
    numerator = np.exp(-beta * APE_theta)
    denominator = np.exp(-beta * APE_theta) + np.exp(-beta * APE_gamma)
    abs_confidence = np.divide(numerator,denominator)
    abs_confidence = abs_confidence[~np.isnan(abs_confidence)]
    confidence = np.mean(abs_confidence)
    return confidence

b_max = 20
fig, ax = plt.subplots()
x = np.linspace(-1.0,1.0,100)
for b in range(0,b_max+1,1):
    y = [confidence_theta(theta_mean, i, b, d = d, N = 10**4) for i in x]
    ax.plot(x,y, color = mpl.cm.viridis(b/b_max))   
ax.set_xlabel('theta_mean - gamma_mean')
ax.set_ylabel('confidence in theta')
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, norm=mpl.colors.Normalize(vmin=0.0, vmax=b_max),orientation='vertical')
cb1.set_label('beta', rotation=90)
plt.gcf().add_axes(ax_cb)  
plt.savefig('results\confidence_theta.png', dpi = 300)
plt.show()



N = 10**5
theta = np.random.normal(theta_mean, theta_sigma, N)
epsilon = np.random.normal(0.0, epsilon_sigma, N)
gamma = np.random.normal(gamma_mean, gamma_sigma, N)


(theta_sigma*np.sqrt(2/np.pi)) - np.sqrt(6*theta_sigma**2/np.pi)


np.mean(np.abs(epsilon)) - np.mean(np.abs(theta + epsilon - gamma))
