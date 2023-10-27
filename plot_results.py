"""Plot results adn save them

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import collections
import os
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import norm
from scipy.stats import probplot
from utility import (
    createFolder, 
    load_object, 
    save_object,
)

fontsize= 18
ticksize = 16
figsize = (9, 9)
params = {'font.family':'serif',
    "figure.figsize":figsize, 
    'figure.dpi': 80,
    'figure.edgecolor': 'k',
    'font.size': fontsize, 
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': ticksize,
    'ytick.labelsize': ticksize
}
plt.rcParams.update(params)

def plot_cumulative_consumers(fileName,Data,y_title,dpi_save,property_y,red_blue_c):

    fig, ax = plt.subplots()

    if red_blue_c:
        for v in range(len(Data.agent_list)):
            if (Data.agent_list[v].dogmatic_state == "theta"):
                color = "blue"
            elif(Data.agent_list[v].dogmatic_state == "gamma"):
                color = "red"
            else:
                color = "black"
            if property_y == "history_theta_variance":
                data_ind = np.cumsum(np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y))))
            else:
                data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y)))
            ax.plot(np.asarray(Data.history_time), ((data_ind)), color = color)
    else:
        for v in range(len(Data.agent_list)):
            data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y)))
            ax.plot(np.asarray(Data.history_time), ((data_ind)))

    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)

    if property_y == "history_expectation_theta_mean":
        
        ax.plot(Data.history_time, (Data.theta_t[::Data.compression_factor]), linestyle='dashed', color="green",  linewidth=2, alpha=0.5)
    #elif property_y == "history_profit":
        #ax.plot(Data.history_time, np.cumsum(data_ind), linestyle='dashed', color="black",  linewidth=2, alpha=0.5)

        #ax.set_ylim([Data.R*Data.W_0 - 0.2, Data.R*Data.W_0 + 0.2])
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_cumulative_consumers_%s" % (property_y)
    #print("f",f)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_histogram_returns(fileName, Data, y_title, dpi_save):
    property_y = "history_p_t"
    fig, ax = plt.subplots()
    data = eval("Data.%s" % property_y)
    prices = np.array(data)

    # Calculate returns
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    rational_prices =(Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:])/ (Data.R - Data.ar_1_coefficient))
    rational_returns = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
    
    # Create a histogram of returns (transparent orange)
    ax.hist(returns, bins=30, alpha=0.5, color='orange', edgecolor='black', density=False, label='Returns Histogram')
    ax.hist(rational_returns, bins=30, alpha=0.5, color='green', edgecolor='black', density=False, label='Returns Histogram')

    # Fit a normal distribution to the data
    mu, std = norm.fit(returns)

    # Plot the PDF of the fitted normal distribution (light blue)
    x = np.linspace(min(returns), max(returns), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'lightblue', linewidth=2, label='Fitted Normal Distribution')
    
    ax.set_title('Histogram of Returns')
    ax.set_xlabel('Returns')
    
    fig.tight_layout()
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + "histogram_returns"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_qq_plot(fileName, Data, y_title, dpi_save):
    property_y = "history_p_t"
    fig, ax = plt.subplots()
    data = eval("Data.%s" % property_y)
    prices = np.array(data)

    # Calculate returns
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    rational_prices =(Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:] )/ (Data.R - Data.ar_1_coefficient))
    rational_returns = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
    # Generate QQ plot
    #probplot(returns, dist="norm", plot=ax)
    probplot(rational_returns, dist = "norm", plot = ax)
    ax.set_title('QQ Plot of Returns')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    
    fig.tight_layout()
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + "qq_plot_returns"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_time_series_market(fileName,Data,y_title,dpi_save,property_y):

    fig, ax = plt.subplots()
    data = eval("Data.%s" % property_y)
    

    # bodge
    
    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)
    #ax.set_ylim([(Data.d)/Data.R - 0.5*((Data.d)/Data.R), (Data.d)/Data.R + 0.5*((Data.d)/Data.R)])
    if property_y == "history_p_t":
        print("date Len is:", len(data), "time len is: ", len(Data.history_time))
        ax.plot(Data.history_time, np.array(data), linestyle='solid', color="black", alpha = 1)
        ax.plot(Data.history_time, (Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:-1])/ (Data.R - Data.ar_1_coefficient)), linestyle='dashed', color="red")
    elif property_y == "history_X_it":
        T,I = np.asarray(data).shape
        # Create a plot for each variable
        for i in range(I):
            ax.plot(Data.history_time, np.cumsum(np.abs(np.asarray(data)[:, i])))   
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_timeseries"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def prod_pos(network_structure: str, network: nx.Graph) -> nx.Graph:

    if network_structure == "small-world":
        layout_type = "circular"
    elif network_structure == "random" or "scale_free":
        layout_type ="kamada_kawai"

    if layout_type == "circular":
        pos_culture_network = nx.circular_layout(network)
    elif layout_type == "spring":
        pos_culture_network = nx.spring_layout(network)
    elif layout_type == "kamada_kawai":
        pos_culture_network = nx.kamada_kawai_layout(network)
    elif layout_type == "planar":
        pos_culture_network = nx.planar_layout(network)
    else:
        raise Exception("Invalid layout given")

    return pos_culture_network

def plot_avg_price_different_seed(fileName,Data_list, transparency_level = 0.2, color1 = '#2A9D8F', color2 = '#F4A261'):
    #order in Data list is: time, p_t, theta_t, compression_factor
    fig, ax = plt.subplots(figsize = (18, 9))
    time = np.asarray(range(1, Data_list[0].total_steps))
    price = [mkt.history_p_t for mkt in Data_list]
    returns = [(np.asarray(price_series[1:]) - np.asarray(price_series[:-1])) / np.asarray(price_series[:-1]) for price_series in price]
    theta = [mkt.theta_t for mkt in Data_list]
    avg_price = np.asarray([np.sum(values)/len(Data_list) for values in zip(*price)])
    avg_theta = np.asarray([np.sum(values)/len(Data_list) for values in zip(*theta)])
    avg_RA_price = np.asarray((Data_list[0].d/ (Data_list[0].R - 1) + (np.asarray(avg_theta[2:-1]))/ (Data_list[0].R - Data_list[0].ar_1_coefficient)))
    std_deviation_price = np.asarray([np.sqrt(sum((x - mean) ** 2 for x in values) / len(Data_list)) for values, mean in zip(zip(*price), avg_price)])
    corr_price = np.mean([np.corrcoef(price_series[1:], price_series[:-1])[0,1] for price_series in price])
    kurtosis_returns = np.mean([kurtosis(return_series) for return_series in returns])
    std_deviation_theta = np.asarray([np.sqrt(sum((x - mean) ** 2 for x in values) / len(Data_list)) for values, mean in zip(zip(*theta), avg_theta)])
    std_deviation_RA_price = np.asarray(np.asarray(std_deviation_theta[2:-1]))/ (Data_list[0].R - Data_list[0].ar_1_coefficient)
    
    print("avg_price is: ", np.mean(avg_price), "RA_price is: ", np.mean(avg_RA_price))
    print("avg_std is: ", np.mean(std_deviation_price**2), "RA_ is: ", np.mean(std_deviation_RA_price**2))
    print("avg_autocorr is: ", corr_price, "RA is: ", Data_list[0].ar_1_coefficient)
    print("kurtosis is: ", kurtosis_returns)
    # Calculate upper and lower bounds for shading
    upper_bound_price = [avg + std_dev for avg, std_dev in zip(avg_price, std_deviation_price)]
    lower_bound_price = [avg - std_dev for avg, std_dev in zip(avg_price, std_deviation_price)]
    upper_bound_RA_price = [avg + std_dev for avg, std_dev in zip(avg_RA_price, std_deviation_RA_price)]
    lower_bound_RA_price = [avg - std_dev for avg, std_dev in zip(avg_RA_price, std_deviation_RA_price)]

    # Create the plot
    ax.plot(time, avg_price, label='Model Price', color=color1)
    ax.fill_between(time, lower_bound_price, upper_bound_price, color=color1, alpha=transparency_level, label='± 1 Std Dev')
    ax.plot(time, avg_RA_price, label='RA Informed Price', color=color2)
    ax.fill_between(time, lower_bound_RA_price, upper_bound_RA_price, color=color2, alpha=transparency_level, label='± 1 Std Dev')
    
    
    ax.set_xlabel("Steps")
    ax.set_ylabel("Price")
    fig.legend(loc='upper right')
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/" + "avg_p_t_multiple_seeds" + "_timeseries"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_histogram_returns_different_seed(fileName, Data_list):
    returns = np.array([])
    rational_returns = np.array([])
    for i in range(len(Data_list)):
        prices = np.array(Data_list[i].history_p_t)
        rational_prices =(Data_list[i].d/ (Data_list[i].R - 1) + (Data_list[i].theta_t[::Data_list[i].compression_factor][2:])/ (Data_list[i].R - Data_list[i].ar_1_coefficient))
        rational_return = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
        ret = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.append(returns, np.array(ret))
        rational_returns = np.append(rational_returns, np.array(rational_return))
    fig, ax = plt.subplots()
    # Create a histogram of returns (transparent orange)
    ax.hist(returns, bins=30, alpha=0.8, color='#2A9D8F', edgecolor='black', density=True, label='Model Returns')
    ax.hist(rational_returns, bins=30, alpha=0.5, color='#F4A261', edgecolor='black', density=True, label='RA Informed Returns')
   
    
    # # Fit a normal distribution to the data
    # mu, std = norm.fit(returns)

    # # Plot the PDF of the fitted normal distribution (light blue)
    # x = np.linspace(min(returns), max(returns), 100)
    # p = norm.pdf(x, mu, std)
    # ax.plot(x, p, '#2A9D8F', linewidth=2, label='Fitted Normal Distribution')
    ax.set_xlabel('Returns')
    fig.legend(loc='upper right')
    fig.tight_layout()
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + "histogram_returns"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_qq_plot_different_seed(fileName, Data_list):
    returns = np.array([])
    rational_returns = np.array([])
    for i in range(len(Data_list)):
        prices = np.array(Data_list[i].history_p_t)
        rational_prices =(Data_list[i].d/ (Data_list[i].R - 1) + (Data_list[i].theta_t[::Data_list[i].compression_factor][2:] )/ (Data_list[i].R - Data_list[i].ar_1_coefficient))
        rational_return = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
        ret = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.append(returns, np.array(ret))
        rational_returns = np.append(rational_returns, np.array(rational_return))

    fig, ax = plt.subplots()

    # Generate QQ plot
    probplot(returns, dist="norm", plot=ax)#E76F51
    # Customize the plot colors
    ax.lines[0].set_markerfacecolor('#264653')  # Change marker color
    ax.lines[0].set_markeredgecolor('#264653')  # Change marker edge color
    #ax.lines[1].set_color('#264653')  # Change line color
    ax.set_title('')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    
    fig.tight_layout()
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + "qq_plot_returns"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

#function to plot the autocorrelation of the price, as a function of the parameter varied K
# where the x-axis is the degree of connectideness for each of the runs within data list and the y-axis is the corresponding autocorrelation
def plot_autocorrelation_price_multi(fileName,Data_list,property_list, property_varied, property_title):
    fig, ax = plt.subplots()
    y_values = []
    for i in range(len(Data_list)):
        y_value = np.corrcoef(Data_list[i].history_p_t,Data_list[i].history_p_t1)[0,1]
        y_values.append(y_value) 
    ax.plot(np.asarray(property_list), y_values, linestyle='solid', color="blue",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
    ax.set_xlabel("%s" % property_varied)
    ax.set_ylabel("Autocorrelation")
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f =plotName + "/plot_autocorrelation_price_multi_%s" % (property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_variance_price_multi(fileName,Data_list,property_list, property_varied, property_title):
    fig, ax = plt.subplots()
    y_values = []
    rational_vars = []
    for i in range(len(Data_list)):
        y_value = np.var(Data_list[i].history_p_t)
        rational_var = np.var(Data_list[i].theta_t/(Data_list[i].R - Data_list[i].ar_1_coefficient))
        y_values.append(y_value) 
        rational_vars.append(rational_var)
    ax.plot(np.asarray(property_list), y_values, linestyle='solid', color="blue",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
    ax.plot(np.asarray(property_list), rational_vars, linestyle='solid', color="red",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
    ax.set_xlabel("%s" % property_varied)
    ax.set_ylabel("Variance")
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f =plotName + "/plot_variance_price_multi_%s" % (property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_avg_price_multi(fileName,Data_list,property_list, property_varied, property_title):
    fig, ax = plt.subplots()
    y_values = []
    rational_vars = []
    for i in range(len(Data_list)):
        y_value = np.mean(Data_list[i].history_p_t)
        rational_var = np.mean(Data_list[i].d/(Data_list[i].R -1) + Data_list[i].theta_t/(Data_list[i].R - Data_list[i].ar_1_coefficient))
        y_values.append(y_value) 
        rational_vars.append(rational_var)
    ax.plot(np.asarray(property_list), y_values, linestyle='solid', color="blue",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
    ax.plot(np.asarray(property_list), rational_vars, linestyle='solid', color="red",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
    ax.set_xlabel("%s" % property_varied)
    ax.set_ylabel("Price")
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f =plotName + "/plot_variance_price_multi_%s" % (property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
  

def plot_skew_price_multi(fileName,Data_list,property_list, property_varied, property_title):
    fig, ax = plt.subplots()
    y_values = []
    for i in range(len(Data_list)):
        y_value = skew(Data_list[i].history_p_t)
        y_values.append(y_value) 
    ax.plot(np.asarray(property_list), y_values, linestyle='solid', color="blue",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
    ax.set_xlabel("%s" % property_varied)
    ax.set_ylabel("Skewness")
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f =plotName + "/plot_skew_price_multi_%s" % (property_varied)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def scatter_explore_single_param(fileName,Data_list,property_list, property_varied, seed_list_len, target_output, color):
    fig, ax = plt.subplots()
    
    y = [simulation.get(target_output) for simulation in Data_list]
    x = [item for item in property_list for _ in range(seed_list_len)]
    print(y[1])
    ax.scatter(x, y, 
        c = color,
        edgecolor="black",
        s=300,
        alpha = 0.5)
    ax.set_xlabel(property_varied)
    ax.set_ylabel(target_output)
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f =plotName + "/scatter_explore_single_param_%s" % (property_varied) + target_output
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


dpi_save = 600
red_blue_c = True

single_shot = 1
single_param_vary = 0
explore_single_param = 0

###PLOT STUFF
node_size = 200
norm_zero_one = Normalize(vmin=0, vmax=1)
cmap = get_cmap("Blues")
fps = 5
interval = 50
layout = "circular"
round_dec = 2


if __name__ == "__main__":
    if single_shot:
        fileName = "results/small-worldsingle_shot_10_05_45_27_10_2023"#"results/single_shot_steps_500_I_100_network_structure_small_world_degroot_aggregation_1"
        createFolder(fileName)
        Data = load_object(fileName + "/Data", "financial_market")
        base_params = load_object(fileName + "/Data", "base_params")
        print("base_params", base_params)
        print("mean price is: ", np.mean(Data.history_p_t), "mean variance is: ", np.var(Data.history_p_t), "autocorr is: ", np.corrcoef(Data.history_p_t[:-1],Data.history_p_t[1:])[0,1])
        print("mean_rational price is: ", np.mean((Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:] )/ (Data.R - Data.ar_1_coefficient))),"mean_rational variance is: ", np.var((Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:])/ (Data.R - Data.ar_1_coefficient))), "mean_rational corr is: ", np.corrcoef(Data.theta_t[:-1],Data.theta_t[1:])[0,1])
        rational_prices =(Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:])/ (Data.R - Data.ar_1_coefficient))
        rational_returns = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
        #print("kurtosis is:", kurtosis(rational_returns))
        plot_history_p_t = plot_time_series_market(fileName,Data,"Price, $p_t$",dpi_save,"history_p_t")  
        #plot_history_x_it = plot_time_series_market(fileName, Data, "Demand", dpi_save, "history_X_it")
        #plot_histogram_returns = plot_histogram_returns(fileName,Data,"returns",dpi_save)
        plot_qq_plot = plot_qq_plot(fileName, Data, "qq_plot", dpi_save)
        #plot_history_profit = plot_cumulative_consumers(fileName,Data,"Cumulative profit",dpi_save,"history_cumulative_profit",red_blue_c)
        #plot_history_expectation_theta_mean = plot_cumulative_consumers(fileName,Data,"Cumulative expectation mean, $E(\mu_{\theta})$",dpi_save,"history_theta_expectation",red_blue_c)
        #plot_history_expectation_theta_variance = plot_cumulative_consumers(fileName,Data,"Cumulative expectation variance, $E(\sigma_{\theta}^2)$",dpi_save,"history_theta_variance",red_blue_c)
        plt.show()
       
    elif single_param_vary:
        fileName = "results/small-worldsingle_vary_set_seed_17_37_51_26_10_2023"
        Data_list = load_object(fileName + "/Data", "financial_market_list")
        property_varied =  load_object(fileName + "/Data", "property_varied")
        property_list = load_object(fileName + "/Data", "property_list")
        property_title = "K"
        if property_varied == "set_seed":
            plot_avg_price_different_seed(fileName,Data_list, transparency_level=0.3)
            plot_histogram_returns_different_seed(fileName,Data_list)
            plot_qq_plot_different_seed(fileName,Data_list)
        else:
            plot_autocorrelation_price_multi(fileName,Data_list,property_list, property_varied, property_title)
            plot_avg_price_multi(fileName,Data_list,property_list, property_varied, property_title)
            plot_variance_price_multi(fileName,Data_list,property_list, property_varied, property_title)
            plot_skew_price_multi(fileName,Data_list,property_list, property_varied, property_title)
        plt.show()
    
    elif explore_single_param:
        fileName = "results/small-worldexplore_singlegamma_sigma_11_36_11_18_10_2023"
        Data_list = load_object(fileName + "/Data", "financial_market_list")
        property_varied =  load_object(fileName + "/Data", "property_varied")
        property_list = load_object(fileName + "/Data", "property_list")
        seed_list_len = load_object(fileName + "/Data", "seed_list_len")
        scatter_explore_single_param(fileName,Data_list,property_list, property_varied, seed_list_len, "dev_price", "red")
        scatter_explore_single_param(fileName,Data_list,property_list, property_varied, seed_list_len, "excess_var", "blue")
        scatter_explore_single_param(fileName,Data_list,property_list, property_varied, seed_list_len, "excess_autocorr", "green")
        scatter_explore_single_param(fileName,Data_list,property_list, property_varied, seed_list_len, "kurtosis", "orange")
        plt.show()
       