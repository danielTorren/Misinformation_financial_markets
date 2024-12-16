"""Plot results adn save them

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import statsmodels.api as sm

from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import probplot
from scipy.stats import kurtosis
from utility import (
    load_object
)

def calc_node_size(Data, min_size, max_size):
    #Get source variance
    list_errors = np.asarray(Data.forecast_errors)
    #print("list_errors", list_errors)
    invert_errors = 1/list_errors
    #print("invert errros", invert_errors)
    # Calculate the minimum and maximum values of your original data
    original_min = np.min(invert_errors)
    original_max = np.max(invert_errors)

    # Perform the rescaling
    node_sizes = ((invert_errors - original_min) / (original_max - original_min)) * (max_size - min_size) + min_size

    return node_sizes


def prod_pos(network_structure: str, network: nx.Graph) -> nx.Graph:

    if network_structure == "small_world":
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


def plot_price_single_run(fileName, Data, dpi = 300):
    fig, ax = plt.subplots(figsize=(18, 9))
    time = np.asarray(range(Data.total_steps))
    price = Data.history_p_t

    # Create the plot
    ax.plot(time, price, label='Model Price')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Price')
    #add the process theta_t to the plot with a different axis
    # ax2 = ax.twinx()
    # ax2.plot(time, Data.history_theta_t, color='red', label='Theta_t', linestyle='dashed')
    # ax2.set_ylabel('Theta_t')
    #fig.legend(loc='upper right')
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/" + "price_single_run" + "_timeseries"
    fig.savefig(f + ".png", dpi=dpi, format="png")



def plot_network(
    fileName: str,
    Data: list,
    network_structure,
    property_value: str,
    colour_informed,
    colour_uninformed,
    colour_misinformed,
    output_folder,
    min_size = 400, 
    max_size = 400,
    dpi = 100
): 

    fig, ax = plt.subplots()
    #colour_adjust = norm_value(data_matrix[-1])
    #ani_step_colours = cmap(colour_adjust)

    """THE AGENT LIST MUST BE IN THE SAME ORDER AS THE ADJACENCY MATRIX FOR THIS WORK"""
    G = Data.network

    # get pos
    if network_structure == "small_world":
        pos = prod_pos(network_structure, G)
    #elif network_structure == "SBM":
    #    pos = nx.bipartite_layout(G, [0] * int(Data.I/2) + [1] * int(Data.I/2))
    else:
        pos = nx.spring_layout(G)
    #purple = (0.5, 0, 0.5)  # RGB values for purple
    #yellow = (0.9, 0.8, 0.2)  # RGB values for yellow
    node_colours = [colour_informed if x == 1 else colour_uninformed if x == 0 else colour_misinformed for x in Data.category_vector ]
    
    #ndoe zise defualt is 300, say between 100, 500

    node_sizes = calc_node_size(Data, min_size, max_size)

    nx.draw(
        G,
        #node_color=ani_step_colours,
        node_color = node_colours,
        ax=ax,
        pos=pos,
        node_size=node_sizes,
        edgecolors="black",
        alpha = 0.9
    )
    values = [colour_informed, colour_uninformed, colour_misinformed]
    c_list = ["Informed", "Uninformed", "Misinformed"]#["Generalists","Specialists"]
    for v in range(len(values)):
        plt.scatter([],[], c=values[v], label="%s" % (c_list[v]))
    ax.legend(loc='upper right')
    plotName = output_folder
    f = plotName + "/" + property_value + "_plot_network_shape" + "_" + network_structure
    #fig.savefig(f + ".eps", dpi=dpi, format="eps")
    fig.savefig(f + ".png", dpi=dpi, format="png")

def scatter_profit_variance_multiple_seeds(
    fileName,
    Data,
    network_structure,
    property_value,
    colour_informed, 
    colour_uninformed_block_1,
    colour_uninformed_block_2,
    colour_misinformed,
    colour_uninformed,
    output_folder,
    dpi = 300
):
    
    # Map dogmatic states to custom colors
    if network_structure == "SBM":
        node_colours = []
        for i,x in enumerate(Data[0].category_vector):
            if x == 1:
                colour_picked = colour_informed
            elif (x == 0) and (i <= Data[0].I/2):
                colour_picked = colour_uninformed_block_1
            elif (x == 0) and (i > Data[0].I/2):
                colour_picked = colour_uninformed_block_2
            else:
                colour_picked = colour_misinformed 
            node_colours.append(colour_picked)
    else:
        node_colours = []
        for i,x in enumerate(Data[0].category_vector):
            if x == 1:
                colour_picked = colour_informed
            elif x == 0:
                colour_picked = colour_uninformed
            else:
                colour_picked = colour_misinformed 
            node_colours.append(colour_picked)

    total_profit = sum(market.cumulative_profit for market in Data)
    average_profit = total_profit / len(Data)
    total_error = sum(market.forecast_errors for market in Data)
    average_error = total_error / len(Data)

    #get the postion of the first informed agent in the network
    informed_agent = [i for i, x in enumerate(Data[0].category_vector) if x == 1][0]
    #get the postion of the first misinformed agent in the network
    misinformed_agent = [i for i, x in enumerate(Data[0].category_vector) if x == -1][0]

    print("profits of informed agent: ", average_profit[informed_agent])
    print("profits of misinformed agent: ", average_profit[misinformed_agent])
    #compute the gini index of the profits of uninformed agents
    uninformed_profits = [average_profit[i] for i, x in enumerate(Data[0].category_vector) if x == 0]
    #compute the gini
    uninformed_profits = np.sort(uninformed_profits)
    #add the minimum value so that all values are non-negative
    uninformed_profits = uninformed_profits + abs(uninformed_profits[0])
    n = len(uninformed_profits)
    gini = 1 - 2/(n-1) * (n - np.sum((np.arange(1, n+1)) * uninformed_profits)/ np.sum(uninformed_profits))
    print("Gini index of uninformed agents: ", gini)


    # Create the barplot with varying color intensities within each category
    fig, ax = plt.subplots()
    ax.scatter(
        list(average_error),
        list(average_profit),
        color=node_colours,
        edgecolor="black",
        s=300,
        alpha = 0.5   # Set the edge color to black
    )
    if network_structure == "SBM":
        values = [colour_informed, colour_uninformed_block_1, colour_uninformed_block_2, colour_misinformed]
        c_list = ["Informed", "Uninformed_Block_1", "Uninformed_Block_2", "Misinformed"]#["Generalists","Specialists"]
    else:
        values = [colour_informed, colour_uninformed, colour_misinformed]
        c_list = ["Informed", "Uninformed", "Misinformed"]#["Generalists","Specialists"]
    
    for v in range(len(values)):
        plt.scatter([],[], c=values[v], label="%s" % (c_list[v]))
    fig.legend(loc='upper right')

    # Customize the plot as needed
    ax.set_title('')
    ax.set_xlabel('Average Forecast Error')
    ax.set_ylabel('Cumulative Profit')
    fig.tight_layout()
    # Save or show the plot
    plotName = output_folder
    f = plotName + "/" + property_value + "_plot_scatter_profit_variance" + "_" + network_structure
    #fig.savefig(f + ".eps", dpi=dpi, format="eps")
    fig.savefig(f + ".png", dpi=dpi, format="png")


############################################################################################################3

def plot_avg_price_different_seed(fileName,Data_list, transparency_level = 0.2, 
                                  color1 = '#2A9D8F', color2 = '#F4A261', dpi = 300):
    #order in Data list is: time, p_t, theta_t, compression_factor
    fig, ax = plt.subplots(figsize = (18, 9))
    time = np.asarray(range(Data_list[0].total_steps))
    price = [mkt.history_p_t for mkt in Data_list]
    returns = [(np.asarray(price_series[1:]) - np.asarray(price_series[:-1])) / np.asarray(price_series[:-1]) for price_series in price]
    theta = [mkt.history_theta_t for mkt in Data_list]
    avg_price = np.asarray([np.sum(values)/len(Data_list) for values in zip(*price)])
    avg_theta = np.asarray([np.sum(values)/len(Data_list) for values in zip(*theta)])
    avg_RA_price = np.asarray((Data_list[0].d/ (Data_list[0].R - 1) + (np.asarray(avg_theta))/ (Data_list[0].R - Data_list[0].ar_1_coefficient)))
    std_deviation_price = np.asarray([np.sqrt(sum((x - mean) ** 2 for x in values) / len(Data_list)) for values, mean in zip(zip(*price), avg_price)])
    corr_price = np.mean([np.corrcoef(price_series[1:], price_series[:-1])[0,1] for price_series in price])
    kurtosis_returns = np.mean([kurtosis(return_series) for return_series in returns])
    std_deviation_theta = np.asarray([np.sqrt(sum((x - mean) ** 2 for x in values) / len(Data_list)) for values, mean in zip(zip(*theta), avg_theta)])
    std_deviation_RA_price = np.asarray(np.asarray(std_deviation_theta))/ (Data_list[0].R - Data_list[0].ar_1_coefficient)
    # element_1 = Data_list[0].proportion_informed/( Data_list[0].epsilon_sigma**2 + Data_list[0].theta_sigma**2/(Data_list[0].R - Data_list[0].ar_1_coefficient)**2)
    # element_2 = Data_list[0].proportion_misinformed/(Data_list[0].epsilon_sigma**2 + (Data_list[0].theta_sigma**2 + Data_list[0].gamma_sigma**2)/(Data_list[0].R - Data_list[0].ar_1_coefficient)**2)
    # element_3 = (1 - Data_list[0].proportion_informed - Data_list[0].proportion_misinformed)/(Data_list[0].epsilon_sigma**2)
    # no_com_price = Data_list[0].d/ (Data_list[0].R - 1) + ((np.asarray(Data_list[0].history_theta_t)/ (Data_list[0].R - Data_list[0].ar_1_coefficient))*element_1 + ((np.asarray(Data_list[0].history_gamma_t) + np.asarray(Data_list[0].history_theta_t))/(Data_list[0].R - Data_list[0].ar_1_coefficient))*element_2 + element_3) / (element_1 + element_2 + element_3)

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
    #fig.savefig(f + ".eps", dpi=dpi, format="eps")
    fig.savefig(f + ".png", dpi=dpi, format="png")

def plot_squared_returns(fileName,Data,
                                  color1 = '#2A9D8F', color2 = '#F4A261', dpi = 300):
    #order in Data list is: time, p_t, theta_t, compression_factor
    fig, ax = plt.subplots(figsize = (18, 9))
    time = np.asarray(range(Data.total_steps))
    price = Data.history_p_t
    returns = (np.asarray(price[1:]) - np.asarray(price[:-1])) / np.asarray(price[:-1]) 
    theta = Data.history_theta_t
    RA_price = np.asarray((Data.d/ (Data.R - 1) + (np.asarray(theta))/ (Data.R - Data.ar_1_coefficient)))
    RA_retruns = (np.asarray(RA_price[1:]) - np.asarray(RA_price[:-1])) / np.asarray(RA_price[:-1])

    # element_1 = Data.proportion_informed/( Data.epsilon_sigma**2 + Data.theta_sigma**2/(Data.R - Data.ar_1_coefficient)**2)
    # element_2 = Data.proportion_misinformed/(Data.epsilon_sigma**2 + (Data.theta_sigma**2 + Data.gamma_sigma**2)/(Data.R - Data.ar_1_coefficient)**2)
    # element_3 = (1 - Data.proportion_informed - Data.proportion_misinformed)/(Data.epsilon_sigma**2)
    # no_com_price = Data.d/ (Data.R - 1) + ((np.asarray(Data.history_theta_t)/ (Data.R - Data.ar_1_coefficient))*element_1 + ((np.asarray(Data.history_gamma_t) + np.asarray(Data.history_theta_t))/(Data.R - Data.ar_1_coefficient))*element_2 + element_3) / (element_1 + element_2 + element_3)
    # no_com_returns = (np.asarray(no_com_price[1:]) - np.asarray(no_com_price[:-1])) / np.asarray(no_com_price[:-1])
    # #compute and plot the acf of the squared returns and the squared returns
    #plot it with confidence intervals
    squared_returns = returns**2
    squared_RA_returns = RA_retruns**2

    sm.graphics.tsa.plot_acf(squared_returns, lags=40, ax=ax, title="Squared Returns ACF")
    sm.graphics.tsa.plot_acf(squared_RA_returns, lags=40, ax=ax, title="Squared Returns ACF")


    fig.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/" + "returns_squared" + "_timeseries"
    #fig.savefig(f + ".eps", dpi=300, format="eps")
    fig.savefig(f + ".png", dpi=300, format="png")





def plot_histogram_returns_different_seed(fileName, Data_list, output_folder, dpi = 300):
    returns = np.array([])
    rational_returns = np.array([])
    for i in range(len(Data_list)):
        prices = np.array(Data_list[i].history_p_t)
        rational_prices =(Data_list[i].d/ (Data_list[i].R - 1) + np.asarray(Data_list[i].history_theta_t)/ (Data_list[i].R - Data_list[i].ar_1_coefficient))
        rational_return = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
        ret = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.append(returns, np.array(ret))
        rational_returns = np.append(rational_returns, np.array(rational_return))
        # element_1 = Data_list[i].proportion_informed/( Data_list[i].epsilon_sigma**2 + Data_list[i].theta_sigma**2/(Data_list[i].R - Data_list[i].ar_1_coefficient)**2)
        # element_2 = Data_list[i].proportion_misinformed/(Data_list[i].epsilon_sigma**2 + (Data_list[i].theta_sigma**2 + Data_list[i].gamma_sigma**2)/(Data_list[i].R - Data_list[i].ar_1_coefficient)**2)
        # element_3 = (1 - Data_list[i].proportion_informed - Data_list[i].proportion_misinformed)/(Data_list[i].epsilon_sigma**2)
        # no_com_price = Data_list[i].d/ (Data_list[i].R - 1) + ((np.asarray(Data_list[i].history_theta_t)/ (Data_list[i].R - Data_list[i].ar_1_coefficient))*element_1 + ((np.asarray(Data_list[i].history_gamma_t) + np.asarray(Data_list[i].history_theta_t))/(Data_list[i].R - Data_list[i].ar_1_coefficient))*element_2 + element_3) / (element_1 + element_2 + element_3)
        # no_com_returns = (np.asarray(no_com_price[1:]) - np.asarray(no_com_price[:-1])) / np.asarray(no_com_price[:-1])
        # #Discard the first 10 values
        # no_com_returns = no_com_returns[10:]

    fig1, ax1 = plt.subplots()
    ax1.plot(returns)
    ax1.plot(rational_returns, alpha = 0.5)

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
    
    plotName = output_folder
    f = plotName + "/" + "histogram_returns"
    fig.savefig(f + ".eps", dpi=dpi, format="eps")
    fig.savefig(f + ".png", dpi=dpi, format="png")


def compute_moments(fileName, Data_list, output_folder):
    returns = np.array([])
    rational_returns = np.array([])
    autocorrelation = np.array([])
    excesse_var = np.array([])
    skewness = np.array([])
    kurtosis = np.array([])
    for i in range(len(Data_list)):
        prices = np.array(Data_list[i].history_p_t)
        autocorrelation = np.append(autocorrelation , np.corrcoef(prices[1:], prices[:-1])[0,1])
        rational_prices =(Data_list[i].d/ (Data_list[i].R - 1) + np.asarray(Data_list[i].history_theta_t)/ (Data_list[i].R - Data_list[i].ar_1_coefficient))
        excesse_var = np.append(excesse_var, np.var(prices) - np.var(rational_prices))
        rational_return = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
        ret = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.append(returns, np.array(ret))
        skewness= np.append(skewness, np.mean(((ret - np.mean(ret)) / np.std(ret)) ** 3))
        kurtosis= np.append(kurtosis, np.mean(((ret - np.mean(ret)) / np.std(ret)) ** 4))
        rational_returns = np.append(rational_returns, np.array(rational_return))

    #compute the average autocorrelation of prices
    avg_autocorrelation = np.mean(autocorrelation)
    #compute the average excess variance of prices
    avg_excess_var = np.mean(excesse_var)
    #compute the average skewness of returns
    avg_skewness = np.mean(skewness)
    #compute the average kurtosis of returns
    avg_kurtosis = np.mean(kurtosis)

    return {"avg_autocorrelation": avg_autocorrelation, "avg_excess_var": avg_excess_var, "avg_skewness": avg_skewness, "avg_kurtosis": avg_kurtosis}


def plot_qq_plot_different_seed(fileName, Data_list, output_folder, dpi = 300):
    returns = np.array([])
    rational_returns = np.array([])
    for i in range(len(Data_list)):
        prices = np.array(Data_list[i].history_p_t)
        rational_prices =(Data_list[i].d/ (Data_list[i].R - 1) + np.asarray(Data_list[i].history_theta_t)/ (Data_list[i].R - Data_list[i].ar_1_coefficient))
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
    
    plotName = output_folder
    f = plotName + "/" + "qq_plot_returns"
    #fig.savefig(f + ".eps", dpi=dpi, format="eps")
    fig.savefig(f + ".png", dpi=dpi, format="png")


def main(
    fileName_single = "results/scale_freesingle_shot_15_29_46_28_03_2024",
    fileName_multi = "results/scale_freesingle_vary_set_seed_09_37_07_01_11_2023",
    output_folder = "Figures/SW",
    network_structure = "scale_free"
):
    
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
        'ytick.labelsize': ticksize,
        'legend.fontsize': fontsize,
        'legend.markerscale': 2.0
    }
    plt.rcParams.update(params) 

    property_value ="error" 
    colour_informed = "#264653" 
    colour_uninformed = "#E9C46A"
    colour_misinformed = "#E76F51"
    colour_uninformed_block_1 = "#2a9d8f"
    colour_uninformed_block_2 = "#F4A261"
    
    min_size = 100
    max_size = 500


    #SINGLE RUN
    Data_single = load_object(fileName_single + "/Data", "financial_market")
    plot_network(fileName_single,Data_single, network_structure, property_value, colour_informed, colour_uninformed, colour_misinformed, output_folder)
    #plot_squared_returns(fileName_single,Data_single)
    #plot_price_single_run(fileName_single, Data_single)

    #MULTIPLE STOCHASTIC SEED RUNS
    Data_multi = load_object(fileName_multi + "/Data", "financial_market_list") 
    scatter_profit_variance_multiple_seeds(fileName_multi,Data_multi, network_structure, property_value, colour_informed, colour_uninformed_block_1, colour_uninformed_block_2, colour_misinformed, colour_uninformed, output_folder, dpi= 100)
    #plot_avg_price_different_seed(fileName_multi,Data_multi, transparency_level=0.3)
    plot_histogram_returns_different_seed(fileName_multi,Data_multi, output_folder, dpi=100)
    plot_qq_plot_different_seed(fileName_multi,Data_multi, output_folder, dpi=100)

    plt.show()


if __name__ == "__main__":
    main(
        fileName_single = "results/SBMsingle_shot_16_59_45_17_06_2024",
        fileName_multi = "results/scale_freesingle_vary_set_seed_09_37_07_01_11_2023",
        network_structure = "SBM"
    )