"""Plot results adn save them

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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

def plot_network(
    fileName: str,
    Data: list,
    network_structure,
    property_value: str,
    colour_informed,
    colour_uninformed,
    colour_misinformed,
    min_size = 400, 
    max_size = 400
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
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_value + "_plot_network_shape" + "_" + network_structure
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def scatter_profit_variance_multiple_seeds(
    fileName,
    Data,
    network_structure,
    property_value,
    colour_informed, 
    colour_uninformed_block_1,
    colour_uninformed_block_2,
    colour_misinformed,
    colour_uninformed
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
    fig.legend(loc='lower right')

    # Customize the plot as needed
    ax.set_title('')
    ax.set_xlabel('Average Forecast Error')
    ax.set_ylabel('Cumulative Profit')
    fig.tight_layout()
    # Save or show the plot
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_value + "_plot_scatter_profit_variance" + "_" + network_structure
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


############################################################################################################3

def plot_avg_price_different_seed(fileName,Data_list, transparency_level = 0.2, color1 = '#2A9D8F', color2 = '#F4A261'):
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
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_histogram_returns_different_seed(fileName, Data_list):
    returns = np.array([])
    rational_returns = np.array([])
    for i in range(len(Data_list)):
        prices = np.array(Data_list[i].history_p_t)
        rational_prices =(Data_list[i].d/ (Data_list[i].R - 1) + np.asarray(Data_list[i].history_theta_t)/ (Data_list[i].R - Data_list[i].ar_1_coefficient))
        rational_return = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
        ret = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.append(returns, np.array(ret))
        rational_returns = np.append(rational_returns, np.array(rational_return))

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
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + "histogram_returns"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")

def plot_qq_plot_different_seed(fileName, Data_list):
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
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + "qq_plot_returns"
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png")


def main(
    fileName_single = "results/scale_freesingle_shot_15_29_46_28_03_2024",
    fileName_multi = "results/scale_freesingle_vary_set_seed_09_37_07_01_11_2023",
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
    plot_network(fileName_single,Data_single, network_structure, property_value, colour_informed, colour_uninformed, colour_misinformed)
    
    #MULTIPLE STOCHASTIC SEED RUNS
    Data_multi = load_object(fileName_multi + "/Data", "financial_market_list") 
    scatter_profit_variance_multiple_seeds(fileName_multi,Data_multi, network_structure, property_value, colour_informed, colour_uninformed_block_1, colour_uninformed_block_2, colour_misinformed, colour_uninformed)
    plot_avg_price_different_seed(fileName_multi,Data_multi, transparency_level=0.3)
    plot_histogram_returns_different_seed(fileName_multi,Data_multi)
    plot_qq_plot_different_seed(fileName_multi,Data_multi)

    plt.show()


if __name__ == "__main__":
    main(
        fileName_single = "results/scale_freesingle_shot_15_29_46_28_03_2024",
        fileName_multi = "results/scale_freesingle_vary_set_seed_09_37_07_01_11_2023",
        network_structure = "scale_free"
    )