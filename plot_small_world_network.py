"""Plot results adn save them

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
import collections
import os
from scipy.stats import skew
from scipy.stats import norm
from scipy.stats import probplot
from utility import (
    createFolder, 
    load_object, 
    save_object,
)
from plot_results import prod_pos

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

def calc_node_size(Data, min_size, max_size):
    #Get source variance
    list_errors = np.asarray([x.avg_sample_variance for x in Data.agent_list])
    #print("list_errors", list_errors)
    invert_errors = 1/list_errors
    #print("invert errros", invert_errors)
    # Calculate the minimum and maximum values of your original data
    original_min = np.min(invert_errors)
    original_max = np.max(invert_errors)

    # Perform the rescaling
    node_sizes = ((invert_errors - original_min) / (original_max - original_min)) * (max_size - min_size) + min_size

    return node_sizes

def create_custom_colormap(color, N = 260):
    # Create a custom colormap with varying intensity from light to dark for a single color
    cmap = LinearSegmentedColormap.from_list("custom", [(1, 1, 1, 0),color], N=N)
    return cmap

def plot_histogram_own_variance(
    fileName: str,
    Data,
    network_structure,
    property_value: str,
    colour_informed, 
    colour_uninformed, 
    colour_misinformed,
    dpi_save,
    min_size=200,
    max_size=400
):
    # Map dogmatic states to custom colors
    node_colours = [
        colour_informed if x.dogmatic_state == "theta" else
        colour_uninformed if x.dogmatic_state == "normal" else
        colour_misinformed
        for x in Data.agent_list
    ]

    # Calculate node sizes
    node_sizes = np.asarray([x.avg_sample_variance for x in Data.agent_list]) #calc_node_size(Data, min_size, max_size)

    # Create a sorted index based on node_sizes in descending order
    sorted_index = np.argsort(node_sizes)[::-1]

    # Sort node_sizes and node_colours based on the sorted index
    sorted_node_sizes = [node_sizes[i] for i in sorted_index]
    sorted_node_colours = [node_colours[i] for i in sorted_index]

    # Create custom colormaps for each category
    cmap_theta = create_custom_colormap(colour_informed)
    cmap_normal = create_custom_colormap(colour_uninformed)
    cmap_misinformed = create_custom_colormap(colour_misinformed)

    # Normalize node size values for each category's colormap
    theta_nodes = [node for (index, node) in
        filter(lambda tup: Data.agent_list[tup[0]].dogmatic_state == "theta", enumerate(node_sizes))
    ]
    normal_nodes = [node for (index, node) in
        filter(lambda tup: Data.agent_list[tup[0]].dogmatic_state == "normal", enumerate(node_sizes))
    ]
    gamma_nodes = [node for (index, node) in
        filter(lambda tup: Data.agent_list[tup[0]].dogmatic_state == "gamma", enumerate(node_sizes))
    ]

    norm_theta = plt.Normalize(min(theta_nodes) - 1, max(theta_nodes))
    norm_normal = plt.Normalize(min(normal_nodes) - 1, max(normal_nodes))
    norm_misinformed = plt.Normalize(min(gamma_nodes) - 1 , max(gamma_nodes))

    # Create the barplot with varying color intensities within each category
    fig, ax = plt.subplots()
    bars = ax.bar(
        range(len(sorted_node_sizes)),
        sorted_node_sizes,
        color=[
            cmap_theta(norm_theta(node_size)) if color == colour_informed else
            cmap_normal(norm_normal(node_size)) if color == colour_uninformed else
            cmap_misinformed(norm_misinformed(node_size))
            for node_size, color in zip(sorted_node_sizes, sorted_node_colours)
        ],
        edgecolor="black",  # Set the edge color to black
        linewidth=1.2
    )

    # Customize the plot as needed
    ax.set_xlabel('Nodes')
    ax.set_ylabel('Node Sizes')

    # Save or show the plot
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_value + "_plot_variance_histogram" + "_" + network_structure
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_histogram_own_variance_SBM(
    fileName: str,
    Data,
    network_structure,
    property_value: str,
    colour_informed, 
    colour_uninformed_block_1,
    colour_uninformed_block_2,
    colour_misinformed,
    dpi_save,
    min_size=200,
    max_size=400
):
    # Map dogmatic states to custom colors
    node_colours = [
        colour_informed if x.dogmatic_state == "theta" else
        colour_uninformed_block_1 if ((x.dogmatic_state == "normal") and (x.id <= Data.I/2)) else
        colour_uninformed_block_2 if ((x.dogmatic_state == "normal") and (x.id > Data.I/2))  else
        colour_misinformed 
        for x in Data.agent_list
    ]

    # Calculate node sizes
    node_sizes = np.asarray([x.avg_sample_variance for x in Data.agent_list]) #calc_node_size(Data, min_size, max_size)

    # Create a sorted index based on node_sizes in descending order
    sorted_index = np.argsort(node_sizes)[::-1]

    # Sort node_sizes and node_colours based on the sorted index
    sorted_node_sizes = [node_sizes[i] for i in sorted_index]
    sorted_node_colours = [node_colours[i] for i in sorted_index]

    
    # Create the barplot with varying color intensities within each category
    fig, ax = plt.subplots()
    bars = ax.bar(
        range(len(sorted_node_sizes)),
        sorted_node_sizes,
        color=sorted_node_colours,
        edgecolor="black",  # Set the edge color to black
        linewidth=1.2
    )

    # Customize the plot as needed
    ax.set_xlabel('Nodes')
    ax.set_ylabel('Node Sizes')

    # Save or show the plot
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_value + "_plot_variance_histogram" + "_" + network_structure
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def scatter_profit_variance_multiple_seeds(
    fileName,
    Data,
    network_structure,
    property_value,
    colour_informed, 
    colour_uninformed_block_1,
    colour_uninformed_block_2,
    colour_misinformed,
    dpi_save,
):
    # Map dogmatic states to custom colors
    if network_structure == "SBM":
        node_colours = [
            colour_informed if x.dogmatic_state == "theta" else
            colour_uninformed_block_1 if ((x.dogmatic_state == "normal") and (x.id <= Data[0].I/2)) else
            colour_uninformed_block_2 if ((x.dogmatic_state == "normal") and (x.id > Data[0].I/2))  else
            colour_misinformed 
            for x in Data[0].agent_list
        ]
    else:
        node_colours = [
            colour_informed if x.dogmatic_state == "theta" else
            colour_uninformed if x.dogmatic_state == "normal" else
            colour_misinformed            
            for x in Data[0].agent_list
        ]
        label = [
            x.dogmatic_state         
            for x in Data[0].agent_list
        ]
        
        

    agent_profits = {}
    agent_errors = {}
    num_agents = Data[0].I # Assumes the number of agents is the same in all markets
    # for agent_id in range(num_agents):
    #     total_profit = sum(market.agent_list[agent_id].cumulative_profit for market in Data)
    #     average_profit = total_profit / len(Data)
    #     agent_profits[agent_id] = average_profit
    #     total_error = sum(market.agent_list[agent_id].avg_sample_variance for market in Data)
    #     average_error = total_error / len(Data)
    #     agent_errors[agent_id] = average_error

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
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")



def scatter_profit_variance(
    fileName: str,
    Data,
    network_structure,
    property_value: str,
    colour_informed, 
    colour_uninformed_block_1,
    colour_uninformed_block_2,
    colour_misinformed,
    dpi_save,
):
    # Map dogmatic states to custom colors
    if network_structure == "SBM":
        node_colours = [
            colour_informed if x.dogmatic_state == "theta" else
            colour_uninformed_block_1 if ((x.dogmatic_state == "normal") and (x.id <= Data.I/2)) else
            colour_uninformed_block_2 if ((x.dogmatic_state == "normal") and (x.id > Data.I/2))  else
            colour_misinformed 
            for x in Data.agent_list
        ]
    else:
        node_colours = [
            colour_informed if x.dogmatic_state == "theta" else
            colour_uninformed if x.dogmatic_state == "normal" else
            colour_misinformed
            for x in Data.agent_list
        ]

    # Calculate variance
    variances = np.asarray([x.avg_sample_variance for x in Data.agent_list]) #calc_node_size(Data, min_size, max_size)

    #Calculate profits

    profits = np.asarray([x.cumulative_profit for x in Data.agent_list])
    
    # Create the barplot with varying color intensities within each category
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        variances,
        profits,
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
    ax.legend(loc='lower left')

    # Customize the plot as needed
    ax.set_xlabel('Average Forecast Error')
    ax.set_ylabel('Cumulative Profit')
    fig.tight_layout()
    # Save or show the plot
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_value + "_plot_scatter_profit_variance" + "_" + network_structure
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")




def plot_network(
    fileName: str,
    Data: list,
    network_structure,
    property_value: str,
    colour_informed,
    colour_uninformed,
    colour_misinformed,
    dpi_save,
    min_size = 400, 
    max_size = 400
): 
    #data_matrix = np.asarray([eval("Data.agent_list[%s].%s" % (v,property_value)) for v in range(Data.I)]).T
    fig, ax = plt.subplots()
    #colour_adjust = norm_value(data_matrix[-1])
    #ani_step_colours = cmap(colour_adjust)

    """THE AGENT LIST MUST BE IN THE SAME ORDER AS THE ADJACENCY MATRIX FOR THIS WORK"""
    G = nx.from_numpy_matrix(Data.adjacency_matrix)

    # get pos
    if network_structure == "small-world":
        pos = prod_pos(network_structure, G)
    #elif network_structure == "SBM":
    #    pos = nx.bipartite_layout(G, [0] * int(Data.I/2) + [1] * int(Data.I/2))
    else:
        pos = nx.spring_layout(G)
    #purple = (0.5, 0, 0.5)  # RGB values for purple
    #yellow = (0.9, 0.8, 0.2)  # RGB values for yellow
    node_colours = [colour_informed if x.dogmatic_state == "theta" else colour_uninformed if x.dogmatic_state == "normal" else colour_misinformed for x in Data.agent_list ]
    
    #ndoe zise defualt is 300, say between 100, 500

    node_sizes = calc_node_size(Data, min_size, max_size)
    #node_colours = [purple if x.dogmatic_state == "theta" else yellow for x in Data.agent_list ]
    #print(" node_colours", node_colours)
    #print("G",G)
    #print("node_colr", node_colours)
    #print("node_size", node_sizes)
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
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_network_SBM(
    fileName: str,
    Data: list,
    network_structure,
    property_value: str,
    colour_informed,
    colour_uninformed_block_1,
    colour_uninformed_block_2,
    colour_misinformed,
    dpi_save,
    min_size = 400, 
    max_size = 400
): 
    #data_matrix = np.asarray([eval("Data.agent_list[%s].%s" % (v,property_value)) for v in range(Data.I)]).T
    fig, ax = plt.subplots()
    #colour_adjust = norm_value(data_matrix[-1])
    #ani_step_colours = cmap(colour_adjust)

    """THE AGENT LIST MUST BE IN THE SAME ORDER AS THE ADJACENCY MATRIX FOR THIS WORK"""
    G = nx.from_numpy_matrix(Data.adjacency_matrix)

    # get pos
    if network_structure == "small-world":
        pos = prod_pos(network_structure, G)
    #elif network_structure == "SBM":
    #    pos = nx.bipartite_layout(G, [0] * int(Data.I/2) + [1] * int(Data.I/2))
    else:
        pos = nx.spring_layout(G)
    #purple = (0.5, 0, 0.5)  # RGB values for purple
    #yellow = (0.9, 0.8, 0.2)  # RGB values for yellow
    #node_colours = [colour_informed if x.dogmatic_state == "theta" else colour_uninformed if x.dogmatic_state == "normal" else colour_misinformed for x in Data.agent_list ]
        # Map dogmatic states to custom colors
    node_colours = [
        colour_informed if x.dogmatic_state == "theta" else
        colour_uninformed_block_1 if ((x.dogmatic_state == "normal") and (x.id < Data.I/2)) else
        colour_uninformed_block_2 if ((x.dogmatic_state == "normal") and (x.id >= Data.I/2))  else
        colour_misinformed 
        for x in Data.agent_list
    ]

    #ndoe zise defualt is 300, say between 100, 500

    node_sizes = calc_node_size(Data, min_size, max_size)
    #node_colours = [purple if x.dogmatic_state == "theta" else yellow for x in Data.agent_list ]
    #print(" node_colours", node_colours)
    #print("G",G)
    #print("node_colr", node_colours)
    #print("node_size", node_sizes)
    nx.draw(
        G,
        #node_color=ani_step_colours,
        node_color = node_colours,
        ax=ax,
        pos=pos,
        node_size=node_sizes,
        edgecolors="black",
    )
    values = [colour_informed, colour_uninformed_block_1, colour_uninformed_block_2, colour_misinformed]
    c_list = ["Informed", "Uninformed_Block_1", "Uninformed_Block_2", "Misinformed"]#["Generalists","Specialists"]
    for v in range(len(values)):
        plt.scatter([],[], c=values[v], label="%s" % (c_list[v]))
    ax.legend(loc='lower right')
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_value + "_plot_network_shape_SBM"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

if __name__ == "__main__":
    network_structure = "scale_free"
    property_value ="error" 
    colour_informed = "#264653" 
    colour_uninformed = "#E9C46A"
    colour_misinformed = "#E76F51"
    colour_uninformed_block_1 = "#2a9d8f"
    colour_uninformed_block_2 = "#F4A261"
    dpi_save = 600
    min_size = 100
    max_size = 500

single_run = 0

if single_run:
    fileName = "results/small-worldsingle_shot_16_20_36_29_10_2023"#"results/single_shot_steps_500_I_100_network_structure_small_world_degroot_aggregation_1"
    Data = load_object(fileName + "/Data", "financial_market")
    base_params = load_object(fileName + "/Data", "base_params")

    plot_network(fileName,Data, network_structure, property_value, colour_informed, colour_uninformed, colour_misinformed, dpi_save)
    #plot_network_SBM(fileName,Data, network_structure, property_value, colour_informed,colour_uninformed_block_1, colour_uninformed_block_2, colour_misinformed, dpi_save)
    #plot_histogram_own_variance(fileName,Data, network_structure, property_value, colour_informed, colour_uninformed, colour_misinformed, dpi_save)
    #plot_histogram_own_variance_SBM(fileName,Data, network_structure, property_value, colour_informed, colour_uninformed_block_1, colour_uninformed_block_2, colour_misinformed, dpi_save)
    #scatter_profit_variance(fileName,Data, network_structure, property_value, colour_informed, colour_uninformed_block_1, colour_uninformed_block_2, colour_misinformed, dpi_save)
    plt.show()
else:
    fileName = "results/scale_freesingle_vary_set_seed_09_37_07_01_11_2023"
    Data = load_object(fileName + "/Data", "financial_market_list")
    scatter_profit_variance_multiple_seeds(fileName,Data, network_structure, property_value, colour_informed, colour_uninformed_block_1, colour_uninformed_block_2, colour_misinformed, dpi_save)
    plt.show()