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

# SMALL_SIZE = 14
# MEDIUM_SIZE = 18
# BIGGER_SIZE = 22

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

fontsize= 14
ticksize = 14
figsize = (12, 12)
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


def plot_time_series_consumers(fileName,Data,y_title,dpi_save,property_y,red_blue_c):
    fig, ax = plt.subplots()

    
    for v in range(len(Data.agent_list)):
        data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y)))
        ax.plot(np.asarray(Data.history_time), data_ind)

    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)

    if property_y == "history_payoff_expectation":
        #Data.theta_t[::Data.compression_factor]
        ax.plot(Data.history_time, Data.d + Data.theta_t[::Data.compression_factor], linestyle='dashed', color="black",  linewidth=2, alpha=0.5) 
    elif property_y == "history_profit":
        ax.plot(Data.history_time, (Data.theta_t - Data.gamma_t)*(Data.theta_t + Data.epsilon_t) - Data.c_info, linestyle='dashdot', color="green" , linewidth=2)
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/timeseries_consumers_%s" % (property_y)
    #print("f",f)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

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

def plot_time_series_consumer_triple(fileName,Data,y_title,dpi_save,property_y,num_signals, titles,red_blue_c):
    fig, axes = plt.subplots(nrows=1, ncols=num_signals, figsize=(10,6))
    #print(property_y)

    for i, ax in enumerate(axes.flat):
        for v in range(len(Data.agent_list)):
            # if (Data.agent_list[v].history_c_bool[0]):
            #     color = "blue"
            # else:
            #     color = "red"
            data_ind = np.asarray(  eval("Data.agent_list[%s].%s" % (str(v), property_y))   )#get the list of data for that specific agent
            #print(data_ind[:, i])
            # ax.plot(np.asarray(Data.history_time), data_ind[:, i], color = color)
            ax.plot(np.asarray(Data.history_time), data_ind[:, i])#plot the ith column in the weighting matrix which is [T x num_signals] where T is total steps
        ax.set_xlabel("Steps")
        #ax.set_ylim([0,1])
        ax.set_ylabel("%s" % y_title)
        ax.set_title(titles[i])
        #ax.set_ylim = ([0,1])

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/plot_time_series_consumer_triple_%s" % property_y
    
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")


def plot_histogram_returns(fileName, Data, y_title, dpi_save):
    property_y = "history_p_t"
    fig, ax = plt.subplots()
    data = eval("Data.%s" % property_y)
    prices = np.array(data)

    # Calculate returns
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    rational_prices =(Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:] * Data.ar_1_coefficient)/ (Data.R - Data.ar_1_coefficient))
    rational_returns = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
    
    # Create a histogram of returns (transparent orange)
    ax.hist(returns, bins=30, alpha=0.5, color='orange', edgecolor='black', density=True, label='Returns Histogram')
    ax.hist(rational_returns, bins=30, alpha=0.5, color='green', edgecolor='black', density=True, label='Returns Histogram')

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
    rational_prices =(Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:] * Data.ar_1_coefficient)/ (Data.R - Data.ar_1_coefficient))
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
        ax.plot(Data.history_time, (Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:] * Data.ar_1_coefficient)/ (Data.R - Data.ar_1_coefficient)), linestyle='dashed', color="red")

        # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        # color = 'tab:green'
        # ax2.set_ylabel('theta_t', color=color )  # we already handled the x-label with ax1
        # ax2.plot(Data.history_time, (Data.theta_t[::Data.compression_factor]), linestyle='dashed',color=color , linewidth=2)
        # #ax2.tick_params(axis='theta_t', labelcolor=color)   
        
        #ax.hline([(Data.d)/Data.R], linestyle='dashed', color="black",  linewidth=2, alpha=0.5)
        #ax.plot(Data.history_time, Data.d + np.asarray(Data.theta_t[::Data.compression_factor])/Data.R, linestyle='dashed', color="black",  linewidth=2, alpha=0.5)
        #print(np.sum(data-(Data.d + Data.theta_t[::Data.compression_factor])/Data.R))
        #ax.axhline(y = (Data.d)/Data.R, linestyle='dashdot', color="red" , linewidth=2)
        #ax.axhline(y = (Data.d + Data.theta_mean)/Data.R, linestyle='dashdot', color="green" , linewidth=2)
        #ax.axhline(y = (Data.d + Data.broadcast_quality*Data.theta_mean +  (1 - Data.broadcast_quality)*Data.gamma_mean)/Data.R, linestyle='dashdot', color="purple" , linewidth=2)
        #labels = labels = ["Price", "Uninformed price"]
        #fig.legend(labels = labels,loc = 'lower right', borderaxespad=10)
    elif property_y == "history_informed_proportion":
        ax.plot(Data.history_time, data , linestyle='solid', color="blue",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
        # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        # color = 'tab:blue'
        # ax2.set_ylabel('informed_profit', color=color )  # we already handled the x-label with ax1
        # ax2.plot(Data.history_time, (Data.p_t + Data.d_t)*(Data.d + Data.theta_t - Data.R*Data.p_t)/(Data.a*Data.epsilon_sigma**2) - Data.c_info, color=color, alpha = 0.7, linestyle = 'dashed')
        # ax2.tick_params(axis='y', labelcolor=color)   
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_timeseries"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")



def plot_time_series_market_pulsing(fileName,Data,y_title,dpi_save):

    fig, ax = plt.subplots()
    data = Data.theta_t[::Data.compression_factor]*Data.gamma_t/np.abs(Data.theta_t[::Data.compression_factor]*Data.gamma_t)

    # bodge
    ax.scatter(Data.history_time, data, color="blue")
    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + "_timeseries_pulsing"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_time_series_market_matrix_transpose(fileName,Data,y_title,dpi_save,property_y):

    fig, ax = plt.subplots()
    for agent in Data.agent_list:
        if agent.dogmatic_state == "theta":
            color = "#264653"
        elif agent.dogmatic_state =="gamma":
            color = "#E9C46A"
        else:
            color = "#E76F51"
        ax.plot(Data.history_time, np.asarray(eval("agent.%s" % property_y)), color=color)
    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_plot_time_series_market_matrix_transpose"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def prod_pos(network_structure: str, network: nx.Graph) -> nx.Graph:

    if network_structure == "small-world":
        layout_type = "circular"
    elif network_structure == "random" or "scale-free":
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

def anim_weighting_matrix_combined(
    fileName: str,
    Data: list,
    cmap_weighting,
    interval: int,
    fps: int,
    round_dec: int,
    weighting_matrix_time_series
):
    
    def update(i, Data, ax, title, weighting_matrix_time_series):

        ax.clear()

        ax.matshow(
            np.asarray(weighting_matrix_time_series[i]),
            cmap=cmap_weighting,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )

        ax.set_xlabel("Individual $j$")
        ax.set_ylabel("Individual $i$")

        title.set_text(
            "Time= {}".format(round(Data.history_time[i], round_dec))
        )

    fig, ax = plt.subplots()

    # plt.tight_layout()

    title = plt.suptitle(t="", fontsize=20)

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_weighting),
        ax=ax,
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label(r"Social network weighting, $\alpha_{i,j}$")

    # need to generate the network from the matrix
    # G = nx.from_numpy_matrix(Data_list[0].history_weighting_matrix[0])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data.history_time)),
        fargs=(Data, ax, title,weighting_matrix_time_series),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = fileName + "/Animations"
    f = animateName + "/nimate_weighting_matrix.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani

def anim_weighting_matrix(
    fileName: str,
    Data: list,
    cmap_weighting,
    interval: int,
    fps: int,
    round_dec: int,
):
    def update(i, Data, ax, title):

        ax.clear()

        ax.matshow(
            Data.history_weighting_matrix[i],
            cmap=cmap_weighting,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )

        ax.set_xlabel("Individual $j$")
        ax.set_ylabel("Individual $i$")

        title.set_text(
            "Time= {}".format(round(Data.history_time[i], round_dec))
        )

    fig, ax = plt.subplots()

    # plt.tight_layout()

    title = plt.suptitle(t="", fontsize=20)

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_weighting),
        ax=ax,
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label(r"Social network weighting, $\alpha_{i,j}$")

    # need to generate the network from the matrix
    # G = nx.from_numpy_matrix(Data_list[0].history_weighting_matrix[0])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data.history_time)),
        fargs=(Data, ax, title),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = fileName + "/Animations"
    f = animateName + "/nimate_weighting_matrix.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani


def plot_network_shape(    
    #fileName, Data, base_params["network_structure"], "c bool","history_c_bool"
    fileName: str,
    Data: list,
    network_structure,
    colour_bar_label:str,
    property_value: str,
    cmap,
    norm_value,
    node_size,
    dpi_save
):  

    data_matrix = np.asarray([eval("Data.agent_list[%s].%s" % (v,property_value)) for v in range(Data.I)]).T

    fig, ax = plt.subplots()

    #colour_adjust = norm_value(data_matrix[-1])
    #ani_step_colours = cmap(colour_adjust)

    G = nx.from_numpy_matrix(Data.adjacency_matrix)

    # get pos
    pos = prod_pos(network_structure, G)
    purple = (0.5, 0, 0.5)  # RGB values for purple
    yellow = (0.9, 0.8, 0.2)  # RGB values for yellow
    node_colours = ["blue" if x.dogmatic_state == "theta" else "red" for x in Data.agent_list ]
    #node_colours = [purple if x.dogmatic_state == "theta" else yellow for x in Data.agent_list ]

    #print(" node_colours", node_colours)

    nx.draw(
        G,
        #node_color=ani_step_colours,
        node_color = node_colours,
        ax=ax,
        pos=pos,
        node_size=node_size,
        edgecolors="black",
    )

    values = ["red", "blue"]
    c_list = ["Generalists","Specialists"]
    for v in range(len(values)):
        plt.scatter([],[], c=values[v], label="%s" % (c_list[v]))

    ax.legend(loc='upper right')

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_value + "_plot_network_shape"
    # fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_time_series_consumer_triple_multi(fileName,Data_list,y_title,dpi_save,property_y,signal, property_varied, property_list,titles):
    
    ncols = len(property_list)
    nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(10,6))
    #print(property_y)

    for i, ax in enumerate(axes.flat):
        for v in range(len(Data_list[i].agent_list)):
            data_ind = np.asarray(eval("Data_list[%s].agent_list[%s].%s" % (str(i),str(v), property_y)))#get the list of data for that specific agent
            ax.plot(np.asarray(Data_list[i].history_time), data_ind[:, signal])#plot the ith column in the weighting matrix which is [T x num_signals] where T is total steps
            #ax.plot(np.asarray(Data_list[i].history_time), data_ind[:, signal], color = color)#plot the ith column in the weighting matrix which is [T x num_signals] where T is total steps
        
        ax.set_xlabel("Steps")
        #ax.set_ylim([0,1])
        
        ax.set_title(titles[i])
        #ax.set_ylim = ([0,1])
    fig.supylabel("%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/plot_time_series_consumer_triple_multi_%s_%s" % (property_varied,signal)
    
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")

def plot_initial_priors_hist(fileName,Data,dpi_save):
    fig, ax = plt.subplots()

    # bodge
    ax.hist(Data.mu_0_i_c, color="blue", label = "Private signal")
    ax.hist(Data.mu_0_i_no_c, color="red", label = "No private signal")
    ax.set_xlabel("Inital prior")
    ax.set_ylabel("Counts")

    plotName = fileName + "/Plots"
    f = plotName + "/" + "_plot_initial_priors_hist"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def degree_distribution_single(   
    fileName,
    Data,
    dpi_save,
):

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    G = Data.network
    
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    ax.bar(deg, cnt, width=0.80, color='b')

    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    #ax.set_ylim(0, 1)

    plotName = fileName + "/Prints"
    f = plotName + "/degree_distribution"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_final_wighting_matrix(
        fileName,
        Data,
        dpi_save
):
#To plot the final weighting matrix
    fig, ax = plt.subplots()
    data = np.asarray(Data.history_weighting_matrix[-1])

    # Create the heatmap
    heatmap = ax.imshow(data, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(heatmap)

    # Add labels, title, and axis ticks
    #ax.set_xlabel('X-axis')
    #ax.set_ylabel('Y-axis')
    ax.set_title('Confidence Matrix')

    #saving
    plotName = fileName + "/Plots"
    f = plotName + "/final_weighting_matrix"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_line_weighting_matrix(
    fileName,
    Data,
    dpi_save,
):
    fig, ax = plt.subplots()
    data = np.asarray(Data.history_weighting_matrix)

    ###dumb way       
    for i in Data.I_array:
        for j in Data.I_array:
            ax.plot(Data.history_time, data[:,i,j])

    ax.set_xlabel("Steps")
    ax.set_ylabel(r"Link strength, $\alpha_{i,j}$")


    plotName = fileName + "/Plots"
    f = plotName + "/plot_line_weighting_matrix"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_node_influence(fileName,Data,dpi_save):

    ######
    #calc the neighbours to normalise
    neighbour_count_vector = np.sum(Data.adjacency_matrix, axis = 1)     
    #print("neighbour_count_vector",neighbour_count_vector,neighbour_count_vector.shape )   
    neighbour_count_vector_omega = np.concatenate((np.asarray([Data.I]), neighbour_count_vector), axis = 0)
    #print("neighbour_count_vector_omega", neighbour_count_vector_omega)

    ###########
    #how many people are using each turn
    history_c_bool_matrix = []
    for v in range(len(Data.agent_list)):
        history_c_bool_matrix.append(Data.agent_list[v].history_c_bool)

    history_c_bool_matrix_array = np.asarray(history_c_bool_matrix).T#row rperesent 1 time step , column is a person

    history_c_bool_matrix_array_sums = np.nansum(history_c_bool_matrix_array, axis = 1)# ow many people use theta signal at time

    
    history_c_bool_matrix_array_sums[history_c_bool_matrix_array_sums == 0] = np.nan
    #print("history_c_bool_matrix_array_sums", history_c_bool_matrix_array_sums)

    ###########################################
    #Get the WEIGHTING maTrIX

    influence_vector_time_series = []
    for t in range(len(Data.history_time)):
        b = np.asarray(Data.weighting_matrix_timeseries[t])
        #print("b",b, b.shape)
        a = np.nansum(b, axis = 0)
        #print("a",a, a.shape)
        norm_vector = np.concatenate(([history_c_bool_matrix_array_sums[t]],neighbour_count_vector_omega), axis = 0)
        #print("norm_vector", norm_vector)
        normalised_influence = a/norm_vector
        #print("normalised_influence", normalised_influence)
        influence_vector_time_series.append(list(normalised_influence))

    influence_vector_time_serie_array = np.asarray(influence_vector_time_series).T
    print("influence_vector_time_serie_array shape",influence_vector_time_serie_array.shape)

    Matrix = np.vstack([np.array([influence_vector_time_serie_array[0]]),influence_vector_time_serie_array[1]])
    network_average = influence_vector_time_serie_array[2:].mean(axis = 0)
    Matrix = np.vstack([Matrix, network_average])
    
    
    fig, ax = plt.subplots(figsize=(10,5))
    cax = ax.matshow(Matrix, cmap=plt.cm.Blues, aspect='auto')
    fig.colorbar(cax)
    ax.set_yticklabels(['']+['CI','FI','Network'])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Node normalised influence")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_node_influence_three"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

    ###############################################################################
    #line plot!
    fig2, ax2 = plt.subplots(figsize=(10,6))
    #### plot the three steps as time series
    ax2.plot(Data.history_time, influence_vector_time_serie_array[0], label = "$\theta$")#theta
    ax2.plot(Data.history_time, influence_vector_time_serie_array[1], label = "$\zeta$")#zeta
    ax2.plot(Data.history_time, network_average, label = "Network")#network

    #ax.set_yticklabels(['']+['CI','FI','Network'])
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Node normalised influence")
    ax2.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_node_influence_norm"
    fig2.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig2.savefig(f + ".png", dpi=dpi_save, format="png")

    #######################################################################################
    #unormalised influence
    
    influence_vector_time_series_unorm = []
    for t in range(len(Data.history_time)):
        b = np.asarray(Data.weighting_matrix_timeseries[t])
        #print("b",b, b.shape)
        a = np.nansum(b, axis = 0)
        influence_vector_time_series_unorm.append(list(a))

    influence_vector_time_serie_array_unorm = np.asarray(influence_vector_time_series_unorm).T
    network_average_unorm = influence_vector_time_serie_array_unorm[2:].mean(axis = 0)

    fig3, ax3 = plt.subplots(figsize=(10,6))
    #### plot the three steps as time series
    ax3.plot(Data.history_time, influence_vector_time_serie_array_unorm[0], label = "$\theta$")#theta
    ax3.plot(Data.history_time, influence_vector_time_serie_array_unorm[1], label = "$\zeta$")#zeta
    ax3.plot(Data.history_time, network_average_unorm, label = "Network")#network

    #ax.set_yticklabels(['']+['CI','FI','Network'])
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Node influence")
    ax3.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_node_influence"
    fig3.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig3.savefig(f + ".png", dpi=dpi_save, format="png")

    ############################################################################################
    #unorm all the lines!

    fig4, ax4 = plt.subplots(figsize=(10,6))
    #### plot the three steps as time series
    ax4.plot(Data.history_time, influence_vector_time_serie_array_unorm[0], label = "$\theta$")#theta
    ax4.plot(Data.history_time, influence_vector_time_serie_array_unorm[1], label = "$\zeta$")#zeta
    for i in range(Data.I):
        ax4.plot(Data.history_time, influence_vector_time_serie_array_unorm[i + 2])#network

    
    ax4.legend()

    #ax.set_yticklabels(['']+['CI','FI','Network'])
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Node influence")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_node_influence_all the lines"
    fig4.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig4.savefig(f + ".png", dpi=dpi_save, format="png")

    # fig, ax = plt.subplots()

    # for v in range(len(Data.agent_list) + 2):
    #     if v == 0:
    #         ax.plot(Data.history_time, influence_vector_time_serie_array[v], label = "Private signal", linestyle = "--")
    #     elif v == 1:
    #         ax.plot(Data.history_time, influence_vector_time_serie_array[v], label = "Public broadcast", linestyle = "dashdot")
    #     else:
    #         ax.plot(Data.history_time, influence_vector_time_serie_array[v])

    # ax.legend()
    # ax.set_xlabel("Steps")
    # ax.set_ylabel("Node normalised influence")



def calc_weighting_matrix_time_series(Data):
    #THIS IS PRODUCIGN MASSIVE FILES, FIX
    """
    influence_vector_time_series = []
    for t in range(len(Data.history_time)):
        weighting_matrix_time_series_at_t = []
        for v in range(len(Data.agent_list)):
            weighting_matrix_time_series_at_t.append(Data.agent_list[v].history_weighting_vector[t])
        influence_vector_time_series.append(weighting_matrix_time_series_at_t)
    #print("nfluence_vector_time_series array ", np.asarray(influence_vector_time_series).shape)

    """

    influence_vector_time_series = np.asarray([[v.history_weighting_vector[t] for v in Data.agent_list] for t in range(len(Data.history_time))])

    """
    def calc_weighting_matrix_time_series_alt(Data):
        weighting_matrix_time_series_i_t = []
        for v in range(len(Data.agent_list)):
            weighting_matrix_time_series_i_k_t.append(Data.agent_list[v].history_weighting_vector)

        weighting_matrix_time_series_t_i_k = np.asarrayy(weighting_matrix_time_series_at_t).T
        return weighting_matrix_time_series_at_t
    """

    return influence_vector_time_series

def plot_multi_time_series(fileName,Data_list,property_list, property_varied, property_title, property_y):
    
    fig, axes = plt.subplots(nrows=1, ncols=len(Data_list), sharey=True, sharex=True, figsize=(10,6))
    #print(property_y)

    for i, ax in enumerate(axes.flat):
        for v in range(len(Data_list[i].agent_list)):
            if (Data_list[i].agent_list[v].dogmatic_state == "theta"):
                color = "blue"
            elif(Data_list[i].agent_list[v].dogmatic_state == "gamma"):
                color = "red"
            else:
                color = "black"
            data_ind = np.asarray(eval("Data_list[%s].agent_list[%s].%s" % (str(i),str(v), property_y)))
            ax.plot(np.asarray(Data_list[i].history_time), data_ind, color = color)

        ax.set_xlabel("Steps")
        ax.set_ylabel("%s" % property_y)
        ax.set_title("%s = %s" % (property_title, property_list[i]))
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_multi_time_series_%s_%s" % (property_varied,property_y)
    #print("f",f)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_price_different_seed(fileName,Data_list, transparency_level = 0.2):
    #order in Data list is: time, p_t, theta_t, compression_factor
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.plot(np.asarray(Data_list[0].history_time), np.array(Data_list[0].history_p_t), linestyle='solid', color="#264653",  linewidth=2, marker = "o", label = 'Simulation Price')
    ax.plot(np.asarray(Data_list[0].history_time), np.array((Data_list[0].d/ (Data_list[0].R - 1) + (Data_list[0].theta_t[::Data_list[0].compression_factor][2:] * Data_list[0].ar_1_coefficient)/ (Data_list[0].R - Data_list[0].ar_1_coefficient))), color="#E76F51", marker = "s", linewidth = 2, label = 'RA Informed Price')
    for i in range(1,len(Data_list)):
        ax.plot(np.array(Data_list[i].history_p_t), linestyle='solid', color="#264653", linewidth=1, alpha = transparency_level)
        ax.plot(np.array((Data_list[i].d/ (Data_list[i].R - 1) + (Data_list[i].theta_t[::Data_list[i].compression_factor] * Data_list[i].ar_1_coefficient)/ (Data_list[i].R - Data_list[i].ar_1_coefficient))), color="#E76F51", alpha = transparency_level, linewidth = 1)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Price")
    fig.legend(loc='upper right')
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f = plotName + "/" + "history_p_t_multiple_seeds" + "_timeseries"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_avg_price_different_seed(fileName,Data_list, transparency_level = 0.2, color1 = '#2A9D8F', color2 = '#F4A261'):
    #order in Data list is: time, p_t, theta_t, compression_factor
    fig, ax = plt.subplots(figsize = (12, 6))
    time = np.asarray(range(1, Data_list[0].total_steps))
    price = [mkt.history_p_t for mkt in Data_list]
    returns = [(np.asarray(price_series[1:]) - np.asarray(price_series[:-1])) / np.asarray(price_series[:-1]) for price_series in price]
    theta = [mkt.theta_t for mkt in Data_list]
    avg_price = np.asarray([np.sum(values)/len(Data_list) for values in zip(*price)])
    avg_theta = np.asarray([np.sum(values)/len(Data_list) for values in zip(*theta)])
    avg_RA_price = np.asarray((Data_list[0].d/ (Data_list[0].R - 1) + (np.asarray(avg_theta[2:]) * Data_list[0].ar_1_coefficient)/ (Data_list[0].R - Data_list[0].ar_1_coefficient)))
    std_deviation_price = np.asarray([np.sqrt(sum((x - mean) ** 2 for x in values) / len(Data_list)) for values, mean in zip(zip(*price), avg_price)])
    corr_price = np.mean([np.corrcoef(price_series[1:], price_series[:-1])[0,1] for price_series in price])
    kurtosis_returns = np.mean([kurtosis(return_series) for return_series in returns])
    std_deviation_theta = np.asarray([np.sqrt(sum((x - mean) ** 2 for x in values) / len(Data_list)) for values, mean in zip(zip(*theta), avg_theta)])
    std_deviation_RA_price = np.asarray(np.asarray(std_deviation_theta[2:]) * Data_list[0].ar_1_coefficient)/ (Data_list[0].R - Data_list[0].ar_1_coefficient)
    
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
    for i in range(len(Data_list)):
        prices = np.array(Data_list[i].history_p_t)
        ret = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.append(returns, np.array(ret))
    print("avg_return is:", np.mean(returns))
    fig, ax = plt.subplots()
    # Create a histogram of returns (transparent orange)
    ax.hist(returns, bins=30, alpha=0.8, color='#F4A261', edgecolor='black', density=True, label='Returns Histogram')

    # Fit a normal distribution to the data
    mu, std = norm.fit(returns)

    # Plot the PDF of the fitted normal distribution (light blue)
    x = np.linspace(min(returns), max(returns), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, '#2A9D8F', linewidth=2, label='Fitted Normal Distribution')
    ax.set_xlabel('Returns')
    
    fig.tight_layout()
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + "histogram_returns"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_qq_plot_different_seed(fileName, Data_list):
    returns = np.array([])
    for i in range(len(Data_list)):
        prices = np.array(Data_list[i].history_p_t)
        ret = (prices[1:] - prices[:-1]) / prices[:-1]
        returns = np.append(returns, np.array(ret))

    fig, ax = plt.subplots()

    # Generate QQ plot
    probplot(returns, dist="norm", plot=ax)
    # Customize the plot colors
    ax.lines[0].set_markerfacecolor('#E76F51')  # Change marker color
    ax.lines[0].set_markeredgecolor('#E76F51')  # Change marker edge color
    ax.lines[1].set_color('#264653')  # Change line color
    ax.set_title('')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    
    fig.tight_layout()
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + "qq_plot_returns"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_time_series_market_multi(fileName,Data_list,property_list, property_varied, property_title, property_y):
    fig, axes = plt.subplots(nrows=len(Data_list), ncols=1, sharey=True, sharex=True, figsize=(10,6))
    for i, ax in enumerate(axes.flat):
        data = eval("Data_list[%s].%s" % (str(i),property_y))
        if property_y == "history_p_t":
            ax.plot(np.asarray(Data_list[i].history_time), np.array(data), linestyle='solid', color="blue",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
            ax.plot(np.asarray(Data_list[i].history_time), np.array((Data_list[i].d/ (Data_list[i].R - 1) + (Data_list[i].theta_t[::Data_list[i].compression_factor][2:] * Data_list[i].ar_1_coefficient)/ (Data_list[i].R - Data_list[i].ar_1_coefficient))), linestyle='dashed', color="red")
        else:
            ax.plot(np.asarray(Data_list[i].history_time), np.array(data), linestyle='solid', color="blue",  linewidth=1, marker = "o", markerfacecolor = 'black', markersize = '5')
        ax.set_xlabel("Steps")
        ax.set_ylabel("%s" % property_y)
        ax.set_title("%s = %s" % (property_title, property_list[i]))
  
    
    fig.tight_layout()
    plotName = fileName + "/Plots"
    f =plotName + "/plot_multi_time_series_%s_%s" % (property_varied,property_y)
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
        rational_var = np.var(Data_list[i].theta_t*Data_list[i].R/(Data_list[i].R - Data_list[i].ar_1_coefficient))
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

dpi_save = 600
red_blue_c = True

single_shot = 1
single_param_vary = 0

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
        fileName = "results/small-worldsingle_shot_10_27_46_05_10_2023"#"results/single_shot_steps_500_I_100_network_structure_small_world_degroot_aggregation_1"
        createFolder(fileName)
        Data = load_object(fileName + "/Data", "financial_market")
        base_params = load_object(fileName + "/Data", "base_params")
        print("base_params", base_params)
        print("mean price is: ", np.mean(Data.history_p_t), "mean variance is: ", np.var(Data.history_p_t), "autocorr is: ", np.corrcoef(Data.history_p_t[:-1],Data.history_p_t[1:])[0,1])
        print("mean_rational price is: ", np.mean((Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:] * Data.ar_1_coefficient)/ (Data.R - Data.ar_1_coefficient))),"mean_rational variance is: ", np.var((Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:] * Data.ar_1_coefficient)/ (Data.R - Data.ar_1_coefficient))), "mean_rational corr is: ", np.corrcoef(Data.theta_t[:-1],Data.theta_t[1:])[0,1])
        rational_prices =(Data.d/ (Data.R - 1) + (Data.theta_t[::Data.compression_factor][2:] * Data.ar_1_coefficient)/ (Data.R - Data.ar_1_coefficient))
        rational_returns = (rational_prices[1:] - rational_prices[:-1]) / rational_prices[:-1]
        print("kurtosis is:", kurtosis(rational_returns))
        #print(Data.history_time)

        #consumers
        #plot_history_c = plot_time_series_consumers(fileName,Data,"c bool",dpi_save,"history_c_bool",red_blue_c)
        #plot_history_profit = plot_time_series_consumers(fileName,Data,"Profit",dpi_save,"history_profit",red_blue_c)
        ##plot_history_lambda_t = plot_time_series_consumers(fileName,Data,"Network signal, $\lambda_{t,i}$",dpi_save,"history_lambda_t",red_blue_c)
        ##
        #plot_history_expectation_theta_mean = plot_time_series_consumers(fileName,Data,"Individual posterior mean",dpi_save,"history_expectation_theta_mean",red_blue_c)
        #plot_history_expectation_theta_variance = plot_time_series_consumers(fileName,Data,"Expectation variance, $E(\sigma_{\theta}^2)$",dpi_save,"history_expectation_theta_variance",red_blue_c)

        #consumer X list and weighting
        ##plot_history_demand = plot_time_series_consumer_triple(fileName,Data,"Theoretical whole demand, $X_k$",dpi_save,"history_theoretical_X_list", 3, ["$X_{\theta}$", "$X_{\zeta}$", "$X_{\lambda}$"],red_blue_c)
 
        #plot_history_weighting = plot_time_series_consumer_triple(fileName,Data,"Signal weighting, $\phi_k$",dpi_save,"history_weighting_vector", 3, ["$S_{ \\theta }$", "$S_{\zeta}$", "$S_{\lambda}$"],red_blue_c)

        #network
        plot_history_p_t = plot_time_series_market(fileName,Data,"Price, $p_t$",dpi_save,"history_p_t")  
        plot_histogram_returns = plot_histogram_returns(fileName,Data,"returns",dpi_save)
        plot_qq_plot = plot_qq_plot(fileName, Data, "qq_plot", dpi_save)
        #plot_history_informed_proportion = plot_time_series_market(fileName,Data,"Informed prop.",dpi_save,"history_informed_proportion")  
        #plot_history_d_t = plot_time_series_market(fileName,Data,"Dividend ,$d_t$",dpi_save,"history_d_t")
        ##plot_history_zeta_t = plot_time_series_market(fileName,Data,"$S_{\omega}$",dpi_save,"zeta_t")
        #plot_network_c = plot_network_shape(fileName, Data, base_params["network_structure"], "c bool","dogmatic_state",cmap, norm_zero_one, node_size,dpi_save)
        ##plot_history_pulsing = plot_time_series_market_pulsing(fileName,Data,"$In phase?$",dpi_save)
        #plot_degree_distribution = degree_distribution_single(fileName,Data,dpi_save)
        #plot_weighting_matrix_relations = plot_line_weighting_matrix(fileName,Data,dpi_save)
        #plot_final_wighting_m = plot_final_wighting_matrix(fileName, Data, dpi_save)
        #plot_node_influencers = plot_node_influence(fileName,Data,dpi_save)

        #network trasnspose
        #plot_history_X_it = plot_time_series_market_matrix_transpose(fileName,Data,"$X_{it}$",dpi_save,"history_X_t")
        

        #cumsum
        ##plot_history_c = plot_cumulative_consumers(fileName,Data,"c bool",dpi_save,"history_c_bool",red_blue_c)
        plot_history_profit = plot_cumulative_consumers(fileName,Data,"Cumulative profit",dpi_save,"history_cumulative_profit",red_blue_c)
        
        ##plot_history_lambda_t = plot_cumulative_consumers(fileName,Data,"Cumulative network signal, $\lambda_{t,i}$",dpi_save,"history_lambda_t",red_blue_c)
        #plot_history_expectation_theta_mean = plot_cumulative_consumers(fileName,Data,"Cumulative expectation mean, $E(\mu_{\theta})$",dpi_save,"history_expectation_theta_mean",red_blue_c)
        #plot_history_expectation_theta_variance = plot_cumulative_consumers(fileName,Data,"Cumulative expectation variance, $E(\sigma_{\theta}^2)$",dpi_save,"history_expectation_theta_variance",red_blue_c)

        #inital prior distributions
        #if base_params["heterogenous_priors"]:
        #    plot_inital_priors = plot_initial_priors_hist(fileName,Data,dpi_save)

        #Animation BROKE
        ##anim_c_bool = anim_value_network(fileName,Data,base_params["network_structure"], "c bool","history_c_bool", fps, round_dec,cmap, interval, norm_zero_one, node_size)
        #anim_weighting_m = anim_weighting_matrix(fileName,Data,cmap, interval, fps, round_dec)
        #anim_weighting_m = anim_weighting_matrix_combined(fileName,Data,cmap, interval, fps, round_dec, weighting_matrix_time_series)

    elif single_param_vary:
        fileName = "results/scale-freesingle_vary_set_seed_17_11_38_05_10_2023"
        Data_list = load_object(fileName + "/Data", "financial_market_list")
        property_varied =  load_object(fileName + "/Data", "property_varied")
        property_list = load_object(fileName + "/Data", "property_list")

        property_title = "K"
        if property_varied == "set_seed":
        
            #plot_price_different_seed(fileName,Data_list, transparency_level=0.3)
            plot_avg_price_different_seed(fileName,Data_list, transparency_level=0.3)
            plot_histogram_returns_different_seed(fileName,Data_list)
            plot_qq_plot_different_seed(fileName,Data_list)
        else:
            #plot_time_series_market_multi(fileName,Data_list,property_list, property_varied, property_title, "history_p_t")
            plot_autocorrelation_price_multi(fileName,Data_list,property_list, property_varied, property_title)
            plot_avg_price_multi(fileName,Data_list,property_list, property_varied, property_title)
            plot_variance_price_multi(fileName,Data_list,property_list, property_varied, property_title)
            plot_skew_price_multi(fileName,Data_list,property_list, property_varied, property_title)
            #plot_multi_time_series(fileName,Data_list,property_list, property_varied, property_title, "history_profit")
            #plot_multi_time_series(fileName,Data_list,property_list, property_varied, property_title, "history_expectation_theta_mean")

            #plot_history_weighting_multi_broadcast = plot_time_series_consumer_triple_multi(fileName,Data_list,"Signal weighting, $\phi_{\omega}$",dpi_save,"history_weighting_vector", 1, property_varied, property_list, titles)
            #plot_history_weighting_multi_network = plot_time_series_consumer_triple_multi(fileName,Data_list,"Signal weighting, $\phi_{\lambda}$",dpi_save,"history_weighting_vector", 2, property_varied, property_list, titles)
        
    plt.show()