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
from utility import (
    createFolder, 
    load_object, 
    save_object,
)

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })


def plot_time_series_consumers(fileName,Data,y_title,dpi_save,property_y,red_blue_c):
    fig, ax = plt.subplots()

    if red_blue_c:
        for v in range(len(Data.agent_list)):
            if (Data.agent_list[v].c_bool):
                color = "blue"
            else:
                color = "red"
            data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y)))
            ax.plot(np.asarray(Data.history_time), data_ind, color = color)
    else:
        for v in range(len(Data.agent_list)):
            data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y)))
            ax.plot(np.asarray(Data.history_time), data_ind)

    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)

    if property_y == "history_expectation_mean":
        #Data.theta_t[::Data.compression_factor]
        ax.plot(Data.history_time, Data.d + Data.theta_t[::Data.compression_factor], linestyle='dashed', color="black",  linewidth=2, alpha=0.5) 
    elif property_y == "history_profit":
        ax.plot(Data.history_time, (Data.theta_t - Data.gamma_t)*(Data.theta_t + Data.epsilon_t) - Data.c_info, linestyle='dashdot', color="green" , linewidth=2)
    elif property_y == "history_expectation_theta_mean":
        ax.axhline(y = 0.0, linestyle='dashdot', color="grey" , linewidth=2, alpha = 0.9)
        #ax.set_ylim([Data.R*Data.W_0 - 0.2, Data.R*Data.W_0 + 0.2])
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
            if (Data.agent_list[v].c_bool):
                color = "blue"
            else:
                color = "red"
            data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y)))
            ax.plot(np.asarray(Data.history_time), np.cumsum(data_ind), color = color)
    else:
        for v in range(len(Data.agent_list)):
            data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y)))
            ax.plot(np.asarray(Data.history_time), np.cumsum(data_ind))

    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)

    if property_y == "history_expectation_mean":
        ax.plot(Data.history_time, np.cumsum(Data.theta_t[::Data.compression_factor]), linestyle='dashed', color="black",  linewidth=2, alpha=0.5)
    elif property_y == "history_profit":
        ax.plot(Data.history_time, np.cumsum([-Data.c_info]*len(Data.history_time)), linestyle='dashed', color="black",  linewidth=2, alpha=0.5)

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

def plot_time_series_market(fileName,Data,y_title,dpi_save,property_y):

    fig, ax = plt.subplots()
    data = eval("Data.%s" % property_y)

    # bodge
    ax.plot(Data.history_time, data, linestyle='solid', color="blue",  linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)
    #ax.set_ylim([(Data.d)/Data.R - 0.5*((Data.d)/Data.R), (Data.d)/Data.R + 0.5*((Data.d)/Data.R)])
    if property_y == "history_p_t":
        #ax.plot(Data.history_time, (Data.d + Data.theta_t[::Data.compression_factor])/Data.R, linestyle='dashed',color="green" , linewidth=2)
        #ax.hline([(Data.d)/Data.R], linestyle='dashed', color="black",  linewidth=2, alpha=0.5)
        #ax.plot(Data.history_time, Data.d + np.asarray(Data.theta_t[::Data.compression_factor])/Data.R, linestyle='dashed', color="black",  linewidth=2, alpha=0.5)
        #print(np.sum(data-(Data.d + Data.theta_t[::Data.compression_factor])/Data.R))
        ax.axhline(y = (Data.d)/Data.R, linestyle='dashdot', color="red" , linewidth=2)
        #ax.axhline(y = (Data.d + Data.theta_mean)/Data.R, linestyle='dashdot', color="green" , linewidth=2)
        #ax.axhline(y = (Data.d + Data.broadcast_quality*Data.theta_mean +  (1 - Data.broadcast_quality)*Data.gamma_mean)/Data.R, linestyle='dashdot', color="purple" , linewidth=2)
        labels = labels = ["Price", "Uninformed price"]
        fig.legend(labels = labels,loc = 'lower right', borderaxespad=10)
    #elif property_y == "history_informed_proportion":
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
    data = np.asarray(eval("Data.%s" % property_y)).T

    #print("Data.agent_list[i].c_bool[0]", Data.agent_list[0].history_c_bool[0], ~(Data.agent_list[0].history_c_bool[0]))
    data_c = [data[i] for i in range(Data.I) if (Data.agent_list[i].c_bool)]
    data_no_c = [data[i] for i in range(Data.I) if not (Data.agent_list[i].c_bool)]
    # bodge
    for v in range(len(data_c)):
        ax.plot(Data.history_time, data_c[v], color="blue")
    for v in range(len(data_no_c)):
        ax.plot(Data.history_time, data_no_c[v], color="red")

    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_plot_time_series_market_matrix_transpose"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def prod_pos(network_structure: str, network: nx.Graph) -> nx.Graph:

    if network_structure == "small_world":
        layout_type = "circular"
    elif network_structure == "barabasi_albert_graph" or "scale_free_directed":
        layout_type ="spring"

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

    colour_adjust = norm_value(data_matrix[-1])
    ani_step_colours = cmap(colour_adjust)

    G = nx.from_numpy_matrix(Data.adjacency_matrix)

    # get pos
    pos = prod_pos(network_structure, G)

    node_colours = ["blue" if x.c_bool else "red" for x in Data.agent_list ]

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
    
    """
    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        location="right"
    )  #
    cbar_culture.set_label(colour_bar_label)
    """

    #print("ani_step_colours",ani_step_colours)

    values = ["red", "blue"]
    c_list = ["Not paying the cost","Paying the cost"]
    for v in range(len(values)):
        plt.scatter([],[], c=values[v], label="%s" % (c_list[v]))

        
    ax.legend()


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
            #if (Data_list[i].agent_list[v].history_c_bool[0]):
            #    color = "blue"
            #else:
            #    color = "red"
            data_ind = np.asarray(eval("Data_list[%s].agent_list[%s].%s" % (str(i),str(v), property_y)))#get the list of data for that specific agent
            #print(data_ind[:, i])
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
        b = np.asarray(Data.history_weighting_matrix[t])
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
        b = np.asarray(Data.history_weighting_matrix[t])
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

dpi_save = 300
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
        fileName = "results/single_shot_19_29_49__22_12_2022"#"results/single_shot_steps_500_I_100_network_structure_small_world_degroot_aggregation_1"
        createFolder(fileName)
        Data = load_object(fileName + "/Data", "financial_market")
        base_params = load_object(fileName + "/Data", "base_params")
        print("base_params", base_params)

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
        #plot_history_p_t = plot_time_series_market(fileName,Data,"Price, $p_t$",dpi_save,"history_p_t")  
        #plot_history_informed_proportion = plot_time_series_market(fileName,Data,"Informed prop.",dpi_save,"history_informed_proportion")  
        #plot_history_d_t = plot_time_series_market(fileName,Data,"Dividend ,$d_t$",dpi_save,"history_d_t")
        ##plot_history_zeta_t = plot_time_series_market(fileName,Data,"$S_{\omega}$",dpi_save,"zeta_t")
        plot_network_c = plot_network_shape(fileName, Data, base_params["network_structure"], "c bool","history_c_bool",cmap, norm_zero_one, node_size,dpi_save)
        ##plot_history_pulsing = plot_time_series_market_pulsing(fileName,Data,"$In phase?$",dpi_save)
        plot_degree_distribution = degree_distribution_single(fileName,Data,dpi_save)
        #plot_weighting_matrix_relations = plot_line_weighting_matrix(fileName,Data,dpi_save)
        
        #plot_node_influencers = plot_node_influence(fileName,Data,dpi_save)

        #network trasnspose
        ##plot_history_X_it = plot_time_series_market_matrix_transpose(fileName,Data,"$X_{it}$",dpi_save,"history_X_it")
        

        #cumsum
        ##plot_history_c = plot_cumulative_consumers(fileName,Data,"c bool",dpi_save,"history_c_bool",red_blue_c)
        #plot_history_profit = plot_cumulative_consumers(fileName,Data,"Cumulative profit",dpi_save,"history_profit",red_blue_c)
        ##plot_history_lambda_t = plot_cumulative_consumers(fileName,Data,"Cumulative network signal, $\lambda_{t,i}$",dpi_save,"history_lambda_t",red_blue_c)
        ##plot_history_expectation_theta_mean = plot_cumulative_consumers(fileName,Data,"Cumulative expectation mean, $E(\mu_{\theta})$",dpi_save,"history_expectation_theta_mean",red_blue_c)
        ##plot_history_expectation_theta_variance = plot_cumulative_consumers(fileName,Data,"Cumulative expectation variance, $E(\sigma_{\theta}^2)$",dpi_save,"history_expectation_theta_variance",red_blue_c)

        #inital prior distributions
        if base_params["heterogenous_priors"]:
            plot_inital_priors = plot_initial_priors_hist(fileName,Data,dpi_save)

        #Animation BROKE
        ##anim_c_bool = anim_value_network(fileName,Data,base_params["network_structure"], "c bool","history_c_bool", fps, round_dec,cmap, interval, norm_zero_one, node_size)
        #anim_weighting_m = anim_weighting_matrix(fileName,Data,cmap, interval, fps, round_dec)
        #anim_weighting_m = anim_weighting_matrix_combined(fileName,Data,cmap, interval, fps, round_dec, weighting_matrix_time_series)

    elif single_param_vary:
        fileName = "results/single_shot_18_57_21__10_12_2022"
        createFolder(fileName)

        Data_list = load_object(fileName + "/Data", "financial_market_list")
        property_varied =  load_object(fileName + "/Data", "property_varied")
        property_list = load_object(fileName + "/Data", "property_list")

        property_title = "$T_{h}$"
        titles = ["%s = %s" % (property_title,i*Data_list[0].steps) for i in property_list]
        #print("titles", titles)


        plot_history_weighting_multi_broadcast = plot_time_series_consumer_triple_multi(fileName,Data_list,"Signal weighting, $\phi_{\omega}$",dpi_save,"history_weighting_vector", 1, property_varied, property_list, titles)
        plot_history_weighting_multi_network = plot_time_series_consumer_triple_multi(fileName,Data_list,"Signal weighting, $\phi_{\lambda}$",dpi_save,"history_weighting_vector", 2, property_varied, property_list, titles)
        
    plt.show()