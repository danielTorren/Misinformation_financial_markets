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
from utility import (
    createFolder, 
    load_object, 
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
            if (Data.agent_list[v].history_c_bool[0]):
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
        ax.plot(Data.history_time, Data.theta_t[::Data.compression_factor], linestyle='dashed', color="black",  linewidth=2, alpha=0.5)
    #elif property_y == "history_profit":
        #ax.axhline(y= Data.R*Data.W_0, linestyle = 'dashed', color = 'black',linewidth=2, alpha=0.5)

        #ax.set_ylim([Data.R*Data.W_0 - 0.2, Data.R*Data.W_0 + 0.2])
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/timeseries_consumers_%s" % (property_y)
    #print("f",f)
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_cumulative_consumers(fileName,Data,y_title,dpi_save,property_y,red_blue_c):

    fig, ax = plt.subplots()

    if red_blue_c:
        for v in range(len(Data.agent_list)):
            if (Data.agent_list[v].history_c_bool[0]):
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
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_time_series_consumer_triple(fileName,Data,y_title,dpi_save,property_y,num_signals, titles,red_blue_c):
    fig, axes = plt.subplots(nrows=1, ncols=num_signals, figsize=(10,6))
    #print(property_y)

    for i, ax in enumerate(axes.flat):
        for v in range(len(Data.agent_list)):
            if (Data.agent_list[v].history_c_bool[0]):
                color = "blue"
            else:
                color = "red"
            data_ind = np.asarray(  eval("Data.agent_list[%s].%s" % (str(v), property_y))   )#get the list of data for that specific agent
            #print(data_ind[:, i])
            ax.plot(np.asarray(Data.history_time), data_ind[:, i], color = color)#plot the ith column in the weighting matrix which is [T x num_signals] where T is total steps
        ax.set_xlabel("Steps")
        #ax.set_ylim([0,1])
        ax.set_ylabel("%s" % y_title)
        ax.set_title(titles[i])
        #ax.set_ylim = ([0,1])

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/plot_time_series_consumer_triple_%s" % property_y
    
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")

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
    # elif property_y == "history_informed_proportion":
    #     ax.plot(Data.history_time, Data.epsilon_t, linestyle='dashed', color="black",  linewidth=2, alpha=0.5)    
    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_timeseries"
    # fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
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
    # fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_time_series_market_matrix_transpose(fileName,Data,y_title,dpi_save,property_y):

    fig, ax = plt.subplots()
    data = np.asarray(eval("Data.%s" % property_y)).T

    #print("Data.agent_list[i].c_bool[0]", Data.agent_list[0].history_c_bool[0], ~(Data.agent_list[0].history_c_bool[0]))
    data_c = [data[i] for i in range(Data.I) if (Data.agent_list[i].history_c_bool[0])]
    data_no_c = [data[i] for i in range(Data.I) if not (Data.agent_list[i].history_c_bool[0])]
    # bodge
    for v in range(len(data_c)):
        ax.plot(Data.history_time, data_c[v], color="blue")
    for v in range(len(data_no_c)):
        ax.plot(Data.history_time, data_no_c[v], color="red")

    ax.set_xlabel("Steps")
    ax.set_ylabel("%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_plot_time_series_market_matrix_transpose"
    # fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
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
    c_list = ["Uninformed","Informed"]
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
            if (Data_list[i].agent_list[v].history_c_bool[0]):
                color = "blue"
            else:
                color = "red"
            data_ind = np.asarray(eval("Data_list[%s].agent_list[%s].%s" % (str(i),str(v), property_y)))#get the list of data for that specific agent
            #print(data_ind[:, i])
            ax.plot(np.asarray(Data_list[i].history_time), data_ind[:, signal], color = color)#plot the ith column in the weighting matrix which is [T x num_signals] where T is total steps
        ax.set_xlabel("Steps")
        #ax.set_ylim([0,1])
        
        ax.set_title(titles[i])
        #ax.set_ylim = ([0,1])
    fig.supylabel("%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/plot_time_series_consumer_triple_multi_%s_%s" % (property_varied,signal)
    
    fig.savefig(f + ".png", dpi=dpi_save, format="png")
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")

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
    #fig.savefig(f + ".png", dpi=dpi_save, format="png")


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
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
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
    # fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
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
    print_simu = 1  # Whether of not to print how long the single shot simulation took
    if print_simu:
        start_time = time.time()

    if single_shot:
        fileName = "results/single_shot_21_56_21__11_12_2022"#"results/single_shot_steps_500_I_100_network_structure_small_world_degroot_aggregation_1"
        createFolder(fileName)
        Data = load_object(fileName + "/Data", "financial_market")
        base_params = load_object(fileName + "/Data", "base_params")

        #print(Data.history_time)

        #consumers
        #plot_history_c = plot_time_series_consumers(fileName,Data,"c bool",dpi_save,"history_c_bool",red_blue_c)
        plot_history_profit = plot_time_series_consumers(fileName,Data,"Profit",dpi_save,"history_profit",red_blue_c)
        #plot_history_lambda_t = plot_time_series_consumers(fileName,Data,"Network signal, $\lambda_{t,i}$",dpi_save,"history_lambda_t",red_blue_c)
        ##
        plot_history_expectation_theta_mean = plot_time_series_consumers(fileName,Data,"Expectation mean, $E(\mu_{\theta})$",dpi_save,"history_expectation_theta_mean",red_blue_c)
        plot_history_expectation_theta_variance = plot_time_series_consumers(fileName,Data,"Expectation variance, $E(\sigma_{\theta}^2)$",dpi_save,"history_expectation_theta_variance",red_blue_c)

        #consumer X list and weighting
        ##plot_history_demand = plot_time_series_consumer_triple(fileName,Data,"Theoretical whole demand, $X_k$",dpi_save,"history_theoretical_X_list", 3, ["$X_{\theta}$", "$X_{\zeta}$", "$X_{\lambda}$"],red_blue_c)
        if not base_params["accuracy_weighting"]:
            plot_history_theoretical_profit = plot_time_series_consumer_triple(fileName,Data,"Theoretical profits, $\pi_k$",dpi_save,"history_theoretical_profit_list", 3, ["$\pi_{\theta}$", "$\pi_{\zeta}$", "$\pi_{\lambda}$"],red_blue_c)
        plot_history_weighting = plot_time_series_consumer_triple(fileName,Data,"Signal weighting, $\phi_k$",dpi_save,"history_weighting_vector", 3, ["$S_{\theta}$", "$S_{\zeta}$", "$S_{\lambda}$"],red_blue_c)

        #network
        plot_history_p_t = plot_time_series_market(fileName,Data,"Price, $p_t$",dpi_save,"history_p_t")  
        plot_history_informed_proportion = plot_time_series_market(fileName,Data,"Informed prop.",dpi_save,"history_informed_proportion")  
        #plot_history_d_t = plot_time_series_market(fileName,Data,"Dividend ,$d_t$",dpi_save,"history_d_t")
        ##plot_history_zeta_t = plot_time_series_market(fileName,Data,"$S_{\omega}$",dpi_save,"zeta_t")
        plot_network_c = plot_network_shape(fileName, Data, base_params["network_structure"], "c bool","history_c_bool",cmap, norm_zero_one, node_size,dpi_save)
        ##plot_history_pulsing = plot_time_series_market_pulsing(fileName,Data,"$In phase?$",dpi_save)
        #plot_degree_distribution = degree_distribution_single(fileName,Data,dpi_save)
        #plot_weighting_matrix_relations = plot_line_weighting_matrix(fileName,Data,dpi_save)

        #network trasnspose
        ##plot_history_X_it = plot_time_series_market_matrix_transpose(fileName,Data,"$X_{it}$",dpi_save,"history_X_it")
        

        #cumsum
        ##plot_history_c = plot_cumulative_consumers(fileName,Data,"c bool",dpi_save,"history_c_bool",red_blue_c)
        plot_history_profit = plot_cumulative_consumers(fileName,Data,"Cumulative profit",dpi_save,"history_profit",red_blue_c)
        ##plot_history_lambda_t = plot_cumulative_consumers(fileName,Data,"Cumulative network signal, $\lambda_{t,i}$",dpi_save,"history_lambda_t",red_blue_c)
        ##plot_history_expectation_theta_mean = plot_cumulative_consumers(fileName,Data,"Cumulative expectation mean, $E(\mu_{\theta})$",dpi_save,"history_expectation_theta_mean",red_blue_c)
        ##plot_history_expectation_theta_variance = plot_cumulative_consumers(fileName,Data,"Cumulative expectation variance, $E(\sigma_{\theta}^2)$",dpi_save,"history_expectation_theta_variance",red_blue_c)

        #inital prior distributions
        if base_params["heterogenous_priors"]:
            plot_inital_priors = plot_initial_priors_hist(fileName,Data,dpi_save)

        #Animation BROKE
        ##anim_c_bool = anim_value_network(fileName,Data,base_params["network_structure"], "c bool","history_c_bool", fps, round_dec,cmap, interval, norm_zero_one, node_size)
        #anim_weighting_m = anim_weighting_matrix(fileName,Data,cmap, interval, fps, round_dec)

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


    if print_simu:
        print(
            "PLOT time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    plt.show()