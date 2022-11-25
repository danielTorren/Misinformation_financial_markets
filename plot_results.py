"""Plot results adn save them

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

###PLOT STUFF
node_size = 100
norm_zero_one = Normalize(vmin=0, vmax=1)
cmap = get_cmap("Blues")
fps = 5
interval = 50
layout = "circular"
round_dec = 2



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

    ax.set_xlabel(r"Steps")
    ax.set_ylabel(r"%s" % y_title)

    if property_y == "history_expectation_mean":
        ax.plot(Data.history_time, Data.theta_t, linestyle='dashed', color="black",  linewidth=2, alpha=0.5)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/timeseries_consumers_%s.eps" % (property_y)
    print("f",f)
    fig.savefig(f, dpi=dpi_save, format="eps")

def plot_time_series_consumer_triple(fileName,Data,y_title,dpi_save,property_y,num_signals, titles,red_blue_c):
    fig, axes = plt.subplots(nrows=1, ncols=num_signals, figsize=(10,6))

    for i, ax in enumerate(axes.flat):
        for v in range(len(Data.agent_list)):
            if (Data.agent_list[v].history_c_bool[0]):
                color = "blue"
            else:
                color = "red"
            data_ind = np.asarray(  eval("Data.agent_list[%s].%s" % (str(v), property_y))   )#get the list of data for that specific agent
            ax.plot(np.asarray(Data.history_time), data_ind[:, i], color = color)#plot the ith column in the weighting matrix which is [T x num_signals] where T is total steps
        ax.set_xlabel(r"Steps")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(titles[i])

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/plot_time_series_consumer_triple_%s.eps" % property_y
    fig.savefig(f, dpi=dpi_save, format="eps")
    
def plot_time_series_market(fileName,Data,y_title,dpi_save,property_y):

    fig, ax = plt.subplots()
    data = eval("Data.%s" % property_y)

    # bodge
    ax.plot(Data.history_time, data, linestyle='solid', color="blue",  linewidth=2)
    ax.set_xlabel(r"Steps")
    ax.set_ylabel(r"%s" % y_title)

    if property_y == "history_p_t":
        ax.plot(Data.history_time, (Data.d + Data.theta_t)/Data.R, linestyle='dashed',color="green" , linewidth=2)
        #ax.plot(Data.history_time, [(Data.d)/Data.R], "--")
        ax.axhline(y = (Data.d)/Data.R, linestyle='dashdot', color="red" , linewidth=2)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")

def plot_time_series_market_pulsing(fileName,Data,y_title,dpi_save):

    fig, ax = plt.subplots()
    data = Data.theta_t*Data.gamma_t/np.abs(Data.theta_t*Data.gamma_t)

    # bodge
    ax.scatter(Data.history_time, data, color="blue")
    ax.set_xlabel(r"Steps")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + "_timeseries_pulsing.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")

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

    ax.set_xlabel(r"Steps")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_plot_time_series_market_matrix_transpose.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")

def prod_pos(layout_type: str, network: nx.Graph) -> nx.Graph:

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

def anim_value_network(
    FILENAME: str,
    Data: list,
    layout: str,
    colour_bar_label:str,
    property_value: str,
    fps:int,
    round_dec:int,
    cmap,
    interval: int,
    norm_value,
    node_size,
):

    data_matrix = np.asarray([eval("Data.agent_list[%s].%s" % (v,property_value)) for v in range(Data.I)]).T

    def update(i, Data, data_matrix, ax, cmap, layout, title,round_dec,norm_value):

        ax.clear()

        #individual_value_list = [eval("Data.agent_list[%s].%s[%s]" % (v,property_value,i)) for v in range(Data.I)]
        #print("individual_value_list",individual_value_list)
        colour_adjust = norm_value(data_matrix[i])
        ani_step_colours = cmap(colour_adjust)

        G = nx.from_numpy_matrix(Data.adjacency_matrix)

        # get pos
        pos = prod_pos(layout, G)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos,
            node_size=node_size,
            edgecolors="black",
        )

        title.set_text(
            "Time= {}".format(round(Data.history_time[i], round_dec))
        )

    fig, ax = plt.subplots()



    title = plt.suptitle(t="", fontsize=20)

    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        location="right",
    )  #
    cbar_culture.set_label(colour_bar_label)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data.history_time)),
        fargs=(Data, data_matrix, ax, cmap, layout, title,round_dec,norm_value),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = (
        animateName
        + "/anim_value_network_%s.mp4" % property_value
    )
    # print("f", f)
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani

def plot_network_shape(    
    fileName: str,
    Data: list,
    layout: str,
    colour_bar_label:str,
    property_value: str,
    cmap,
    norm_value,
    node_size,
):  
    data_matrix = np.asarray([eval("Data.agent_list[%s].%s" % (v,property_value)) for v in range(Data.I)]).T

    fig, ax = plt.subplots()

    colour_adjust = norm_value(data_matrix[-1])
    ani_step_colours = cmap(colour_adjust)

    G = nx.from_numpy_matrix(Data.adjacency_matrix)

    # get pos
    pos = prod_pos(layout, G)

    nx.draw(
        G,
        node_color=ani_step_colours,
        ax=ax,
        pos=pos,
        node_size=node_size,
        edgecolors="black",
    )

    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        location="right",
    )  #
    cbar_culture.set_label(colour_bar_label)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_value + "_plot_network_shape.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


dpi_save = 1200
red_blue_c = True

if __name__ == "__main__":

    fileName = "results/single_shot_steps_3000_I_200_network_structure_small_world_degroot_aggregation_1"
    createFolder(fileName)
    Data = load_object(fileName + "/Data", "financial_market")

    #consumers
    #plot_history_c = plot_time_series_consumers(fileName,Data,"c bool",dpi_save,"history_c_bool",red_blue_c)
    plot_history_profit = plot_time_series_consumers(fileName,Data,"Profit",dpi_save,"history_profit",red_blue_c)
    plot_history_lambda_t = plot_time_series_consumers(fileName,Data,r"Network signal, $\lambda_{t,i}$",dpi_save,"history_lambda_t",red_blue_c)
    plot_history_expectation_theta_mean = plot_time_series_consumers(fileName,Data,"Expectation mean, $E(\mu_{\theta})$",dpi_save,"history_expectation_theta_mean",red_blue_c)
    plot_history_expectation_theta_variance = plot_time_series_consumers(fileName,Data,"Expectation variance, $E(\sigma_{\theta}^2)$",dpi_save,"history_expectation_theta_variance",red_blue_c)

    #consumer X list and weighting
    plot_history_demand = plot_time_series_consumer_triple(fileName,Data,"Theoretical whole demand, $X_k$",dpi_save,"history_X_list", 3, [r"$X_{\theta}$", r"$X_{\zeta}$", r"$X_{\lambda}$"],red_blue_c)
    plot_history_weighting = plot_time_series_consumer_triple(fileName,Data,"Signal weighting, $\phi_k$",dpi_save,"history_weighting_vector", 3, [r"$S_{\theta}$", r"$S_{\zeta}$", r"$S_{\lambda}$"],red_blue_c)

    #network
    plot_history_p_t = plot_time_series_market(fileName,Data,"Price, $p_t$",dpi_save,"history_p_t")    
    #plot_history_d_t = plot_time_series_market(fileName,Data,"Dividend ,$d_t$",dpi_save,"history_d_t")
    #plot_history_zeta_t = plot_time_series_market(fileName,Data,"$S_{\omega}$",dpi_save,"zeta_t")
    #plot_network_c = plot_network_shape(fileName, Data, layout, "c bool","history_c_bool",cmap, norm_zero_one, node_size)
    #plot_history_pulsing = plot_time_series_market_pulsing(fileName,Data,"$In phase?$",dpi_save)

    #network trasnspose
    plot_history_X_it = plot_time_series_market_matrix_transpose(fileName,Data,"$X_{it}$",dpi_save,"history_X_it")

    #Animation BROKE
    #anim_c_bool = anim_value_network(fileName,Data,layout, "c bool","history_c_bool", fps, round_dec,cmap, interval, norm_zero_one, node_size)

    plt.show()