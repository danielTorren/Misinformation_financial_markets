"""Plot results adn save them

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
import numpy as np
from utility import (
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

def plot_time_series_consumers(fileName,Data,y_title,dpi_save,property_y):
    fig, ax = plt.subplots()

    for v in range(len(Data.agent_list)):
        data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property_y)))
        ax.plot(np.asarray(Data.history_time), data_ind)
        ax.set_xlabel(r"Steps")
        ax.set_ylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_time_series_consumers_%s.eps" % (property_y)
    fig.savefig(f, dpi=dpi_save, format="eps")

def plot_time_series_consumer_triple(fileName,Data,y_title,dpi_save,property_y,num_signals, titles):
    fig, axes = plt.subplots(nrows=1, ncols=num_signals)

    for i, ax in enumerate(axes.flat):
        for v in range(len(Data.agent_list)):
            #print("evalll",Data.agent_list[v].history_weighting_vector, )
            #print("shapelll",np.asarray(Data.agent_list[v].history_weighting_vector).shape())
            data_ind = np.asarray(  eval("Data.agent_list[%s].%s" % (str(v), property_y))   )#get the list of data for that specific agent
            #print("dataind",data_ind)
            #print("secltc",data_ind[:, i])
            ax.plot(np.asarray(Data.history_time), data_ind[:, i])#plot the ith column in the weighting matrix which is [T x num_signals] where T is total steps
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
    ax.plot(Data.history_time, data)
    ax.set_xlabel(r"Steps")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property_y + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")

dpi_save = 1200

if __name__ == "__main__":

    fileName = "results/test"

    Data = load_object(fileName + "/Data", "financial_market")

    ####produce plots
    #consumers
    plot_history_profit = plot_time_series_consumers(fileName,Data,"profit",dpi_save,"history_profit")
    plot_history_S_rho = plot_time_series_consumers(fileName,Data,"S_rho",dpi_save,"history_S_rho")
    plot_history_expectation_mean = plot_time_series_consumers(fileName,Data,"expectation_mean",dpi_save,"history_expectation_mean")
    plot_history_expectation_variance = plot_time_series_consumers(fileName,Data,"expectation_variance",dpi_save,"history_expectation_variance")

    #consumer weighting
    plot_history_weighting = plot_time_series_consumer_triple(fileName,Data,"Signal weighting",dpi_save,"history_weighting_vector", 3, [r"$S_{\tau}$", r"$S_{\omega}$", r"$S_{\rho}$"])

    #network
    # plot1 plot_history_p_t = plot_time_series_market(fileName,Data,"p_t",dpi_save,"history_p_t")
    plot_history_d_t = plot_time_series_market(fileName,Data,"d_t",dpi_save,"history_d_t")

    plt.show()