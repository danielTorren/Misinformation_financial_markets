"""Contains functions that are not crucial to the simulation itself and are shared amongst files.
A module that aides in preparing folders, saving, loading and generating data for plots.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import pickle
import os
import datetime

# modules
def createFolder(fileName: str) -> str:
    """
    Check if folders exist and if they dont create results folder in which place Data, Plots, Animations
    and Prints folders

    Parameters
    ----------
    fileName:
        name of file where results may be found

    Returns
    -------
    None
    """

    # print(fileName)
    # check for resutls folder
    if str(os.path.exists("results")) == "False":
        os.mkdir("results")

    # check for runName folder
    if str(os.path.exists(fileName)) == "False":
        os.mkdir(fileName)

    # make data folder:#
    dataName = fileName + "/Data"
    if str(os.path.exists(dataName)) == "False":
        os.mkdir(dataName)
    # make plots folder:
    plotsName = fileName + "/Plots"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make animation folder:
    plotsName = fileName + "/Animations"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make prints folder:
    plotsName = fileName + "/Prints"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)


def save_object(data, fileName, objectName):
    """save single object as a pickle object

    Parameters
    ----------
    data: object,
        object to be saved
    fileName: str
        where to save it e.g in the results folder in data or plots folder
    objectName: str
        what name to give the saved object

    Returns
    -------
    None
    """
    with open(fileName + "/" + objectName + ".pkl", "wb") as f:
        pickle.dump(data, f)


def load_object(fileName, objectName) -> dict:
    """load single pickle file

    Parameters
    ----------
    fileName: str
        where to load it from e.g in the results folder in data folder
    objectName: str
        what name of the object to load is

    Returns
    -------
    data: object
        the pickle file loaded
    """
    with open(fileName + "/" + objectName + ".pkl", "rb") as f:
        data = pickle.load(f)
    return data

def produce_name(root:str, parameters: dict, parameters_name_list: list) -> str:
    """produce a file name from a subset list of parameters and values  to create a unique identifier for each simulation run

    Parameters
    ----------
    params_dict: dict[dict],
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
        See generate_data function for an example
    parameters_name_list: list
        list of parameters to be used in the filename

    Returns
    -------
    fileName: str
        name of file where results may be found composed of value from the different assigned parameters.
    """

    fileName = "results/" + root
    for key, value in parameters.items():
        if key in parameters_name_list:
            fileName = fileName + "_" + str(key) + "_" + str(value)
    return fileName

def produce_name_datetime(root):
    fileName = "results/" + root +  "_" + datetime.datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    return fileName