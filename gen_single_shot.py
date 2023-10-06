"""Generate and save model data

Author: Tommaso Di Francesco and Daniel Torren Peraire  Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
# imports
import json
import pyperclip
from utility import (
    createFolder, 
    save_object,
    produce_name_datetime,
)
from generate_data import (
    generate_data_single
    )

if __name__ == "__main__":
    #load in exogenous parameters
    f = open("constants/base_params.json")
    params = json.load(f)

    rootName = params["network_type"] +"single_shot"
    fileName = produce_name_datetime(rootName)

    print("FILENAME:", fileName)
    #copy the filename variable to the clipboard
    pyperclip.copy(fileName)
    print_simu = 1
    Data = generate_data_single(params,print_simu)  # run the simulation

    createFolder(fileName)#put after run so that you dont create folder unless you got to the end of the simulation

    save_object(Data, fileName + "/Data", "financial_market")
    save_object(params, fileName + "/Data", "base_params")

    