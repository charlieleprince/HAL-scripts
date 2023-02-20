# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from datetime import date
import pandas as pd
from datetime import date
import json
from pathlib import Path
import os
logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "3ter. Combine and export" # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


def main(self):
    """
    the script also have to define a `main` function. When playing a script,
    HAL runs `main` passes one (and only one) argument "self" that is the
    HAL mainwindow object (granting access to all the gui attributes and methods)

    Note : ce script reprend totalement le script MCP_combine mais sauve les données
    dans un fichier CSV unique plutôt que les plotter.
    """

    # get metadata from current selection
    metadata = getSelectionMetaDataFromCache(self, update_cache=True)

    # -- get selected data
    selection = self.runList.selectedItems()
    if not selection:
        return
    # -- init object data
    # get object data type
    data_class = self.dataTypeComboBox.currentData()
    data = data_class()
    # get path
    item = selection[0]
    data.path = item.data(QtCore.Qt.UserRole)
    X, Y, T = data.getrawdata()

    for k in range(len(selection) - 1):
        item = selection[k + 1]
        data.path = item.data(QtCore.Qt.UserRole)
        if not data.path.suffix == ".atoms":
            return
        # get data
        Xa, Ya, Ta = data.getrawdata()
        X = np.concatenate([X, Xa])
        Y = np.concatenate([Y, Ya])
        T = np.concatenate([T, Ta])


    ## Now we create the dataframe
    data = np.array([X, Y, T])
    data = np.transpose(data)
    data = pd.DataFrame(data, columns =['X','Y', 'T'])


    ## J'arrange ensuite les données pour enlever ce qui n'est pas dans la ROI.
    ##1. Je récupère la ROI par défaut (c'est celle-ci qu'on utilise)
    root = Path().home()
    default_roi_dir = root / ".HAL"
    default_roi_file_name = default_roi_dir / "default_mcp_roi.json"
    with open(default_roi_file_name, encoding="utf8") as f:
        defaultroi = json.load(f)
    ROI0 = {}
    Xmin = defaultroi["ROI 0"]["Xmin"]
    Xmax = defaultroi["ROI 0"]["Xmax"]
    Ymin = defaultroi["ROI 0"]["Ymin"]
    Ymax = defaultroi["ROI 0"]["Ymax"]
    Tmin = defaultroi["ROI 0"]["Tmin"]
    Tmax = defaultroi["ROI 0"]["Tmax"]

    # 2. on sélectionne les données en utilisant la puissance de pandas
    selection = data[ (data["X"] > Xmin) & (data["X"]<Xmax)  &   (data["Y"] > Ymin) & (data["Y"]<Ymax)   &     (data["T"] > Tmin) & (data["T"]<Tmax)]


    # get the directory
    today = date.today()
    directory_windows = os.path.join("C:\\Users","Helium 1","LabWiki","Journal",
    today.strftime("%Y"),today.strftime("%m"),today.strftime("%d"))

    directory_linux = os.path.join("~/", "LabWiki","Journal",today.strftime("%Y"),
    today.strftime("%m"),today.strftime("%d"))


    if os.path.exists(directory_windows):
        file_name =  os.path.join(directory_windows, "file.csv")
        selection.to_csv(file_name, encoding = 'utf-8', sep=';', index = False)
        print("###################################")
        print("CSV saved in {}".format(file_name))
        print("###################################")
    elif  os.path.exists(directory_linux):
        file_name =  os.path.join(directory_linux, "file.csv")
        selection.to_csv(file_name, encoding = 'utf-8', sep=';', index = False)
        print("###################################")
        print("CSV saved in {}".format(file_name))
        print("###################################")
    else:
        file_name =  "file.csv"
        selection.to_csv(file_name, encoding = 'utf-8', sep=';', index = False)
        print("###################################")
        print("CSV saved in {}".format(os.getcwd()))
        print("###################################")
