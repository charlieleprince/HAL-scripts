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

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "4. Plot time histograms"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


def get_histo(savelist, T, cycle):
    bin_heights, bin_borders, _ = plt.hist(
        T, bins=np.linspace(np.min(T), np.max(T), 300)
    )
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    savelist.append([bin_centers, bin_heights, cycle])
    plt.close()
    return savelist


def main(self):
    """
    the script also have to define a `main` function. When playing a script,
    HAL runs `main` passes one (and only one) argument "self" that is the
    HAL mainwindow object (granting access to all the gui attributes and methods)
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
    nb_of_subplots = len(selection)

    # fig, axs = plt.subplots(nb_of_subplots, 1)
    savelist = []
    for k in range(len(selection)):
        item = selection[k]
        data.path = item.data(QtCore.Qt.UserRole)
        cycle = int(str(data.path.stem).split("_")[1])
        if not data.path.suffix == ".atoms":
            return
        # get data
        X, Y, T = data.getrawdata()
        savelist = get_histo(savelist, T, cycle)

    plt.figure()
    for k in range(len(selection)):
        plt.plot(savelist[k][0], savelist[k][1], label=str(savelist[k][2]))
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of events")
    plt.grid(True)
    plt.legend()
    plt.show()
    # plt.close()
