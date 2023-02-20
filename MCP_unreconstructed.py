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
NAME = "1bis. Plot unreconstructed data"  # display name, used in menubar and command palette
CATEGORY = "MCP - single file"  # category (note that CATEGORY="" is a valid choice)


def plot_unreconstructed_data(T_raw):

    bin_heights_raw, bin_borders_raw, _ = plt.hist(
        T_raw, bins=np.linspace(np.min(T_raw), np.max(T_raw), 300)
    )
    bin_centers_raw = bin_borders_raw[:-1] + np.diff(bin_borders_raw) / 2
    return (bin_centers_raw, bin_heights_raw)


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
    item = selection[0]
    data.path = item.data(QtCore.Qt.UserRole)
    X, Y, T = data.getrawdata()
    T_x1, T_x2, T_y1, T_y2 = data.getunreconstructeddata()
    #T_raw = data.getdatafromsingleline()
    (bin_centers_raw_x1, bin_heights_raw_x1) = plot_unreconstructed_data(T_x1)
    (bin_centers_raw_x2, bin_heights_raw_x2) = plot_unreconstructed_data(T_x2)
    (bin_centers_raw_y1, bin_heights_raw_y1) = plot_unreconstructed_data(T_y1)
    (bin_centers_raw_y2, bin_heights_raw_y2) = plot_unreconstructed_data(T_y2)
    (bin_centers, bin_heights) = plot_unreconstructed_data(T)

    fig2 = plt.figure()
    #plt.hist(T_raw, bins=np.linspace(0, np.max(T_raw), 300), color="black")
    #plt.hist(T, bins=np.linspace(0, np.max(T), 300))
    plt.plot(bin_centers_raw_x1, bin_heights_raw_x1,label='X1')
    plt.plot(bin_centers_raw_x2, bin_heights_raw_x2,label='X2')
    plt.plot(bin_centers_raw_y1, bin_heights_raw_y1,label='Y1')
    plt.plot(bin_centers_raw_y2, bin_heights_raw_y2,label='Y2')
    plt.plot(bin_centers, bin_heights, linestyle="dotted", color="black",label='reconstructed')
    plt.xlabel("time (ms)")
    plt.ylabel("number of events")
    plt.legend()
    plt.grid(True)
    fig2.show()
