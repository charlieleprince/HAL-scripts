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
NAME = "3. Combine"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


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

    fig3 = plt.figure()
    plt.hist2d(X, Y, bins=np.linspace(-40, 40, 2 * 81), cmap=plt.cm.jet)
    plt.colorbar()
    fig3.show()

    fig2 = plt.figure()
    plt.hist(T, bins=np.linspace(0, np.max(T), 300))
    plt.xlabel("time (ms)")
    plt.ylabel("number of events")
    plt.grid(True)
    fig2.show()
