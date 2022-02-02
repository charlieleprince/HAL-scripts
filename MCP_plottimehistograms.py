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

    fig, axs = plt.subplots(nb_of_subplots, 1)

    for k in range(len(selection)):
        item = selection[k]
        data.path = item.data(QtCore.Qt.UserRole)
        cycle = int(str(data.path.stem).split("_")[1])
        if not data.path.suffix == ".atoms":
            return
        # get data
        X, Y, T = data.getrawdata()
        axs[k].hist(T, bins=np.linspace(0, np.max(T), 300), color="tab:blue")
        axs[k].yaxis.set_label_position("right")
        axs[k].set_ylabel(cycle)

    # fig.tight_layout()
    fig.supxlabel("Time (ms)")
    fig.supylabel("Number of events")
    plt.show()
