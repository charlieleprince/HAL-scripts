# -*- coding: utf-8 -*-
"""
Author   : alex
Created  : 2021-06-24 16:47:55

Comments : copy this script to the user script folder (~/.HAL/user_scripts)
"""

# the script is a standard python file
# you can import global python modules
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt

# and also local modules from HAL
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache

# you can of course write some python
logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "phase space density"  # display name, used in menubar and command palette
CATEGORY = "plot"  # category (note that CATEGORY="" is a valid choice)


def main(self):
    """
    the script also have to define a `main` function. When playing a script,
    HAL runs `main` passes one (and only one) argument "self" that is the
    HAL mainwindow object (granting access to all the gui attributes and methods)
    """

    # get metadata from current selection
    metadata = getSelectionMetaDataFromCache(self, update_cache=True)
    # let's plot
    if metadata:
        fig = plt.figure()
        for dset, data in metadata.items():
            if "file" in data and {"timestamp", "size"} <= data["file"].keys():
                cycle = data["file"]["cycle"]
                Ncal = data["fit"]["ROI 0::Ncal"]
                sx = data["fit"]["ROI 0::sx"]
                sy = data["fit"]["ROI 0::sy"]
                PSD = np.array(Ncal) / ((np.array(sx) ** 4) * (np.array(sy) ** 2))
                plt.plot(cycle, PSD, ":o", label="PSD")
        plt.legend()
        plt.grid()
        plt.ylabel("Phase space density")
        plt.xlabel("cycle")
        plt.title(f"Hope you will condense soon")
        plt.show()
