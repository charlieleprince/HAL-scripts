# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "temperature"  # display name, used in menubar and command palette
CATEGORY = "fit"  # category (note that CATEGORY="" is a valid choice)

# helium parameters and constants
u = 1.66053906660e-27
m = 4.00260325413 * u
k_B = 1.380649e-23

# TOOLS
def linfunc(x, a, b):
    return a * x + b


def sigma(t, a, b):
    return np.sqrt(b + a * (t ** 2))


def main(self):
    """
    the script also have to define a `main` function. When playing a script,
    HAL runs `main` passes one (and only one) argument "self" that is the
    HAL mainwindow object (granting access to all the gui attributes and methods)
    """

    # get metadata from current selection
    metadata = getSelectionMetaDataFromCache(self, update_cache=True)

    # let's go
    if metadata:
        for dset, data in metadata.items():
            if "file" in data:
                TOF = np.array(data["qc3"]["TOF"]) * (1e-3)
                sx = np.array(data["fit"]["ROI 0::sx"]) * (1e-6)
                sy = np.array(data["fit"]["ROI 0::sy"]) * (1e-6)

        # fit
        poptx, pcovx = opt.curve_fit(linfunc, TOF ** 2, sx ** 2)
        popty, pcovy = opt.curve_fit(linfunc, TOF ** 2, sy ** 2)

        # fit results
        # x
        sigma_vx_fit = np.sqrt(poptx[0])
        Tx_fit = m * sigma_vx_fit ** 2 / k_B
        std_err_sigma_vx = np.sqrt(pcovx[0][0])
        std_err_Tx = m * (std_err_sigma_vx) / k_B
        sigma_0x = np.sqrt(poptx[1])
        std_err_sigma0x = np.sqrt(pcovx[1][1]) / (2 * np.sqrt(poptx[1]))
        # y
        sigma_vy_fit = np.sqrt(popty[0])
        Ty_fit = m * sigma_vy_fit ** 2 / k_B
        std_err_sigma_vy = np.sqrt(pcovy[0][0])
        std_err_Ty = m * (std_err_sigma_vy) / k_B
        sigma_0y = np.sqrt(popty[1])
        std_err_sigma0y = np.sqrt(pcovy[1][1]) / (2 * np.sqrt(popty[1]))
        # plot fit results:
        t = np.linspace(np.min(TOF), np.max(TOF), 100)

        sigma_x_fit = []
        sigma_y_fit = []
        for k in range(len(t)):
            sigma_x_fit.append(sigma(t[k], poptx[0], poptx[1]))
            sigma_y_fit.append(sigma(t[k], popty[0], popty[1]))

        # plot
        fig = plt.figure()
        plt.plot(TOF * 1e3, sx * 1e6, ":o", label="sx")
        plt.plot(TOF * 1e3, sy * 1e6, ":o", label="sy")
        plt.plot(t * 1e3, np.array(sigma_x_fit) * 1e6, label="fit x")
        plt.plot(t * 1e3, np.array(sigma_y_fit) * 1e6, label="fit y")
        plt.legend()
        plt.grid()
        plt.ylabel("size (µm)")
        plt.xlabel("Time of flight (ms)")
        plt.title(f"Fit temperature")
        fig.autofmt_xdate()
        plt.show()

        # display fit results
        Text = "[fit results]\n"
        Text += f"Tx = {np.round(Tx_fit*1e6,2)} ± {2*np.round(std_err_Tx*1e6,2)} µK\n"
        Text += f"Ty = {np.round(Ty_fit*1e6,2)} ± {2*np.round(std_err_Ty*1e6,2)} µK\n"
        Text += "--------------\n"
        Text += f"s0x = {np.round(sigma_0x*1e6,1)} ± {2*np.round(std_err_sigma0x*1e6,1)} µm\n"
        Text += f"s0y = {np.round(sigma_0y*1e6,1)} ± {2*np.round(std_err_sigma0y*1e6,1)} µm\n"
    self.metaDataText.setPlainText(Text)
