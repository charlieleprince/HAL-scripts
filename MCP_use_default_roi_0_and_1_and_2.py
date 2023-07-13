# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from pathlib import Path
from scipy.optimize import curve_fit
import json

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "2.Ter. Get number of atoms in default ROI0 and ROI1 and Roi2"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


k_B = 1.3806e-23
m = 4 * 1.66e-27
g = 9.81


# layout tools


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * standard_deviation ** 2))


def ROI_data(ROI, X, Y, T):
    ROI_indices = (
        (T > ROI["Tmin"])
        & (T < ROI["Tmax"])
        & (X > ROI["Xmin"])
        & (X < ROI["Xmax"])
        & (Y > ROI["Ymin"])
        & (Y < ROI["Ymax"])
    )
    T_ROI = T[ROI_indices]
    X_ROI = X[ROI_indices]
    Y_ROI = Y[ROI_indices]
    return (X_ROI, Y_ROI, T_ROI)


def exportROIinfo(to_mcp, ROI, nb):
    to_mcp.append(
        {
            "name": "--ROI" + str(nb) + ":Tmin",
            "value": ROI["Tmin"],
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp.append(
        {
            "name": "--ROI" + str(nb) + ":Tmax",
            "value": ROI["Tmax"],
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )


def fit_time_histo(T):
    bin_heights, bin_borders, _ = plt.hist(
        T, bins=np.linspace(np.min(T), np.max(T), 300)
    )
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    guess_amplitude = np.max(bin_heights)
    guess_mean = bin_centers[list(bin_heights).index(np.max(bin_heights))]
    guess_stdev = 5.0
    p0 = [guess_mean, guess_amplitude, guess_stdev]
    popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=p0)
    return (popt, pcov)

def add_roi_fit(to_mcp_dictionary, X_ROI, Y_ROI, T_ROI, roi_num = 0):
    to_mcp_dictionary.append(
        {
            "name": "N_ROI{}".format(roi_num),
            "value": len(X_ROI),
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp_dictionary.append(
        {
            "name": "T_ROI{}".format(roi_num),
            "value": np.mean(T_ROI),
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp_dictionary.append(
        {
            "name": "dT_ROI{}".format(roi_num),
            "value": np.std(T_ROI),
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp_dictionary.append(
        {
            "name": "X_ROI{}".format(roi_num),
            "value": np.mean(X_ROI),
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp_dictionary.append(
        {
            "name": "Y_ROI{}".format(roi_num),
            "value": np.mean(Y_ROI),
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp_dictionary.append(
        {
            "name": "dX_ROI{}".format(roi_num),
            "value": np.std(X_ROI),
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp_dictionary.append(
        {
            "name": "dY_ROI{}".format(roi_num),
            "value": np.std(Y_ROI),
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )

    return to_mcp_dictionary

# main
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
    if not data.path.suffix == ".atoms":
        return
    # get data
    X, Y, T = data.getrawdata()

    # default ROI

    root = Path().home()
    default_roi_dir = root / ".HAL"
    default_roi_file_name = default_roi_dir / "default_mcp_roi.json"

    if not default_roi_file_name.is_file():
        default_roi_dict = {
            "ROI 0": {"Xmin": 0, "Xmax": 0, "Ymin": 0, "Ymax": 0, "Tmin": 0, "Tmax": 0},
            "ROI 1": {"Xmin": 0, "Xmax": 0, "Ymin": 0, "Ymax": 0, "Tmin": 0, "Tmax": 0},
            "ROI 2": {"Xmin": 0, "Xmax": 0, "Ymin": 0, "Ymax": 0, "Tmin": 0, "Tmax": 0},
            "ROI 3": {"Xmin": 0, "Xmax": 0, "Ymin": 0, "Ymax": 0, "Tmin": 0, "Tmax": 0},
        }
        default_roi_file_name = default_roi_dir / "default_mcp_roi.json"
        with open(default_roi_file_name, "w", encoding="utf-8") as file:
            json.dump(default_roi_dict, file, ensure_ascii=False, indent=4)
    with open(default_roi_file_name, encoding="utf8") as f:
        defaultroi = json.load(f)

    ROI0 = defaultroi["ROI 0"]
    ROI1 = defaultroi["ROI 1"]
    ROI2 = defaultroi["ROI 2"]

    for k in range(len(selection)):
        item = selection[k]
        data.path = item.data(QtCore.Qt.UserRole)
        if not data.path.suffix == ".atoms":
            return
        # get data
        X, Y, T = data.getrawdata()

        to_mcp_dictionary = []
        ####
        ## NOMBRE TOTAL D'ATOME
        ####
        to_mcp_dictionary.append(
            {
                "name": "N_tot",
                "value": len(X),
                "display": "%.3g",
                "unit": "",
                "comment": "",
            }
        )

        #####
        ## NOMBRE D'ATOME DANS CHAQUE ROI
        #####
        (X_ROI0, Y_ROI0, T_ROI0) = ROI_data(ROI0, X, Y, T)
        (X_ROI1, Y_ROI1, T_ROI1) = ROI_data(ROI1, X, Y, T)
        (X_ROI2, Y_ROI2, T_ROI2) = ROI_data(ROI2, X, Y, T)

        N0 = len(X_ROI0)
        N1 = len(X_ROI1)
        N2 = len(X_ROI2)

        ####
        ## SAUVEGARDE DES PARAMÈTRES DE ROI
        ####
        exportROIinfo(to_mcp_dictionary, ROI0, 0)
        exportROIinfo(to_mcp_dictionary, ROI1, 1)
        exportROIinfo(to_mcp_dictionary, ROI2, 2)

        ####
        ## NOMBRE D'ATOME PAR ROI
        ####
        to_mcp_dictionary = add_roi_fit(to_mcp_dictionary, X_ROI0, Y_ROI0, T_ROI0, roi_num = 0 )
        to_mcp_dictionary = add_roi_fit(to_mcp_dictionary, X_ROI1, Y_ROI1, T_ROI1, roi_num = 1 )
        to_mcp_dictionary = add_roi_fit(to_mcp_dictionary, X_ROI2, Y_ROI2, T_ROI2, roi_num = 2 )


        N012 = N0 + N1 + N2

        if N012 > 0:
            to_mcp_dictionary.append(
                {
                    "name": "N_0 / N_tot",
                    "value": N0 / N012,
                    "display": "%.3g",
                    "unit": "",
                    "comment": "",
                }
            )
            to_mcp_dictionary.append(
                {
                    "name": "N_1 / N_tot",
                    "value": N1 / N012,
                    "display": "%.3g",
                    "unit": "",
                    "comment": "",
                }
            )
            to_mcp_dictionary.append(
                {
                    "name": "N_2 / N_tot",
                    "value": N2 / N012,
                    "display": "%.3g",
                    "unit": "",
                    "comment": "",
                }
            )

        MCP_stats_folder = data.path.parent / ".MCPstats"
        MCP_stats_folder.mkdir(exist_ok=True)
        file_name = MCP_stats_folder / data.path.stem
        with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
            json.dump(to_mcp_dictionary, file, ensure_ascii=False, indent=4)

    plt.close()
