# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import curve_fit
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "7. BEC temperature"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


k_B = 1.3806e-23
m = 4 * 1.66e-27
g = 9.81


def read_metadata(metadata, nb):
    Xmin = metadata["current selection"]["mcp"]["--ROI" + nb + ":Xmin"][0]
    Xmax = metadata["current selection"]["mcp"]["--ROI" + nb + ":Xmax"][0]
    Ymin = metadata["current selection"]["mcp"]["--ROI" + nb + ":Ymin"][0]
    Ymax = metadata["current selection"]["mcp"]["--ROI" + nb + ":Ymax"][0]
    Tmin = metadata["current selection"]["mcp"]["--ROI" + nb + ":Tmin"][0]
    Tmax = metadata["current selection"]["mcp"]["--ROI" + nb + ":Tmax"][0]
    return (Xmin, Xmax, Ymin, Ymax, Tmin, Tmax)


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * standard_deviation**2))


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


def fit_transverse_distribution_Y(Y):
    bin_heights, bin_borders, _ = plt.hist(
        Y, bins=np.linspace(np.min(Y), np.max(Y), 300)
    )
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    guess_amplitude = np.max(bin_heights)
    guess_mean = bin_centers[list(bin_heights).index(np.max(bin_heights))]
    guess_stdev = 5.0
    p0 = [guess_mean, guess_amplitude, guess_stdev]
    popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=p0)
    return (popt, pcov)


def exportROIinfo(to_mcp, ROI, nb):
    to_mcp.append(
        {
            "name": "--ROI" + str(nb) + ":Xmin",
            "value": ROI["Xmin"],
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp.append(
        {
            "name": "--ROI" + str(nb) + ":Xmax",
            "value": ROI["Xmax"],
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp.append(
        {
            "name": "--ROI" + str(nb) + ":Ymin",
            "value": ROI["Ymin"],
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
    to_mcp.append(
        {
            "name": "--ROI" + str(nb) + ":Ymax",
            "value": ROI["Ymax"],
            "display": "%.3g",
            "unit": "",
            "comment": "",
        }
    )
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


def ROIdata(metadata, nb, X, Y, T):
    (Xmin, Xmax, Ymin, Ymax, Tmin, Tmax) = read_metadata(metadata, nb)
    T_ROI = T[
        (T > Tmin) & (T < Tmax) & (X > Xmin) & (X < Xmax) & (Y > Ymin) & (Y < Ymax)
    ]
    X_ROI = X[
        (T > Tmin)
        & (T < Tmax)
        & (X > Xmin)
        & (X < Xmax)
        & (X < Xmax)
        & (Y > Ymin)
        & (Y < Ymax)
    ]
    Y_ROI = Y[
        (T > Tmin)
        & (T < Tmax)
        & (X > Xmin)
        & (X < Xmax)
        & (X < Xmax)
        & (Y > Ymin)
        & (Y < Ymax)
    ]
    return (X_ROI, Y_ROI, T_ROI)


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
    # T_raw_ROI = T_raw[ROI_indices2]
    return (X_ROI, Y_ROI, T_ROI)


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
    root = Path().home()
    default_roi_dir = root / ".HAL"
    default_roi_file_name = default_roi_dir / "default_mcp_roi.json"

    for k in range(len(selection) - 1):
        item = selection[k + 1]
        data.path = item.data(QtCore.Qt.UserRole)
        X, Y, T = data.getrawdata()
        if not data.path.suffix == ".atoms":
            return
        # get data
        Xa, Ya, Ta = data.getrawdata()
        X = np.concatenate([X, Xa])
        Y = np.concatenate([Y, Ya])
        T = np.concatenate([T, Ta])

    atoms_df = pd.DataFrame({"X": X, "Y": Y, "T": T})
    atoms_df = atoms_df.loc[atoms_df["T"] > ROI0["Tmin"]]
    atoms_df = atoms_df.loc[atoms_df["T"] < ROI0["Tmax"]]
    atoms_df = atoms_df.loc[atoms_df["X"] > ROI0["Xmin"]]
    atoms_df = atoms_df.loc[atoms_df["X"] < ROI0["Xmax"]]
    atoms_df = atoms_df.loc[atoms_df["Y"] > ROI0["Ymin"]]
    atoms_df = atoms_df.loc[atoms_df["Y"] < ROI0["Ymax"]]
        

        (popt, pcov) = fit_time_histo(T_ROI)
        Temperature_t = m * (g**2) * ((popt[2]) ** 2) / k_B  # µK

        (popt, pcov) = fit_time_histo(Y_ROI)
        t_fall = 308.0
        Temperature_y = ((popt[2] / t_fall) ** 2) * m / k_B

        print(f"Time temperature: {Temperature_t} µK")
        print(f"Y temperature: {Temperature_y*1e6} µK")

        MCP_stats_folder = data.path.parent / ".MCPstats"
        MCP_stats_folder.mkdir(exist_ok=True)
        file_name = MCP_stats_folder / data.path.stem
        with open(str(file_name) + ".json", "r", encoding="utf-8") as file:
            current_mcp_metadata = json.load(file)

        current_mcp_metadata.append(
            {
                "name": "ROI0 arrival time",
                "value": popt[0],
                "display": "%.2f",
                "unit": "ms",
                "comment": "",
            }
        )
        current_mcp_metadata.append(
            {
                "name": "ROI0 time width",
                "value": popt[2] * 1e3,
                "display": "%.2f",
                "unit": "µs",
                "comment": "",
            }
        )
        current_mcp_metadata.append(
            {
                "name": "ROI0 temperature",
                "value": Temperature_t,
                "display": "%.2f",
                "unit": "µK",
                "comment": "",
            }
        )
        with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
            json.dump(current_mcp_metadata, file, ensure_ascii=False, indent=4)

        plt.close()