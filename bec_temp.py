# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
import json
from pathlib import Path

from matplotlib import pyplot as plt

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

    item = selection[0]
    data.path = item.data(QtCore.Qt.UserRole)
    X, Y, T = data.getrawdata()
    if "N_ROI0" in metadata["current selection"]["mcp"]:
        (X, Y, T) = ROIdata(metadata, "0", X, Y, T)
    else:
        with open(default_roi_file_name, encoding="utf8") as f:
            defaultroi = json.load(f)

        ROI0 = {}
        ROI0["Xmin"] = defaultroi["ROI 0"]["Xmin"]
        ROI0["Xmax"] = defaultroi["ROI 0"]["Xmax"]
        ROI0["Ymin"] = defaultroi["ROI 0"]["Ymin"]
        ROI0["Ymax"] = defaultroi["ROI 0"]["Ymax"]
        ROI0["Tmin"] = defaultroi["ROI 0"]["Tmin"]
        ROI0["Tmax"] = defaultroi["ROI 0"]["Tmax"]

        (X, Y, T) = ROI_data(ROI0, X, Y, T)

        to_mcp_dictionary = []
        to_mcp_dictionary.append(
            {
                "name": "N_tot",
                "value": len(X),
                "display": "%.3g",
                "unit": "",
                "comment": "",
            }
        )
        exportROIinfo(to_mcp_dictionary, ROI0, 0)

        MCP_stats_folder = data.path.parent / ".MCPstats"
        MCP_stats_folder.mkdir(exist_ok=True)
        file_name = MCP_stats_folder / data.path.stem
        with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
            json.dump(to_mcp_dictionary, file, ensure_ascii=False, indent=4)

    for k in range(len(selection) - 1):
        item = selection[k]
        data.path = item.data(QtCore.Qt.UserRole)
        if not data.path.suffix == ".atoms":
            return
        # get data
        Xa, Ya, Ta = data.getrawdata()
        if "N_ROI0" in metadata["current selection"]["mcp"]:
            (Xa, Ya, Ta) = ROIdata(metadata, "0", Xa, Ya, Ta)
        else:
            with open(default_roi_file_name, encoding="utf8") as f:
                defaultroi = json.load(f)

            ROI0 = {}
            ROI0["Xmin"] = defaultroi["ROI 0"]["Xmin"]
            ROI0["Xmax"] = defaultroi["ROI 0"]["Xmax"]
            ROI0["Ymin"] = defaultroi["ROI 0"]["Ymin"]
            ROI0["Ymax"] = defaultroi["ROI 0"]["Ymax"]
            ROI0["Tmin"] = defaultroi["ROI 0"]["Tmin"]
            ROI0["Tmax"] = defaultroi["ROI 0"]["Tmax"]

            (Xa, Ya, Ta) = ROI_data(ROI0, Xa, Ya, Ta)

            to_mcp_dictionary = []
            to_mcp_dictionary.append(
                {
                    "name": "N_tot",
                    "value": len(X),
                    "display": "%.3g",
                    "unit": "",
                    "comment": "",
                }
            )
            exportROIinfo(to_mcp_dictionary, ROI0, 0)

            MCP_stats_folder = data.path.parent / ".MCPstats"
            MCP_stats_folder.mkdir(exist_ok=True)
            file_name = MCP_stats_folder / data.path.stem
            with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
                json.dump(to_mcp_dictionary, file, ensure_ascii=False, indent=4)

        X = np.concatenate([X, Xa])
        Y = np.concatenate([Y, Ya])
        T = np.concatenate([T, Ta])

    df = pd.DataFrame({"X": X, "Y": Y, "T": T})
    del X, Y, T

    # print(np.histogram(df["T"], bins=400))
    # fig2 = plt.figure()
    # plt.hist(
    #     df["T"],
    #     bins=np.linspace(df["T"].min(), df["T"].max(), 1000),
    #     histtype="step",
    #     log=True,
    # )
    # plt.xlabel("time (ms)")
    # plt.ylabel("number of events")
    # plt.grid(True)
    # fig2.show()

    fig2 = plt.figure()
    plt.hist(
        df["Y"],
        bins="auto",
        histtype="step",
        log=True,
    )
    plt.xlabel("Y (mm)")
    plt.ylabel("number of events")
    plt.grid(True)
    fig2.show()

    # filtering
    left_wing_min = 299.2
    right_wing_max = 317.2
    center = 307.9
    bec_widths = np.linspace(0.5, 6, 20)
    temperatures_t = []
    sigma_temperatures_t = []
    temperatures_Y = []
    sigma_temperatures_Y = []

    for bec_width in bec_widths:

        df_left_wing = df.loc[
            ((df["T"] > left_wing_min) & (df["T"] < center - bec_width))
        ]
        df_right_wing = df.loc[
            ((df["T"] > center + bec_width) & (df["T"] < right_wing_max))
        ]

        df_wings = pd.concat([df_left_wing, df_right_wing], ignore_index=True)
        Y_bin = np.histogram(df_wings["Y"], bins="auto")
        Y_bin_centers = Y_bin[1][:-1] + np.diff(Y_bin[1]) / 2
        Y_sigma_bin = np.sqrt(Y_bin[0])

        T_bin_left = np.histogram(df_left_wing["T"], bins="auto")
        T_sigma_bin_left = np.sqrt(T_bin_left[0]) / (
            T_bin_left[0].max() - T_bin_left[0].min()
        )

        T_bin_right = np.histogram(df_right_wing["T"], bins="auto")
        T_sigma_bin_right = np.sqrt(T_bin_right[0]) / (
            T_bin_right[0].max() - T_bin_right[0].min()
        )

        T_bin_left_norm = np.interp(
            T_bin_left[0], (T_bin_left[0].min(), T_bin_left[0].max()), (0.0, 1.0)
        )
        T_bin_left_centers = T_bin_left[1][:-1] + np.diff(T_bin_left[1]) / 2

        T_bin_right_norm = np.interp(
            T_bin_right[0], (T_bin_right[0].min(), T_bin_right[0].max()), (0.0, 1.0)
        )
        T_bin_right_centers = T_bin_right[1][:-1] + np.diff(T_bin_right[1]) / 2

        T_binned_wings_norm = np.array(
            [
                np.concatenate([T_bin_left_centers, T_bin_right_centers]),
                np.concatenate([T_bin_left_norm, T_bin_right_norm]),
                np.concatenate([T_sigma_bin_left, T_sigma_bin_right]),
            ]
        )

        # time fit
        guess_amplitude = np.max(T_binned_wings_norm[1])
        guess_mean = center
        guess_stdev = 1.0
        p0 = [guess_mean, guess_amplitude, guess_stdev]
        popt, pcov = curve_fit(
            gaussian,
            T_binned_wings_norm[0],
            T_binned_wings_norm[1],
            p0=p0,
            sigma=T_binned_wings_norm[2],
        )
        perr = np.sqrt(np.diag(pcov))

        temperature_t = m * (g**2) * ((popt[2]) ** 2) / k_B
        sigma_temperature_t = 2 * temperature_t * perr[2] / popt[2]
        temperatures_t.append(temperature_t)
        sigma_temperatures_t.append(sigma_temperature_t)

        # spatial fit
        guess_amplitude = np.max(Y_bin[1])
        guess_mean = 0.0
        guess_stdev = 1.0
        p0 = [guess_mean, guess_amplitude, guess_stdev]
        popt, pcov = curve_fit(
            gaussian,
            Y_bin[0],
            Y_bin_centers,
            p0=p0,
            sigma=Y_sigma_bin,
        )
        perr = np.sqrt(np.diag(pcov))

        temperature_Y = (m / k_B) * (popt[2] / center) ** 2
        sigma_temperature_Y = 2 * temperature_Y * perr[2] / popt[2]
        temperatures_Y.append(temperature_Y)
        sigma_temperatures_Y.append(sigma_temperature_Y)

    fig = plt.figure()
    plt.errorbar(bec_widths, temperatures_t, yerr=sigma_temperatures_t, fmt="o")
    plt.xlabel("BEC truncated width (ms)")
    plt.ylabel("Temperature (µK)")
    plt.grid(True)
    plt.show()

    fig = plt.figure()
    plt.errorbar(bec_widths, temperatures_Y, yerr=sigma_temperatures_Y, fmt="o")
    plt.xlabel("BEC truncated width (ms)")
    plt.ylabel("Temperature (K)")
    plt.grid(True)
    plt.show()

    # fig3 = plt.figure()
    # plt.hist(
    #     df_left_wing["T"],
    #     bins=np.linspace(df_left_wing["T"].min(), df_left_wing["T"].max(), 100),
    #     histtype="step",
    #     log=False,
    #     density=True,
    # )
    # plt.hist(
    #     df_right_wing["T"],
    #     bins=np.linspace(df_right_wing["T"].min(), df_right_wing["T"].max(), 100),
    #     histtype="step",
    #     log=False,
    #     density=True,
    # )
    # plt.xlabel("time (ms)")
    # plt.ylabel("number of events")
    # plt.grid(True)
    # fig3.show()

    # fig4 = plt.figure()
    # plt.plot(bin_left_centers, bin_left_norm)
    # plt.plot(bin_right_centers, bin_right_norm)
    # fig4.show()

    # fig4 = plt.figure()
    # plt.plot(binned_wings_norm[0], binned_wings_norm[1])
    # fig4.show()

    fig4 = plt.figure()
    plt.hist(
        df_wings["Y"],
        bins="auto",
        histtype="step",
    )
    plt.xlabel("Y (mm)")
    plt.ylabel("number of events")
    plt.grid(True)
    fig4.show()
