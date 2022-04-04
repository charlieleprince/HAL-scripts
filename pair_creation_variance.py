# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
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
NAME = "Pair variances"  # display name, used in menubar and command palette
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

    # fig2 = plt.figure()
    # plt.hist(
    #     df["T"],
    #     bins="auto",
    #     histtype="step",
    #     log=True,
    # )
    # plt.xlabel("T (mm)")
    # plt.ylabel("number of events")
    # plt.grid(True)
    # fig2.show()

    # filtering

    # large temporal box
    df = df.loc[(df["T"] > 309.0) & (df["T"] < 326.5)]

    fig, axs = plt.subplots(1, 2)

    axs[0].hist2d(
        df["X"],
        df["T"],
        bins=[40, 100],
    )
    axs[0].set(xlabel="X (mm)")
    axs[0].set(ylabel="T (ms)")

    axs[1].hist2d(
        df["Y"],
        df["T"],
        bins=[40, 100],
    )
    axs[1].set(xlabel="Y (mm)")
    axs[1].set(ylabel="T (ms)")
    fig.show()

    # small boxes
    spatial_width = 5.0
    temporal_width = 0.1

    box1_temporal_center = 310.0
    box2_temporal_centers = np.linspace(320, 326, 10)

    box_spatial_center_X = -13.0
    box_spatial_center_Y = 0.0

    df_correlations = pd.DataFrame(columns=["box2_temporal_center", "Variance"])
    for box2_temporal_center in box2_temporal_centers:

        DeltaN = []
        SumN = []

        for k in range(len(selection) - 1):
            item = selection[k]
            data.path = item.data(QtCore.Qt.UserRole)
            if not data.path.suffix == ".atoms":
                return
            # get data
            Xa, Ya, Ta = data.getrawdata()
            df = pd.DataFrame({"X": Xa, "Y": Ya, "T": Ta})

            df_box1 = df.loc[
                (df["T"] > box1_temporal_center - 0.5 * temporal_width)
                & (df["T"] < box1_temporal_center + 0.5 * temporal_width)
                & (df["X"] > box_spatial_center_X - 0.5 * spatial_width)
                & (df["X"] < box_spatial_center_X + 0.5 * spatial_width)
                & (df["Y"] > box_spatial_center_Y - 0.5 * spatial_width)
                & (df["Y"] < box_spatial_center_Y + 0.5 * spatial_width)
            ]

            df_box2 = df.loc[
                (df["T"] > box2_temporal_center - 0.5 * temporal_width)
                & (df["T"] < box2_temporal_center + 0.5 * temporal_width)
                & (df["X"] > box_spatial_center_X - 0.5 * spatial_width)
                & (df["X"] < box_spatial_center_X + 0.5 * spatial_width)
                & (df["Y"] > box_spatial_center_Y - 0.5 * spatial_width)
                & (df["Y"] < box_spatial_center_Y + 0.5 * spatial_width)
            ]

            DeltaN.append(len(df_box1) - len(df_box2))
            SumN.append(len(df_box1) + len(df_box2))

        DeltaN = np.array(DeltaN)
        SumN = np.array(SumN)
        var = np.var(DeltaN) / np.mean(SumN)
        df_correlations = pd.concat(
            [
                df_correlations,
                pd.DataFrame(
                    [[box2_temporal_center, var]],
                    columns=["box2_temporal_center", "Variance"],
                ),
            ],
            ignore_index=True,
        )

    fig = plt.figure()
    plt.plot(df_correlations["box2_temporal_center"], df_correlations["Variance"])
    plt.show()
    # left_wing_min = 299.2
    # right_wing_max = 317.2
    # center = 307.9
    # bec_widths = np.linspace(0.5, 6, 20)
    # temperatures_t = []
    # sigma_temperatures_t = []
    # temperatures_Y = []
    # sigma_temperatures_Y = []
