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
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = (
    "3bis. Combine and watch ROI"  # display name, used in menubar and command palette
)
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


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

    root = Path().home()
    default_roi_dir = root / ".HAL"
    default_roi_file_name = default_roi_dir / "default_mcp_roi.json"
    # get path
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
                "display": "%o",
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
        item = selection[k + 1]
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
                    "display": "%o",
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

    fig3 = plt.figure()
    plt.hist2d(X, Y, bins=np.linspace(-40, 40, 2 * 81), cmap=plt.cm.jet)
    plt.colorbar()
    fig3.show()

    fig2 = plt.figure()
    plt.hist(T, bins=np.linspace(np.min(T), np.max(T), 300))
    plt.xlabel("time (ms)")
    plt.ylabel("number of events")
    plt.grid(True)
    fig2.show()

    fig4 = plt.figure()
    plt.hist2d(
        X,
        T,
        bins=[np.linspace(-40, 40, 2 * 81), np.linspace(np.min(T), np.max(T), 2 * 81)],
        cmap=plt.cm.jet,
    )
    plt.xlabel("X")
    plt.ylabel("T")
    plt.colorbar()
    fig4.show()

    fig5 = plt.figure()
    plt.hist2d(
        Y,
        T,
        [np.linspace(-40, 40, 2 * 81), np.linspace(np.min(T), np.max(T), 2 * 81)],
        cmap=plt.cm.jet,
    )
    plt.xlabel("Y")
    plt.ylabel("T")
    plt.colorbar()
    fig5.show()
