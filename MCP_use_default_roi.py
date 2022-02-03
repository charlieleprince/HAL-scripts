# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import numpy as np
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
NAME = "2. Export ROI info using default ROI0"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)

# layout tools


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


def _roi_changed(self):
    # move self.label
    position = self.pos()
    self.label.setPos(position[0], position[1])


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

    ROI0 = {}
    ROI0["Xmin"] = defaultroi["ROI 0"]["Xmin"]
    ROI0["Xmax"] = defaultroi["ROI 0"]["Xmax"]
    ROI0["Ymin"] = defaultroi["ROI 0"]["Ymin"]
    ROI0["Ymax"] = defaultroi["ROI 0"]["Ymax"]
    ROI0["Tmin"] = defaultroi["ROI 0"]["Tmin"]
    ROI0["Tmax"] = defaultroi["ROI 0"]["Tmax"]

    for k in range(len(selection)):
        item = selection[k]
        data.path = item.data(QtCore.Qt.UserRole)
        if not data.path.suffix == ".atoms":
            return
        # get data
        X, Y, T = data.getrawdata()

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

        (X_ROI0, Y_ROI0, T_ROI0) = ROI_data(ROI0, X, Y, T)

        exportROIinfo(to_mcp_dictionary, ROI0, 0)
        to_mcp_dictionary.append(
            {
                "name": "N_ROI0",
                "value": len(X_ROI0),
                "display": "%o",
                "unit": "",
                "comment": "",
            }
        )

        MCP_stats_folder = data.path.parent / ".MCPstats"
        MCP_stats_folder.mkdir(exist_ok=True)
        file_name = MCP_stats_folder / data.path.stem
        with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
            json.dump(to_mcp_dictionary, file, ensure_ascii=False, indent=4)
