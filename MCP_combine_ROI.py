# -*- coding: utf-8 -*-

# IMPORTS
# --------------------------------------------------------------------------------------

# built-in python libs
# --------------------
import logging
import json
from pathlib import Path

# third party imports
# -------------------
# Qt
from PyQt5.QtCore import Qt

# misc.
import numpy as np
from matplotlib import pyplot as plt

# local libs
# ----------
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from .libs.roi import exportROIinfo, filter_data_to_ROI
from .libs.constants import *

# --------------------------------------------------------------------------------------

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = (
    "3bis. Combine and watch ROI"  # display name, used in menubar and command palette
)
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)
logger = logging.getLogger(__name__)


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

    # Initialize data
    X, Y, T = np.empty([3, 0])

    for item in selection:
        data.path = item.data(Qt.UserRole)
        # if not data.path.suffix == ".atoms":
        #     return

        # get data
        X_item, Y_item, T_item = data.getrawdata()

        # first data filter
        if "N_ROI0" in metadata["current selection"]["mcp"]:
            (X_item, Y_item, T_item) = filter_data_to_ROI(
                X_item,
                Y_item,
                T_item,
                from_metadata=True,
                metadata=metadata,
                metadata_ROI_nb=0,
            )
        else:
            with open(default_roi_file_name, encoding="utf8") as f:
                default_roi = json.load(f)

            def_ROI = {
                "Xmin": default_roi["ROI 0"]["Xmin"],
                "Xmax": default_roi["ROI 0"]["Xmax"],
                "Ymin": default_roi["ROI 0"]["Ymin"],
                "Ymax": default_roi["ROI 0"]["Ymax"],
                "Tmin": default_roi["ROI 0"]["Tmin"],
                "Tmax": default_roi["ROI 0"]["Tmax"],
            }

            (X_item, Y_item, T_item) = filter_data_to_ROI(
                X_item, Y_item, T_item, from_metadata=False, ROI=def_ROI
            )

            to_mcp_dictionaries = []
            to_mcp_dictionaries.append(
                {
                    "name": "N_tot",
                    "value": len(X),
                    "display": "%.3g",
                    "unit": "",
                    "comment": "",
                }
            )
            exportROIinfo(to_mcp_dictionaries, def_ROI, 0)

            MCP_stats_folder = data.path.parent / ".MCPstats"
            MCP_stats_folder.mkdir(exist_ok=True)
            file_name = MCP_stats_folder / data.path.stem
            with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
                json.dump(to_mcp_dictionaries, file, ensure_ascii=False, indent=4)

        X = np.concatenate([X, X_item])
        Y = np.concatenate([Y, Y_item])
        T = np.concatenate([T, T_item])

    fig3 = plt.figure()
    plt.hist2d(X, Y, bins=np.linspace(-40, 40, 2 * 81), cmap=plt.cm.jet)
    plt.colorbar()
    fig3.show()

    fig2 = plt.figure()
    plt.hist(T, bins=np.linspace(np.min(T), np.max(T), 150))
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
