# -*- coding: utf-8 -*-

# IMPORTS
# --------------------------------------------------------------------------------------

# built-in python libs
import logging
import json
from pathlib import Path

# third party imports
# -------------------
# Qt
from PyQt5.QtCore import Qt

# misc.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# local libs
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from .libs.roi import exportROIinfo, filter_data_to_ROI
from .libs.constants import *

# --------------------------------------------------------------------------------------

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "Autocorrelations"  # display name, used in menubar and command palette
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

    item = selection[0]
    data.path = item.data(Qt.UserRole)
    X, Y, T = data.getrawdata()
    if "N_ROI0" in metadata["current selection"]["mcp"]:
        (X, Y, T) = filter_data_to_ROI(
            X, Y, T, from_metadata=True, metadata=metadata, metadata_ROI_nb=0
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
        (X, Y, T) = filter_data_to_ROI(X, Y, T, from_metadata=False, ROI=def_ROI)

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
        exportROIinfo(to_mcp_dictionary, def_ROI, 0)

        MCP_stats_folder = data.path.parent / ".MCPstats"
        MCP_stats_folder.mkdir(exist_ok=True)
        file_name = MCP_stats_folder / data.path.stem
        with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
            json.dump(to_mcp_dictionary, file, ensure_ascii=False, indent=4)

    for k in range(len(selection) - 1):
        item = selection[k]
        data.path = item.data(Qt.UserRole)
        if not data.path.suffix == ".atoms":
            return
        # get data
        Xa, Ya, Ta = data.getrawdata()
        if "N_ROI0" in metadata["current selection"]["mcp"]:
            (Xa, Ya, Ta) = filter_data_to_ROI(
                Xa, Ya, Ta, from_metadata=True, metadata=metadata, metadata_ROI_nb=0
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

            (Xa, Ya, Ta) = filter_data_to_ROI(
                Xa, Ya, Ta, from_metadata=False, ROI=def_ROI
            )

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
            exportROIinfo(to_mcp_dictionary, def_ROI, 0)

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

    # filtering

    # large temporal box
    df = df.loc[(df["T"] > 309.0) & (df["T"] < 326.5)]

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
            data.path = item.data(Qt.UserRole)
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
