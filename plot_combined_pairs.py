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
from .libs.analysis import spacetime_to_velocities_converter
from .libs.constants import *

# --------------------------------------------------------------------------------------

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["ECGaramond"]})
rc("text", usetex=True)
red = "#B00028"
# --------------------------------------------------------------------------------------

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "Plot combined pairs"  # display name, used in menubar and command palette
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

    N_file = 0
    for item in selection:
        N_file += 1
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

    print(X)
    print(T)
    print(N_file)
    v_x, v_y, v_z = spacetime_to_velocities_converter(X, Y, T)
    df = pd.DataFrame({"v_x": v_x, "v_y": v_y, "v_z": v_z})
    del X, Y, T

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(4.5, 4.5))

    axs[0].hist2d(
        df["v_x"],
        df["v_z"],
        bins=[40, 100],
        cmap="turbo",
        rasterized=True,
    )
    axs[0].set(xlabel=r"$v_x$ (mm/s)")
    axs[0].set(ylabel=r"$v_z$ (mm/s)")

    axs[1].hist2d(
        df["v_y"],
        df["v_z"],
        bins=[40, 100],
        cmap="turbo",
        rasterized=True,
    )
    axs[1].set(xlabel=r"$v_y$ (mm/s)")

    axs[0].plot(
        [df["v_x"].min(), df["v_x"].max()],
        [40, 40],
        color=red,
    )
    axs[0].plot(
        [df["v_x"].min(), df["v_x"].max()],
        [80, 80],
        color=red,
    )
    axs[0].plot(
        [df["v_x"].min(), df["v_x"].max()],
        [110, 110],
        color="tab:orange",
    )
    axs[0].plot(
        [df["v_x"].min(), df["v_x"].max()],
        [150, 150],
        color="tab:orange",
    )

    axs[1].plot(
        [df["v_y"].min(), df["v_y"].max()],
        [40, 40],
        color=red,
    )
    axs[1].plot(
        [df["v_y"].min(), df["v_y"].max()],
        [80, 80],
        color=red,
    )
    axs[1].plot(
        [df["v_y"].min(), df["v_y"].max()],
        [110, 110],
        color="tab:orange",
    )
    axs[1].plot(
        [df["v_y"].min(), df["v_y"].max()],
        [150, 150],
        color="tab:orange",
    )

    plt.tight_layout()
    plt.savefig("pairs.pdf", bbox_inches="tight")

    fig.show()

    df_2 = df.loc[
        (df["v_z"] > 110)
        & (df["v_z"] < 150)
        # & (df["v_x"] > -100)
        # & (df["v_x"] < 0)
        # & (df["v_y"] > -50)
        # & (df["v_y"] < 50)
    ]

    df = df.loc[
        (df["v_z"] > 40)
        & (df["v_z"] < 80)
        # & (df["v_x"] > -100)
        # & (df["v_x"] < 0)
        # & (df["v_y"] > -50)
        # & (df["v_y"] < 50)
    ]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6.5, 4.5))

    axs[0, 0].set(ylabel=r"$\overline{N}_v$ (mm/s)${}^{-1}$")
    axs[1, 0].set(ylabel=r"$\overline{N}_v$ (mm/s)${}^{-1}$")
    axs[1, 0].set(xlabel=r"$v_x$ (mm/s)")
    axs[1, 1].set(xlabel=r"$v_y$ (mm/s)")
    axs[1, 2].set(xlabel=r"$v_z$ (mm/s)")

    axs[0, 1].sharey(axs[0, 0])
    axs[0, 2].sharey(axs[0, 0])
    axs[1, 1].sharey(axs[1, 0])
    axs[1, 2].sharey(axs[1, 0])

    axs[0, 0].sharex(axs[1, 0])
    axs[0, 1].sharex(axs[1, 1])
    axs[0, 2].sharex(axs[1, 2])

    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[0, 2].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[1, 2].grid()

    hist_Vx = np.histogram(df["v_x"], bins=100)
    bin_width_Vx = hist_Vx[1][1] - hist_Vx[1][0]
    bin_centers_Vx = hist_Vx[1][:-1] + 0.5 * bin_width_Vx

    hist_Vy = np.histogram(df["v_y"], bins=100)
    bin_width_Vy = hist_Vy[1][1] - hist_Vy[1][0]
    bin_centers_Vy = hist_Vy[1][:-1] + 0.5 * bin_width_Vy

    hist_Vz = np.histogram(df["v_z"], bins=100)
    bin_width_Vz = hist_Vz[1][1] - hist_Vz[1][0]
    bin_centers_Vz = hist_Vz[1][:-1] + 0.5 * bin_width_Vz

    axs[0, 0].plot(bin_centers_Vx, hist_Vx[0] / (N_file * bin_width_Vx), color=red)
    axs[0, 1].plot(bin_centers_Vy, hist_Vy[0] / (N_file * bin_width_Vy), color=red)
    axs[0, 2].plot(bin_centers_Vz, hist_Vy[0] / (N_file * bin_width_Vz), color=red)

    hist_Vx = np.histogram(df_2["v_x"], bins=100)
    bin_width_Vx = hist_Vx[1][1] - hist_Vx[1][0]
    bin_centers_Vx = hist_Vx[1][:-1] + 0.5 * bin_width_Vx

    hist_Vy = np.histogram(df_2["v_y"], bins=100)
    bin_width_Vy = hist_Vy[1][1] - hist_Vy[1][0]
    bin_centers_Vy = hist_Vy[1][:-1] + 0.5 * bin_width_Vy

    hist_Vz = np.histogram(df_2["v_z"], bins=100)
    bin_width_Vz = hist_Vz[1][1] - hist_Vz[1][0]
    bin_centers_Vz = hist_Vz[1][:-1] + 0.5 * bin_width_Vz

    axs[1, 0].plot(bin_centers_Vx, hist_Vx[0] / (N_file * bin_width_Vx), color=red)
    axs[1, 1].plot(bin_centers_Vy, hist_Vy[0] / (N_file * bin_width_Vy), color=red)
    axs[1, 2].plot(bin_centers_Vz, hist_Vy[0] / (N_file * bin_width_Vz), color=red)

    plt.tight_layout()
    # plt.savefig("pairs.pdf", bbox_inches="tight")

    fig.show()
