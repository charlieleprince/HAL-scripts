# -*- coding: utf-8 -*-

# IMPORTS
# --------------------------------------------------------------------------------------

# built-in python libs
import logging
import json
import itertools as itt
from pathlib import Path

# third party imports
# -------------------
# Qt
from PyQt5.QtCore import Qt
from PyQt5 import QtCore

# misc.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# local libs
from .libs.analysis import spacetime_to_velocities_converter
from .libs.constants import *

# --------------------------------------------------------------------------------------


# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "Pair variances"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)
logger = logging.getLogger(__name__)


def main(self):
    """
    the script also have to define a `main` function. When playing a script,
    HAL runs `main` passes one (and only one) argument "self" that is the
    HAL mainwindow object (granting access to all the gui attributes and methods)
    """

    # -- get selected data
    selection = self.runList.selectedItems()
    if not selection:
        return
    # -- init object data
    # get object data type
    data_class = self.dataTypeComboBox.currentData()
    data = data_class()

    # box parameters (mm/s)
    # ----------------------------------------------------------------------------------

    # sizes
    delta_v_perp = 10.0  # 0.28
    delta_v_z = 2.0  # 0.48

    # centers
    box_1_vx = -40.5
    box_1_vy = 2.5

    box_2_vx = -43.7
    box_2_vy = 1.6

    vz_min, vz_max = 50.0, 140.0
    n_boxes = int(np.ceil((vz_max - vz_min) / delta_v_z))
    print(n_boxes)
    box_vz_array = np.linspace(50.0, 140.0, n_boxes)

    # ----------------------------------------------------------------------------------

    df_correlations = pd.DataFrame(
        list(itt.product(box_vz_array, box_vz_array)), columns=["vz box1", "vz box2"]
    )
    df_correlations["N box1"] = [[] for _ in range(n_boxes**2)]
    df_correlations["N box2"] = [[] for _ in range(n_boxes**2)]
    df_correlations["N1 x N2"] = [[] for _ in range(n_boxes**2)]

    for idx_file, item in enumerate(selection):
        data.path = item.data(Qt.UserRole)

        print(f"{idx_file/len(selection)*100:.2f}%")

        for idx1, box_1_vz in enumerate(box_vz_array):

            # get data
            X_item, Y_item, T_item = data.getrawdata()
            v_x, v_y, v_z = spacetime_to_velocities_converter(X_item, Y_item, T_item)
            df_item = pd.DataFrame({"v_x": v_x, "v_y": v_y, "v_z": v_z})

            # filter data to box 1
            df_box1 = df_item.loc[
                (df_item["v_x"] > box_1_vx - delta_v_perp / 2)
                & (df_item["v_x"] < box_1_vx + delta_v_perp / 2)
                & (df_item["v_y"] > box_1_vy - delta_v_perp / 2)
                & (df_item["v_y"] < box_1_vy + delta_v_perp / 2)
                & (df_item["v_z"] > box_1_vz - delta_v_z / 2)
                & (df_item["v_z"] < box_1_vz + delta_v_z / 2)
            ]

            N_box1 = len(df_box1)

            for idx2, box_2_vz in enumerate(box_vz_array):

                # filter data to box 1
                df_box2 = df_item.loc[
                    (df_item["v_x"] > box_2_vx - delta_v_perp / 2)
                    & (df_item["v_x"] < box_2_vx + delta_v_perp / 2)
                    & (df_item["v_y"] > box_2_vy - delta_v_perp / 2)
                    & (df_item["v_y"] < box_2_vy + delta_v_perp / 2)
                    & (df_item["v_z"] > box_2_vz - delta_v_z / 2)
                    & (df_item["v_z"] < box_2_vz + delta_v_z / 2)
                ]

                N_box2 = len(df_box2)
                df_correlations.iloc[idx1 * n_boxes + idx2]["N box1"].append(N_box1)
                df_correlations.iloc[idx1 * n_boxes + idx2]["N box2"].append(N_box2)
                df_correlations.iloc[idx1 * n_boxes + idx2]["N1 x N2"].append(
                    N_box1 * N_box2
                )


    # filtering

    # large temporal box
    df = df.loc[(df["T"] > 309.0) & (df["T"] < 326.5)]

    # fig, axs = plt.subplots(1, 2)

    # axs[0].hist2d(
    #     df["X"],
    #     df["T"],
    #     bins=[40, 100],
    # )
    # axs[0].set(xlabel="X (mm)")
    # axs[0].set(ylabel="T (ms)")

    # axs[1].hist2d(
    #     df["Y"],
    #     df["T"],
    #     bins=[40, 100],
    # )
    # axs[1].set(xlabel="Y (mm)")
    # axs[1].set(ylabel="T (ms)")
    # fig.show()

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
