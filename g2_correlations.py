# -*- coding: utf-8 -*-

# IMPORTS
# --------------------------------------------------------------------------------------

# built-in python libs
import logging
import itertools as itt
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
from .libs.analysis import spacetime_to_velocities_converter
from .libs.constants import *

# --------------------------------------------------------------------------------------

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "2D g(2)"  # display name, used in menubar and command palette
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
        # get data
        X_item, Y_item, T_item = data.getrawdata()
        v_x, v_y, v_z = spacetime_to_velocities_converter(X_item, Y_item, T_item)
        df_item = pd.DataFrame({"v_x": v_x, "v_y": v_y, "v_z": v_z})

        for idx1, box_1_vz in enumerate(box_vz_array):


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

    df_correlations["N box1 average"] = np.array(
        [np.array(l).mean() for l in df_correlations["N box1"].to_numpy()]
    )
    df_correlations["N box1 std"] = np.array(
        [np.array(l).std() for l in df_correlations["N box1"].to_numpy()]
    )
    df_correlations["N box2 average"] = np.array(
        [np.array(l).mean() for l in df_correlations["N box2"].to_numpy()]
    )
    df_correlations["N box2 std"] = np.array(
        [np.array(l).std() for l in df_correlations["N box2"].to_numpy()]
    )

    df_correlations["N1 x N2 average"] = np.array(
        [np.array(l).mean() for l in df_correlations["N1 x N2"].to_numpy()]
    )
    df_correlations["N1 x N2 std"] = np.array(
        [np.array(l).std() for l in df_correlations["N1 x N2"].to_numpy()]
    )

    # saving results
    root = Path.home()
    df_correlations.to_csv(root / "correlations.csv")

    # plot
    # box_2_vz = df_correlations["Box2 v_z"].to_numpy()
    # g2 = df_correlations["N_Box1 x N_Box2"].to_numpy() / (
    #     N_box1 * df_correlations["N_Box2"].to_numpy()
    # )
    # print(g2)
    # plt.figure()
    # plt.plot(box_2_vz, g2)
    # plt.show()
