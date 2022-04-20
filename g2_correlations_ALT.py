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
import seaborn as sns

# local libs
from .libs.analysis import spacetime_to_velocities_converter
from .libs.constants import *

# --------------------------------------------------------------------------------------

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "2D g(2) ALT"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)
logger = logging.getLogger(__name__)


def box_mask(dataframe: pd.DataFrame, parameters: dict) -> pd.Series:
    if parameters["mode"] == "cylindrical":
        box = (
                (dataframe["v_x"] - parameters["center"]["v_x"]) ** 2
                + (dataframe["v_y"] - parameters["center"]["v_y"]) ** 2
            < (parameters["size"]["diameter"] / 2)**2
        ) & (
            np.abs(dataframe["v_z"] - parameters["center"]["v_z"])
            < parameters["size"]["height"] / 2
        )
    elif parameters["mode"] == "rectangle":
        box = (
            (
                np.abs(dataframe["v_x"] - parameters["center"]["v_x"])
                < parameters["size"]["v_x"] / 2
            )
            & (
                np.abs(dataframe["v_y"] - parameters["center"]["v_y"])
                < parameters["size"]["v_y"] / 2
            )
            & (
                np.abs(dataframe["v_z"] - parameters["center"]["v_z"])
                < parameters["size"]["v_z"] / 2
            )
        )
    else:
        raise

    return box


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

    v_z_height = 2.0
    transverse_diameter = 4.0

    box1_parameters = {
        "mode": "cylindrical",
        "center": {
            "v_x": -45,#-40.5,
            "v_y": 0,#2.5,
        },
        "size": {
            "diameter": transverse_diameter,
            "height": v_z_height,
        },
    }

    box2_parameters = {
        "mode": "cylindrical",
        "center": {
            "v_x": -45,#-43.7,
            "v_y": 0,#1.6,
        },
        "size": {
            "diameter": transverse_diameter,
            "height": v_z_height,
        },
    }

    vz_min, vz_max = 50.0, 140.0
    n_boxes = int(np.ceil((vz_max - vz_min) / v_z_height))
    print(n_boxes)
    box_vz_array = np.linspace(vz_min, vz_max, n_boxes)
    # ----------------------------------------------------------------------------------

    # data loading:
    # all the files are successively opened and the data are loaded in the same large
    # DataFrame `df_data`. The memory of the run (= data file) is kept in the DataFrame
    # with the series `run`.
    # ----------------------------------------------------------------------------------

    # preparing the DataFrame
    df_data = pd.DataFrame(columns=["run", "v_x", "v_y", "v_z"])

    # actual data loading: loop over the `*.atoms` files
    for idx_file, item in enumerate(selection):
        data.path = item.data(Qt.UserRole)

        # get raw data (in ms)
        X_item, Y_item, T_item = data.getrawdata()

        # converting the raw data into velocities units
        v_x, v_y, v_z = spacetime_to_velocities_converter(X_item, Y_item, T_item)

        # storing data in the DataFrame
        df_item = pd.DataFrame(
            {
                "run": (idx_file + 1)
                * np.ones(v_x.size),  # careful: "run" is a float here...
                "v_x": v_x,
                "v_y": v_y,
                "v_z": v_z,
            }
        )
        df_data = pd.concat([df_data, df_item], ignore_index=True)

    # we recast the ranks of the runs to genuine integers.
    df_data["run"] = df_data["run"].astype(int)
    # ----------------------------------------------------------------------------------

    # computation of the correlations:
    #
    # ----------------------------------------------------------------------------------

    # we prepare a series that will contain the number of atoms inside a given box
    # the initialization to the value "0" is important: by default we consider that the
    # atoms are not in the box, the method pd.DataFrame.mask() will toggle it to "1"
    # if the atom is indeed inside the box.
    df_data["in box"] = 0

    df_correlations = pd.DataFrame(
        list(itt.product(box_vz_array, box_vz_array)), columns=["vz box1", "vz box2"]
    )
    df_correlations["N box1 average"] = 0.0
    df_correlations["N box1 std"] = 0.0
    df_correlations["N box2 average"] = 0.0
    df_correlations["N box2 std"] = 0.0
    df_correlations["N1 x N2 average"] = 0.0
    df_correlations["N1 x N2 std"] = 0.0

    for idx1, box_1_vz in enumerate(box_vz_array):
        print(f"{idx1/len(box_vz_array)*100:.2f}%")

        df1 = df_data.copy()

        box1_parameters["center"]["v_z"] = box_1_vz
        box1 = box_mask(df1, box1_parameters)

        df1["in box"] = df1["in box"].mask(box1, 1)
        df1 = (
            df1.rename(columns={"in box": "N"})
            .groupby("run")
            .sum()
    #        .drop(["v_x", "v_z"], axis=1)
        )

        N_box1_av = df1["N"].mean()
        N_box1_std = df1["N"].std()

        for idx2, box_2_vz in enumerate(box_vz_array):

            df2 = df_data.copy()

            box2_parameters["center"]["v_z"] = box_2_vz
            box2 = box_mask(df2, box2_parameters)

            df2["in box"] = df2["in box"].mask(box2, 1)
            df2 = (
                df2.rename(columns={"in box": "N"})
                .groupby("run")
                .sum()
    #            .drop(["v_x", "v_z"], axis=1)
            )

            N_box2_av = df2["N"].mean()
            N_box2_std = df2["N"].std()

            product_av = (df1["N"] * df2["N"]).mean()
            product_std = (df1["N"] * df2["N"]).std()

            df_correlations.at[idx1 * n_boxes + idx2, "N box1 average"] = N_box1_av
            df_correlations.at[idx1 * n_boxes + idx2, "N box1 std"] = N_box1_std
            df_correlations.at[idx1 * n_boxes + idx2, "N box2 average"] = N_box2_av
            df_correlations.at[idx1 * n_boxes + idx2, "N box2 std"] = N_box2_std
            df_correlations.at[idx1 * n_boxes + idx2, "N1 x N2 average"] = product_av
            df_correlations.at[idx1 * n_boxes + idx2, "N1 x N2 std"] = product_std
    # ----------------------------------------------------------------------------------

    # computation of the g(2)
    # ----------------------------------------------------------------------------------

    # general case (crossed correlations)
    df_correlations["g2"] = df_correlations["N1 x N2 average"] / (
        df_correlations["N box1 average"] * df_correlations["N box2 average"]
    )
    # cleanup in case of boxes with 0 atoms on average...
    df_correlations = df_correlations.fillna(0)

    # local g(2) (we remove the shot noise)
    df_correlations.loc[
        df_correlations["vz box1"] == df_correlations["vz box2"], "g2"
    ] = (
        df_correlations.loc[
            df_correlations["vz box1"] == df_correlations["vz box2"], "N1 x N2 average"
        ]
        - df_correlations.loc[
            df_correlations["vz box1"] == df_correlations["vz box2"], "N box1 average"
        ]
    ) / (
        df_correlations.loc[
            df_correlations["vz box1"] == df_correlations["vz box2"], "N box1 average"
        ]
        * df_correlations.loc[
            df_correlations["vz box1"] == df_correlations["vz box2"], "N box2 average"
        ]
    )
    # ----------------------------------------------------------------------------------

    # Saving the results in CSV file
    # ----------------------------------------------------------------------------------
    root = Path.home()
    df_correlations.to_csv(root / "correlations.csv")
    # ----------------------------------------------------------------------------------

    # plotting
    # ----------------------------------------------------------------------------------
    df_pivoted_correlations = df_correlations.pivot(
        index="vz box1", columns="vz box2", values="g2"
    )

    fig, ax = plt.subplots(figsize=(13, 10))

    box_vz_array = np.linspace(vz_min, vz_max, n_boxes)

    num_ticks = 20
    # the index of the position of yticks
    yticks = np.linspace(0, len(box_vz_array) - 1, num_ticks, dtype=int)
    # the content of labels of these yticks
    ticklabels = [f"{int(box_vz_array[idx])}" for idx in yticks]

    sns.heatmap(
        df_pivoted_correlations,
        center=0,
        cmap="YlGnBu",
        ax=ax,
        vmin=0.5,
        vmax=3,
    )

    ax.set_xticks(yticks)
    ax.set_xticklabels(ticklabels)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ticklabels)

    ax.invert_yaxis()
    plt.show()
