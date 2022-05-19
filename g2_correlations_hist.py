# -*- coding: utf-8 -*-

# IMPORTS
# --------------------------------------------------------------------------------------

# built-in python libs
from fileinput import filename
import logging
from itertools import product
from pathlib import Path
import json
from datetime import datetime

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
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache

# --------------------------------------------------------------------------------------

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "2D g(2) histograms"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)
logger = logging.getLogger(__name__)


def box_mask(dataframe: pd.DataFrame, parameters: dict) -> pd.Series:
    if parameters["mode"] == "cylindrical":
        box = (dataframe["v_x"] - parameters["center"]["v_x"]) ** 2 + (
            dataframe["v_y"] - parameters["center"]["v_y"]
        ) ** 2 < (parameters["size"]["diameter"] / 2) ** 2
    elif parameters["mode"] == "rectangle":
        box = (
            np.abs(dataframe["v_x"] - parameters["center"]["v_x"])
            < parameters["size"]["v_x"] / 2
        ) & (
            np.abs(dataframe["v_y"] - parameters["center"]["v_y"])
            < parameters["size"]["v_y"] / 2
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

    path_config = Path(self._user_scripts_folder) / "config" / "g2_correlations.json"

    with path_config.open("r") as config_file:
        dict_conf = json.load(config_file)

    vz_min = dict_conf["global"]["vz_min"]
    vz_max = dict_conf["global"]["vz_max"]
    v_z_height = dict_conf["global"]["v_z_height"]

    transverse_diameter = dict_conf["global"].get("transverse_diameter", None)

    box1_parameters = dict_conf["box1"]
    box2_parameters = dict_conf["box2"]

    if transverse_diameter:
        box1_parameters["size"]["diameter"] = transverse_diameter
        box2_parameters["size"]["diameter"] = transverse_diameter

    # preparation of the boxes Vz center values
    n_boxes = int(np.ceil((vz_max - vz_min) / v_z_height))
    bins_edges = np.linspace(vz_min, vz_max, n_boxes + 1)

    # We print some information about the chosen parameters for the calculations
    msg = (
        "----------------------------------------\n"
        "  INFORMATIONS ABOUT THE CALCULATION:\n"
        "----------------------------------------\n"
        "\n"
        f"Vz range: [{vz_min} mm/s, {vz_max} mm/s]\n"
        f"Temporal width of a box: {v_z_height}\n"
        f"Number of boxes per axis: {n_boxes}\n\n"
    )
    print(msg)
    print("  BOX 1 center and size:\n----------------------------------------\n")
    print(json.dumps(box1_parameters, indent=4))
    print("\n")
    print("  BOX 2 center and size:\n----------------------------------------\n")
    print(json.dumps(box2_parameters, indent=4))
    print("\n")
    # ----------------------------------------------------------------------------------

    # data loading:
    # all the files are successively opened and the data are loaded in the same large
    # DataFrame `df_data`. The memory of the run (= data file) is kept in the DataFrame
    # with the series `run`.
    # ----------------------------------------------------------------------------------

    # first we prepare a tag containing sequence infos (for naming the result file)
    metadata = getSelectionMetaDataFromCache(self)
    sequences_list = metadata["current selection"]["file"]["sequence"]
    dates_list = metadata["current selection"]["file"]["date"]
    sources_list = [
        date + "_seq" + str(sequences_list[i]) for i, date in enumerate(dates_list)
    ]
    tag = str(set(sources_list))

    # preparing the DataFrame
    df_data = pd.DataFrame(columns=["run", "v_x", "v_y", "v_z"])

    # actual data loading: loop over the `*.atoms` files
    for idx_file, item in enumerate(selection):
        data.path = item.data(Qt.UserRole)

        # get sequence info

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
        # first filter of the data to region of interest (accelerates the computation a lot)
        df_item = df_item.loc[(df_item["v_z"] < vz_max) & (df_item["v_z"] > vz_min)]
        df_data = pd.concat([df_data, df_item], ignore_index=True)

    # we recast the ranks of the runs to genuine integers.
    df_data["run"] = df_data["run"].astype(int)
    # ----------------------------------------------------------------------------------

    # computation of the correlations:
    # We call "axis" a column centered on given (Vx_c, Vy_c) coordinates, that we can
    # slice into n_boxes Vz boxes. We compute the temporal correlations between axis1
    # and axis2.
    # ----------------------------------------------------------------------------------

    axis1 = box_mask(df_data, box1_parameters)
    axis2 = box_mask(df_data, box2_parameters)

    df_axis1 = df_data.loc[axis1].reset_index(drop=True).drop(columns=["v_x", "v_y"])
    df_axis2 = df_data.loc[axis2].reset_index(drop=True).drop(columns=["v_x", "v_y"])

    df_histograms = pd.DataFrame(
        {
            "box1: number of atoms": df_axis1.groupby("run")["v_z"].apply(
                lambda series: np.histogram(series, bins_edges)[0]
            ),
            "box2: number of atoms": df_axis2.groupby("run")["v_z"].apply(
                lambda series: np.histogram(series, bins_edges)[0]
            ),
        }
    )
    df_histograms["Vz bin index"] = [np.arange(n_boxes) for _ in df_histograms.index]
    df_histograms = (
        df_histograms.reset_index()
        .explode(
            column=["box1: number of atoms", "box2: number of atoms", "Vz bin index"]
        )
        .reset_index(drop=True)
    )

    def couples(bin_indices):
        return list(product(bin_indices, bin_indices))

    def pop_box1(pop1, pop2):
        populations_products = list(product(pop1, pop2))
        return [pop[0] for pop in populations_products]

    def pop_box2(pop1, pop2):
        populations_products = list(product(pop1, pop2))
        return [pop[1] for pop in populations_products]

    def pop_product(pop1, pop2):
        populations_products = list(product(pop1, pop2))
        return [pop[0] * pop[1] for pop in populations_products]

    df_pairs = pd.DataFrame()

    df_pairs["Vz bins couple"] = df_histograms.groupby("run")["Vz bin index"].apply(
        couples
    )
    df_pairs["N1"] = df_histograms.groupby("run").apply(
        lambda df: pop_box1(df["box1: number of atoms"], df["box2: number of atoms"])
    )
    df_pairs["N2"] = df_histograms.groupby("run").apply(
        lambda df: pop_box2(df["box1: number of atoms"], df["box2: number of atoms"])
    )
    df_pairs["N1*N2"] = df_histograms.groupby("run").apply(
        lambda df: pop_product(df["box1: number of atoms"], df["box2: number of atoms"])
    )

    df_pairs = pd.DataFrame(
        df_pairs.explode(["Vz bins couple", "N1", "N2", "N1*N2"], ignore_index=True)
    )

    # print(df_pairs)

    df_pairs[["bin1 index", "bin2 index"]] = pd.DataFrame(
        df_pairs["Vz bins couple"].tolist(), index=df_pairs.index
    )
    df_pairs = df_pairs.drop(columns=["Vz bins couple"])

    averages_axis1 = df_pairs.groupby("bin1 index")["N1"].mean()
    averages_axis2 = df_pairs.groupby("bin2 index")["N2"].mean()

    df_prod = (
        pd.DataFrame(df_pairs.groupby(["bin1 index", "bin2 index"])["N1*N2"].mean())
        .reset_index()
        .rename(columns={"N1*N2": "<N1*N2>"})
    )

    df_prod["<N1>"] = np.array(
        list(zip(*[list(averages_axis1) for _ in range(n_boxes)]))
    ).flatten()

    df_prod["<N2>"] = list(pd.concat([averages_axis2] * (n_boxes)))

    df_prod["g2"] = df_prod["<N1*N2>"] / (df_prod["<N1>"] * df_prod["<N2>"])

    condition = df_prod["bin1 index"] == df_prod["bin2 index"]
    df_prod.loc[condition, "g2"] = (
        df_prod.loc[condition, "<N1*N2>"]
        - 0.5 * (df_prod.loc[condition, "<N1>"] + df_prod.loc[condition, "<N2>"])
    ) / (df_prod.loc[condition, "<N1>"] * df_prod.loc[condition, "<N2>"])

    df_prod_piv = df_prod.pivot(index="bin1 index", columns="bin2 index", values="g2")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_prod_piv, vmin=0.8, vmax=1.5)
    box_vz_array = np.linspace(vz_min, vz_max, n_boxes)
    num_ticks = 20
    # the index of the position of yticks
    ticks = np.linspace(0, len(box_vz_array) - 1, num_ticks, dtype=int)
    # the content of labels of these yticks
    ticklabels = [f"{int(box_vz_array[idx])}" for idx in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.set_xlabel("Vz box1 (mm/s)")
    ax.set_xlabel("Vz box2 (mm/s)")
    ax.invert_yaxis()
    plt.show()
