# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.optimize as opt
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "2. Plot histogram data from ROI only"  # display name, used in menubar and command palette
CATEGORY = "MCP - single file"  # category (note that CATEGORY="" is a valid choice)


def plot_unreconstructed_data(T_raw):

    bin_heights_raw, bin_borders_raw, _ = plt.hist(
        T_raw, bins=np.linspace(np.min(T_raw), np.max(T_raw), 300)
    )
    bin_centers_raw = bin_borders_raw[:-1] + np.diff(bin_borders_raw) / 2
    return (bin_centers_raw, bin_heights_raw)


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


def read_metadata(metadata, nb):
    Xmin = metadata["current selection"]["mcp"]["--ROI" + nb + ":Xmin"][0]
    Xmax = metadata["current selection"]["mcp"]["--ROI" + nb + ":Xmax"][0]
    Ymin = metadata["current selection"]["mcp"]["--ROI" + nb + ":Ymin"][0]
    Ymax = metadata["current selection"]["mcp"]["--ROI" + nb + ":Ymax"][0]
    Tmin = metadata["current selection"]["mcp"]["--ROI" + nb + ":Tmin"][0]
    Tmax = metadata["current selection"]["mcp"]["--ROI" + nb + ":Tmax"][0]
    return (Xmin, Xmax, Ymin, Ymax, Tmin, Tmax)


def addROI(metadata, ax, nb, color):
    (Xmin, Xmax, Ymin, Ymax, Tmin, Tmax) = read_metadata(metadata, nb)
    cube0 = np.array(
        [
            [Xmin, Ymin, Tmin],
            [Xmax, Ymin, Tmin],
            [Xmin, Ymax, Tmin],
            [Xmin, Ymin, Tmax],
            [Xmax, Ymax, Tmin],
            [Xmax, Ymin, Tmax],
            [Xmin, Ymax, Tmax],
            [Xmax, Ymax, Tmax],
        ]
    )
    hull = ConvexHull(cube0)

    for s in hull.simplices:
        tri = Poly3DCollection(cube0[s])
        tri.set_color(color)
        tri.set_alpha(1)
        ax.add_collection3d(tri)


def plotfigs(ax, X, Y, T, T_raw):
    (bin_centers_raw, bin_heights_raw) = plot_unreconstructed_data(T_raw)
    ax[0].hist2d(X, Y, bins=np.linspace(-40, 40, 2 * 81), cmap=plt.cm.jet)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[1].hist(
        T_raw, bins=np.linspace(np.min(T_raw), np.max(T_raw), 300), color="black"
    )
    ax[1].hist(T, bins=np.linspace(np.min(T), np.max(T), 300), color="tab:blue")
    # ax[1].plot(bin_centers_raw, bin_heights_raw, linestyle="dotted", color="black")
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("number of events")


def plot_XT_data(X, T):
    fig = plt.figure()
    plt.hist2d(
        X,
        T,
        bins=[np.linspace(-40, 40, 2 * 81), np.linspace(np.min(T), np.max(T), 2 * 81)],
        cmap=plt.cm.jet,
    )
    plt.xlabel("X")
    plt.ylabel("T")
    plt.colorbar()
    fig.show()


def plot_YT_data(Y, T):
    fig = plt.figure()
    plt.hist2d(
        Y,
        T,
        [np.linspace(-40, 40, 2 * 81), np.linspace(np.min(T), np.max(T), 2 * 81)],
        cmap=plt.cm.jet,
    )
    plt.xlabel("Y")
    plt.ylabel("T")
    plt.colorbar()
    fig.show()


def displayROIs(ax, color, metadata, ROI_name, nb):
    (Xmin, Xmax, Ymin, Ymax, Tmin, Tmax) = read_metadata(metadata, nb)
    rect_0_histo = patches.Rectangle(
        (Xmin, Ymin),
        Xmax - Xmin,
        Ymax - Ymin,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
    )
    ax[0].add_patch(rect_0_histo)
    ax[0].text(
        Xmin,
        Ymax,
        ROI_name,
        color=color,
    )
    ax[1].axvline(Tmin, linestyle="dotted", color=color)
    ax[1].axvline(Tmax, linestyle="dotted", color=color)
    ax[1].axvspan(Tmin, Tmax, alpha=0.2, color=color)


def ROIdata(metadata, nb, X, Y, T, T_raw):
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
    T_raw_ROI = T_raw[(T_raw > Tmin) & (T_raw < Tmax)]
    return (X_ROI, Y_ROI, T_ROI, T_raw_ROI)


def ROI_data(ROI, X, Y, T, T_raw):
    ROI_indices = (
        (T > ROI["Tmin"])
        & (T < ROI["Tmax"])
        & (X > ROI["Xmin"])
        & (X < ROI["Xmax"])
        & (Y > ROI["Ymin"])
        & (Y < ROI["Ymax"])
    )
    ROI_indices2 = (T_raw > ROI["Tmin"]) & (T_raw < ROI["Tmax"])
    T_ROI = T[ROI_indices]
    X_ROI = X[ROI_indices]
    Y_ROI = Y[ROI_indices]
    T_raw_ROI = T_raw[ROI_indices2]
    return (X_ROI, Y_ROI, T_ROI, T_raw_ROI)


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
    T_raw = data.getdatafromsingleline()
    if "N_ROI0" in metadata["current selection"]["mcp"]:
        (X_ROI, Y_ROI, T_ROI, T_raw_ROI) = ROIdata(metadata, "0", X, Y, T, T_raw)
        fig2D0, ax2D0 = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8)
        )
        plotfigs(ax2D0, X_ROI, Y_ROI, T_ROI, T_raw_ROI)
        plot_XT_data(X_ROI, T_ROI)
        plot_YT_data(Y_ROI, T_ROI)
        fig2D0.show()
    if "N_ROI1" in metadata["current selection"]["mcp"]:
        (X_ROI, Y_ROI, T_ROI) = ROIdata(metadata, "1", X, Y, T)
        fig2D1, ax2D1 = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8)
        )
        plotfigs(ax2D1, X_ROI, Y_ROI, T_ROI, T_raw_ROI)
        plot_XT_data(X_ROI, T_ROI)
        plot_YT_data(Y_ROI, T_ROI)
        fig2D1.show()
    if "N_ROI2" in metadata["current selection"]["mcp"]:
        (X_ROI, Y_ROI, T_ROI) = ROIdata(metadata, "2", X, Y, T)
        fig2D2, ax2D2 = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8)
        )
        plotfigs(ax2D2, X_ROI, Y_ROI, T_ROI, T_raw_ROI)
        plot_XT_data(X_ROI, T_ROI)
        plot_YT_data(Y_ROI, T_ROI)
        fig2D2.show()
    if "N_ROI3" in metadata["current selection"]["mcp"]:
        (X_ROI, Y_ROI, T_ROI) = ROIdata(metadata, "3", X, Y, T)
        fig2D3, ax2D3 = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8)
        )
        plotfigs(ax2D3, X_ROI, Y_ROI, T_ROI, T_raw_ROI)
        plot_XT_data(X_ROI, T_ROI)
        plot_YT_data(Y_ROI, T_ROI)
        fig2D3.show()

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

        (X_ROI, Y_ROI, T_ROI, T_raw_ROI) = ROI_data(ROI0, X, Y, T, T_raw)

        fig2D, ax2D = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8)
        )
        plotfigs(ax2D, X_ROI, Y_ROI, T_ROI, T_raw_ROI)
        fig2D.show()
        plot_XT_data(X_ROI, T_ROI)
        plot_YT_data(Y_ROI, T_ROI)

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
