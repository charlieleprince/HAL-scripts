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
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "watch ROI"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


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
    X, Y, T = data.getrawdata()

    fig1 = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(X, Y, T, marker=".")
    if "N_ROI0" in metadata["current selection"]["mcp"]:
        Xmin = metadata["current selection"]["mcp"]["--ROI0:Xmin"][0]
        Xmax = metadata["current selection"]["mcp"]["--ROI0:Xmax"][0]
        Ymin = metadata["current selection"]["mcp"]["--ROI0:Ymin"][0]
        Ymax = metadata["current selection"]["mcp"]["--ROI0:Ymax"][0]
        Tmin = metadata["current selection"]["mcp"]["--ROI0:Tmin"][0]
        Tmax = metadata["current selection"]["mcp"]["--ROI0:Tmax"][0]
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
            tri.set_color("tab:orange")
            tri.set_alpha(1)
            ax.add_collection3d(tri)
    if "N_ROI1" in metadata["current selection"]["mcp"]:
        Xmin = metadata["current selection"]["mcp"]["--ROI1:Xmin"][0]
        Xmax = metadata["current selection"]["mcp"]["--ROI1:Xmax"][0]
        Ymin = metadata["current selection"]["mcp"]["--ROI1:Ymin"][0]
        Ymax = metadata["current selection"]["mcp"]["--ROI1:Ymax"][0]
        Tmin = metadata["current selection"]["mcp"]["--ROI1:Tmin"][0]
        Tmax = metadata["current selection"]["mcp"]["--ROI1:Tmax"][0]
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
            tri.set_color("tab:green")
            tri.set_alpha(1)
            ax.add_collection3d(tri)
    if "N_ROI2" in metadata["current selection"]["mcp"]:
        Xmin = metadata["current selection"]["mcp"]["--ROI2:Xmin"][0]
        Xmax = metadata["current selection"]["mcp"]["--ROI2:Xmax"][0]
        Ymin = metadata["current selection"]["mcp"]["--ROI2:Ymin"][0]
        Ymax = metadata["current selection"]["mcp"]["--ROI2:Ymax"][0]
        Tmin = metadata["current selection"]["mcp"]["--ROI2:Tmin"][0]
        Tmax = metadata["current selection"]["mcp"]["--ROI2:Tmax"][0]
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
            tri.set_color("tab:red")
            tri.set_alpha(1)
            ax.add_collection3d(tri)
    if "N_ROI3" in metadata["current selection"]["mcp"]:
        Xmin = metadata["current selection"]["mcp"]["--ROI3:Xmin"][0]
        Xmax = metadata["current selection"]["mcp"]["--ROI3:Xmax"][0]
        Ymin = metadata["current selection"]["mcp"]["--ROI3:Ymin"][0]
        Ymax = metadata["current selection"]["mcp"]["--ROI3:Ymax"][0]
        Tmin = metadata["current selection"]["mcp"]["--ROI3:Tmin"][0]
        Tmax = metadata["current selection"]["mcp"]["--ROI3:Tmax"][0]
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
            tri.set_color("tab:purple")
            tri.set_alpha(1)
            ax.add_collection3d(tri)
    plt.xlabel("X")
    plt.ylabel("Y")
    fig1.show()
