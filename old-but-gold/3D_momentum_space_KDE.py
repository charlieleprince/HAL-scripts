# -*- coding: utf-8 -*-

from pathlib import Path
import json
import logging
from statistics import mode
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from mayavi import mlab
import numpy as np
import pandas as pd
from scipy import stats

from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache

logger = logging.getLogger(__name__)

NAME = "Plot KDE 3D momentum space"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


def main(self):

    root = Path().home()
    default_roi_dir = root / ".HAL"
    default_roi_file_name = default_roi_dir / "default_mcp_roi.json"

    with open(default_roi_file_name, encoding="utf8") as f:
        defaultroi = json.load(f)

    ROI0 = {}
    ROI0["Xmin"] = defaultroi["ROI 0"]["Xmin"]
    ROI0["Xmax"] = defaultroi["ROI 0"]["Xmax"]
    ROI0["Ymin"] = defaultroi["ROI 0"]["Ymin"]
    ROI0["Ymax"] = defaultroi["ROI 0"]["Ymax"]
    ROI0["Tmin"] = defaultroi["ROI 0"]["Tmin"]
    ROI0["Tmax"] = defaultroi["ROI 0"]["Tmax"]

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

    for k in range(len(selection) - 1):
        item = selection[k + 1]
        data.path = item.data(QtCore.Qt.UserRole)
        if not data.path.suffix == ".atoms":
            return
        # get data
        Xa, Ya, Ta = data.getrawdata()
        X = np.concatenate([X, Xa])
        Y = np.concatenate([Y, Ya])
        T = np.concatenate([T, Ta])

    atoms_df = pd.DataFrame({"X": X, "Y": Y, "T": T})
    atoms_df = atoms_df.loc[atoms_df["T"] > ROI0["Tmin"]]
    atoms_df = atoms_df.loc[atoms_df["T"] < ROI0["Tmax"]]
    atoms_df = atoms_df.loc[atoms_df["X"] > ROI0["Xmin"]]
    atoms_df = atoms_df.loc[atoms_df["X"] < ROI0["Xmax"]]
    atoms_df = atoms_df.loc[atoms_df["Y"] > ROI0["Ymin"]]
    atoms_df = atoms_df.loc[atoms_df["Y"] < ROI0["Ymax"]]

    values = np.array([atoms_df["X"], atoms_df["Y"], atoms_df["T"]])

    kde = stats.gaussian_kde(values)
    atoms_df["density"] = kde(values)

    # Create a regular 3D grid with 50 points in each dimension
    xmin, ymin, zmin = ROI0["Xmin"], ROI0["Ymin"], ROI0["Tmin"]
    xmax, ymax, zmax = ROI0["Xmax"], ROI0["Ymax"], ROI0["Tmax"]
    xi, yi, zi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:100j]

    # Evaluate the KDE on a regular grid...
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    density = kde(coords).reshape(xi.shape)

    # Visualize the density estimate as isosurfaces
    cont = mlab.figure()
    mlab.contour3d(
        xi,
        yi,
        zi,
        density,
        opacity=0.5,
        extent=[0, 1, 0, 1, 0, 1],
        contours=8,
        figure=cont,
    )
    mlab.axes()
    mlab.show()
