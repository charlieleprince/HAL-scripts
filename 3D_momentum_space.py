# -*- coding: utf-8 -*-

from pathlib import Path
import json
import logging
from statistics import mode
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import scipy.optimize as opt
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache

logger = logging.getLogger(__name__)

NAME = "Plot 3D momentum space"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)


def main(self):

    root = Path().home()
    default_roi_dir = root / ".HAL"
    default_roi_file_name = default_roi_dir / "default_mcp_roi.json"

    if not default_roi_file_name.is_file():
        default_roi_dict = {
            "ROI 0": {
                "Xmin": None,
                "Xmax": None,
                "Ymin": None,
                "Ymax": None,
                "Tmin": None,
                "Tmax": None,
            },
            "ROI 1": {
                "Xmin": None,
                "Xmax": None,
                "Ymin": None,
                "Ymax": None,
                "Tmin": None,
                "Tmax": None,
            },
            "ROI 2": {
                "Xmin": None,
                "Xmax": None,
                "Ymin": None,
                "Ymax": None,
                "Tmin": None,
                "Tmax": None,
            },
            "ROI 3": {
                "Xmin": None,
                "Xmax": None,
                "Ymin": None,
                "Ymax": None,
                "Tmin": None,
                "Tmax": None,
            },
        }
        default_roi_file_name = default_roi_dir / "default_mcp_roi.json"
        with open(default_roi_file_name, "w", encoding="utf-8") as file:
            json.dump(default_roi_dict, file, ensure_ascii=False, indent=4)
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
    size = np.ones(len(T))

    atoms_df = pd.DataFrame({"X": X, "Y": Y, "T": T})

    x_bin = np.linspace(atoms_df["X"].min(), atoms_df["X"].max(), 100)
    y_bin = np.linspace(atoms_df["Y"].min(), atoms_df["Y"].max(), 100)
    t_bin = np.linspace(290, 325, 100)

    groups = atoms_df.groupby(np.digitize(atoms_df.X, x_bin))
    print(groups.size())
    # atoms_binned_df = {"X": }

    fig = px.scatter_3d(
        atoms_df,
        x="X",
        y="Y",
        z="T",
        # mode="markers",
        size=size,
        color=T,  # set color to an array/list of desired values
        # colorscale="Viridis",  # choose a colorscale
        opacity=0.7,
    )
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

    print(atoms_df)
