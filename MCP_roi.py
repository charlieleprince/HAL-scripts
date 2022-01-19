# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.optimize as opt
import PySimpleGUI as sg
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "ROI"  # display name, used in menubar and command palette
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

    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8))
    ax[0].hist2d(X, Y, bins=np.linspace(-40, 40, 2 * 81), cmap=plt.cm.jet)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[1].hist(T, bins=np.linspace(0, 180, 300))
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("number of events")
    fig.show()

    nb_of_ROIs = 0
    layout = [
        [sg.Checkbox("ROI::0", default=True, key="ROI0", text_color="orange")],
        [
            sg.Text("Tmin"),
            sg.Input(size=(16, 1), default_text="150", key="Tmin0"),
            sg.Text("Tmax"),
            sg.Input(size=(16, 1), default_text="165", key="Tmax0"),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(size=(16, 1), default_text="-15", key="Xmin0"),
            sg.Text("Xmax"),
            sg.Input(size=(16, 1), default_text="-10", key="Xmax0"),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(size=(16, 1), default_text="-12", key="Ymin0"),
            sg.Text("Ymax"),
            sg.Input(size=(16, 1), default_text="15", key="Ymax0"),
        ],
        [sg.Checkbox("ROI::1", default=False, key="ROI1", text_color="darkgreen")],
        [
            sg.Text("Tmin"),
            sg.Input(size=(16, 1), default_text="100", key="Tmin1"),
            sg.Text("Tmax"),
            sg.Input(size=(16, 1), default_text="125", key="Tmax1"),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(size=(16, 1), default_text="-10", key="Xmin1"),
            sg.Text("Xmax"),
            sg.Input(size=(16, 1), default_text="0", key="Xmax1"),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(size=(16, 1), default_text="-30", key="Ymin1"),
            sg.Text("Ymax"),
            sg.Input(size=(16, 1), default_text="-20", key="Ymax1"),
        ],
        [sg.Checkbox("ROI::2", default=False, key="ROI2", text_color="red")],
        [
            sg.Text("Tmin"),
            sg.Input(size=(16, 1), key="Tmin2"),
            sg.Text("Tmax"),
            sg.Input(size=(16, 1), key="Tmax2"),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(size=(16, 1), key="Xmin2"),
            sg.Text("Xmax"),
            sg.Input(size=(16, 1), key="Xmax2"),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(size=(16, 1), key="Ymin2"),
            sg.Text("Ymax"),
            sg.Input(size=(16, 1), key="Ymax2"),
        ],
        [sg.Checkbox("ROI::3", default=False, key="ROI3", text_color="purple")],
        [
            sg.Text("Tmin"),
            sg.Input(size=(16, 1), key="Tmin3"),
            sg.Text("Tmax"),
            sg.Input(size=(16, 1), key="Tmax3"),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(size=(16, 1), key="Xmin3"),
            sg.Text("Xmax"),
            sg.Input(size=(16, 1), key="Xmax3"),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(size=(16, 1), key="Ymin3"),
            sg.Text("Ymax"),
            sg.Input(size=(16, 1), key="Ymax3"),
        ],
        [sg.Button("Ok"), sg.Button("Cancel")],
    ]

    # Create the Window
    window = sg.Window("Welcome to the ROI layout", layout)
    # Event Loop to process "events" and get the "values" of the inputs
    # test = 0
    ROI0 = {}
    ROI1 = {}
    ROI2 = {}
    ROI3 = {}
    while True:
        event, values = window.read()
        if (
            event == sg.WIN_CLOSED or event == "Cancel"
        ):  # if user closes window or clicks cancel
            break
        elif event == "Ok":
            ROI0["enabled"] = values["ROI0"]
            ROI1["enabled"] = values["ROI1"]
            ROI2["enabled"] = values["ROI2"]
            ROI3["enabled"] = values["ROI3"]

            if ROI0["enabled"]:
                ROI0["Tmin"] = float(values["Tmin0"])
                ROI0["Tmax"] = float(values["Tmax0"])
                ROI0["Xmin"] = float(values["Xmin0"])
                ROI0["Xmax"] = float(values["Xmax0"])
                ROI0["Ymin"] = float(values["Ymin0"])
                ROI0["Ymax"] = float(values["Ymax0"])
            if ROI1["enabled"]:
                ROI1["Tmin"] = float(values["Tmin1"])
                ROI1["Tmax"] = float(values["Tmax1"])
                ROI1["Xmin"] = float(values["Xmin1"])
                ROI1["Xmax"] = float(values["Xmax1"])
                ROI1["Ymin"] = float(values["Ymin1"])
                ROI1["Ymax"] = float(values["Ymax1"])
            break

    window.close()
    plt.close(fig)
    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8))
    T_remaining = T
    X_remaining = X
    Y_remaining = Y

    if ROI0["enabled"]:
        color = "tab:orange"
        rect_0_histo = patches.Rectangle(
            (ROI0["Xmin"], ROI0["Ymin"]),
            ROI0["Xmax"] - ROI0["Xmin"],
            ROI0["Ymax"] - ROI0["Ymin"],
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax[0].add_patch(rect_0_histo)
        ax[0].text(
            ROI0["Xmin"],
            ROI0["Ymax"],
            "ROI::0",
            color=color,
        )

        ax[1].axvline(ROI0["Tmin"], linestyle="dotted", color=color)
        ax[1].axvline(ROI0["Tmax"], linestyle="dotted", color=color)
        ax[1].axvspan(ROI0["Tmin"], ROI0["Tmax"], alpha=0.2, color=color)

        ROI0_indexes = (
            (T_remaining > ROI0["Tmin"])
            & (T_remaining < ROI0["Tmax"])
            & (X_remaining > ROI0["Xmin"])
            & (X_remaining < ROI0["Xmax"])
            & (Y_remaining > ROI0["Ymin"])
            & (Y_remaining < ROI0["Ymax"])
        )

        remaining_indexes = (
            ~(T_remaining > ROI0["Tmin"])
            | ~(T_remaining < ROI0["Tmax"])
            | ~(X_remaining > ROI0["Xmin"])
            | ~(X_remaining < ROI0["Xmax"])
            | ~(Y_remaining > ROI0["Ymin"])
            | ~(Y_remaining < ROI0["Ymax"])
        )

        T_ROI0 = T_remaining[ROI0_indexes]
        X_ROI0 = X_remaining[ROI0_indexes]
        Y_ROI0 = Y_remaining[ROI0_indexes]
        T_remaining = T_remaining[remaining_indexes]
        X_remaining = X_remaining[remaining_indexes]
        Y_remaining = Y_remaining[remaining_indexes]

    if ROI1["enabled"]:
        color = "tab:green"
        rect_1_histo = patches.Rectangle(
            (ROI1["Xmin"], ROI1["Ymin"]),
            ROI1["Xmax"] - ROI1["Xmin"],
            ROI1["Ymax"] - ROI1["Ymin"],
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax[0].add_patch(rect_1_histo)
        ax[0].text(
            ROI1["Xmin"],
            ROI1["Ymax"],
            "ROI::1",
            color=color,
        )

        ax[1].axvline(ROI1["Tmin"], linestyle="dotted", color=color)
        ax[1].axvline(ROI1["Tmax"], linestyle="dotted", color=color)
        ax[1].axvspan(ROI1["Tmin"], ROI1["Tmax"], alpha=0.2, color=color)

        ROI1_indexes = (
            (T_remaining > ROI1["Tmin"])
            & (T_remaining < ROI1["Tmax"])
            & (X_remaining > ROI1["Xmin"])
            & (X_remaining < ROI1["Xmax"])
            & (Y_remaining > ROI1["Ymin"])
            & (Y_remaining < ROI1["Ymax"])
        )

        remaining_indexes = (
            ~(T_remaining > ROI1["Tmin"])
            | ~(T_remaining < ROI1["Tmax"])
            | ~(X_remaining > ROI1["Xmin"])
            | ~(X_remaining < ROI1["Xmax"])
            | ~(Y_remaining > ROI1["Ymin"])
            | ~(Y_remaining < ROI1["Ymax"])
        )

        T_ROI1 = T_remaining[ROI1_indexes]
        X_ROI1 = X_remaining[ROI1_indexes]
        Y_ROI1 = Y_remaining[ROI1_indexes]
        T_remaining = T_remaining[remaining_indexes]
        X_remaining = X_remaining[remaining_indexes]
        Y_remaining = Y_remaining[remaining_indexes]
    ax[0].hist2d(X, Y, bins=np.linspace(-40, 40, 2 * 81), cmap=plt.cm.jet)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[1].hist(T, bins=np.linspace(0, 180, 300))
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("number of events")
    fig.show()
    fig1 = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(X_remaining, Y_remaining, T_remaining, marker=".")
    if ROI0["enabled"]:
        ax.scatter3D(X_ROI0, Y_ROI0, T_ROI0, marker=".")
    if ROI1["enabled"]:
        ax.scatter3D(X_ROI1, Y_ROI1, T_ROI1, marker=".")
    plt.xlabel("X")
    plt.ylabel("Y")
    fig1.show()
