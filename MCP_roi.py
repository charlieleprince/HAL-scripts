# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
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
            sg.Input(size=(16, 1), key="Tmin1"),
            sg.Text("Tmax"),
            sg.Input(size=(16, 1), key="Tmax1"),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(size=(16, 1), key="Xmin1"),
            sg.Text("Xmax"),
            sg.Input(size=(16, 1), key="Xmax1"),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(size=(16, 1), key="Ymin1"),
            sg.Text("Ymax"),
            sg.Input(size=(16, 1), key="Ymax1"),
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
    list_of_rois = []
    ROI0 = {}
    while True:
        event, values = window.read()
        if (
            event == sg.WIN_CLOSED or event == "Cancel"
        ):  # if user closes window or clicks cancel
            break
        elif event == "Ok":
            # print("You entered ", values[0], values[1])
            ROI0_enabled = values["ROI0"]
            # print(values["Tmin0"])
            if ROI0_enabled:
                list_of_rois.append("0")
                ROI0["Tmin"] = float(values["Tmin0"])
                ROI0["Tmax"] = float(values["Tmax0"])
                ROI0["Xmin"] = float(values["Xmin0"])
                ROI0["Xmax"] = float(values["Xmax0"])
                ROI0["Ymin"] = float(values["Ymin0"])
                ROI0["Ymax"] = float(values["Ymax0"])
            break

    print(ROI0)

    window.close()

    ROI0_indexes = (
        (T > ROI0["Tmin"])
        & (T < ROI0["Tmax"])
        & (X > ROI0["Xmin"])
        & (X < ROI0["Xmax"])
        & (Y > ROI0["Ymin"])
        & (Y < ROI0["Ymax"])
    )

    remaining_indexes = (
        ~(T > ROI0["Tmin"])
        | ~(T < ROI0["Tmax"])
        | ~(X > ROI0["Xmin"])
        | ~(X < ROI0["Xmax"])
        | ~(Y > ROI0["Ymin"])
        | ~(Y < ROI0["Ymax"])
    )

    T_ROI = T[ROI0_indexes]
    X_ROI = X[ROI0_indexes]
    Y_ROI = Y[ROI0_indexes]
    T_remaining = T[remaining_indexes]
    X_remaining = X[remaining_indexes]
    Y_remaining = Y[remaining_indexes]

    fig1 = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(X_remaining, Y_remaining, T_remaining, marker=".")
    ax.scatter3D(X_ROI, Y_ROI, T_ROI, marker=".")
    plt.xlabel("X")
    plt.ylabel("Y")
    fig1.show()
