# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import scipy.optimize as opt
import PySimpleGUI as sg
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from pathlib import Path

import json

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "1. Choose ROI and export ROI info"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)

# layout tools


EMPTY_DICT = {
    "ROI 0": {
        "Xmin": 0,
        "Xmax": 0,
        "Ymin": 0,
        "Ymax": 0,
        "Tmin": 0,
        "Tmax": 0,
    },
    "ROI 1": {
        "Xmin": 0,
        "Xmax": 0,
        "Ymin": 0,
        "Ymax": 0,
        "Tmin": 0,
        "Tmax": 0,
    },
    "ROI 2": {
        "Xmin": 0,
        "Xmax": 0,
        "Ymin": 0,
        "Ymax": 0,
        "Tmin": 0,
        "Tmax": 0,
    },
    "ROI 3": {
        "Xmin": 0,
        "Xmax": 0,
        "Ymin": 0,
        "Ymax": 0,
        "Tmin": 0,
        "Tmax": 0,
    },
}


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def plotfigs(ax, X, Y, T):
    ax[0].hist2d(X, Y, bins=np.linspace(-40, 40, 2 * 81), cmap=plt.cm.jet)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].grid(True)
    ax[1].hist(T, bins=np.linspace(0, np.max(T), 300), color="tab:blue")
    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("number of events")


def isROIset(values, str):
    if values["Tmin" + str] == "":
        return False
    if values["Tmax" + str] == "":
        return False
    if values["Xmin" + str] == "":
        return False
    if values["Xmax" + str] == "":
        return False
    if values["Ymin" + str] == "":
        return False
    if values["Ymax" + str] == "":
        return False
    return True


def setROIvalues(dict, values, str):
    if isROIset(values, str) is False:
        return
    dict["Tmin"] = float(values["Tmin" + str])
    dict["Tmax"] = float(values["Tmax" + str])
    dict["Xmin"] = float(values["Xmin" + str])
    dict["Xmax"] = float(values["Xmax" + str])
    dict["Ymin"] = float(values["Ymin" + str])
    dict["Ymax"] = float(values["Ymax" + str])


def displayROIs(ax, color, ROI, ROI_name, values, str):
    if isROIset(values, str) is False:
        return
    rect_0_histo = patches.Rectangle(
        (ROI["Xmin"], ROI["Ymin"]),
        ROI["Xmax"] - ROI["Xmin"],
        ROI["Ymax"] - ROI["Ymin"],
        linewidth=2,
        edgecolor=color,
        facecolor="none",
    )
    ax[0].add_patch(rect_0_histo)
    ax[0].text(
        ROI["Xmin"],
        ROI["Ymax"],
        ROI_name,
        color=color,
    )
    ax[1].axvline(ROI["Tmin"], linestyle="dotted", color=color)
    ax[1].axvline(ROI["Tmax"], linestyle="dotted", color=color)
    ax[1].axvspan(ROI["Tmin"], ROI["Tmax"], alpha=0.2, color=color)


def get_enabled_rois(ROI0, ROI1, ROI2, ROI3, values):
    ROI0["enabled"] = values["ROI0"]
    ROI1["enabled"] = values["ROI1"]
    ROI2["enabled"] = values["ROI2"]
    ROI3["enabled"] = values["ROI3"]


def ROI_data(ROI, X, Y, T):
    ROI_indices = (
        (T > ROI["Tmin"])
        & (T < ROI["Tmax"])
        & (X > ROI["Xmin"])
        & (X < ROI["Xmax"])
        & (Y > ROI["Ymin"])
        & (Y < ROI["Ymax"])
    )
    T_ROI = T[ROI_indices]
    X_ROI = X[ROI_indices]
    Y_ROI = Y[ROI_indices]
    return (X_ROI, Y_ROI, T_ROI)


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


def _roi_changed(self):
    # move self.label
    position = self.pos()
    self.label.setPos(position[0], position[1])


# main
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
    if not data.path.suffix == ".atoms":
        return
    # get data
    X, Y, T = data.getrawdata()

    # default ROI

    root = Path().home()
    default_roi_dir = root / ".HAL"
    default_roi_file_name = default_roi_dir / "default_mcp_roi.json"

    if not default_roi_file_name.is_file():
        default_roi_dict = {
            "ROI 0": {"Xmin": 0, "Xmax": 0, "Ymin": 0, "Ymax": 0, "Tmin": 0, "Tmax": 0},
            "ROI 1": {"Xmin": 0, "Xmax": 0, "Ymin": 0, "Ymax": 0, "Tmin": 0, "Tmax": 0},
            "ROI 2": {"Xmin": 0, "Xmax": 0, "Ymin": 0, "Ymax": 0, "Tmin": 0, "Tmax": 0},
            "ROI 3": {"Xmin": 0, "Xmax": 0, "Ymin": 0, "Ymax": 0, "Tmin": 0, "Tmax": 0},
        }
        default_roi_file_name = default_roi_dir / "default_mcp_roi.json"
        with open(default_roi_file_name, "w", encoding="utf-8") as file:
            json.dump(default_roi_dict, file, ensure_ascii=False, indent=4)
    with open(default_roi_file_name, encoding="utf8") as f:
        defaultroi = json.load(f)

    # gui layout

    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8))
    plotfigs(ax, X, Y, T)

    col1 = [
        [sg.Checkbox("ROI::0", default=True, key="ROI0", text_color="orange")],
        [
            sg.Text("Tmin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 0"]["Tmin"]), key="Tmin0"
            ),
            sg.Text("Tmax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 0"]["Tmax"]), key="Tmax0"
            ),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 0"]["Xmin"]), key="Xmin0"
            ),
            sg.Text("Xmax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 0"]["Xmax"]), key="Xmax0"
            ),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 0"]["Ymin"]), key="Ymin0"
            ),
            sg.Text("Ymax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 0"]["Ymax"]), key="Ymax0"
            ),
        ],
        [sg.Checkbox("ROI::1", default=False, key="ROI1", text_color="darkgreen")],
        [
            sg.Text("Tmin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 1"]["Tmin"]), key="Tmin1"
            ),
            sg.Text("Tmax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 1"]["Tmax"]), key="Tmax1"
            ),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 1"]["Xmin"]), key="Xmin1"
            ),
            sg.Text("Xmax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 1"]["Xmax"]), key="Xmax1"
            ),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 1"]["Ymin"]), key="Ymin1"
            ),
            sg.Text("Ymax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 1"]["Ymax"]), key="Ymax1"
            ),
        ],
        [sg.Checkbox("ROI::2", default=False, key="ROI2", text_color="red")],
        [
            sg.Text("Tmin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 2"]["Tmin"]), key="Tmin2"
            ),
            sg.Text("Tmax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 2"]["Tmax"]), key="Tmax2"
            ),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 2"]["Xmin"]), key="Xmin2"
            ),
            sg.Text("Xmax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 2"]["Xmax"]), key="Xmax2"
            ),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 2"]["Ymin"]), key="Ymin2"
            ),
            sg.Text("Ymax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 2"]["Ymax"]), key="Ymax2"
            ),
        ],
        [sg.Checkbox("ROI::3", default=False, key="ROI3", text_color="purple")],
        [
            sg.Text("Tmin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 3"]["Tmin"]), key="Tmin3"
            ),
            sg.Text("Tmax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 3"]["Tmax"]), key="Tmax3"
            ),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 3"]["Xmin"]), key="Xmin3"
            ),
            sg.Text("Xmax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 3"]["Xmax"]), key="Xmax3"
            ),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 3"]["Ymin"]), key="Ymin3"
            ),
            sg.Text("Ymax"),
            sg.Input(
                size=(16, 1), default_text=str(defaultroi["ROI 3"]["Ymax"]), key="Ymax3"
            ),
        ],
        [
            sg.Button("Ok"),
            sg.Button("Watch ROIs"),
            sg.Button("Cancel"),
            sg.Checkbox("Set to default", default=True, key="set to default"),
        ],
    ]

    col2 = [[sg.Canvas(key="-CANVAS-")]]

    layout = [[sg.Frame(layout=col1, title=""), sg.Frame(layout=col2, title="")]]

    # Create the Window
    window = sg.Window("Welcome to the ROI manager", layout, finalize=True)

    # Associate fig with Canvas.
    fig_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)
    fig_agg.draw()

    # Initialize ROIs
    ROI0 = {}
    ROI0["enabled"] = False
    ROI1 = {}
    ROI1["enabled"] = False
    ROI2 = {}
    ROI2["enabled"] = False
    ROI3 = {}
    ROI3["enabled"] = False

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if (
            event == sg.WIN_CLOSED or event == "Cancel"
        ):  # if user closes window or clicks cancel
            break

        if event == "Watch ROIs":
            ax[0].cla()
            ax[1].cla()
            if values["ROI0"]:
                setROIvalues(ROI0, values, "0")
                color = "tab:orange"
                plotfigs(ax, X, Y, T)
                displayROIs(ax, color, ROI0, "ROI::0", values, "0")
            if values["ROI1"]:
                setROIvalues(ROI1, values, "1")
                color = "tab:green"
                plotfigs(ax, X, Y, T)
                displayROIs(ax, color, ROI1, "ROI::1", values, "1")
            if values["ROI2"]:
                setROIvalues(ROI2, values, "2")
                color = "tab:red"
                plotfigs(ax, X, Y, T)
                displayROIs(ax, color, ROI2, "ROI::2", values, "2")
            if values["ROI3"]:
                setROIvalues(ROI3, values, "3")
                color = "tab:purple"
                plotfigs(ax, X, Y, T)
                displayROIs(ax, color, ROI3, "ROI::3", values, "3")
            fig_agg.draw()

        if event == "Ok":
            if values["set to default"]:
                new_dict = EMPTY_DICT
                setROIvalues(new_dict["ROI 0"], values, "0")
                setROIvalues(new_dict["ROI 1"], values, "1")
                setROIvalues(new_dict["ROI 2"], values, "2")
                setROIvalues(new_dict["ROI 3"], values, "3")
                with open(default_roi_file_name, "w", encoding="utf-8") as file:
                    json.dump(new_dict, file, ensure_ascii=False, indent=4)
            get_enabled_rois(ROI0, ROI1, ROI2, ROI3, values)
            if ROI0["enabled"]:
                setROIvalues(ROI0, values, "0")
            if ROI1["enabled"]:
                setROIvalues(ROI1, values, "1")
            if ROI2["enabled"]:
                setROIvalues(ROI2, values, "2")
            if ROI3["enabled"]:
                setROIvalues(ROI3, values, "3")
            break
    plt.close()
    window.close()
    # now the ROIs are set

    for k in range(len(selection)):
        item = selection[k]
        data.path = item.data(QtCore.Qt.UserRole)
        if not data.path.suffix == ".atoms":
            return
        # get data
        X, Y, T = data.getrawdata()

        to_mcp_dictionary = []
        to_mcp_dictionary.append(
            {
                "name": "N_tot",
                "value": len(X),
                "display": "%o",
                "unit": "",
                "comment": "",
            }
        )

        if ROI0["enabled"]:
            (X_ROI0, Y_ROI0, T_ROI0) = ROI_data(ROI0, X, Y, T)

            exportROIinfo(to_mcp_dictionary, ROI0, 0)
            to_mcp_dictionary.append(
                {
                    "name": "N_ROI0",
                    "value": len(X_ROI0),
                    "display": "%o",
                    "unit": "",
                    "comment": "",
                }
            )

        if ROI1["enabled"]:
            (X_ROI1, Y_ROI1, T_ROI1) = ROI_data(ROI1, X, Y, T)
            exportROIinfo(to_mcp_dictionary, ROI1, 1)
            to_mcp_dictionary.append(
                {
                    "name": "N_ROI1",
                    "value": len(X_ROI1),
                    "display": "%o",
                    "unit": "",
                    "comment": "",
                }
            )
        if ROI2["enabled"]:
            (X_ROI2, Y_ROI2, T_ROI2) = ROI_data(ROI2, X, Y, T)
            exportROIinfo(to_mcp_dictionary, ROI2, 2)
            to_mcp_dictionary.append(
                {
                    "name": "N_ROI2",
                    "value": len(X_ROI2),
                    "display": "%o",
                    "unit": "",
                    "comment": "",
                }
            )
        if ROI3["enabled"]:
            (X_ROI3, Y_ROI3, T_ROI3) = ROI_data(ROI3, X, Y, T)
            exportROIinfo(to_mcp_dictionary, ROI3, 3)
            to_mcp_dictionary.append(
                {
                    "name": "N_ROI3",
                    "value": len(X_ROI3),
                    "display": "%o",
                    "unit": "",
                    "comment": "",
                }
            )
        MCP_stats_folder = data.path.parent / ".MCPstats"
        MCP_stats_folder.mkdir(exist_ok=True)
        file_name = MCP_stats_folder / data.path.stem
        with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
            json.dump(to_mcp_dictionary, file, ensure_ascii=False, indent=4)
