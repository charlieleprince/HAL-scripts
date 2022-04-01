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
NAME = "0. MCP visualizer"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)

# layout tools
# sg.theme("DarkBlack")
sg.theme("LightGrey1")

# https://stackoverflow.com/questions/66662800/update-element-using-a-function-pysimplegui
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


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
    T_raw = data.getdatafromsingleline()
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

    # print(data.path)
    # print(data.path.parent)
    # print(list(data.path.parent.iterdir()))
    list_of_files = []

    currentDir = data.path.parent
    for currentFile in currentDir.iterdir():
        if currentFile.suffix == ".atoms":
            list_of_files.append(currentFile.stem)
    seq_dir = str(currentDir)
    seq_number = seq_dir.split("\\")[len(seq_dir.split("\\")) - 1]

    list_of_files.reverse()
    # gui layout

    fig2D, ax2D = plt.subplots(figsize=(6, 6))
    ax2D.hist2d(X, Y, bins=np.linspace(-40, 40, 2 * 81), cmap=plt.cm.jet)
    ax2D.set_xlabel("X")
    ax2D.set_ylabel("Y")
    ax2D.grid(True)

    fig1D, ax1D = plt.subplots(figsize=(6, 3))
    ax1D.hist(T_raw, bins=np.linspace(np.min(T_raw), np.max(T_raw), 300), color="black")
    ax1D.hist(T, bins=np.linspace(0, np.max(T), 300), color="tab:blue")
    # ax1D.plot(bin_centers_raw, bin_heights_raw, linestyle="dotted", color="black")
    ax1D.set_xlabel("time (ms)")
    ax1D.set_ylabel("number of events")

    data_col = [
        [
            sg.Checkbox(
                list_of_files[k],
                default=False,
                key=list_of_files[k],
            )
        ]
        for k in range(len(list_of_files))
    ]

    l2col1 = [
        [sg.Text("ROI selection", font="Helvetica 10 bold", justification="center")],
        [sg.Checkbox("Plot data from ROI", default=False, key="ROI0")],
        [
            sg.Text("Tmin"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Tmin"]), key="Tmin0"
            ),
            sg.Text("Tmax"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Tmax"]), key="Tmax0"
            ),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Xmin"]), key="Xmin0"
            ),
            sg.Text("Xmax"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Xmax"]), key="Xmax0"
            ),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Ymin"]), key="Ymin0"
            ),
            sg.Text("Ymax"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Ymax"]), key="Ymax0"
            ),
        ],
        [
            sg.Checkbox("Set to default", default=True, key="set to default"),
        ],
        [sg.Text("2D graph options", font="Helvetica 10 bold", justification="center")],
        [
            sg.Checkbox("XY", default=True, key="XY"),
            sg.Checkbox("XT", default=False, key="XT"),
            sg.Checkbox("YT", default=False, key="YT"),
            sg.Checkbox("Grid", default=True, key="grid"),
        ],
        [
            sg.Text("Number of bins"),
            sg.Input(size=(6, 1), default_text=160, key="bins2D"),
        ],
        [
            sg.Combo(
                ["colormap 1", "colormap 2", "colormap 3"],
                default_value="colormap 1",
                enable_events=True,
                key="combo",
            )
        ],
        [sg.Text("1D graph options", font="Helvetica 10 bold", justification="center")],
        [
            sg.Checkbox("T", default=True, key="T"),
            sg.Checkbox("X", default=False, key="X"),
            sg.Checkbox("Y", default=False, key="Y"),
            sg.Checkbox("Grid", default=True, key="grid1D"),
        ],
        [
            sg.Text("Number of bins"),
            sg.Input(size=(6, 1), default_text=300, key="bins1D"),
        ],
        [
            sg.Checkbox(
                "Max number of events", default=False, key="max events enabled"
            ),
            sg.Input(size=(6, 1), default_text=300, key="max events"),
        ],
        [
            sg.Checkbox(
                "Plot unreconstructed data", default=False, key="unreconstructed"
            )
        ],
        [sg.Checkbox("logscale", default=False, key="logscale")],
        [sg.Button("Update", button_color=("white", "green"))],
    ]

    l2col2 = [[sg.Canvas(key="-CANVAS-")]]

    l2col3 = [
        [sg.Checkbox("test", default=True, key="testkey", text_color="orange")],
    ]

    l1col1 = [[sg.Text("Bonjour")]]
    l1col2 = [[sg.Text("WORK IN PROGRESS")]]
    l1col3 = [[sg.Button("testbouton")]]
    # l3col1 = [[sg.Button("testbouton")]]
    l3col2 = [[sg.Canvas(key="-CANVAS2-")]]
    l3col3 = [
        [sg.Button("Open 2D graph"), sg.Button("Open 1D graph")],
        [
            sg.Button("Open 3D graph"),
        ],
    ]

    data_options_col = [
        [
            sg.Checkbox(
                "Select all",
                default=False,
                key="select all",
            )
        ],
        [
            sg.Checkbox(
                "Deselect all",
                default=False,
                key="deselect all",
            )
        ],
    ]

    l3col1 = [
        [
            sg.Button(
                "refresh",
                button_color=("white", "green"),
            ),
            sg.Text("Sequence"),
            sg.Input(size=(6, 1), default_text=seq_number, key="selected_seq"),
        ],
        [
            sg.Column(data_col, scrollable=True, vertical_scroll_only=True),
            sg.Column(data_options_col),
        ],
    ]

    layout = [
        [
            sg.Frame(layout=l1col1, title="", size=(400, 100)),
            sg.Frame(layout=l1col2, title="", size=(550, 100)),
            sg.Frame(layout=l1col3, title="Quote of the day", size=(300, 100)),
        ],
        [
            sg.Frame(layout=l2col1, title="Options", size=(400, 550)),
            sg.Frame(layout=l2col2, title="2D graph", size=(550, 550)),
            sg.Frame(layout=l2col3, title="qc3 parameters", size=(300, 550)),
        ],
        [
            sg.Frame(layout=l3col1, title="Data", size=(400, 250)),
            sg.Frame(layout=l3col2, title="1D graph", size=(550, 250)),
            sg.Frame(layout=l3col3, title="Export", size=(300, 250)),
        ],
    ]

    # Create the Window
    window = sg.Window("MCP visualizer tool", layout, finalize=True)

    # Associate fig with Canvas.
    fig_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig2D)
    fig_agg.draw()
    fig_agg = draw_figure(window["-CANVAS2-"].TKCanvas, fig1D)
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
                plotfigs(ax, X, Y, T, T_raw)
                displayROIs(ax, color, ROI0, "ROI::0", values, "0")
            if values["ROI1"]:
                setROIvalues(ROI1, values, "1")
                color = "tab:green"
                plotfigs(ax, X, Y, T, T_raw)
                displayROIs(ax, color, ROI1, "ROI::1", values, "1")
            if values["ROI2"]:
                setROIvalues(ROI2, values, "2")
                color = "tab:red"
                plotfigs(ax, X, Y, T, T_raw)
                displayROIs(ax, color, ROI2, "ROI::2", values, "2")
            if values["ROI3"]:
                setROIvalues(ROI3, values, "3")
                color = "tab:purple"
                plotfigs(ax, X, Y, T, T_raw)
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
                "display": "%.3g",
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
                    "display": "%.3g",
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
                    "display": "%.3g",
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
                    "display": "%.3g",
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
                    "display": "%.3g",
                    "unit": "",
                    "comment": "",
                }
            )

        if ROI0["enabled"] & ROI1["enabled"]:
            (X_ROI0, Y_ROI0, T_ROI0) = ROI_data(ROI0, X, Y, T)
            nb0 = len(X_ROI0)
            (X_ROI1, Y_ROI1, T_ROI1) = ROI_data(ROI1, X, Y, T)
            nb1 = len(X_ROI1)
            nb0norm = nb0 / (nb0 + nb1)
            to_mcp_dictionary.append(
                {
                    "name": "N_ROI0/(N_ROI0+N_ROI1)",
                    "value": nb0norm,
                    "display": "%.3g",
                    "unit": "",
                    "comment": "",
                }
            )

        MCP_stats_folder = data.path.parent / ".MCPstats"
        MCP_stats_folder.mkdir(exist_ok=True)
        file_name = MCP_stats_folder / data.path.stem
        with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
            json.dump(to_mcp_dictionary, file, ensure_ascii=False, indent=4)
