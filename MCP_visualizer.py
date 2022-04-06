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
import pickle
import io
import json

# https://pysimplegui.readthedocs.io/en/latest/#persistent-window-example-running-timer-that-updates
logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "0. MCP visualizer"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)

# layout tools
# sg.theme("DarkBlack")
sg.theme("LightGrey1")


def getrawdata(path):
    """loads data"""
    time_resolution = 1.2e-10
    time_to_pos = 2 * 0.98e-9
    atoms_file = np.fromfile(path, dtype="uint64")
    times_file_path = str(path.parent) + "/" + str(path.stem) + ".times"
    times_file = np.fromfile(times_file_path, dtype="uint64")
    times = times_file * time_resolution
    atoms = atoms_file * time_resolution

    events_list = atoms.reshape(int(len(atoms) / 4), 4).T

    Xmcp = (events_list[1] - events_list[0]) / time_to_pos
    Ymcp = (events_list[3] - events_list[2]) / time_to_pos

    X = (Xmcp + Ymcp) / np.sqrt(2)
    Y = (Ymcp - Xmcp) / np.sqrt(2)
    T = (events_list[0] + events_list[1] + events_list[2] + events_list[3]) / 4

    T = T * 1e3
    T_raw = times * 1e3
    return (X, Y, T, T_raw)


# https://stackoverflow.com/questions/66662800/update-element-using-a-function-pysimplegui
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def show_figure2D(fig):
    # create a dummy figure and use its
    # manager to display "fig"
    dummy = plt.figure(figsize=(6, 6))
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


def show_figure1D(fig):
    # create a dummy figure and use its
    # manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


def generate_list(prefix, nbfiles):
    u = []
    for i in range(nbfiles):
        k = i + 1
        if len(str(k)) == 1:
            u.append(prefix + "_00" + str(k))
        elif len(str(k)) == 2:
            u.append(prefix + "_0" + str(k))
        else:
            u.append(prefix + "_" + str(k))
        # u.reverse()
    return u


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


def update_plot(values, X, Y, T, T_raw, ax1D, fig_agg1D, ax2D, fig_agg2D, nb_of_cycles):
    cmaps = [name for name in plt.colormaps() if not name.endswith("_r")]
    if values["ROI0"]:
        Xdata, Ydata, Tdata = X, Y, T
        ROI_dict = {}
        ROI_dict["Tmin"] = float(values["Tmin"])
        ROI_dict["Tmax"] = float(values["Tmax"])
        ROI_dict["Xmin"] = float(values["Xmin"])
        ROI_dict["Xmax"] = float(values["Xmax"])
        ROI_dict["Ymin"] = float(values["Ymin"])
        ROI_dict["Ymax"] = float(values["Ymax"])
        (X, Y, T) = ROI_data(ROI_dict, X, Y, T)

    ax1D.cla()
    ax2D.cla()

    bins = int(values["bins1D"])
    if values["grid1D"]:
        ax1D.grid(True)
    if values["T"]:
        if values["unreconstructed"]:
            bin_heights, bin_borders, _ = plt.hist(
                T_raw, bins=np.linspace(np.min(T_raw), np.max(T_raw), bins)
            )
            plt.close()
            widths = np.diff(bin_borders)
            ax1D.bar(bin_borders[:-1], bin_heights, widths, color="black")
        bin_heights, bin_borders, _ = plt.hist(
            T, bins=np.linspace(np.min(T), np.max(T), bins)
        )
        plt.close()
        widths = np.diff(bin_borders)
        bin_heights = np.array(bin_heights) / nb_of_cycles
        ax1D.bar(bin_borders[:-1], bin_heights, widths)

        # ax1D.hist(T, bins=np.linspace(np.min(T), np.max(T), bins), color="tab:blue")
        if values["ROI0"]:
            ax1D.set_xlim(float(values["Tmin"]), float(values["Tmax"]))
        if not values["ROI0"]:
            ax1D.set_xlim(np.min(T), np.max(T))
        ax1D.set_xlabel("time (ms)")
        ax1D.set_ylabel("number of events")
    if values["X"]:
        bin_heights, bin_borders, _ = plt.hist(X, bins=np.linspace(-40, 40, bins))
        plt.close()
        widths = np.diff(bin_borders)
        bin_heights = np.array(bin_heights) / nb_of_cycles
        ax1D.bar(bin_borders[:-1], bin_heights, widths, color="tab:blue")
        # ax1D.hist(X, bins=np.linspace(-40, 40, bins), color="tab:blue")
        ax1D.set_xlim(-40, 40)
        ax1D.set_xlabel("X (mm)")
        ax1D.set_ylabel("number of events")
    if values["Y"]:
        bin_heights, bin_borders, _ = plt.hist(Y, bins=np.linspace(-40, 40, bins))
        plt.close()
        widths = np.diff(bin_borders)
        bin_heights = np.array(bin_heights) / nb_of_cycles
        ax1D.bar(bin_borders[:-1], bin_heights, widths, color="tab:blue")
        ax1D.set_xlabel("Y (mm)")
        ax1D.set_xlim(-40, 40)
        ax1D.set_ylabel("number of events")
    if values["max events enabled"]:
        ax1D.set_ylim(0, float(values["max events"]))
    if values["logscale"]:
        ax1D.set_yscale("log")

    if not values["colormap"] in cmaps:
        return
    cmap = plt.get_cmap(values["colormap"])
    if values["XY"]:
        ax2D.hist2d(
            X,
            Y,
            bins=np.linspace(-40, 40, int(values["bins2D"])),
            cmap=cmap,
        )
        ax2D.set_xlabel("X")
        ax2D.set_ylabel("Y")
    if values["XT"]:

        if values["ROI0"]:
            ax2D.set_ylim(float(values["Tmin"]), float(values["Tmax"]))
            ax2D.hist2d(
                X,
                T,
                bins=[
                    np.linspace(-40, 40, int(values["bins2D"])),
                    np.linspace(
                        float(values["Tmin"]),
                        float(values["Tmax"]),
                        int(values["bins2D"]),
                    ),
                ],
                cmap=cmap,
            )
        if not values["ROI0"]:
            ax2D.hist2d(
                X,
                T,
                bins=[
                    np.linspace(-40, 40, int(values["bins2D"])),
                    np.linspace(np.min(T), np.max(T), int(values["bins2D"])),
                ],
                cmap=cmap,
            )
        ax2D.set_xlabel("X")
        ax2D.set_ylabel("T")
    if values["YT"]:

        if values["ROI0"]:
            ax2D.set_ylim(float(values["Tmin"]), float(values["Tmax"]))
            ax2D.hist2d(
                Y,
                T,
                bins=[
                    np.linspace(-40, 40, int(values["bins2D"])),
                    np.linspace(
                        float(values["Tmin"]),
                        float(values["Tmax"]),
                        int(values["bins2D"]),
                    ),
                ],
                cmap=cmap,
            )
        if not values["ROI0"]:
            ax2D.hist2d(
                Y,
                T,
                bins=[
                    np.linspace(-40, 40, int(values["bins2D"])),
                    np.linspace(np.min(T), np.max(T), int(values["bins2D"])),
                ],
                cmap=cmap,
            )

        ax2D.set_xlabel("Y")
        ax2D.set_ylabel("T")
    if values["grid2D"]:
        ax2D.grid(True)

    fig_agg1D.draw()
    fig_agg2D.draw()

    if values["ROI0"]:
        X, Y, T = Xdata, Ydata, Tdata


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

    list_of_files = []
    total_cycles = 1

    currentDir = data.path.parent
    for currentFile in currentDir.iterdir():
        if currentFile.suffix == ".atoms":
            list_of_files.append(currentFile.stem)
    seq_dir = str(currentDir)
    seq_number = str(currentDir.name)

    list_of_files.reverse()
    # gui layout
    cmaps = [name for name in plt.colormaps() if not name.endswith("_r")]
    fig2D, ax2D = plt.subplots(figsize=(6, 6))
    ax2D.hist2d(X, Y, bins=np.linspace(-40, 40, 160), cmap=plt.cm.jet)
    ax2D.set_xlabel("X")
    ax2D.set_ylabel("Y")

    fig1D, ax1D = plt.subplots(figsize=(6, 3))
    # ax1D.hist(T_raw, bins=np.linspace(np.min(T_raw), np.max(T_raw), 300), color="black")
    bin_heights, bin_borders, _ = plt.hist(T, bins=np.linspace(0, np.max(T), 300))
    plt.close()
    widths = np.diff(bin_borders)
    ax1D.bar(bin_borders[:-1], bin_heights, widths, color="tab:blue")
    # ax1D.hist(T, bins=np.linspace(0, np.max(T), 300), color="tab:blue")
    # ax1D.plot(bin_centers_raw, bin_heights_raw, linestyle="dotted", color="black")
    ax1D.set_xlim(np.min(T), np.max(T))
    ax1D.set_xlabel("time (ms)")
    ax1D.set_ylabel("number of events")

    nbfiles = 100
    all_buttons = generate_list(seq_number, nbfiles)

    cycles_in_seq = len(list_of_files)
    data_buttons = []
    for k in range(len(all_buttons)):
        if all_buttons[k] in list_of_files:
            data_buttons.append(
                [sg.Button(all_buttons[k], key=all_buttons[k][4:], visible=True)]
            )
        else:
            data_buttons.append(
                [
                    sg.Button(
                        all_buttons[k],
                        key=all_buttons[k][4:],
                        visible=False,
                    )
                ]
            )
    data_buttons.reverse()
    data_col = data_buttons

    qc3 = ""
    parameters_file = data.path.parent / (data.path.stem + ".json")
    if parameters_file.is_file():
        with open(parameters_file, encoding="utf-8") as file:
            sequence_parameters = json.load(file)
        for k in range(len(sequence_parameters)):
            qc3 += (
                sequence_parameters[k]["name"]
                + " : "
                + str(np.round(sequence_parameters[k]["value"], 3))
                + "\n"
            )

    l2col1 = [
        [sg.Text("ROI selection", font="Helvetica 10 bold", justification="center")],
        [sg.Checkbox("Plot data from ROI", default=False, key="ROI0")],
        [
            sg.Text("Tmin"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Tmin"]), key="Tmin"
            ),
            sg.Text("Tmax"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Tmax"]), key="Tmax"
            ),
        ],
        [
            sg.Text("Xmin"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Xmin"]), key="Xmin"
            ),
            sg.Text("Xmax"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Xmax"]), key="Xmax"
            ),
        ],
        [
            sg.Text("Ymin"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Ymin"]), key="Ymin"
            ),
            sg.Text("Ymax"),
            sg.Input(
                size=(6, 1), default_text=str(defaultroi["ROI 0"]["Ymax"]), key="Ymax"
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
        ],
        [
            sg.Text("Number of bins"),
            sg.Input(size=(6, 1), default_text=160, key="bins2D"),
            sg.Checkbox("Grid", default=False, key="grid2D"),
        ],
        [
            sg.Combo(
                cmaps,
                default_value="jet",
                enable_events=True,
                key="colormap",
            )
        ],
        [sg.Text("1D graph options", font="Helvetica 10 bold", justification="center")],
        [
            sg.Checkbox("T", default=True, key="T"),
            sg.Checkbox("X", default=False, key="X"),
            sg.Checkbox("Y", default=False, key="Y"),
            sg.Checkbox("logscale", default=False, key="logscale"),
        ],
        [
            sg.Text("Number of bins"),
            sg.Input(size=(6, 1), default_text=300, key="bins1D"),
            sg.Checkbox("Grid", default=False, key="grid1D"),
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
        [sg.Button("Update", button_color=("white", "green"), key="update")],
    ]

    l2col2 = [[sg.Canvas(key="-CANVAS-")]]

    qc3col = [[sg.Text(qc3, font="Helvetica 10", key="qc3params")]]

    l2col3 = [
        [
            sg.Column(
                qc3col,
                size=(300, 550),
                scrollable=True,
                # vertical_scroll_only=True,
                key="qc3column",
            ),
        ]
    ]

    l1col1 = [[sg.Text("Bonjour")]]
    l1col2 = [[sg.Text("WORK IN PROGRESS")]]
    l1col3 = [[sg.Button("testbouton")]]
    # l3col1 = [[sg.Button("testbouton")]]
    l3col2 = [[sg.Canvas(key="-CANVAS2-")]]
    l3col3 = [
        [sg.Button("Open 1D graph")],
        [sg.Button("Open 2D graph")],
        [sg.Button("Open 3D graph")],
    ]

    data_options_col = [
        [sg.Button("Combine seq")],
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
            sg.Column(
                data_col,
                size=(100, 250),
                scrollable=True,
                vertical_scroll_only=True,
                key="cycles",
            ),
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
    window.refresh()
    window["cycles"].contents_changed()

    # Associate fig with Canvas.
    fig_agg2D = draw_figure(window["-CANVAS-"].TKCanvas, fig2D)
    fig_agg2D.draw()
    fig_agg1D = draw_figure(window["-CANVAS2-"].TKCanvas, fig1D)
    fig_agg1D.draw()

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

        if event == "update":
            update_plot(
                values, X, Y, T, T_raw, ax1D, fig_agg1D, ax2D, fig_agg2D, total_cycles
            )

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
            break

        if event == "Open 1D graph":
            buf = io.BytesIO()
            pickle.dump(fig1D, buf)
            buf.seek(0)
            fig2 = pickle.load(buf)
            # fig_to_plot = fig2
            show_figure1D(fig2)
            fig2.show()
        if event == "Open 2D graph":
            buf = io.BytesIO()
            pickle.dump(fig2D, buf)
            buf.seek(0)
            fig2 = pickle.load(buf)
            show_figure2D(fig2)
            fig2.show()
        if event == "Open 3D graph":
            fig3D = plt.figure()
            ax = plt.axes(projection="3d")
            ax.scatter3D(X, Y, T, marker=".")
            plt.xlabel("X")
            plt.ylabel("Y")
            fig3D.show()

        if event == "refresh":
            sequence = values["selected_seq"]
            # window["_123_011_"].Update("wesh la zone")
            if sequence != seq_number:
                seq_number = sequence
                currentDir = data.path.parent.parent / str(seq_number)
                list_of_files = []
                if not currentDir.exists():
                    break
                for currentFile in currentDir.iterdir():
                    if currentFile.suffix == ".atoms":
                        list_of_files.append(currentFile.stem)
                seq_dir = str(currentDir)
                all_buttons_new = generate_list(seq_number, nbfiles)
                for k in range(len(all_buttons)):
                    window[all_buttons[k][4:]].update(all_buttons_new[k])
                all_buttons = all_buttons_new
                for k in range(len(all_buttons)):
                    if all_buttons[k] in list_of_files:
                        window[all_buttons[k][4:]].update(visible=True)
                    else:
                        window[all_buttons[k][4:]].update(visible=False)
                window["cycles"].Widget.canvas.yview_moveto(1.0)

            if sequence == seq_number:
                new_list_of_files = []
                for currentFile in currentDir.iterdir():
                    if currentFile.suffix == ".atoms":
                        new_list_of_files.append(currentFile.stem)
                # print(new_list_of_files)
                for k in range(len(all_buttons)):
                    if (all_buttons[k] in new_list_of_files) and (
                        all_buttons[k] not in list_of_files
                    ):
                        window[all_buttons[k][4:]].update(visible=True)
                window["cycles"].Widget.canvas.yview_moveto(0.0)
            window.refresh()
            window["cycles"].contents_changed()
            update_plot(
                values, X, Y, T, T_raw, ax1D, fig_agg1D, ax2D, fig_agg2D, total_cycles
            )

        for k in range(nbfiles):
            if event == all_buttons[k][4:]:
                new_path = (
                    data.path.parent.parent
                    / str(all_buttons[k][:3])
                    / (str(all_buttons[k]) + ".atoms")
                )
                data.path = new_path
                (X, Y, T, T_raw) = getrawdata(new_path)
                total_cycles = 1
                update_plot(
                    values,
                    X,
                    Y,
                    T,
                    T_raw,
                    ax1D,
                    fig_agg1D,
                    ax2D,
                    fig_agg2D,
                    total_cycles,
                )

                qc3 = ""
                parameters_file = data.path.parent / (data.path.stem + ".json")
                if parameters_file.is_file():
                    with open(parameters_file, encoding="utf-8") as file:
                        sequence_parameters = json.load(file)
                    for k in range(len(sequence_parameters)):
                        qc3 += (
                            sequence_parameters[k]["name"]
                            + " : "
                            + str(np.round(sequence_parameters[k]["value"], 3))
                            + "\n"
                        )
                window["qc3params"].update(qc3)
                window.refresh()
                window["qc3column"].contents_changed()

        if event == "Combine seq":
            X = []
            Y = []
            T = []
            for k in range(len(list_of_files)):
                new_path = (
                    data.path.parent.parent
                    / str(list_of_files[k][:3])
                    / (str(list_of_files[k]) + ".atoms")
                )
                data.path = new_path
                (Xa, Ya, Ta, T_raw) = getrawdata(new_path)
                X = np.concatenate([X, Xa])
                Y = np.concatenate([Y, Ya])
                T = np.concatenate([T, Ta])
            total_cycles = len(list_of_files)
            update_plot(
                values,
                X,
                Y,
                T,
                T_raw,
                ax1D,
                fig_agg1D,
                ax2D,
                fig_agg2D,
                len(list_of_files),
            )

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
