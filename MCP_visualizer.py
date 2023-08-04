# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import numpy as np

# import scipy.optimize as opt
from scipy.stats import gaussian_kde
import PySimpleGUI as sg
from PyQt5 import QtCore
from pathlib import Path
import pickle
import io
import json
import os


logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "0. MCP visualizer"  # display name, used in menubar and command palette
CATEGORY = "MCP"  # category (note that CATEGORY="" is a valid choice)

# layout tools
# sg.theme("DarkBlack")
sg.theme("LightGrey1")

g = 9.81


GREINER_CM = [
    [1.0000, 1.0000, 1.0000, 1.0000],
    [0.9474, 0.9474, 1.0000, 1.0000],
    [0.8947, 0.8947, 1.0000, 1.0000],
    [0.8421, 0.8421, 1.0000, 1.0000],
    [0.7895, 0.7895, 1.0000, 1.0000],
    [0.7368, 0.7368, 1.0000, 1.0000],
    [0.6842, 0.6842, 1.0000, 1.0000],
    [0.6316, 0.6316, 1.0000, 1.0000],
    [0.5789, 0.5789, 1.0000, 1.0000],
    [0.5263, 0.5263, 1.0000, 1.0000],
    [0.4737, 0.4737, 1.0000, 1.0000],
    [0.4211, 0.4211, 1.0000, 1.0000],
    [0.3684, 0.3684, 1.0000, 1.0000],
    [0.3158, 0.3158, 1.0000, 1.0000],
    [0.2632, 0.2632, 1.0000, 1.0000],
    [0.2105, 0.2105, 1.0000, 1.0000],
    [0.1579, 0.1579, 1.0000, 1.0000],
    [0.1053, 0.1053, 1.0000, 1.0000],
    [0.0526, 0.0526, 1.0000, 1.0000],
    [0.0000, 0.0000, 1.0000, 1.0000],
    [0.0000, 0.0769, 1.0000, 1.0000],
    [0.0000, 0.1538, 1.0000, 1.0000],
    [0.0000, 0.2308, 1.0000, 1.0000],
    [0.0000, 0.3077, 1.0000, 1.0000],
    [0.0000, 0.3846, 1.0000, 1.0000],
    [0.0000, 0.4615, 1.0000, 1.0000],
    [0.0000, 0.5385, 1.0000, 1.0000],
    [0.0000, 0.6154, 1.0000, 1.0000],
    [0.0000, 0.6923, 1.0000, 1.0000],
    [0.0000, 0.7692, 1.0000, 1.0000],
    [0.0000, 0.8462, 1.0000, 1.0000],
    [0.0000, 0.9231, 1.0000, 1.0000],
    [0.0000, 1.0000, 1.0000, 1.0000],
    [0.0769, 1.0000, 0.9231, 1.0000],
    [0.1538, 1.0000, 0.8462, 1.0000],
    [0.2308, 1.0000, 0.7692, 1.0000],
    [0.3077, 1.0000, 0.6923, 1.0000],
    [0.3846, 1.0000, 0.6154, 1.0000],
    [0.4615, 1.0000, 0.5385, 1.0000],
    [0.5385, 1.0000, 0.4615, 1.0000],
    [0.6154, 1.0000, 0.3846, 1.0000],
    [0.6923, 1.0000, 0.3077, 1.0000],
    [0.7692, 1.0000, 0.2308, 1.0000],
    [0.8462, 1.0000, 0.1538, 1.0000],
    [0.9231, 1.0000, 0.0769, 1.0000],
    [1.0000, 1.0000, 0.0000, 1.0000],
    [1.0000, 0.9231, 0.0000, 1.0000],
    [1.0000, 0.8462, 0.0000, 1.0000],
    [1.0000, 0.7692, 0.0000, 1.0000],
    [1.0000, 0.6923, 0.0000, 1.0000],
    [1.0000, 0.6154, 0.0000, 1.0000],
    [1.0000, 0.5385, 0.0000, 1.0000],
    [1.0000, 0.4615, 0.0000, 1.0000],
    [1.0000, 0.3846, 0.0000, 1.0000],
    [1.0000, 0.3077, 0.0000, 1.0000],
    [1.0000, 0.2308, 0.0000, 1.0000],
    [1.0000, 0.1538, 0.0000, 1.0000],
    [1.0000, 0.0769, 0.0000, 1.0000],
    [1.0000, 0.0000, 0.0000, 1.0000],
    [0.9000, 0.0000, 0.0000, 1.0000],
    [0.8000, 0.0000, 0.0000, 1.0000],
    [0.7000, 0.0000, 0.0000, 1.0000],
    [0.6000, 0.0000, 0.0000, 1.0000],
    [0.5000, 0.0000, 0.0000, 1.0000],
]
greiner = ListedColormap(GREINER_CM)


def setROIvalues(dict, values, str):
    dict["Tmin"] = float(values["Tmin"])
    dict["Tmax"] = float(values["Tmax"])
    dict["Xmin"] = float(values["Xmin"])
    dict["Xmax"] = float(values["Xmax"])
    dict["Ymin"] = float(values["Ymin"])
    dict["Ymax"] = float(values["Ymax"])


def convert_to_speed(X, Y, T):
    L_fall = 46.5e-2  # in meters
    # transverse momenta
    v_x, v_y = (X / T) * 1e3, Y / T * 1e3
    # momenta along gravity axis
    v_z = (0.5 * g * T) - (L_fall / T) * 1e6

    return (v_x, v_y, v_z)


def convert_boundaries_to_speed(Tmin, Tmax):
    L_fall = 46.5e-2  # in meters
    v_xmin = (-40.0 / Tmin) * 1e3
    v_xmax = (40.0 / Tmin) * 1e3
    v_zmin = (0.5 * g * Tmin) - (L_fall / Tmin) * 1e6
    v_zmax = (0.5 * g * Tmax) - (L_fall / Tmax) * 1e6

    return (v_xmin, v_xmax, v_zmin, v_zmax)


def getrawdata(path):
    """loads data"""
    v_perp_x = 1.02  # mm/ns
    v_perp_y = 1.13  # mm/ns
    time_resolution = 1.2e-10
    # time_to_pos = 2 * 0.98e-9
    atoms_file = np.fromfile(path, dtype="uint64")
    times_file_path = str(path.parent) + "/" + str(path.stem) + ".times"
    times_file = np.fromfile(times_file_path, dtype="uint64")
    T_x2 = times_file * time_resolution * 1e3
    atoms = atoms_file * time_resolution

    events_list = atoms.reshape(int(len(atoms) / 4), 4).T

    Xmcp = 0.5 * v_perp_x * 1e9 * (events_list[1] - events_list[0])
    Ymcp = 0.5 * v_perp_y * 1e9 * (events_list[3] - events_list[2])

    X = (Xmcp + Ymcp) / np.sqrt(2)
    Y = (Ymcp - Xmcp) / np.sqrt(2)
    T = (events_list[0] + events_list[1] + events_list[2] + events_list[3]) / 4

    T = T * 1e3

    timesx1_file_path = str(path.parent) + "/" + str(path.stem) + ".timesx1"
    if os.path.exists(timesx1_file_path):
        timesx1_file = np.fromfile(timesx1_file_path, dtype="uint64")
        T_x1 = timesx1_file * time_resolution * 1e3
        timesy1_file_path = str(path.parent) + "/" + str(path.stem) + ".timesy1"
        timesy1_file = np.fromfile(timesy1_file_path, dtype="uint64")
        T_y1 = timesy1_file * time_resolution * 1e3
        timesy2_file_path = str(path.parent) + "/" + str(path.stem) + ".timesy2"
        timesy2_file = np.fromfile(timesy2_file_path, dtype="uint64")
        T_y2 = timesy2_file * time_resolution * 1e3
    else:
        T_x1 = []
        T_y1 = []
        T_y2 = []

    T_raw = [T_x1, T_x2, T_y1, T_y2]
    return (X, Y, T, T_raw)


def getunreconstructed(path):
    """loads data"""
    time_resolution = 1.2e-10

    timesx1_file_path = str(path.parent) + "/" + str(path.stem) + ".timesx1"
    if timesx1_file_path.is_file():
        timesx1_file = np.fromfile(timesx1_file_path, dtype="uint64")
        T_x1 = timesx1_file * time_resolution * 1e3
    if not timesx1_file_path.is_file():
        T_x1 = []
    timesy1_file_path = str(path.parent) + "/" + str(path.stem) + ".timesy1"
    timesy1_file = np.fromfile(timesy1_file_path, dtype="uint64")
    T_y1 = timesy1_file * time_resolution * 1e3
    timesy2_file_path = str(path.parent) + "/" + str(path.stem) + ".timesy2"
    timesy2_file = np.fromfile(timesy2_file_path, dtype="uint64")
    T_y2 = timesy2_file * time_resolution * 1e3
    return (T_x1, T_y1, T_y2)


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


def generate_list2(prefix, min, max):
    u = []
    cyclemin = int(min)
    cyclemax = int(max)
    nbfiles = cyclemax - cyclemin + 1
    for i in range(nbfiles):
        k = i + cyclemin
        if len(str(k)) == 1:
            u.append(prefix + "_00" + str(k))
        elif len(str(k)) == 2:
            u.append(prefix + "_0" + str(k))
        else:
            u.append(prefix + "_" + str(k))
        # u.reverse()
    return u


def ROI_unreconstructed(ROI, T_raw):
    T_ROI = []
    for k in range(len(T_raw)):
        if type(T_raw[k]) == np.ndarray:
            ROI_indices = (T_raw[k] > ROI["Tmin"]) & (T_raw[k] < ROI["Tmax"])
            T_ROI.append(T_raw[k][ROI_indices])
        else:
            T_ROI.append([])
    return T_ROI


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
    cmaps.insert(0, "greiner")
    xy_lim = 40.0
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
        T_raw = ROI_unreconstructed(ROI_dict, T_raw)
        if values["conversion"]:
            X, Y, T = convert_to_speed(X, Y, T)
            (v_xmin, v_xmax, v_zmin, v_zmax) = convert_boundaries_to_speed(
                float(values["Tmin"]), float(values["Tmax"])
            )
            xy_lim = v_xmax
    ax1D.cla()
    ax2D.cla()
    bins = int(values["bins1D"])
    if values["grid1D"]:
        ax1D.grid(True)
    if values["T"]:
        x1Dlabel = "time (ms)"
        if values["unreconstructed"]:
            if values["X1"] and T_raw[0] != []:
                bin_heights, bin_borders = np.histogram(
                    T_raw[0], bins=np.linspace(np.min(T_raw[0]), np.max(T_raw[0]), bins)
                )
                bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
                # widths = np.diff(bin_borders)
                # ax1D.bar(bin_borders[:-1], bin_heights, widths, color="black")
                ax1D.plot(
                    bin_centers, bin_heights, linewidth="1", color="red", label="X1"
                )
                ax1D.legend()
                plt.legend()
            if values["X2"]:
                bin_heights, bin_borders = np.histogram(
                    T_raw[1], bins=np.linspace(np.min(T_raw[1]), np.max(T_raw[1]), bins)
                )
                bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
                # widths = np.diff(bin_borders)
                # ax1D.bar(bin_borders[:-1], bin_heights, widths, color="black")
                ax1D.plot(
                    bin_centers, bin_heights, linewidth="1", color="orange", label="X2"
                )
                ax1D.legend()
                plt.legend()
            if values["Y1"] and T_raw[2] != []:
                bin_heights, bin_borders = np.histogram(
                    T_raw[2], bins=np.linspace(np.min(T_raw[2]), np.max(T_raw[2]), bins)
                )
                bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
                # widths = np.diff(bin_borders)
                # ax1D.bar(bin_borders[:-1], bin_heights, widths, color="black")
                ax1D.plot(
                    bin_centers, bin_heights, linewidth="1", color="green", label="Y1"
                )
                ax1D.legend()
                plt.legend()
            if values["Y2"] and T_raw[3] != []:
                bin_heights, bin_borders = np.histogram(
                    T_raw[3], bins=np.linspace(np.min(T_raw[3]), np.max(T_raw[3]), bins)
                )
                bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
                # widths = np.diff(bin_borders)
                # ax1D.bar(bin_borders[:-1], bin_heights, widths, color="black")
                ax1D.plot(
                    bin_centers, bin_heights, linewidth="1", color="purple", label="Y2"
                )
                ax1D.legend()
                plt.legend()
        bin_heights, bin_borders = np.histogram(
            T, bins=np.linspace(np.min(T), np.max(T), bins)
        )
        widths = np.diff(bin_borders)
        bin_heights = np.array(bin_heights, dtype=object) / nb_of_cycles
        ax1D.bar(bin_borders[:-1], bin_heights, widths)

        if values["ROI0"]:
            ax1D.set_xlim(float(values["Tmin"]), float(values["Tmax"]))
            if values["conversion"]:
                ax1D.set_xlim(v_zmin, v_zmax)
                x1Dlabel = "vz (mm/s)"
        if not values["ROI0"]:
            ax1D.set_xlim(np.min(T), np.max(T))
        ax1D.set_xlabel(x1Dlabel)
        ax1D.set_ylabel("number of events")
    if values["X"]:
        x1Dlabel = "X (mm)"
        bin_heights, bin_borders = np.histogram(
            X, bins=np.linspace(-xy_lim, xy_lim, bins)
        )
        widths = np.diff(bin_borders)
        bin_heights = np.array(bin_heights, dtype=object) / nb_of_cycles
        ax1D.bar(bin_borders[:-1], bin_heights, widths, color="tab:blue")
        # ax1D.hist(X, bins=np.linspace(-40, 40, bins), color="tab:blue")
        ax1D.set_xlim(-xy_lim, xy_lim)
        if values["ROI0"] and values["conversion"]:
            x1Dlabel = "vx (mm/s)"
        ax1D.set_xlabel(x1Dlabel)
        ax1D.set_ylabel("number of events")
    if values["Y"]:
        x1Dlabel = "Y (mm)"
        bin_heights, bin_borders = np.histogram(
            Y, bins=np.linspace(-xy_lim, xy_lim, bins)
        )
        widths = np.diff(bin_borders)
        bin_heights = np.array(bin_heights, dtype=object) / nb_of_cycles
        ax1D.bar(bin_borders[:-1], bin_heights, widths, color="tab:blue")
        if values["ROI0"] and values["conversion"]:
            x1Dlabel = "vy (mm/s)"
        ax1D.set_xlabel(x1Dlabel)
        ax1D.set_xlim(-xy_lim, xy_lim)
        ax1D.set_ylabel("number of events")
    if values["max events enabled"]:
        ax1D.set_ylim(0, float(values["max events"]))
    if values["logscale"]:
        ax1D.set_yscale("log")

    if not values["colormap"] in cmaps:
        return
    if values["colormap"] == "greiner":
        cmap = greiner
    else:
        cmap = plt.get_cmap(values["colormap"])
    coeff = 1.0
    if values["2dmax"]:
        coeff = float(values["max plot2d"]) / 100
    if values["XY"]:
        x2dlabel = "X (mm)"
        y2dlabel = "Y (mm)"
        if values["ROI0"] and values["conversion"]:
            x2dlabel = "vx (mm/s)"
            y2dlabel = "vy (mm/s)"
        hist = ax2D.hist2d(
            X, Y, bins=np.linspace(-xy_lim, xy_lim, int(values["bins2D"]))
        )

        ax2D.hist2d(
            X,
            Y,
            bins=np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
            cmap=cmap,
            vmax=coeff * max(hist[0].flatten()),
        )
        ax2D.set_xlabel(x2dlabel)
        ax2D.set_ylabel(y2dlabel)
    if values["XT"]:
        x2dlabel = "X (mm)"
        y2dlabel = "T (ms)"
        if values["ROI0"] and values["conversion"]:
            x2dlabel = "vx (mm/s)"
            y2dlabel = "vz (mm/s)"
        if values["ROI0"]:
            ax2D.set_ylim(float(values["Tmin"]), float(values["Tmax"]))
            tmin = float(values["Tmin"])
            tmax = float(values["Tmax"])
            if values["conversion"]:
                ax2D.set_ylim(v_zmin, v_zmax)
                tmin = v_zmin
                tmax = v_zmax
            hist = ax2D.hist2d(
                X,
                T,
                bins=[
                    np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
                    np.linspace(
                        tmin,
                        tmax,
                        int(values["bins2D"]),
                    ),
                ],
            )
            ax2D.hist2d(
                X,
                T,
                bins=[
                    np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
                    np.linspace(
                        tmin,
                        tmax,
                        int(values["bins2D"]),
                    ),
                ],
                cmap=cmap,
                vmax=coeff * max(hist[0].flatten()),
            )
        if not values["ROI0"]:
            hist = ax2D.hist2d(
                X,
                T,
                bins=[
                    np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
                    np.linspace(np.min(T), np.max(T), int(values["bins2D"])),
                ],
                cmap=cmap,
            )
            ax2D.hist2d(
                X,
                T,
                bins=[
                    np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
                    np.linspace(np.min(T), np.max(T), int(values["bins2D"])),
                ],
                cmap=cmap,
                vmax=coeff * max(hist[0].flatten()),
            )
        ax2D.set_xlabel(x2dlabel)
        ax2D.set_ylabel(y2dlabel)
    if values["YT"]:
        x2dlabel = "Y (mm)"
        y2dlabel = "T (ms)"
        if values["ROI0"] and values["conversion"]:
            x2dlabel = "vy (mm/s)"
            y2dlabel = "vz (mm/s)"
        if values["ROI0"]:
            ax2D.set_ylim(float(values["Tmin"]), float(values["Tmax"]))
            tmin = float(values["Tmin"])
            tmax = float(values["Tmax"])
            if values["conversion"]:
                ax2D.set_ylim(v_zmin, v_zmax)
                tmin = v_zmin
                tmax = v_zmax
            hist = ax2D.hist2d(
                Y,
                T,
                bins=[
                    np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
                    np.linspace(
                        tmin,
                        tmax,
                        int(values["bins2D"]),
                    ),
                ],
            )
            ax2D.hist2d(
                Y,
                T,
                bins=[
                    np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
                    np.linspace(
                        tmin,
                        tmax,
                        int(values["bins2D"]),
                    ),
                ],
                cmap=cmap,
                vmax=coeff * max(hist[0].flatten()),
            )
        if not values["ROI0"]:
            hist = ax2D.hist2d(
                Y,
                T,
                bins=[
                    np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
                    np.linspace(np.min(T), np.max(T), int(values["bins2D"])),
                ],
            )
            ax2D.hist2d(
                Y,
                T,
                bins=[
                    np.linspace(-xy_lim, xy_lim, int(values["bins2D"])),
                    np.linspace(np.min(T), np.max(T), int(values["bins2D"])),
                ],
                cmap=cmap,
                vmax=coeff * max(hist[0].flatten()),
            )

        ax2D.set_xlabel(x2dlabel)
        ax2D.set_ylabel(y2dlabel)
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
    sample = data.path.stem.split("_")[1]
    if not data.path.suffix == ".atoms":
        return
    # get data
    X, Y, T = data.getrawdata()
    # T_x2 = data.getdatafromsingleline()
    (T_x1, T_x2, T_y1, T_y2) = data.getunreconstructeddata()
    T_raw = [T_x1, T_x2, T_y1, T_y2]
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
    cmaps.insert(0, "greiner")
    fig2D, ax2D = plt.subplots(figsize=(6, 6))
    ax2D.hist2d(X, Y, bins=np.linspace(-40, 40, 160), cmap=plt.cm.nipy_spectral)
    ax2D.set_xlabel("X")
    ax2D.set_ylabel("Y")

    fig1D, ax1D = plt.subplots(figsize=(6, 3))
    # ax1D.hist(T_raw, bins=np.linspace(np.min(T_raw), np.max(T_raw), 300), color="black")
    bin_heights, bin_borders = np.histogram(T, bins=np.linspace(0, np.max(T), 300))
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
                + str(np.round(sequence_parameters[k]["value"], 4))
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
        [sg.Checkbox("Convert to speed", default=False, key="conversion")],
        [
            sg.Button(
                "Set to default", button_color=("black", "white"), key="set to default"
            )
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
            sg.Checkbox("Colormap max", default=False, key="2dmax"),
            sg.Input(size=(6, 1), default_text=100, key="max plot2d"),
            sg.Text("%"),
        ],
        [
            sg.Combo(
                cmaps,
                default_value="nipy_spectral",
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
            sg.Checkbox("Plot raw data", default=False, key="unreconstructed"),
            sg.Checkbox("X1", default=False, key="X1"),
            sg.Checkbox("X2", default=True, key="X2"),
            sg.Checkbox("Y1", default=False, key="Y1"),
            sg.Checkbox("Y2", default=False, key="Y2"),
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

    l1col1 = [[sg.Text("Welcome to the HAL visualizer")]]
    name_of_data = sample
    l1col2 = [
        [
            sg.Text("Selected data:", font="Helvetica 10 bold"),
            sg.Text(name_of_data, key="name"),
        ],
        #       [
        #          sg.Text("Number of atoms in ROI:", font="Helvetica 12 bold"),
        #         sg.Text(name_of_data, key="nb of atoms in ROI"),
        #      ],
        #      [
        #          sg.Text("Total number of atoms:", font="Helvetica 10 bold"),
        #          sg.Text(name_of_data, key="nb of atoms"),
        #      ],
    ]
    l1col3 = []
    # l3col1 = [[sg.Button("testbouton")]]
    l3col2 = [[sg.Canvas(key="-CANVAS2-")]]
    l3col3 = [
        [sg.Button("Open 1D graph")],
        [sg.Button("Open 2D graph")],
        [sg.Button("Open 3D graph")],
    ]

    data_options_col = [
        [sg.Button("Average seq")],
        [
            sg.Button("Average cycles"),
            sg.Input(size=(4, 1), default_text=str(1), key="seqmin"),
            sg.Text("to"),
            sg.Input(size=(4, 1), default_text=str(2), key="seqmax"),
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

        if event == "set to default":
            with open(default_roi_file_name, encoding="utf8") as f:
                defaultroi = json.load(f)
            setROIvalues(defaultroi["ROI 0"], values, "0")
            with open(default_roi_file_name, "w", encoding="utf-8") as file:
                json.dump(defaultroi, file, ensure_ascii=False, indent=4)

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
            (Xdata, Ydata, Tdata) = (X, Y, T)
            if values["ROI0"]:
                ROI_dict = {}
                ROI_dict["Tmin"] = float(values["Tmin"])
                ROI_dict["Tmax"] = float(values["Tmax"])
                ROI_dict["Xmin"] = float(values["Xmin"])
                ROI_dict["Xmax"] = float(values["Xmax"])
                ROI_dict["Ymin"] = float(values["Ymin"])
                ROI_dict["Ymax"] = float(values["Ymax"])
                (Xdata, Ydata, Tdata) = ROI_data(ROI_dict, X, Y, T)
            fig3D = plt.figure()
            ax = plt.axes(projection="3d")
            xyz = np.vstack([Xdata, Ydata, Tdata])
            z2 = gaussian_kde(xyz)(xyz)
            ax.scatter3D(Xdata, Ydata, Tdata, c=z2, marker=".")
            plt.xlabel("X")
            plt.ylabel("Y")
            fig3D.show()

        if event == "refresh":
            sequence = values["selected_seq"]
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
            list_of_files = new_list_of_files

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
                # print(all_buttons[k][4:])
                name_of_data = all_buttons[k][4:]
                qc3 = ""
                parameters_file = data.path.parent / (data.path.stem + ".json")
                if parameters_file.is_file():
                    with open(parameters_file, encoding="utf-8") as file:
                        sequence_parameters = json.load(file)
                    for k in range(len(sequence_parameters)):
                        qc3 += (
                            sequence_parameters[k]["name"]
                            + " : "
                            + str(np.round(sequence_parameters[k]["value"], 4))
                            + "\n"
                        )
                window["qc3params"].update(qc3)
                window["name"].update(name_of_data)
                window.refresh()
                window["qc3column"].contents_changed()

        if event == "Average seq":
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
                (Xa, Ya, Ta, T_rawa) = getrawdata(new_path)
                X = np.concatenate([X, Xa])
                Y = np.concatenate([Y, Ya])
                T = np.concatenate([T, Ta])
                # T_raw = np.concatenate([T_raw, T_rawa])
                T_raw = T_rawa  # Traw has a strange shape (4)
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
            name_of_data = (
                str(list_of_files[len(list_of_files) - 1]).split("_")[1]
                + " - "
                + str(list_of_files[0]).split("_")[1]
            )
            window["name"].update(name_of_data)
        if event == "Average cycles":
            X = []
            Y = []
            T = []
            list_of_files_to_average = generate_list2(
                values["selected_seq"], values["seqmin"], values["seqmax"]
            )
            cancel_average = False
            total_cycles = len(list_of_files_to_average)
            for k in range(len(list_of_files_to_average)):
                if not list_of_files_to_average[k] in list_of_files:
                    total_cycles -= 1
                else:
                    new_path = (
                        data.path.parent.parent
                        / str(list_of_files_to_average[k][:3])
                        / (str(list_of_files_to_average[k]) + ".atoms")
                    )
                    data.path = new_path
                    (Xa, Ya, Ta, T_rawa) = getrawdata(new_path)
                    X = np.concatenate([X, Xa])
                    Y = np.concatenate([Y, Ya])
                    T = np.concatenate([T, Ta])
                    # T_raw = np.concatenate([T_raw, T_rawa])
                    T_raw = T_rawa
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
            name_of_data = (
                str(list_of_files_to_average[0]).split("_")[1]
                + " - "
                + str(
                    list_of_files_to_average[len(list_of_files_to_average) - 1]
                ).split("_")[1]
            )
            window["name"].update(name_of_data)

    plt.close()
    window.close()
    # now the ROIs are set
