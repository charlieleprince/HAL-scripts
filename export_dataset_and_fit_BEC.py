# !/usr/bin/env python
#  -*-mode:Python; coding:utf8 -*-

# ------------------------------------
# Created on Tu Sep 2023 by Victor Gondret
# Contact : victor.gondret@institutoptique
#
# MIT Copyright (c) 2023 - Helium1@LCF
# Institut d'Optique Graduate School
# Université Paris-Saclay
# ------------------------------------
#
"""
Decription of fit_paire_position.py 

This code was designed to be run from HAL (https://github.com/adareau/HAL).
The principle is he following : when the application is run, the model Model is instanciated. 
Once it is instanciate, the application gets back all parameters of the model from the function self.model.get_parameters(). 
This generates a dictionary from model.__dict__ in which each element is a number/list/string or boolean. 
From this dictionary, the application built in a scrollable area a list of QLabels and Qlines to updtate parameters of the model. 
The figure that is shown on the right is defined in the PlotZaxis class. It does not contains a lot but the method update_plot that is called each time the user pushes the button 'Update Plot'.  This function obviously need the model to be rightly updated.
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QScrollArea,
)
from PyQt5.QtGui import QIcon, QPixmap, QImage, QClipboard
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import os
from heliumtools.correlations import Correlation
from scipy.optimize import curve_fit
import seaborn as sns
from PIL import Image
from heliumtools.misc.gather_data import apply_ROI
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache, _loadFileMetaData

# Tous les paramètres du code affichés dans le graphique doivent être définis dans le dictionnaire ci-dessus
# ils doivent tous être des flottants.
PARAMETERS = {
    "Arrival_time": 307.5,
    "BoxZsize": 1.3,
    "Transverse_size_1": 10,
    "Transverse size 2": 20,
    "Transverse size 3": 30,
    "Inertial Frame Vx": 0,
    "Inertial Frame Vy": 0,
    "Inertial Frame Vz": 93.8,
    "Fit Vz min": 18,
    "Fit Vz max": 34,
}


HOM_FOLDER = "/mnt/manip_E/2023/09/HOM/01"

atoms = pd.read_pickle(
    os.path.join(HOM_FOLDER, "atoms_BSdelay_0000_us.pkl")
).reset_index(drop=True)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "1. Density along z"  # display name, used in menubar and command palette
CATEGORY = "Lattice Pairs"  # category (note that CATEGORY="" is a valid choice)


class Model:
    def __init__(self, atoms, metadata):
        self.atoms = atoms
        self.metadat = metadata
        self.inertial_frame = [0, 0, 93]
        self.BEC_arrival_time = 307.5
        self.boxZsize = 1.3
        self.boxXYsize = 10
        self.Xposition = 0
        self.Yposition = 0
        self.range_for_fit = [18, 35]
        self.vperp_list = [10, 20, 30]
        self.plot_range = [-40, 40]

    def get_parameters(self):
        """return every attribut of the class as a strig if it is a boolean, a string or a number."""
        parameters = {}
        for key, value in self.__dict__.items():
            if type(value) in [int, float, str, bool, list]:
                parameters[key.replace("_", " ")] = str(value)
        return parameters

    def update_parameter(self, key, str_val):
        """This function update the parameters dictionary, giving a key of the dictionary and a str_val"""
        key = key.replace(" ", "_")
        if key not in self.__dict__.keys():
            logger.warning(f"WARNING : The {key} is not in the model.")
            return str_val
        value = self.__dict__[key]
        try:
            if type(value) == bool:
                if str_val.lower() in "vrai true":
                    self.__dict__[key] = True
                elif str_val.lower() in "faux false":
                    self.__dict__[key] = False
                else:
                    logger.warning(
                        f"WARNING: parameter boolean {key} was not recognized."
                    )
            elif type(value) in [float, int]:
                value = float(str_value)
                if int(value) == value:
                    self.__dict__[key] = int(value)
                else:
                    self.__dict__[key] = value
            elif type(value) == str:
                self.__dict__[key] = str_val
            elif type(value) == list:
                str_val = str_val.replace("[", "").replace("]", "").split(",")
                value = []
                for i in str_val:
                    try:
                        if int(i) == float(i):
                            value.append(int(i))
                        else:
                            value.append(float(i))
                    except:
                        logger.warning(
                            f"Warning: conversion of {i} to float in {key} failed."
                        )
                self.__dict__[key] = value
            else:
                logger.warning(
                    f"WARNING : I did not recognized the type of the parameter {key}"
                )
            return str(self.__dict__[key])
        except:
            return str(self.__dict__[key])


class PlotZaxis:
    """This class contains the figure that is shown to the user, on the right of the figure."""

    def __init__(self, fig, ax, canvas):
        self.fig = fig
        self.ax = ax
        self.canvas = canvas

    def update_plot(self, model):
        self.ax.clear()

        ROI = {
            "Vz": {"min": -70, "max": 70},
            "Vy": {"min": -70, "max": 70},
            "Vx": {"max": 75, "min": -75},
        }
        inertial_frame = {
            "Vx": model.inertial_frame[0],
            "Vy": model.inertial_frame[1],
            "Vz": model.inertial_frame[2],
        }
        corr = Correlation(
            model.atoms,
            ROI=ROI,
            bec_arrival_time=model.BEC_arrival_time,
            ref_frame_speed=inertial_frame,
        )
        for i, vperp_max in enumerate(model.vperp_list):
            my_box = {"Vperp": {"minimum": -1, "maximum": vperp_max}}
            at = corr.get_atoms_in_box(corr.atoms, my_box)
            hist, bins = np.histogram(
                at["Vz"],
                bins=np.arange(
                    model.plot_range[0], model.plot_range[1], model.boxZsize
                ),
            )
            x = (bins[0:-1] + bins[1:]) / 2
            self.ax.scatter(
                x,
                hist / corr.n_cycles,
                alpha=0.4,
                label=r"$\Delta V_z={:.1f}, \, \Delta V_\perp={:.0f}$ mm/s ".format(
                    model.boxZsize, vperp_max
                ),
            )
            ###### Fits de la paire 1
            hist1, bins1 = np.histogram(
                at["Vz"],
                bins=np.arange(
                    -model.range_for_fit[1], -model.range_for_fit[0], model.boxZsize
                ),
            )
            hist1 = hist1 / corr.n_cycles
            bin_centers1 = np.array(bins1[:-1] + np.diff(bins1) / 2)
            max_index = np.argmax(hist1)
            p0 = [
                bin_centers1[max_index],
                np.max(hist1),
                np.mean(hist1 * (bin_centers1 - bin_centers1[max_index])),
            ]
            try:
                popt, pcov_paire_1 = curve_fit(gaussian, bin_centers1, hist1, p0=p0)
                fit_absc = np.linspace(np.min(bin_centers1), np.max(bin_centers1), 50)
                self.ax.plot(
                    fit_absc, gaussian(fit_absc, *popt), "--", color="C" + str(i)
                )
                self.ax.text(
                    1.1 * popt[0],
                    1.1 * popt[1],
                    "{:.1f}".format(popt[0]),
                    color="C" + str(i),
                    ha="center",
                    bbox=dict(facecolor="white", alpha=0.3, boxstyle="round"),
                )
            except:
                pass

            ###### Fits de la paire 2
            hist1, bins1 = np.histogram(
                at["Vz"],
                bins=np.arange(
                    model.range_for_fit[0], model.range_for_fit[1], model.boxZsize
                ),
            )
            hist1 = hist1 / corr.n_cycles
            bin_centers1 = np.array(bins1[:-1] + np.diff(bins1) / 2)
            max_index = np.argmax(hist1)
            # p0 = mean, amplitude, standard_deviation

            p0 = [
                bin_centers1[max_index],
                np.max(hist1),
                np.mean(hist1 * (bin_centers1 - bin_centers1[max_index]) ** 2),
            ]
            try:
                popt, pcov_paire_1 = curve_fit(
                    gaussian, bin_centers1, hist1, p0=p0, maxfev=5000
                )
                fit_absc = np.linspace(np.min(bin_centers1), np.max(bin_centers1), 50)
                self.ax.plot(
                    fit_absc, gaussian(fit_absc, *popt), "--", color="C" + str(i)
                )
                self.ax.text(
                    1.1 * popt[0],
                    1.1 * popt[1],
                    "{:.1f}".format(popt[0]),
                    color="C" + str(i),
                    ha="center",
                    bbox=dict(facecolor="white", alpha=0.3, boxstyle="round"),
                )
            except:
                pass
        self.ax.set_xlabel("Velocity along z (mm/s)")
        self.ax.set_ylabel(r"Atoms per box $(mm/s)^{-3}$")
        self.ax.grid(True)
        self.ax.set_title(
            r"Inertial frame : {} mm/s".format(model.inertial_frame), fontsize="medium"
        )
        self.fig.tight_layout()
        self.canvas.draw()


class DensityApplication(QWidget):
    def __init__(self, atoms, metadata):
        super().__init__()
        self.model = Model(atoms, metadata)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Density Visualizer.")
        self.setGeometry(100, 100, 1000, 400)
        try:
            self.setWindowIcon(
                QIcon(os.path.join("icons", "chameau.png"))
            )  # Remplacez 'chemin_vers_votre_icone.ico' par le chemin de votre icône
        except:
            logger.warning(
                "WARNING : Our camel is missing. Please find him as fast as possible !!"
            )
        # Left part of the windows : parameters
        self.layout_left = QVBoxLayout()
        # title of the coluns
        label = QLabel("<html><b> Parameters </b></html>")
        label.setAlignment(Qt.AlignCenter)  # Alignement centré
        self.layout_left.addWidget(label)
        # Define a scroll area for parameters
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)

        self.labels_of_parameters = []  # liste contenant les QLabel des paramètres
        self.values_of_parameters = []  # liste contenant les QLine edits des paramètres
        # on parourt la liste des paramètres du modèle
        for name, value in self.model.get_parameters().items():
            self.labels_of_parameters.append(QLabel(name))
            self.values_of_parameters.append(QLineEdit())
            # set the default value in it
            self.values_of_parameters[-1].setText(str(value))
            # creat a horizontal layout foqvalue.text()r the name and the value of the parameter
            layout = QHBoxLayout()
            layout.addWidget(self.labels_of_parameters[-1])
            layout.addWidget(self.values_of_parameters[-1])
            # and store it into the vertical left layout
            self.scroll_layout.addLayout(layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.layout_left.addWidget(self.scroll_area)

        ## now we add the button to update the figure
        self.update_figure_button = QPushButton("Update Figure")
        self.update_figure_button.clicked.connect(self.update_plot)  # connect action
        self.layout_left.addWidget(self.update_figure_button)

        # Right part of the windows
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout_right = QVBoxLayout()
        self.layout_right.addWidget(self.canvas)
        self.plot = PlotZaxis(self.figure, self.ax, self.canvas)

        ## now we add the button to update the figure
        self.copy_to_clipboard_button = QPushButton("Copy to Clip Board")
        self.copy_to_clipboard_button.clicked.connect(
            self.copy_to_clipboard
        )  # connect action
        self.layout_right.addWidget(self.copy_to_clipboard_button)
        # and plot now

        # Built the windows Layout
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.layout_left)
        self.main_layout.addLayout(self.layout_right)

        self.setLayout(self.main_layout)
        self.update_plot()

    def update_plot(self):
        for qlabel, qvalue in zip(self.labels_of_parameters, self.values_of_parameters):
            model_value = self.model.update_parameter(qlabel.text(), qvalue.text())
            qvalue.setText(model_value)
        self.plot.update_plot(self.model)

    def copy_to_clipboard(self):
        # Sauvegarder la figure en tant qu'image PNG temporaire
        temp_file = "temp_figure.png"
        self.figure.savefig(temp_file, dpi=300, bbox_inches="tight")

        # Charger l'image temporaire avec QImage
        image = QImage(temp_file)

        # Convertir QImage en QPixmap
        pixmap = QPixmap.fromImage(image)

        # Copier le QPixmap dans le presse-papiers
        clipboard = QApplication.clipboard()
        clipboard.setPixmap(pixmap)

        # Supprimer le fichier temporaire
        del image
        os.remove(temp_file)


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * standard_deviation**2))


def main(hal_main_window):
    """
    the script also have to define a `main` function. When playing a script,
    HAL runs `main` passes one (and only one) argument "hal_main_window" that is the HAL mainwindow object (granting access to all the gui attributes and methods)
    """
    # -- get selected data
    selection = hal_main_window.runList.selectedItems()
    atoms_path = []
    for k in range(len(selection)):
        atoms_path.append(str(item.data(QtCore.Qt.UserRole)))
    atoms.reset_index(drop=True, inplace=True)
    metadata = getSelectionMetaDataFromCache(hal_main_window, update_cache=True)
    # app = QApplication(sys.argv)
    density_app = DensityApplication(atoms, metadata)
    density_app.show()
    # sys.exit(app.exec_())


if __name__ == "__main__":
    atom_paths = glob.glob("/mnt/manip_E/2023/09/12/040/*.atoms")
    app = QApplication(sys.argv)
    density_app = DensityApplication()
    density_app.show()
    sys.exit(app.exec_())
