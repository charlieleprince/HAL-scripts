# -*- coding: utf-8 -*-
# the script is a standard python file
# you can import global python modules
import logging
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from pathlib import Path
import pandas as pd
# and also local modules from HAL
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache

# you can of course write some python
logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "Picoscope Raw Datas"  # display name, used in menubar and command palette
CATEGORY = "plot"  # category (note that CATEGORY="" is a valid choice)


def main(self):
    """
    the script also have to define a `main` function. When playing a script,
    HAL runs `main` passes one (and only one) argument "self" that is the
    HAL mainwindow object (granting access to all the gui attributes and methods)
    """
    # -- get selected data
    selection = self.runList.selectedItems()
    print("Action Picoscope")
    if not selection:
        Text = "Please, select at least \n one file if you want to \n look at Picoscope Datas."
        self.metaDataText.setPlainText(Text)
        return
    # get path
    item = selection[0]  # item is a PyQt5.QtWidgets.QListWidgetItem object
    print(item)
    data = item.data(Qt.UserRole)
    # if data.path is something like /mnt/manip_E/2023/02/12/003/003_018.png
    # data.stem is 003_018
    # and data.parent is /mnt/manip_E/2023/02/12/003/
    file_name = data.with_suffix(".picoscope_raw_data")
    print("--------------")
    print(file_name)
    if not file_name.is_file():
        Text = f"No Picoscope Raw Data \n file was found for this cycle. \n Please choose a cycle for wich \n picoscope datas were taken. \n {file_name}"
        self.metaDataText.setPlainText(Text)
        return
    # load data


    df = pd.read_pickle(file_name)
    column_names = list(df.columns)

    graph_titles = [
        col_name
        for col_name in column_names
        if not ("Time" in col_name or "fitted" in col_name)
    ]
    time = df["Time"]
    
    if np.max(time) < 1.3e-6:
        time = time *1e9
        xlabel = "Time (ns)"
    elif np.max(time) < 1.3e-3:
        time = time *1e6
        xlabel = "Time (us)"
    elif  np.max(time) < 1.3:
        time = time *1e3
        xlabel = "Time (ms)"
    else:
        xlabel = "Time (s)"
    fig, axs = plt.subplots(2, 2)
    for i, ax in enumerate(axs.flat):
        name = graph_titles[i]
        s = df[name]
        sfit = df[name + " fitted"]
        ax.plot(
            time,
            s,
            ls="None",
            marker=".",
            label="Exp",
            alpha=0.8,
        )
        ax.plot(
            time,
            sfit,
            ls="-",
            marker="None",
            label="Fit",
            alpha=0.8,
        )
        ax.set(xlabel=xlabel, ylabel=name + " (mV)")
        # ax.label_outer()
        ax.legend()
    plt.tight_layout()
    plt.show()
    return
