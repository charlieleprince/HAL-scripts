# -*- coding: utf-8 -*-
# the script is a standard python file
# you can import global python modules
import logging
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from pathlib import Path

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
    file_name = data.with_suffix(".mon_suffixe")
    file_name = Path(str(file_name).replace(".mon_suffixe", "_picoscope_raw.npy"))
    print("--------------")
    print(file_name)
    if not file_name.is_file():
        Text = f"No Picoscope Raw Data \n file was found for this cycle. \n Please choose a cycle for wich \n picoscope datas were taken. \n {file_name}"
        self.metaDataText.setPlainText(Text)
        return
    # load data
    data = np.load(file_name)
    # data[0] = Temps // data[i] = Voie i avec 1 <-> A ; 2 <-> B ; 3 <-> C ; 4 <-> D
    plt.clf()
    try:
        fig = plt.figure(1)
        plt.plot(data[0], data[1], ls="-", label="A")
        plt.plot(data[0], data[2], ls="-.", label="B")
        plt.plot(data[0], data[3], ls="--", label="C")
        plt.plot(data[0], data[4], ls=":", label="D")
        plt.xlabel("Time (ms)")
        plt.ylabel("Measured signal (V)")
        plt.legend()
        plt.show()
    except (TypeError, ValueError, IndexError):
        print(f"Error was found in the numpy picoscope datas file : {file_name}")
    return
