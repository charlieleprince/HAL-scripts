# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.optimize as opt
from datetime import datetime
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "Get script variables"  # display name, used in menubar and command palette
CATEGORY = "Qcontrol3"  # category (note that CATEGORY="" is a valid choice)

def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            #print(f"=={key}==")
            yield from recursive_items(value)
        else:
            yield (key, value)



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

    root = Path().home()
    default_roi_dir = root / ".HAL"
    default_roi_file_name = default_roi_dir / "default_mcp_roi.json"
    # get path
    item = selection[0]
    data.path = item.data(QtCore.Qt.UserRole)

    sequence_parameters_file = data.path.parent / "sequence_parameters.json"
    with open(sequence_parameters_file, encoding="utf-8") as file:
        sequence_parameters=json.load(file)


    Text =  "[qc3 parameters]\n"


    for key, value in sequence_parameters.items():
        if type(value) is dict:
            Text += f"======{key}=======\n"
            small_dict = value
            for key, value in small_dict.items():
                if type(value) is dict:
                    Text += f"---{key}---\n"
                    small_dict = value
                    for key, value in small_dict.items():
                        if type(value) is dict:
                            Text += f"--{key}--\n"
                            small_dict = value
                            for key, value in small_dict.items():
                                if type(value) is dict:
                                    Text += f"-{key}-\n"
                                    small_dict = value

                                else:
                                    Text+= f"  {key} : {value}\n"
                        else:
                            Text+= f"  {key} : {value}\n"
                else:
                    Text+= f"  {key} : {value}\n"
        else:
            Text+= f"  {key} : {value}\n"




    self.metaDataText.setPlainText(Text)


    """
        MCP_stats_folder = data.path.parent / ".MCPstats"
        MCP_stats_folder.mkdir(exist_ok=True)
        file_name = MCP_stats_folder / data.path.stem
        with open(str(file_name) + ".json", "w", encoding="utf-8") as file:
            json.dump(to_mcp_dictionary, file, ensure_ascii=False, indent=4)
    """
