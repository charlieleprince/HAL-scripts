# -*- coding: utf-8 -*-


import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
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
NAME = "3. Plot time histogram"  # display name, used in menubar and command palette
CATEGORY = "HAL - MCP display"  # category (note that CATEGORY="" is a valid choice)

root = Path().home()
default_roi_dir = root / ".HAL"
HAL_display = default_roi_dir / "user_modules" / "plot_roi_only.json"


def main(self):
    """
    the script also have to define a `main` function. When playing a script,
    HAL runs `main` passes one (and only one) argument "self" that is the
    HAL mainwindow object (granting access to all the gui attributes and methods)
    """

    with open(HAL_display, "r", encoding="utf-8") as file:
        current_mcp_display = json.load(file)

    current_mcp_display["plot time histogram"] = "true"

    with open(HAL_display, "w", encoding="utf-8") as file:
        json.dump(current_mcp_display, file, ensure_ascii=False, indent=4)
