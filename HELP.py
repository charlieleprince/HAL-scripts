# -*- coding: utf-8 -*-


import logging
import PySimpleGUI as sg
from pathlib import Path
import PySimpleGUI as sg
from PIL import Image

# open method used to open different extension image file

logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "MCP SCRIPTS"  # display name, used in menubar and command palette
CATEGORY = "help"  # category (note that CATEGORY="" is a valid choice)

# layout tools


def go_to_line(st, nb):
    u = list(st)
    for k in range(int(len(u) / nb)):
        if u[nb * (k + 1)] == " ":
            u.insert(nb * (k + 1) + 2, "\n")
        else:
            v = 0
            j = 1
            while v != " ":
                v = u[nb * (k + 1) - j]
                j = j + 1
            u.insert(nb * (k + 1) + 2 - j, "\n")
    return "".join(u)


# main
nb_of_chars = 180


def main(self):
    # -- display window

    # sg.theme("Default")  # Add a touch of color
    # All the stuff inside your window.
    layout = [
        [sg.Text("General information", text_color="black", font="Helvetica 13 bold")],
        [
            sg.Text(
                go_to_line(
                    "The MCP data are extracted from the .times and .atoms files. "
                    + "The picture displayed in the main gui is a X-Y histogram "
                    + "integrated over the whole acquisition time of the MCP (no ROI)."
                    + " The number of bins is defined in the mcp.py file in the data folder of the HAL user modules."
                    + "The scripts make it possible to define ROIs and get their properties"
                    + " such as the number of atoms, the average arrival time or the temperature."
                    + "There are two categories of MCP scripts depending on the number of files you selected.",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            )
        ],
        [
            sg.Text(
                go_to_line(
                    "For the plot of time histograms, reconstructed data are represented in blue, while unreconstructed "
                    + "data are in black behind the blue histogram.",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            )
        ],
        [sg.Text("Scripts description", text_color="black", font="Helvetica 13 bold")],
        [sg.Text("MCP", text_color="black", font="Helvetica 12 bold")],
        [
            sg.Text(
                "  1. Choose ROI and get ROI info",
                text_color="darkslategray",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                "Opens a gui to define at most 4 ROIs and set the default ROI, then exports ROI number of atoms to metadata.",
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  2. Gets number of atoms in default ROI0",
                text_color="dark blue",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                "Exports default ROI0 number of atoms to metadata (no need to define a new ROI).",
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  2Bis. Get number of atoms & fit arrival times in default ROI0",
                text_color="dark blue",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Same as 2. + fits the time histogram with a Gaussian to export arrival time and time width to metadata.",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  3. Combine",
                text_color="darkslategray",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Plots the X-Y histogram integrated over the data from all the selected files.",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  4. Plot time histograms",
                text_color="darkslategray",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Plots the time histogram of all the selected files on the same graph.",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  5. Plot time histograms using default ROI0",
                text_color="dark blue",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Plots the time histogram of all the selected files on the same graph within the range of the ROI.",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  6. Get arrival time and temperature",
                text_color="indigo",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Fits the time histogram with a Gaussian to export arrival time, time width and temperature to metadata.",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [sg.Text("MCP - single file", text_color="black", font="Helvetica 12 bold")],
        [
            sg.Text(
                "  1. Plot time histogram",
                text_color="darkslategray",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Plots the time histogram using all data (no ROI)",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  2. Plot histogram data from ROI only",
                text_color="dark blue",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Plots the time and X-Y histograms using the data from default ROI or metadata ROI if defined",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  3. Watch ROI",
                text_color="indigo",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Plots the ROIs defined in metadata",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text(
                "  4. Get arrival time and temperature",
                text_color="dark blue",
                font="Helvetica 11 bold",
            ),
            sg.Text(
                go_to_line(
                    "Plots and fits the time histogram using default ROI0 or ROI0 from metadata if defined and export to metadata",
                    nb_of_chars,
                ),
                font="Helvetica 11",
            ),
        ],
        [
            sg.Text("■", text_color="darkslategray", font="Helvetica 11"),
            sg.Text("You do not need any ROI", font="Helvetica 10"),
            sg.Text("■", text_color="dark blue", font="Helvetica 11"),
            sg.Text("You need the default ROI to be (well) set", font="Helvetica 10"),
            sg.Text("■", text_color="indigo", font="Helvetica 11"),
            sg.Text("You need a ROI to be defined as metadata", font="Helvetica 10"),
        ],
        [sg.Button("Ok thank you have a nice day"), sg.Button("More on the MCP")],
    ]

    # Create the Window
    window = sg.Window("Welcome to the HAL help for MCP scripts", layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if (
            event == sg.WIN_CLOSED or event == "Ok thank you have a nice day"
        ):  # if user closes window or clicks cancel
            break
        elif event == "More on the MCP":
            root = Path().home()
            mcp_info = root / ".HAL/user_scripts/mcp_info/a.png"
            im = Image.open(mcp_info)
            im.show()

    window.close()
