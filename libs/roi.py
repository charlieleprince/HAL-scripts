import numpy as np


def read_metadata(metadata, nb):
    Xmin = metadata["current selection"]["mcp"]["--ROI" + str(nb) + ":Xmin"][0]
    Xmax = metadata["current selection"]["mcp"]["--ROI" + str(nb) + ":Xmax"][0]
    Ymin = metadata["current selection"]["mcp"]["--ROI" + str(nb) + ":Ymin"][0]
    Ymax = metadata["current selection"]["mcp"]["--ROI" + str(nb) + ":Ymax"][0]
    Tmin = metadata["current selection"]["mcp"]["--ROI" + str(nb) + ":Tmin"][0]
    Tmax = metadata["current selection"]["mcp"]["--ROI" + str(nb) + ":Tmax"][0]
    return (Xmin, Xmax, Ymin, Ymax, Tmin, Tmax)


def exportROIinfo(to_mcp: list, ROI, nb) -> None:
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


def filter_data_to_ROI(
    X, Y, T, from_metadata=True, metadata=None, metadata_ROI_nb=0, ROI={}
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if from_metadata is True:
        (Xmin, Xmax, Ymin, Ymax, Tmin, Tmax) = read_metadata(metadata, metadata_ROI_nb)
        T_ROI = T[
            (T > Tmin) & (T < Tmax) & (X > Xmin) & (X < Xmax) & (Y > Ymin) & (Y < Ymax)
        ]
        X_ROI = X[
            (T > Tmin)
            & (T < Tmax)
            & (X > Xmin)
            & (X < Xmax)
            & (X < Xmax)
            & (Y > Ymin)
            & (Y < Ymax)
        ]
        Y_ROI = Y[
            (T > Tmin)
            & (T < Tmax)
            & (X > Xmin)
            & (X < Xmax)
            & (X < Xmax)
            & (Y > Ymin)
            & (Y < Ymax)
        ]
        return (X_ROI, Y_ROI, T_ROI)
    else:
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
        # T_raw_ROI = T_raw[ROI_indices2]
        return (X_ROI, Y_ROI, T_ROI)
