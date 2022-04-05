# -*- coding: utf-8 -*-

# IMPORTS
# --------------------------------------------------------------------------------------

# built-in python libs
from typing import (
    Union,
    Optional,
)  # NB: deprecated in python version>=3.10 (should be updated to new notation "|": cf PEP 604 https://docs.python.org/3/library/typing.html#typing.Union)

# third party imports
# -------------------
import numpy as np

# local libs
from .constants import *

# --------------------------------------------------------------------------------------

import numpy as np


def spacetime_to_velocities_converter(
    X: Union[float, np.ndarray],
    Y: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    k_lattice: Optional[float] = None,
) -> tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """Converts the space-time coordinates of atoms detected by the MCP into their corresponding
    speed coordinates. We assume the free fall without interactions. The vertical axis
    is oriented such that a positive v_z correspond to a atom with velocity opposite
    to gravity (and therefore T>T_fall). If a value is given for k_lattice, then the
    MOMENTA in units of k_lattice are returned.

    Parameters
    ----------
    X : Union[float, np.ndarray]
        X (reconstructed) coordinates expressed in MILLIMETERS.
    Y : Union[float, np.ndarray]
        Y (reconstructed) coordinates expressed in MILLIMETERS.
    T : Union[float, np.ndarray]
        T (reconstructed) coordinates expressed in MILLISECONDS.
    k_lattice : Optional[float], optional
        must be expressed in SI units (kg.m/s). If not None, returns (v_x, v_y, v_z) in
        units of k_lattice. By default None

    Returns
    -------
    tuple[ Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray] ]
        (v_x, v_y, v_z) expressed in mm/s OR in units of k_lattice.
    """
    # transverse momenta
    v_x, v_y = X / t_fall, Y / t_fall

    # momenta along gravity axis
    v_z = (0.5 * g * T) - (L_fall / T) * 1e6

    # eventual unit conversion
    if k_lattice is not None:
        v_x = m * v_x / k_lattice
        v_y = m * v_y / k_lattice
        v_z = m * v_z / k_lattice

    return v_x, v_y, v_z
