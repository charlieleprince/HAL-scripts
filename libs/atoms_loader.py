# -*- coding: utf-8 -*-

from pathlib import Path
import itertools as itt
import collections
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np


N_proc = cpu_count() - 2


def df_from_path_tuple(path_list):
    dfs = []
    for file in path_list:
        try:
            dfs.append(
                pd.read_csv(
                    file,
                    sep="\t",
                    names=["X1", "X2", "Y1", "Y2"],
                    usecols=[0, 1, 2, 3],
                    header=None,
                )
            )
        except:
            pass
    return (pd.concat(dfs, ignore_index=True).dropna() / 120.0e-12).astype(int)


def df_from_path_tuple_bin(path_list):
    dt = np.dtype([("X1", "u8"), ("X2", "u8"), ("Y1", "u8"), ("Y2", "u8")])
    dfs = []
    for file in path_list:
        try:
            data = np.fromfile(file, dtype=dt)
            dfs.append(pd.DataFrame(data))
        except:
            pass
    return pd.concat(dfs, ignore_index=True).dropna().astype(int)


def multiproc_raw_atoms_loader(data, rawmode="bin"):

    filenames = data
    if rawmode == "bin":
        chunk_length = int(
            np.ceil(collections.Counter(p.suffix for p in filenames)[".atoms"] / N_proc)
        )
        chunked_lists: tuple[tuple[Path], ...] = tuple(
            itt.zip_longest(*[filenames] * int(chunk_length), fillvalue=None)
        )

        result_list = []
        with Pool(processes=8) as pool:
            for res in pool.imap(df_from_path_tuple_bin, chunked_lists):
                result_list.append(res)
        df = pd.concat(result_list, ignore_index=True)
        del result_list

    elif rawmode == "txt":
        chunk_length = int(
            np.ceil(collections.Counter(p.suffix for p in filenames)[".atoms"] / N_proc)
        )
        chunked_lists: tuple[tuple[Path], ...] = tuple(
            itt.zip_longest(*[filenames] * int(chunk_length), fillvalue=None)
        )

        result_list = []
        with Pool(processes=8) as pool:
            for res in pool.imap(df_from_path_tuple, chunked_lists):
                result_list.append(res)
        df = pd.concat(result_list, ignore_index=True)
        del result_list

    else:
        raise

    return df
