# ======================================================================= 
# This file is part of the CS519_Project_3 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import pandas as pd
import numpy as np
import os


def preprocess_REALDISP():
    num_subjects = 17
    subjects = range(1, num_subjects + 1)
    file_names = ["subject%d_ideal.log" % x for x in subjects]
    path = r"D:\Library\Desktop\realistic_sensor_displacement"
    frames = []

    for counter, name in enumerate(file_names):
        # Reading the file
        raw_ds = pd.read_csv(os.path.join(path, name), sep="\t", header=None)

        # Removing the missing values
        raw_ds.replace(["na", "nan", "NaN", "NaT", "inf", "-inf", "?"], np.nan, inplace=True)
        raw_ds = raw_ds.dropna()#.values.astype(float)

        # Removing the timestamps (columns 1 and 2)
        raw_ds = raw_ds.drop(raw_ds.columns[[0, 1]], axis=1)

        frames.append(raw_ds)

    # Merging subject files
    return pd.concat(frames)
