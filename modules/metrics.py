import geopandas as gpd
import pandas as pd
import libpysal as ps
import numpy as np
from modules.utility import *
from esda.moran import Moran_Local


"""
Communality: the percentage of visits that a user makes to a given location, relative to the total number of visits to that location
input:
    gdf: geodataframe
    total_users_col: column name of the total users column
    total_visits_col: column name of the total visits column
    suffix: suffix to add to the column names
output:
    gdf: geodataframe with communality column
"""


def communality_calc(gdf, total_users_col, total_visits_col, suffix=""):
    gdf = gdf.copy()
    gdf["visits_per_user" + suffix] = gdf[total_visits_col] / gdf[total_users_col]
    gdf["communality" + suffix] = (gdf["visits_per_user" + suffix] /
                                   gdf[total_visits_col]) * 100
    return gdf


"""
Calculate relative frequency of a column in a geodataframe

input:
    gdf: geodataframe
    cols: list of columns to calculate relative frequency for
output:
    gdf: geodataframe with relative frequency columns
"""


def relative_frequency_calc(gdf, cols):
    gdf = gdf.copy()
    for col in cols:
        gdf["rel_freq_" + col] = (gdf[col] / gdf[cols].sum(axis=1)) * 100
    gdf["num_cat_cols_with_value"] = (gdf[cols] > 0).sum(axis=1)
    return gdf


"""
Local Moran's I: a measure of spatial autocorrelation
input:
    gdf: geodataframe
    col: column name of the column to calculate local Moran's I for
    w: weights matrix
    suffix: suffix to add to the column names
output:
    gdf: geodataframe with local Moran's I column
"""


def local_moran_calc(gdf, col, w, suffix=""):
    gdf = gdf.copy()
    with np.errstate(invalid='ignore'):
        lm = Moran_Local(gdf[col], w)
    gdf["local_moran_quad" + suffix] = lm.q
    gdf["local_moran_Is" + suffix] = lm.Is
    gdf["local_moran_p_sim" + suffix] = lm.p_sim
    gdf.loc[gdf["local_moran_p_sim" + suffix] > 0.05,
            "local_moran_quad" + suffix] = 0
    return gdf


"""
Shannon Entropy / Urban Complexity, as suggested by the following paper: "Comparing two methods for Urban Complexity calculation using Shannon-Wiener index" by Jesús López Baeza, Damiano Cerrone, and Kristjan Männigo
input:
    gdf: geodataframe
    count_cols: list of column names of the columns to calculate Shannon Entropy for
    base: base of the logarithm
    suffix: suffix to add to the column name
output:
    gdf: geodataframe with Shannon Entropy column
"""


def shannon_entropy(gdf, count_by_cat_cols, base=2, suffix=""):
    gdf = gdf.copy()
    for index, row in gdf.iterrows():
        counts = [row[col] for col in count_by_cat_cols]
        probs = [count / sum(counts) for count in counts]
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -sum([prob * (np.log(prob) / np.log(base))
                            for prob in probs])
            if np.isnan(entropy):
                entropy = 0
            gdf.loc[index, "shannon_entropy" + suffix] = entropy
    return gdf


"""
Shannon Entropy / Urban Complexity, as suggested by the following paper: "Comparing two methods for Urban Complexity calculation using Shannon-Wiener index" by Jesús López Baeza, Damiano Cerrone, and Kristjan Männigo (local weighted version)
input:
    gdf: geodataframe
    count_cols: list of column names of the columns to calculate Shannon Entropy for
    base: base of the logarithm
    suffix: suffix to add to the column name
output:
    gdf: geodataframe with Shannon Entropy column
"""


def shannon_entropy_local_weighted(gdf, count_by_cat_cols, base=2, suffix=""):
    gdf = gdf.copy()
    for index, row in gdf.iterrows():
        counts = [row[col] for col in count_by_cat_cols]
        probs = [count / sum(counts) for count in counts]
        probs = [prob for prob in probs if prob != 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -sum([prob * (np.log(prob) / np.log(base))
                            for prob in probs])
            if np.isnan(entropy):
                entropy = 0
            gdf.loc[index, "shannon_entropy" + suffix] = entropy * \
                (len(probs) / len(count_by_cat_cols))
    return gdf
