import geopandas as gpd
import pandas as pd
import pysal as ps
import numpy as np
from modules.utility import *


"""
Communality: the percentage of visits that a user makes to a given location, relative to the total number of visits to that location
input:
    gdf: geodataframe
    total_users_col: column name of the total users column
    total_visits_col: column name of the total visits column
output:
    gdf: geodataframe with communality column
"""


def communality_calc(gdf, total_users_col, total_visits_col, suffix=""):
    gdf["visits_per_user_" + suffix] = gdf[total_visits_col] / gdf[total_users_col]
    gdf["communality_" + suffix] = gdf["visits_per_user_" + suffix] / \
        gdf[total_visits_col]
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
    for col in cols:
        gdf["rel_freq_" + col] = (gdf[col] / gdf[cols].sum(axis=1)) * 100
    return gdf


"""
Local Moran's I: a measure of spatial autocorrelation
input:
    gdf: geodataframe
    col: column name of the column to calculate local Moran's I for
    w: weights matrix
output:
    gdf: geodataframe with local Moran's I column
"""


def local_moran_calc(gdf, col, w, suffix=""):
    gdf["local_moran_" + suffix] = ps.Moran_Local(gdf[col], w)
    return gdf


"""
Shannon Entropy / Urban Complexity, as suggested by the following paper: "Comparing two methods for Urban Complexity calculation using Shannon-Wiener index" by Jesús López Baeza, Damiano Cerrone, and Kristjan Männigo
input:
    gdf: geodataframe
    count_cols: list of column names of the columns to calculate Shannon Entropy for
output:
    gdf: geodataframe with Shannon Entropy column
"""


def shannon_entropy(gdf, count_by_cat_cols, base=2, suffix=""):
    for index, row in gdf.iterrows():
        counts = [row[col] for col in count_by_cat_cols]
        probs = [count / sum(counts) for count in counts]
        entropy = -sum([prob * (np.log(prob) / np.log(base))
                        for prob in probs])
        gdf.loc[index, "shannon_entropy_" + suffix] = entropy

    return gdf


"""
Shannon Entropy / Urban Complexity, as suggested by the following paper: "Comparing two methods for Urban Complexity calculation using Shannon-Wiener index" by Jesús López Baeza, Damiano Cerrone, and Kristjan Männigo -- hacky version, removing probs if they are 0 and then multiplying the result by the share of categories available at the location
"""


def shannon_entropy_local_weighted(gdf, count_by_cat_cols, base=2, suffix=""):
    for index, row in gdf.iterrows():
        counts = [row[col] for col in count_by_cat_cols]
        probs = [count / sum(counts) for count in counts]
        probs = [prob for prob in probs if prob != 0]
        entropy = -sum([prob * (np.log(prob) / np.log(base))
                       for prob in probs])
        gdf.loc[index, "shannon_entropy_" + suffix] = entropy * \
            (len(probs) / len(count_by_cat_cols))
    return gdf
