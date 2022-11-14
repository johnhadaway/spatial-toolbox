import geopandas as gpd
import pandas as pd
import pysal as ps
from utility import *


"""
Communality: the percentage of visits that a user makes to a given location, relative to the total number of visits to that location
input:
    gdf: geodataframe
    total_users_col: column name of the total users column
    total_visits_col: column name of the total visits column
output:
    gdf: geodataframe with communality column
"""


def communality_calc(gdf, total_users_col, total_visits_col, prefix=""):
    gdf[prefix + "visits_per_user"] = gdf[total_visits_col]/gdf[total_users_col]
    gdf[prefix + "communality"] = gdf["visits_per_user" + prefix] / \
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
        gdf[col+"_rel_freq"] = gdf[col]/gdf[cols].sum(axis=1)
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


def local_moran_calc(gdf, col, w):
    gdf["local_moran"] = ps.Moran_Local(gdf[col], w)
    return gdf


# ENTROPY
# different functions for different entropy calculations, and one function to call them all (takes a geodataframe and a list of column names, whose values are counts, and returns a geodataframe with the entropy values added as columns)

"""
"""
