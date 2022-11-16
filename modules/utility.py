import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen, Rook, KNN


"""
Trivially convert the CRS of a geodataframe to a specific CRS
input:
    gdf: geodataframe
    crs: CRS to convert to
output:
    gdf: geodataframe with converted CRS
"""


def convert_crs(gdf, crs):
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return gdf


"""
Generate a weights matrix for a geodataframe
input:
    gdf: geodataframe
    w_type: type of weights matrix to generate (rook, queen, knn)
    id_col: column name of the id column
    k: number of neighbors for knn weights matrix
output:
    w: weights matrix
"""


def weights_matrix(gdf, w_type='rook', id_col='id', k=None):
    if w_type == 'rook':
        w = Rook.from_dataframe(
            gdf, idVariable=id_col, silence_warnings=True)
    elif w_type == 'queen':
        w = Queen.from_dataframe(
            gdf, idVariable=id_col, silence_warnings=True)
    elif (w_type == 'knn') and (k is not None):
        w = KNN.from_dataframe(
            gdf, k=4, idVariable=id_col, silence_warnings=True)
    else:
        w = None
    return w
