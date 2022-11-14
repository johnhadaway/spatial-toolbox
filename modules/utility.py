import geopandas as gpd
import pandas as pd
import pysal as ps
import matplotlib.pyplot as plt


"""
Trivially convert the CRS of a geodataframe to a specified CRS
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
Plot the geodataframe with a column of values
input:
    gdf: geodataframe
    column: column name of the column to plot
    cmap: colormap to use
    figsize: figure size
output:
    fig: figure
    ax: axis
"""


def plot_gdf(gdf, column=None, cmap='viridis', figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(ax=ax, column=column, cmap=cmap)
    plt.show()
    return fig, ax


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
        w = ps.weights.Rook.from_dataframe(gdf, idVariable=id_col)
    elif w_type == 'queen':
        w = ps.weights.Queen.from_dataframe(gdf, idVariable=id_col)
    elif (w_type == 'knn') and (k is not None):
        w = ps.weights.KNN.from_dataframe(gdf, k=4, idVariable=id_col)
    else:
        w = None
    return w
