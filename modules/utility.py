import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


def convert_crs(gdf, crs):
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return gdf


def plot_gdf(gdf, column=None, cmap='viridis', figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(ax=ax, column=column, cmap=cmap)
    plt.show()
    return fig, ax
