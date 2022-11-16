import matplotlib.pyplot as plt
import geopandas as gpd


"""
Trivially plot a geodataframe's column using Matplotlib
input:
    gdf: geodataframe
    column: column name of the column to plot
    cmap: colormap to use
    figsize: figure size
output:
    fig: figure
    ax: axis
"""


def plot_gdf(gdf, col=None, cmap='viridis', figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(ax=ax, column=col, cmap=cmap)
    plt.show()
    return fig, ax
