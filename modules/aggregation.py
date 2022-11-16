import geopandas as gpd
import pandas as pd
from modules.utility import *


"""
Aggregate data in a point layer, by category, to a polygon layer

input:
    point_gdf: point geodataframe
    poly_gdf: polygon geodataframe
    poly_id_col: column name of the polygon id column
    point_cat_col: column name of the categorical column
    point_val_cols: list of column names of the values in the point geodataframe to aggregate
    agg_funcs: list of aggregation functions to apply to the point values (default: sum)
output:
    poly_gdf_agg: geopandas dataframe of polygons with aggregated point values
"""


def aggregate_by_cat_to_poly(point_gdf, poly_gdf, poly_id_col, point_cat_col, point_val_cols, agg_funcs=["sum"]):
    point_gdf = convert_crs(point_gdf, poly_gdf.crs)
    poly_sindex = poly_gdf.sindex

    point_gdf[poly_id_col] = point_gdf.apply(
        lambda x: list(poly_sindex.intersection(x.geometry.bounds)), axis=1)
    point_gdf[poly_id_col] = point_gdf[poly_id_col].apply(
        lambda x: poly_gdf.iloc[x][poly_id_col].values[0] if len(x) > 0 else None)

    point_gdf_grouped = point_gdf.groupby([poly_id_col, point_cat_col])
    point_gdf_agg = point_gdf_grouped[point_val_cols].agg(agg_funcs)

    total_data = point_gdf_agg.groupby(level=0).sum()
    total_data.columns = ["_".join(col).strip()
                          for col in total_data.columns.values]
    total_data = total_data.reset_index()
    total_data = total_data.drop(
        [col for col in total_data.columns if any(x in col for x in ["mean", "max", "min"])], axis=1)

    point_gdf_agg = point_gdf_agg.unstack()
    point_gdf_agg.columns = ['_'.join(col).strip()
                             for col in point_gdf_agg.columns.values]
    point_gdf_agg = point_gdf_agg.reset_index()
    point_gdf_agg = point_gdf_agg.merge(
        total_data, on=poly_id_col, how='left')
    point_gdf_agg = point_gdf_agg.fillna(0)

    poly_gdf_agg = poly_gdf[poly_gdf.columns].merge(
        point_gdf_agg, left_on=poly_id_col, right_on=poly_id_col, how='left')

    return poly_gdf_agg


"""
Isolate polygons in an origin polygon layer whose centroids are within a specific row
in a destination polygon layer, and save as a new geodataframe.

input:
    input_poly_gdf: input polygon geodataframe
    input_poly_id_col: column name of the input polygon id column
    dest_poly_gdf: destination polygon geodataframe
    dest_poly_id_col: column name of the destination polygon id column
    row_id: the single row id of the destination polygon layer to filter by
output:
    within_gdf: geopandas dataframe of polygons within the selected destination polygon
"""


def isolate_poly_within_dest_poly_by_centroid(input_poly_gdf, input_poly_id_col, dest_poly_gdf, dest_poly_id_col, row_id):
    input_poly_gdf = convert_crs(input_poly_gdf, dest_poly_gdf.crs)

    if row_id not in dest_poly_gdf[dest_poly_id_col].values:
        print("Row ID not in destination polygon layer")
        return None

    input_poly_gdf_centroid = input_poly_gdf.copy()
    input_poly_gdf_centroid['geometry'] = input_poly_gdf_centroid.centroid

    dest_poly_gdf = dest_poly_gdf[dest_poly_gdf[dest_poly_id_col] == row_id]
    input_poly_gdf_centroid_sindex = input_poly_gdf_centroid.sindex

    within_ids = []
    for index, row in dest_poly_gdf.iterrows():
        possible_matches_index = list(
            input_poly_gdf_centroid_sindex.intersection(row.geometry.bounds))
        possible_matches = input_poly_gdf_centroid.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(
            row.geometry)]
        within_ids.extend(precise_matches[input_poly_id_col].tolist())

    within_gdf = input_poly_gdf[input_poly_gdf[input_poly_id_col].isin(
        within_ids)]

    return within_gdf


"""
Give selected attributes of polygons in a polygon layer to polygons whose 
centroids fall within them, and save as a new geodataframe.

input:
    input_poly_gdf: input polygon geodataframe
    dest_poly_gdf: destination polygon geodataframe
    poly_attr_cols: list of column names of the attributes in the dest polygon geodataframe to give to the input polygons
output:
    poly_gdf: geopandas dataframe of points with polygon attributes
"""


def give_poly_attributes_to_poly_centroid_within_poly(input_poly_gdf, dest_poly_gdf, poly_attr_cols):
    input_poly_gdf = convert_crs(input_poly_gdf, dest_poly_gdf.crs)

    input_poly_gdf_centroid = input_poly_gdf.copy()
    input_poly_gdf_centroid['geometry'] = input_poly_gdf_centroid.centroid

    for index, row in input_poly_gdf_centroid.iterrows():
        possible_matches_index = list(
            dest_poly_gdf.sindex.intersection(row.geometry.bounds))
        possible_matches = dest_poly_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(
            row.geometry)]
        if len(precise_matches) > 0:
            for attr_col in poly_attr_cols:
                input_poly_gdf_centroid.loc[index,
                                            attr_col] = precise_matches.iloc[0][attr_col]

    input_poly_gdf = input_poly_gdf.merge(
        input_poly_gdf_centroid[poly_attr_cols], left_index=True, right_index=True, how='left')

    return input_poly_gdf


"""
Give selected attributes of polygons in a polygon layer to the points within 
them, and save as a new geodataframe

input:
    point_gdf: point geodataframe
    poly_gdf: polygon geodataframe
    poly_id_col: column name of the polygon id column
    poly_attr_cols: list of column names of the attributes in the polygon geodataframe to give to the points
output:
    point_gdf: geopandas dataframe of points with polygon attributes
"""


def give_poly_attributes_to_points_within_poly(point_gdf, poly_gdf, poly_id_col, poly_attr_cols):
    point_gdf = convert_crs(point_gdf, poly_gdf.crs)
    poly_sindex = poly_gdf.sindex

    point_gdf[poly_id_col] = point_gdf.apply(
        lambda x: list(poly_sindex.intersection(x.geometry.bounds)), axis=1)
    point_gdf[poly_id_col] = point_gdf[poly_id_col].apply(
        lambda x: poly_gdf.iloc[x][poly_id_col].values[0] if len(x) > 0 else None)

    point_gdf = point_gdf.merge(
        poly_gdf[poly_attr_cols + [poly_id_col]], on=poly_id_col, how='left')

    return point_gdf


"""
Generate an h3 hexagon grid from a geodataframe
input:
    gdf: geodataframe
    resolution: h3 resolution
output:
    gdf: geodataframe with h3 hexagons, and a column with the h3 resolution
"""


def generate_h3_grid(gdf, resolution=9):
    gdf = gdf.copy()
    gdf = gdf.h3.polyfill(
        resolution=resolution, return_geometry=True)
    gdf = gdf.h3.get_resolution()
    return gdf
