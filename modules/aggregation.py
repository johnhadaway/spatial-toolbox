import geopandas as gpd
import pandas as pd
from modules.utility import *


"""
Aggregate points, by category, to polygon

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
