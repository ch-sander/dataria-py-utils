"""
SPARQL-based geospatial data processing and interactive mapping.

This module provides functions to:
- Convert SPARQL query results into GeoDataFrames,
- Serialize them as GeoJSON,
- Generate interactive maps using GeoPandas' `.explore()` (via folium).
"""


from SPARQLWrapper import SPARQLWrapper, JSON
import json
import geopandas as gpd
import shapely.wkt
from shapely.geometry import shape
import pandas as pd
from .DATA import sparql_to_dataframe

def dataframe_to_geodataframe(df, geo_var, json_filename="result_explore.geojson"):
    """
    Convert a Pandas DataFrame into a GeoPandas GeoDataFrame using a geometry column.

    This function extracts a geometry column from the input DataFrame, converts it into a spatial
    GeoDataFrame, and optionally saves the result as a GeoJSON file.

    Args:
        df (pd.DataFrame): Input DataFrame containing geometry data.
        geo_var (str): Name of the column with Shapely geometry objects.
        json_filename (str, optional): Path to save the resulting GeoJSON file.

    Returns:
        gpd.GeoDataFrame: A spatially enabled GeoDataFrame with WGS84 coordinates.

    Raises:
        ValueError: If the geometry column is missing or empty.
    """
    # Validate that the geometry column exists
    if geo_var not in df.columns:
        raise ValueError(f"The specified geometry column '{geo_var}' does not exist in the DataFrame.")

    # Ensure the geometry column contains valid Shapely geometries
    if df[geo_var].isnull().all():
        raise ValueError(f"The geometry column '{geo_var}' contains no valid geometries.")

    # Create the GeoDataFrame and drop the geometry column from the DataFrame
    gdf = gpd.GeoDataFrame(df.drop(columns=[geo_var]), geometry=df[geo_var], crs="EPSG:4326")

    if len(json_filename) > 0 and json_filename is not None:
        try:
            gdf.to_file(json_filename, driver="GeoJSON")
        except Exception as e:
            print(f"Warning: Failed to save GeoJSON file. Error: {e}")

    return gdf

def explore(df=None,
            gdf=None,
            endpoint_url=None,
            query=None,
            geo_var='geom',
            cluster_weight_var='cluster',
            csv_filename="query_geodata.csv",
            json_filename="result_explore.geojson",
            html_filename="result_map.html",
            **explore_kwargs):
    """
    Generate an interactive map from SPARQL query results or GeoDataFrames.

    This function fetches data via SPARQL (if needed), transforms it into a GeoDataFrame,
    and visualizes the result as an interactive Leaflet map using GeoPandas’ `.explore()` method.
    Results can be exported as CSV, GeoJSON, and HTML files.

    Args:
        df (pd.DataFrame, optional): Input DataFrame. Ignored if `gdf` is provided.
        gdf (gpd.GeoDataFrame, optional): Input GeoDataFrame. Takes precedence over `df`.
        endpoint_url (str, optional): SPARQL endpoint to query.
        query (str, optional): SPARQL query string.
        geo_var (str): Column name containing geometry data (default: 'geom').
        cluster_weight_var (str): Optional column used to color the map.
        csv_filename (str): Optional file path to export query results as CSV.
        json_filename (str): Optional file path to save the data as GeoJSON.
        html_filename (str): Optional file path to save the interactive map as HTML.
        **explore_kwargs: Additional keyword arguments for `GeoDataFrame.explore()`.

    Returns:
        folium.Map: An interactive map object rendered with folium.

    Raises:
        ValueError: If required inputs are missing or data transformation fails.
    """
    if gdf is None and df is None and (endpoint_url is None or query is None):
        raise ValueError("Either `gdf`, `df`, or both `endpoint_url` and `query` must be provided.")

    if df is None and endpoint_url and query:
        try:
            # Fetch data and create DataFrame
            df = sparql_to_dataframe(endpoint_url, query, csv_filename)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")
    
    if gdf is None and df is not None:
        try:
            # Create GeoDataFrame
            gdf = dataframe_to_geodataframe(df, geo_var, json_filename)
        except ValueError as ve:
            raise ValueError(f"GeoDataFrame creation failed. Ensure '{geo_var}' exists and contains valid geometries. Error: {ve}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred while creating GeoDataFrame. Error: {e}")
        
    if gdf is None:
        raise ValueError("Failed to create a GeoDataFrame. Ensure valid inputs.")

    # Check for cluster_weight_var in the GeoDataFrame
    if cluster_weight_var:
        if cluster_weight_var in gdf.columns:
            explore_kwargs['column'] = cluster_weight_var
        else:
            print(f"Warning: Specified cluster_weight_var '{cluster_weight_var}' does not exist in the GeoDataFrame. Skipping column setting.")

    # Automatically set tooltips and popups if not specified
    non_geom_cols = [c for c in gdf.columns if c != 'geometry']
    if 'tooltip' not in explore_kwargs:
        explore_kwargs['tooltip'] = non_geom_cols
    if 'popup' not in explore_kwargs:
        explore_kwargs['popup'] = non_geom_cols

    try:
        # Create folium map
        m = gdf.explore(**explore_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to generate map using gdf.explore(). Error: {e}")

    # Optionally save GeoJSON and HTML
    if len(json_filename) > 0 and json_filename is not None:
        try:
            gdf.to_file(json_filename, driver="GeoJSON")
        except Exception as e:
            print(f"Warning: Failed to save GeoJSON file. Error: {e}")
            
    if len(html_filename) > 0 and html_filename is not None:
        try:
            m.save(html_filename)
        except Exception as e:
            print(f"Warning: Failed to save HTML map file. Error: {e}")

    return m