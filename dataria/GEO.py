from SPARQLWrapper import SPARQLWrapper, JSON
import json
import geopandas as gpd
import shapely.wkt
from shapely.geometry import shape
import pandas as pd
from .DATA import sparql_to_dataframe

def dataframe_to_geodataframe(df, geo_var, save_GeoJSON=True):
    """
    Convert a Pandas DataFrame into a GeoPandas GeoDataFrame using a specified geometry column.
    The geometry column is removed from the DataFrame columns and set as the geometry in the GeoDataFrame.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        geo_var (str): The column name to use as geometry.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with a defined geometry column.
    """
    # Validate that the geometry column exists
    if geo_var not in df.columns:
        raise ValueError(f"The specified geometry column '{geo_var}' does not exist in the DataFrame.")

    # Ensure the geometry column contains valid Shapely geometries
    if df[geo_var].isnull().all():
        raise ValueError(f"The geometry column '{geo_var}' contains no valid geometries.")

    # Create the GeoDataFrame and drop the geometry column from the DataFrame
    gdf = gpd.GeoDataFrame(df.drop(columns=[geo_var]), geometry=df[geo_var], crs="EPSG:4326")

    if save_GeoJSON:
        try:
            gdf.to_file("result_query.geojson", driver="GeoJSON")
        except Exception as e:
            print(f"Warning: Failed to save GeoJSON file. Error: {e}")

    return gdf

def explore(df=None,
            gdf=None,
            endpoint_url=None,
            query=None,
            geo_var='geom',
            label_var='label',
            cluster_weight_var='cluster',
            save_CSV=True,
            save_GeoJSON=True,
            save_HTML=True,
            **explore_kwargs):
    """
    Executes a SPARQL query against a specified endpoint, transforms the results into a GeoDataFrame,
    converts them to GeoJSON, and displays them using folium (via gdf.explore).

    Args:
        df : DataFrame
            An existent DataFrame. If absent, endpoint_url and query must be given. 
        gdf : GeoDataFrame
            An existent GeoDataFrame. If absent, df or endpoint_url and query must be defined
        endpoint_url : str
            The SPARQL endpoint URL to query. Ignored if df or gdf are defined.
        query : str
            The SPARQL query to be executed. Ignored if df or gdf are defined.
        geo_var : str, optional
            The variable name in the SPARQL query that contains geometry data. Default is 'geom'.
        label_var : str, optional
            The variable name in the SPARQL query for labels (e.g., rdfs:label). Default is 'label'.
        cluster_weight_var : str, optional
            The variable name containing clustering or weighting values to customize map visualization.
            If None or not present, this column will be ignored. Default is 'cluster'.
        save_CSV : bool, optional
            If True, saves the resulting dataframe as an CSV file. Default is True.
        save_GeoJSON : bool, optional
            If True, saves the GeoDataFrame as a GeoJSON file. Default is True.
        save_HTML : bool, optional
            If True, saves the resulting map as an HTML file. Default is True.
        **explore_kwargs:
            Additional keyword arguments passed to gdf.explore() for customizing the map.

    Returns:
        folium.Map
            The generated folium map with the SPARQL query results.

    Raises:
        ValueError
            If the SPARQL query results cannot be transformed into a valid DataFrame or GeoDataFrame.
    """
    if gdf is None and df is None and (endpoint_url is None or query is None):
        raise ValueError("Either `gdf`, `df`, or both `endpoint_url` and `query` must be provided.")

    if df is None and endpoint_url and query:
        try:
            # Fetch data and create DataFrame
            df = sparql_to_dataframe(endpoint_url, query, save_CSV)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")
    
    if gdf is None and df is not None:
        try:
            # Create GeoDataFrame
            gdf = dataframe_to_geodataframe(df, geo_var, save_GeoJSON)
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
    if save_GeoJSON:
        try:
            gdf.to_file("result_explore.geojson", driver="GeoJSON")
        except Exception as e:
            print(f"Warning: Failed to save GeoJSON file. Error: {e}")

    if save_HTML:
        try:
            m.save("result_map.html")
        except Exception as e:
            print(f"Warning: Failed to save HTML map file. Error: {e}")

    return m