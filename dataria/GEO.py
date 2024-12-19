from SPARQLWrapper import SPARQLWrapper, JSON
import json
import geopandas as gpd
import shapely.wkt
from shapely.geometry import shape

def explore(endpoint_url,
                         query,
                         geo_var='geom',
                         label_var='label',
                         cluster_weight_var=None,
                         **explore_kwargs):
    """
    Executes a SPARQL query against a specified endpoint, transforms the results into a GeoDataFrame,
    converts them to GeoJSON, and displays them using folium (via gdf.explore).

    Parameters:
    -----------
    endpoint_url : str
        The SPARQL endpoint URL.
    query : str
        The SPARQL query to be executed.
    geo_var : str, optional
        The variable name in the SPARQL query that holds the geometry. Default: 'geom'.
    label_var : str, optional
        The variable name for a label (e.g., rdfs:label). Default: 'label'.
    cluster_weight_var : str, optional
        The variable name that contains a clustering or weighting value to be used
        as a column in the gdf.explore() function. If None, no column is used.
    **explore_kwargs:
        Additional keyword arguments passed to gdf.explore().

    Returns:
    --------
    m : folium.Map
        The generated folium map.
    """

    # Execute SPARQL query
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # Extract variable names
    vars_ = results['head']['vars']

    # Process results
    rows = []
    for b in results['results']['bindings']:
        row = {}
        geom_val = None
        for var in vars_:
            if var in b:
                val = b[var]['value']
                dtype = b[var].get('datatype', '')

                # Identify geometry
                if var == geo_var:
                    # Geometry can be WKT or GeoJSON
                    if "wktLiteral" in dtype:
                        geom_val = shapely.wkt.loads(val)
                    elif "geoJSONLiteral" in dtype:
                        # Parse GeoJSON
                        geom_val = shape(json.loads(val))
                    else:
                        # If unknown, try WKT (or ignore if it's not a standardized literal)
                        try:
                            geom_val = shapely.wkt.loads(val)
                        except:
                            pass
                else:
                    # Try to convert values to numeric if the datatype indicates a number
                    if dtype in [
                        "http://www.w3.org/2001/XMLSchema#integer",
                        "http://www.w3.org/2001/XMLSchema#float",
                        "http://www.w3.org/2001/XMLSchema#double",
                        "http://www.w3.org/2001/XMLSchema#decimal"
                    ]:
                        # Convert to the appropriate numeric type
                        try:
                            if dtype == "http://www.w3.org/2001/XMLSchema#integer":
                                val = int(val)
                            else:
                                val = float(val)
                        except ValueError:
                            # If conversion fails, keep as string
                            pass
                    row[var] = val
            else:
                row[var] = None

        if geom_val is not None and isinstance(geom_val, (shapely.geometry.base.BaseGeometry)):
            row['geometry'] = geom_val
            rows.append(row)

    if not rows or all('geometry' not in row or row['geometry'] is None for row in rows):
        raise ValueError("No valid geometries in the data. Check the variable 'geo_var'.")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Determine column for gdf.explore (Cluster/Weight)
    # If cluster_weight_var is given and exists, use it
    if cluster_weight_var and cluster_weight_var in gdf.columns:
        explore_kwargs['column'] = cluster_weight_var

    # Use all columns as tooltip/popup
    # gdf.explore allows tooltip and popup as a list of columns
    # We use all except 'geometry'
    non_geom_cols = [c for c in gdf.columns if c != 'geometry']
    # If tooltip/popup not set, set them
    if 'tooltip' not in explore_kwargs:
        explore_kwargs['tooltip'] = non_geom_cols
    if 'popup' not in explore_kwargs:
        explore_kwargs['popup'] = non_geom_cols

    # Create folium map
    m = gdf.explore(**explore_kwargs)
    # Save the GeoDataFrame as GeoJSON
    gdf.to_file("myquery.geojson", driver="GeoJSON")

    # Save the Folium map as HTML
    m.save("result_map.html")


    return m