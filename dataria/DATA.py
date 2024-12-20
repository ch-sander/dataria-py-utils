from SPARQLWrapper import SPARQLWrapper, JSON
import json
import shapely.wkt
from shapely.geometry import shape
import pandas as pd
from datetime import datetime, timezone

def sparql_to_dataframe(endpoint_url, query, save_CSV="query_result.csv"):
    """
    Executes a SPARQL query and returns the results as a Pandas DataFrame.
    Geometries and date values are parsed based on their data types.

    Args:
        endpoint_url (str): The SPARQL endpoint URL.
        query (str): The SPARQL query string.
        save_CSV (str): If not None, saves the result as a CSV file to the given path.

    Returns:
        pd.DataFrame: A DataFrame containing the SPARQL query results.
    """
    # Initialize SPARQL Wrapper
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query and get results
    results = sparql.query().convert()

    # Extract variable names
    vars_ = results['head']['vars']

    # Process results into rows
    rows = []
    for b in results['results']['bindings']:
        row = {}
        for var in vars_:
            if var in b:
                val = b[var]['value']
                dtype = b[var].get('datatype', '')

                # Parse geometries based on data type
                if "wktLiteral" in dtype:
                    try:
                        row[var] = shapely.wkt.loads(val)
                    except Exception as e:
                        print(f"Error parsing WKT for variable '{var}': {e}")
                        row[var] = None
                elif "geoJSONLiteral" in dtype:
                    try:
                        row[var] = shape(json.loads(val))
                    except Exception as e:
                        print(f"Error parsing GeoJSON for variable '{var}': {e}")
                        row[var] = None
                elif dtype in [
                    "http://www.w3.org/2001/XMLSchema#date",
                    "http://www.w3.org/2001/XMLSchema#dateTime"
                ]:
                    parsed_date = parse_xsd_date_or_datetime(val, dtype)
                    row[var] = parsed_date
                else:
                    # Convert to numeric types if possible
                    if dtype in [
                        "http://www.w3.org/2001/XMLSchema#integer",
                        "http://www.w3.org/2001/XMLSchema#float",
                        "http://www.w3.org/2001/XMLSchema#double",
                        "http://www.w3.org/2001/XMLSchema#decimal"
                    ]:
                        try:
                            val = int(val) if dtype == "http://www.w3.org/2001/XMLSchema#integer" else float(val)
                        except ValueError:
                            pass  # Keep as string if conversion fails

                    row[var] = val
            else:
                row[var] = None
        rows.append(row)

    df_result = pd.DataFrame(rows)

    # Optionally save the result as CSV
    if save_CSV:
        df_result.to_csv(save_CSV, index=False)

    return df_result

import pandas as pd

# def date_to_epoch_ms(date_str):
#     try:
#         dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
#         dt = dt.replace(tzinfo=timezone.utc)
#         epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
#         delta = dt - epoch
#         return int(delta.total_seconds() * 1000)
#     except (ValueError, TypeError) as e:
#         print(f"Error converting date: {date_str} - {e}")
#         return None

def iso_to_period(iso_string):
    parts = iso_string.split("T")[0]
    try:
      year, month, day = map(int, parts.split("-"))
      return pd.Period(year=year, month=month, day=day, freq='D')
    except:
      return pd.NaT

def parse_xsd_date_or_datetime(iso_string, dtype, unix_year=1700):
    """
    Parses an xsd:date or xsd:dateTime string into a Pandas Timestamp or Period.
    For dates before the specified `unix_year`, a Period is used or the string is retained.
    
    Args:
        iso_string (str): The ISO date or datetime string.
        dtype (str): The data type (xsd:date or xsd:dateTime).
        unix_year (int): The year before which dates will be converted to periods or retained as strings.
    
    Returns:
        pd.Timestamp, pd.Period, or str: The parsed date/time or the original string in case of errors.
    """
    try:
        # Remove the trailing "Z" if present
        if iso_string.endswith("Z"):
            iso_string = iso_string[:-1]
        
        # Extract the year from the iso_string
        # ISO 8601 format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
        year_str = iso_string.split("-")[0]
        year = int(year_str)

        if year < unix_year:
            # Handle dates before the unix_year
            # Create a Period with daily frequency
            return iso_to_period(iso_string)
        else:
            # For dates within the supported range, parse normally
            return pd.to_datetime(iso_string, errors='coerce')
    
    except Exception as e:
        print(f"Error parsing '{dtype}' with value '{iso_string}': {e}")
        return pd.NaT  # Fallback to the original string