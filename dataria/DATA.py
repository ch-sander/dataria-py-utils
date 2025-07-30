from SPARQLWrapper import SPARQLWrapper, JSON
import json
import shapely.wkt
from shapely.geometry import shape
import pandas as pd
from datetime import datetime, timezone

def sparql_to_dataframe(endpoint_url, query, csv_filename="query_result.csv"):
    """
    Execute a SPARQL query and convert the results into a Pandas DataFrame.

    Supports parsing of geometry (WKT, GeoJSON), numeric types, and xsd:date/xsd:dateTime fields.
    Can optionally save results as CSV.

    Args:
        endpoint_url (str): SPARQL endpoint URL.
        query (str): SPARQL query string.
        csv_filename (str): Path to save the CSV result. If None, no file is written.

    Returns:
        pd.DataFrame: The query results as a DataFrame with parsed values.
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
    if len(csv_filename) > 0 and csv_filename is not None:
        df_result.to_csv(csv_filename, index=False)

    return df_result


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
    """
    Convert an ISO date string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) to a Pandas Period.

    Intended for historical date handling when datetime parsing is not viable.

    Args:
        iso_string (str): ISO 8601 date string.

    Returns:
        pd.Period or pd.NaT: A Period with daily frequency or NaT if parsing fails.
    """
    parts = iso_string.split("T")[0]
    try:
      year, month, day = map(int, parts.split("-"))
      return pd.Period(year=year, month=month, day=day, freq='D')
    except:
      return pd.NaT

def parse_xsd_date_or_datetime(iso_string, dtype, unix_year=1950):
    """
    Parse xsd:date or xsd:dateTime strings into Pandas datetime or period values.

    Automatically handles early historical dates by converting them into Periods
    to avoid datetime parsing issues (e.g., pre-1950).

    Args:
        iso_string (str): ISO-formatted date or datetime string.
        dtype (str): Expected data type (e.g. xsd:date or xsd:dateTime).
        unix_year (int): Dates earlier than this year are treated as historical (default: 1950).

    Returns:
        pd.Timestamp or pd.Period: Parsed date/time object, or NaT on failure.
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

def get_token_matrix(series, sep=" ", dummies=True):
    """
    Generate a token matrix (either binary or count-based) from a Pandas Series.

    Args:
        series (pd.Series): The input column containing string data.
        sep (str): Separator used to split tokens.
        dummies (bool): If True, return binary (0/1) presence; if False, return token counts.

    Returns:
        pd.DataFrame: A DataFrame with one column per token and one row per entry.
    """
    series = series.fillna("").astype(str)
    
    if dummies:
        return series.str.get_dummies(sep=sep)
    else:
        return (
            series
            .str.split(sep)
            .apply(lambda tokens: pd.Series(tokens).value_counts())
            .fillna(0)
            .astype(int)
        )