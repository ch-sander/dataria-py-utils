import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from .DATA import sparql_to_dataframe

def calculate_correlation(df=None,
    endpoint_url=None,
    query=None,
    col1=None, col2=None, sep=',', edges=0, save_CSV=True, csv_filename="correlations.csv", heatmap=True, heatmap_kwargs={}, save_PNG=True):
    """
    Calculate correlations between two DataFrame columns.
    Handles both numerical and string columns. String columns are converted into dummy variables.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        endpoint_url (str): The SPARQL endpoint URL to query. Ignored if df or gdf are defined.
        query (str): The SPARQL query to be executed. Ignored if df or gdf are defined.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        sep (str, optional): Separator for string columns (default: ',').
        edges (int, optional): Number of top and bottom rows to return from the correlation DataFrame. Default is 0 (no truncation).
        save_CSV (bool, optional): If True, saves the correlation DataFrame as a CSV file. Default is True.
        csv_filename (str, optional): The filename for saving the CSV file. Default is "correlations.csv".
        heatmap (bool, optional): If True, plots a heatmap of correlations. Default is True.
        heatmap_kwargs (dict, optional): Additional arguments for the heatmap plot.
        save_PNG (bool, optional): If True, saves the correlation heatmap as a PNG file. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with correlation and p-value for each dummy variable if needed.
                      Otherwise, a single correlation and p-value.
    """
    
    if df is None and endpoint_url and query:
        try:
            # Fetch data and create DataFrame
            df = sparql_to_dataframe(endpoint_url, query)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")    
    
    # Validate that columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"One or both columns '{col1}' and '{col2}' do not exist in the DataFrame.")

    # Case 1: Both columns are numeric
    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        correlation, p_val = pearsonr(df[col1], df[col2])
        return pd.DataFrame({'Correlation': [correlation], 'P-Value': [p_val]}, index=[f'{col1} vs {col2}'])

    # Case 2: One or both columns are strings -> Create dummy variables
    if pd.api.types.is_string_dtype(df[col1]):
        col1_dummies = df[col1].str.get_dummies(sep=sep)
    else:
        col1_dummies = df[[col1]]  # Keep original column

    if pd.api.types.is_string_dtype(df[col2]):
        col2_dummies = df[col2].str.get_dummies(sep=sep)
    else:
        col2_dummies = df[[col2]]  # Keep original column

    # Combine dummies with the original DataFrame
    combined_df = pd.concat([df, col1_dummies, col2_dummies], axis=1)

    # Initialize a dictionary to store correlations
    correlations = {}

    # Iterate over all combinations of dummy columns
    for col1_dummy in col1_dummies.columns:
        for col2_dummy in col2_dummies.columns:
            try:
                correlation, p_val = pearsonr(combined_df[col1_dummy], combined_df[col2_dummy])
                correlations[f'{col1_dummy} vs {col2_dummy}'] = {'Correlation': correlation, 'P-Value': p_val}
            except Exception as e:
                print(f"Failed to calculate correlation for {col1_dummy} vs {col2_dummy}: {e}")

    # Convert correlations to a DataFrame
    correlation_df = pd.DataFrame.from_dict(correlations, orient='index')
    correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

    # Truncate the DataFrame if edges > 0
    if edges > 0:
        correlation_df = pd.concat([correlation_df.head(edges), correlation_df.tail(edges)])

    # Save the correlation DataFrame as a CSV file
    if save_CSV:
        try:
            correlation_df.to_csv(csv_filename)
        except Exception as e:
            print(f"Failed to save CSV file '{csv_filename}': {e}")

    # Plot a heatmap if requested
    if heatmap:
        try:
            plot_correlation_heatmap(correlation_df, save_PNG, **heatmap_kwargs)
        except Exception as e:
            print(f"Failed to plot heatmap: {e}")

    return correlation_df

def plot_correlation_heatmap(correlation_df, corr_col='Correlation', save_PNG=True, title="Correlation Heatmap", figsize=(10, 8)):
    """
    Create a heatmap plot for correlation values.

    Args:
        correlation_df (pd.DataFrame): A DataFrame containing correlation results. 
                                       Index should contain variable names.
        title (str, optional): Title of the heatmap. Default is "Correlation Heatmap".
        figsize (tuple, optional): Size of the heatmap figure. Default is (10, 8).
    """
    # Reshape the DataFrame for the heatmap
    heatmap_data = correlation_df[corr_col].unstack() if isinstance(correlation_df.index, pd.MultiIndex) else correlation_df

    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5, 
        cbar_kws={'label': corr_col}
    )
    plt.title(title)
    if save_PNG:
        plt.savefig("correlations.png", dpi=300, format='png')
    plt.show()