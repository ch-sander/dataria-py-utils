import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import upsetplot as up
from fuzzywuzzy import fuzz
from .DATA import sparql_to_dataframe

def correlation(df=None,
    endpoint_url=None,
    query=None,
    col1=None, col2=None, sep=',', edges=0, csv_filename="correlations.csv", heatmap=True, heatmap_kwargs={}, save_PNG=True, verbose=True):
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
        csv_filename (str, optional): The filename for saving the CSV file.
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
            df = sparql_to_dataframe(endpoint_url, query, csv_filename=f"query_{csv_filename}" if csv_filename is not None else None)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")    
    
    # Validate that columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"One or both columns '{col1}' and '{col2}' do not exist in the DataFrame.")
    
    df = df.dropna(subset=[col1, col2]) # only rows with valid content

    if col1 == col2:
        # Case: Both columns are the same
        if pd.api.types.is_numeric_dtype(df[col1]):
            # Both numerical, calculate the self-correlation
            correlation, p_val = pearsonr(df[col1], df[col2])
            return pd.DataFrame({'Correlation': [correlation], 'P-Value': [p_val]}, index=[f'{col1} vs {col2}'])
        elif pd.api.types.is_string_dtype(df[col1]):
            # Both are string, create dummy variables once
            dummies = df[col1].str.get_dummies(sep=sep)
            
            if dummies.shape[1] < 2:
                raise ValueError(f"Not enough dummy variables in '{col1}' to calculate correlations.")
            
            correlation_matrix = dummies.corr(method='pearson')
            correlation_matrix = correlation_matrix.where(~np.eye(correlation_matrix.shape[0], dtype=bool))
            correlation_df = correlation_matrix.unstack().dropna().reset_index()
            correlation_df.columns = ['Var_1', 'Var_2', 'Correlation']
            correlation_df = correlation_df[correlation_df['Var_1'] < correlation_df['Var_2']]
            
    else:
        # Case 1: Both columns are numeric
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            correlation, p_val = pearsonr(df[col1], df[col2])
            return pd.DataFrame({'Correlation': [correlation], 'P-Value': [p_val]}, index=[f'{col1} vs {col2}'])

        # Case 2: One or both columns are strings -> Create dummy variables
        if pd.api.types.is_string_dtype(df[col1]):
            col1_dummies = df[col1].str.get_dummies(sep=sep)
        else:
            col1_dummies = df[[col1]]

        if pd.api.types.is_string_dtype(df[col2]):
            col2_dummies = df[col2].str.get_dummies(sep=sep)
        else:
            col2_dummies = df[[col2]]

        # Combine dummies with the original DataFrame
        df = df.drop(columns=[col1, col2])
        combined_df = pd.concat([df, col1_dummies, col2_dummies], axis=1)        

        # Initialize a dictionary to store correlations
        correlations = {}

        # Iterate over all combinations of dummy columns
        for col1_dummy in col1_dummies.columns:
            for col2_dummy in col2_dummies.columns:
                try:
                    if combined_df[col1_dummy].std() == 0 or combined_df[col2_dummy].std() == 0:
                        continue
                    correlation, p_val = pearsonr(combined_df[col1_dummy], combined_df[col2_dummy])
                    correlations[f'{col1_dummy} vs {col2_dummy}'] = {'Correlation': correlation, 'P-Value': p_val}
                except Exception as e:
                    print(f"Failed to calculate correlation for {col1_dummy} vs {col2_dummy}: {e}")

        # Convert correlations to a DataFrame
        correlation_df = pd.DataFrame.from_dict(correlations, orient='index')
    
    correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)
    if verbose:
        print(correlation_df.info())
        correlation_df.describe()


    # Truncate the DataFrame if edges > 0
    if edges > 0:
        correlation_df = correlation_df.dropna()        
        correlation_df = pd.concat([correlation_df.head(edges), correlation_df.tail(edges)])

    # Save the correlation DataFrame as a CSV file
    if len(csv_filename) > 0 and csv_filename is not None:
        try:
            correlation_df.to_csv(csv_filename)
        except Exception as e:
            print(f"Failed to save CSV file '{csv_filename}': {e}")

    # Plot a heatmap if requested
    if heatmap:
        try:
            plot_correlation_heatmap(correlation_df=correlation_df, save_PNG=save_PNG, **heatmap_kwargs)
        except Exception as e:
            print(f"Failed to plot heatmap: {e}")

    return correlation_df

def plot_correlation_heatmap(correlation_df, corr_col='Correlation', save_PNG=True, title="Correlation Heatmap", figsize=(10, 8), **heatmap_kwargs):
    """
    Create a heatmap plot for correlation values.

    Args:
        correlation_df (pd.DataFrame): A DataFrame containing correlation results. 
                                       Index should contain variable names.
        title (str, optional): Title of the heatmap. Default is "Correlation Heatmap".
        figsize (tuple, optional): Size of the heatmap figure. Default is (10, 8).
    """
    correlation_df = correlation_df.dropna()
    if {corr_col, 'P-Value'}.issubset(correlation_df.columns) and 'Var_1' not in correlation_df.columns:
        temp_df = correlation_df.copy().reset_index()
        plt.figure(figsize=figsize)
        sns.barplot(
            x=corr_col,
            y='index',
            data=temp_df,
            palette='viridis',
            hue='index',
            **heatmap_kwargs
        )
        plt.tight_layout()
    else:
      if {'Var_1', 'Var_2', corr_col}.issubset(correlation_df.columns):
          # Long format
          heatmap_matrix = correlation_df.pivot(index='Var_1', columns='Var_2', values=corr_col)
          # As the matrix is symmetrical, fill in the missing values
          heatmap_matrix = heatmap_matrix.fillna(heatmap_matrix.T)
      elif correlation_df.shape[0] == correlation_df.shape[1]:
          # Square matrix, use it directly
          heatmap_matrix = correlation_df
      elif correlation_df.shape[1] == 1:
          # Single correlation value, create a 1x1 matrix
          correlation = correlation_df.iloc[0,0]
          heatmap_matrix = pd.DataFrame([[correlation]], index=['Var1'], columns=['Var2'])
      else:
          try:
              heatmap_matrix = correlation_df[corr_col].unstack() if isinstance(correlation_df.index, pd.MultiIndex) else correlation_df
          
          except:
              raise ValueError("Unknown format of the correlation DataFrame.")

      # Create the heatmap
      plt.figure(figsize=figsize)
      sns.heatmap(
          heatmap_matrix,
          annot=True, 
          cmap='coolwarm', 
          fmt=".4f", 
          linewidths=.5, 
          cbar_kws={'label': corr_col},
          **heatmap_kwargs
      )
      plt.tight_layout()
    plt.title(title)
    if save_PNG:
        plt.savefig("correlations.png", dpi=300, format='png')
    plt.show()

def upset(
    df=None, endpoint_url=None, query=None, col_item="item", col_sets="set", sep=",", 
    csv_filename="upset_data.csv", plot_upset=True, png_filename="upset_plot.png", verbose=True, **upset_kwargs
):
    """
    Converts a DataFrame with an item column and a delimited set column into a format suitable for upset.js.

    Args:
        df (pd.DataFrame): Input DataFrame with two columns (item, sets).
        endpoint_url (str): SPARQL endpoint URL (ignored if df is provided).
        query (str): SPARQL query to fetch data (ignored if df is provided).
        col_item (str): Column name for the item (default: "item").
        col_sets (str): Column name for the sets (default: "set").
        sep (str): Separator used in the 'sets' column (default: ',').
        csv_filename (str): Filename for saving the CSV file.
        plot_upset (bool): If True, generates an UpSet plot.
        png_filename (str): Filename for saving the PNG file.
        **upset_kwargs: Additional keyword arguments passed to up.UpSet()
    Returns:
        pd.DataFrame: Transformed DataFrame suitable for upset.js.
    """

    if df is None and endpoint_url and query:
        df = sparql_to_dataframe(endpoint_url, query, csv_filename=f"query_{csv_filename}" if csv_filename is not None else None)
    if col_item not in df.columns or col_sets not in df.columns:
        raise ValueError(f"Expected columns '{col_item}' and '{col_sets}' in DataFrame.")
    df_expanded = df[col_sets].str.get_dummies(sep=sep)
    df_final = pd.concat([df[[col_item]], df_expanded], axis=1)

    if len(csv_filename) > 0 and csv_filename is not None:
        df_final.to_csv(csv_filename, index=False)

    if plot_upset:
        upset_data = df_final.set_index(col_item).astype(bool)
        upset_data = upset_data.groupby(list(upset_data.columns)).size()
        plot = up.UpSet(upset_data,**upset_kwargs)
        plot.plot()
        plt.title(f"UpSet Plot")

        if len(png_filename) > 0 and png_filename is not None:
            plt.savefig(png_filename, dpi=300)

        plt.show()
        if verbose:
            print(upset_data.info())
            upset_data.describe()
        
    return df_final

def fuzzy_compare(df1=None,df2=None,
    endpoint_url=None,
    query=None,
    grouping_var=None, label_var=None, element_var=None, threshold=95, match_all=False, unique_rows=False, csv_filename="comparison.csv", verbose= True):
    """
    Args:
        df1 (pd.DataFrame): The input DataFrame containing the data.
        df2 (pd.DataFrame): The input DataFrame containing the data.
        endpoint_url (str): The SPARQL endpoint URL to query. Ignored if df or gdf are defined.
        query (str): The SPARQL query to be executed. Ignored if df or gdf are defined.
        grouping_var (str, optional): The name of column to aggregate on.
        label_var (str, optional): The name of the column to match on.
        element_var (str): The name of the column to compare.
        threshold (int): The threshold for fuzzy compare.
        match_all (bool, optional): If True, only return grouped results for a full match accross all elements.
        unique_rows (bool, optional): If True, only return one row per hit.
        csv_filename (str, optional): The filename for saving the CSV file. Default is "comparison.csv".

    Returns:
        pd.DataFrame: A DataFrame with each element compared against all other elements.
    """

    if df1 is None and endpoint_url and query:
        try:
            # Fetch data and create DataFrame
            df1 = sparql_to_dataframe(endpoint_url, query, csv_filename=f"query1_{csv_filename}" if csv_filename is not None else None)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")

    if df2 is None and endpoint_url and query:
        try:
            # Fetch data and create DataFrame
            df2 = sparql_to_dataframe(endpoint_url, query, csv_filename=f"query2_{csv_filename}" if csv_filename is not None else None)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")

    if df2 is None:
        df2 = df1

    # Validate that columns exist in the DataFrame
    if element_var not in df1.columns or element_var not in df2.columns:
        raise ValueError(f"Column '{element_var}' do not exist in the DataFrame.")

    grouping = False
    # Grouping only if grouping_var is set and present in both DataFrames.
    if grouping_var and grouping_var in df1.columns and grouping_var in df2.columns:
        grouping = True
        groups_df1 = {g: group for g, group in df1.groupby(grouping_var)}
        groups_df2 = {g: group for g, group in df2.groupby(grouping_var)}
    else:
        groups_df1 = {'all': df1}
        groups_df2 = {'all': df2}

    # Optional: Check whether there are values in the Label column in both DataFrames.
    if label_var in df1.columns and label_var in df2.columns and df1[label_var].notna().any() and df2[label_var].notna().any():
        check_label = True
    else:
        check_label = False

    matches = []

    # Compare all combinations of the (optionally grouped) DataFrames
    for group_key2, group2 in groups_df2.items():
        for group_key1, group1 in groups_df1.items():
            if grouping and group_key2 == group_key1:
                continue
            if grouping and group_key1 >= group_key2 and unique_rows:
                continue
            for _, row2 in group2.iterrows():
                for _, row1 in group1.iterrows():
                    # If check_label, the labels must match.
                    # Maybe add score = 0 for non existing row1?
                    if check_label and row2[label_var] != row1[label_var]:
                        continue

                    if threshold >= 100:
                        if row2[element_var].lower() == row1[element_var].lower():
                            score = 100
                        else:
                            score = 0
                    else:
                        score = fuzz.ratio(row2[element_var].lower(), row1[element_var].lower())

                    matches.append({
                        'group2': group_key2,
                        'group1': group_key1,
                        'label': row2[label_var] if check_label else None,
                        'element2': row2[element_var],
                        'element1': row1[element_var],
                        'score': score
                    })
    
    matches_df = pd.DataFrame(matches)
    aggregated = pd.DataFrame()
    if verbose:
        print(matches_df.info())
        matches_df.describe()

    if not match_all:
        matches_df = matches_df[matches_df['score'] >= threshold]

    if not matches_df.empty:
        aggregated = matches_df.groupby(['group1', 'group2']).agg(
            Labels=('label', lambda x: ", ".join(sorted(set(x)))),
            df1_Elements=('element1', lambda x: ", ".join(sorted(set(x)))),
            df2_Elements=('element2', lambda x: ", ".join(sorted(set(x)))),
            Num_Matches=('score', 'count'),
            Average_Score=('score', 'mean'),
            Min_Score=('score', 'min'),
            Max_Score=('score', 'max')
        ).reset_index()
        if verbose:
            print(aggregated.info())
            aggregated.describe()
    else:
        aggregated = pd.DataFrame()
        print("aggregated dataframe is empty!")
        return aggregated
    


    if match_all and grouping:
        aggregated = aggregated[aggregated['Min_Score'] >= threshold]

    aggregated = aggregated[aggregated['Max_Score'] >= threshold]
    print(len(aggregated))
    # Save the correlation DataFrame as a CSV file
    if len(csv_filename) > 0 and csv_filename is not None:
        try:
            aggregated.to_csv(csv_filename)
        except Exception as e:
            print(f"Failed to save CSV file '{csv_filename}': {e}")

    return aggregated