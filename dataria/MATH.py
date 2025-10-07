import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import upsetplot as up
from rapidfuzz import fuzz, process
from .DATA import sparql_to_dataframe, get_token_matrix

def correlation(df=None,
    endpoint_url=None,
    query=None,
    col1=None, col2=None, sep=';', edges=0, csv_filename="correlations.csv", heatmap=True, heatmap_kwargs={}, dummies_matrix=True, save_PNG=True, verbose=True):
    """
    Compute correlations between two columns of a DataFrame, including support for categorical (string) data.

    This function calculates Pearson correlations, handles dummy encoding for categorical data, supports SPARQL-based data retrieval, and can generate a heatmap of the results.

    Args:
        df (pd.DataFrame, optional): The input DataFrame. If not provided, SPARQL must be used.
        endpoint_url (str, optional): SPARQL endpoint URL.
        query (str, optional): SPARQL query string.
        col1 (str): Name of the first column to compare.
        col2 (str): Name of the second column to compare.
        sep (str, optional): Separator for multi-value string fields. Default is ','.
        edges (int, optional): If > 0, only returns top and bottom N correlations.
        csv_filename (str, optional): File path to save the result as CSV.
        heatmap (bool, optional): Whether to generate a heatmap. Default is True.
        heatmap_kwargs (dict, optional): Additional kwargs passed to the heatmap function.
        dummies_matrix (bool, optional): Treat col1 and col2 as binary dummies. Default is True, else bag of words, count frequency.
        save_PNG (bool, optional): Whether to save the heatmap as a PNG file.
        verbose (bool, optional): Whether to print insights into the dataframe.

    Returns:
        pd.DataFrame: A DataFrame containing correlation values and p-values.
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
            dummies = get_token_matrix(df[col1],sep,dummies_matrix)
            # dummies = df[col1].str.get_dummies(sep=sep)
  
            if dummies.shape[1] < 2:
                raise ValueError(f"Not enough dummy variables in '{col1}' to calculate correlations.")
            
            correlation_matrix = dummies.corr(method='pearson')
            correlation_matrix = correlation_matrix.where(~np.eye(correlation_matrix.shape[0], dtype=bool))
            correlation_df = correlation_matrix.unstack().dropna().reset_index()
            correlation_df.columns = ['Var_1', 'Var_2', 'Correlation']
            correlation_df = correlation_df[correlation_df['Var_1'] < correlation_df['Var_2']]
            
    else:
        # Case: Both columns are different

        if pd.api.types.is_string_dtype(df[col1]):
            col1_dummies_raw = get_token_matrix(df[col1],sep,dummies_matrix) # df[col1].astype(str).str.get_dummies(sep=sep)
        else:
            col1_dummies_raw = df[[col1]].astype(float)

        if pd.api.types.is_string_dtype(df[col2]):
            col2_dummies_raw = get_token_matrix(df[col2],sep,dummies_matrix) # df[col2].astype(str).str.get_dummies(sep=sep)
        else:
            col2_dummies_raw = df[[col2]].astype(float)

        shared_columns = set(col1_dummies_raw.columns) & set(col2_dummies_raw.columns)
        col1_dummies = col1_dummies_raw.add_prefix(f"{col1}__") if shared_columns else col1_dummies_raw
        col2_dummies = col2_dummies_raw.add_prefix(f"{col2}__") if shared_columns else col2_dummies_raw

        # Align indexes
        col1_dummies = col1_dummies.reset_index(drop=True)
        col2_dummies = col2_dummies.reset_index(drop=True)

        # Combine dummies with the original DataFrame
        df = df.drop(columns=[col1, col2])
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
    Generate a heatmap plot from a correlation DataFrame.

    Supports various matrix formats (wide, long, 1x1) and visual customization.

    Args:
        correlation_df (pd.DataFrame): DataFrame containing correlation values.
        corr_col (str): Name of the correlation column. Default is "Correlation".
        save_PNG (bool): Whether to save the heatmap as a PNG.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure (width, height).
        **heatmap_kwargs: Additional arguments for `seaborn.heatmap` or `seaborn.barplot`.

    Returns:
        None
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
    df=None, endpoint_url=None, query=None, col_item="item", col_sets="set", sep=";", 
    csv_filename="upset_data.csv", plot_upset=True, png_filename="upset_plot.png", verbose=True, **upset_kwargs
):
    """
    Generate an UpSet plot and/or a DataFrame suitable for upset.js from set membership data.

    Useful for visualizing overlapping categories or tag combinations. Data can be provided via SPARQL or directly as a DataFrame.

    Args:
        df (pd.DataFrame, optional): Input DataFrame.
        endpoint_url (str, optional): SPARQL endpoint URL.
        query (str, optional): SPARQL query.
        col_item (str): Name of the item column. Default is "item".
        col_sets (str): Name of the set membership column. Default is "set".
        sep (str): Separator used in set column. Default is ','.
        csv_filename (str): File path to save the transformed DataFrame.
        plot_upset (bool): Whether to generate an UpSet plot.
        png_filename (str): File path to save the UpSet plot as PNG.
        verbose (bool, optional): Whether to print insights into the dataframe.
        **upset_kwargs: Additional arguments for `up.UpSet()`.

    Returns:
        pd.DataFrame: The transformed DataFrame (one-hot encoded for sets).
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

def fuzzy_compare(df1, df2=None,
                         match_by=None,
                         grouping_var=None,
                         treat_empty_as_match=False,
                         match_all=True,
                         unique_rows=False,
                         additional_vars=None,
                         csv_filename=None,
                         verbose=True):
    """
    Perform a fast, memory-efficient fuzzy comparison between two DataFrames
    with strict AND logic across multiple columns.

    Each (column, threshold) pair in `match_by` must satisfy its threshold
    for a group to be retained if `match_all=True`.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primary DataFrame for comparison.
    df2 : pd.DataFrame, optional
        Secondary DataFrame to compare against. If None, `df1` is used (self-join).
    match_by : list of (str, int)
        List of tuples specifying columns to compare and their fuzzy thresholds
        (0–100). Columns with threshold >=100 are treated as exact matches.
    grouping_var : str, optional
        Column name used to define group identities between df1 and df2.
    treat_empty_as_match : bool, default=False
        If True, empty strings ("") count as perfect matches (score = 100).
    match_all : bool, default=True
        If True, groups are retained only if *all* match_by conditions meet
        their respective thresholds (strict AND logic). If False, any matching
        condition is sufficient (OR logic).
    unique_rows : bool, default=False
        For self-joins, exclude symmetric duplicates.
    additional_vars : list of str, optional
        Additional columns to include in the aggregated output for context.
    csv_filename : str, optional
        Path to save the aggregated results as CSV. If None, results are not saved.
    verbose : bool, default=True
        If True, print progress and summary information.

    Returns
    -------
    pd.DataFrame
        Aggregated match statistics per (group1, group2) pair, including
        per-column min/avg/max fuzzy scores and optional contextual fields.
    """

    if df2 is None:
        df2 = df1
    self_join = df1 is df2

    if not match_by:
        raise ValueError("Parameter 'match_by' must be specified (list of (column, threshold)).")

    def normalize(s):
        return s.fillna("").astype(str).str.strip().str.lower().to_numpy()

    def uniq(vals):
        return ", ".join(sorted({str(v) for v in vals if pd.notna(v)}))

    df1 = df1.reset_index(drop=True).copy()
    df2 = df2.reset_index(drop=True).copy()
    df1["_i"] = np.arange(len(df1))
    df2["_j"] = np.arange(len(df2))

    exact_cols = [col for col, th in match_by if th >= 100]
    fuzzy_cols = [(col, th) for col, th in match_by if th < 100]

    if exact_cols:
        if verbose:
            print(f"Exact match on: {exact_cols}")
        for col in exact_cols:
            df1[f"__{col}"] = normalize(df1[col])
            df2[f"__{col}"] = normalize(df2[col])
        on_cols = [f"__{c}" for c in exact_cols]
        merged = pd.merge(df1, df2, on=on_cols, suffixes=("_df1", "_df2"))
    else:
        merged = df1.assign(_key=1).merge(df2.assign(_key=1), on="_key", suffixes=("_df1", "_df2")).drop(columns="_key")

    if merged.empty:
        if verbose:
            print("No candidate pairs after exact match filtering.")
        return pd.DataFrame()

    if verbose:
        print(f"Candidate pairs after exact match filter: {len(merged)}")

    keep_mask = np.ones(len(merged), dtype=bool)

    for col, th in fuzzy_cols:
        if f"{col}_df1" not in merged.columns or f"{col}_df2" not in merged.columns:
            if verbose:
                print(f"Column '{col}' missing in one dataframe — skipped.")
            continue

        s1 = normalize(merged[f"{col}_df1"])
        s2 = normalize(merged[f"{col}_df2"])
        sims = process.cpdist(s1, s2, scorer=fuzz.ratio, workers=-1)

        if treat_empty_as_match:
            empty_mask = (s1 == "") | (s2 == "")
            sims[empty_mask] = 100.0

        if match_all:
            keep_mask &= sims >= th
        else:
            keep_mask |= sims >= th

        merged[f"score_{col}"] = sims

        if verbose:
            n_ok = int((sims >= th).sum())
            print(f"  '{col}' >= {th}: {n_ok}/{len(sims)}")

    merged = merged[keep_mask]
    if merged.empty:
        if verbose:
            print("No pairs left after fuzzy filtering.")
        return pd.DataFrame()

    if grouping_var and grouping_var in df1.columns and grouping_var in df2.columns:
        merged["group1"] = merged[f"{grouping_var}_df1"]
        merged["group2"] = merged[f"{grouping_var}_df2"]
    else:
        merged["group1"] = "all"
        merged["group2"] = "all"

    if self_join and unique_rows and grouping_var:
        merged = merged[merged["group1"] < merged["group2"]]

    if verbose:
        print("Aggregating results...")

    agg_dict = {"Num_Matches": ("_i", "count")}
    for col, _ in fuzzy_cols:
        agg_dict[f"Min_{col}"] = (f"score_{col}", "min")
        agg_dict[f"Max_{col}"] = (f"score_{col}", "max")
        agg_dict[f"Avg_{col}"] = (f"score_{col}", "mean")

    for col in additional_vars or []:
        if f"{col}_df1" in merged.columns and f"{col}_df2" in merged.columns:
            agg_dict[f"df1_{col}"] = (f"{col}_df1", uniq)
            agg_dict[f"df2_{col}"] = (f"{col}_df2", uniq)

    agg = merged.groupby(["group1", "group2"]).agg(**agg_dict).reset_index()

    if match_all and fuzzy_cols:
        for col, th in fuzzy_cols:
            agg = agg[agg[f"Min_{col}"] >= th]

    if csv_filename:
        try:
            agg.to_csv(csv_filename, index=False)
        except Exception as e:
            print(f"Failed to save CSV file '{csv_filename}': {e}")

    if verbose:
        print(agg.info())
        print(agg.describe())

    return agg

def fuzzy_compare_deprecated(df1=None, df2=None,
                  additional_vars_df1=None, additional_vars_df2=None,
                  endpoint_url=None, query=None,
                  grouping_var=None, label_var=None, element_var=None,
                  threshold=95, match_all=False, unique_rows=False,
                  csv_filename="comparison.csv", verbose=True):
    """
    Fuzzy string matching between two DataFrames (or SPARQL query results) based on a common element column.

    Supports optional grouping, label filtering, and aggregation of match statistics.

    Args:
        df1 (pd.DataFrame, optional): First DataFrame.
        df2 (pd.DataFrame, optional): Second DataFrame. If not provided, df1 is used.
        additional_vars_df1 (list, optional): List of columns from df1 that will be aggregated in the result.
        additional_vars_df2 (list, optional): List of columns from df2 that will be aggregated in the result.
        endpoint_url (str, optional): SPARQL endpoint (used if df1 is None).
        query (str, optional): SPARQL query (used if df1 is None).
        grouping_var (str, optional): Column name used for grouping (must exist in both DataFrames).
        label_var (str or list, optional): Label conditions, one of:
            - str or (col, "identical") or (col, 100): require exact match on this column
            - (col, int < 100): require fuzzy match on this column with given threshold
        element_var (str): Column containing the string values to compare with fuzzy matching.
        threshold (int, optional): Fuzzy matching threshold for element_var (0–100). Default: 95.
        match_all (bool, optional): If True, only include groups where all matches exceed the threshold.
        unique_rows (bool, optional): If True, suppress duplicate pairings (self-joins).
        csv_filename (str, optional): File path to save the aggregated results. If None, skip saving.
        verbose (bool, optional): If True, print debug info and head of results.

    Returns:
        pd.DataFrame: Aggregated match statistics between df1 and df2 (or within df1).

    """
    def _uniq_join(vals):
        return ", ".join(sorted({str(v) for v in vals if pd.notna(v)}))
    
    def normalize_series(s):
        return s.fillna("").astype(str).str.lower().to_numpy()
    
    def _non_empty_pair(a, b): 
        return (a.strip() != "") and (b.strip() != "")
    # Load from SPARQL if needed
    if df1 is None and endpoint_url and query:
        try:
            df1 = sparql_to_dataframe(endpoint_url, query,
                                      csv_filename=f"query_{csv_filename}" if csv_filename else None)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")

    if df2 is None:
        df2 = df1

    # detect self-join (common when df2 is None)
    self_join = (df1 is df2)

    additional_vars_df1 = [c for c in (additional_vars_df1 or []) if c in df1.columns]
    additional_vars_df2 = [c for c in (additional_vars_df2 or []) if c in df2.columns] if not self_join else additional_vars_df1

    if element_var not in df1.columns or element_var not in df2.columns:
        raise ValueError(f"Column '{element_var}' not found in DataFrames.")

    identical_vars, fuzzy_vars, different_vars  = [], {}, []
    if label_var:
        if isinstance(label_var, str):
            identical_vars.append(label_var)
        elif isinstance(label_var, list):
            for lv in label_var:
                if isinstance(lv, tuple) and len(lv) == 2:
                    col, cond = lv
                    if cond == "identical" or cond == 100:
                        identical_vars.append(col)
                    elif cond == "different":
                        different_vars.append(col)
                    elif isinstance(cond, int) and 0 <= cond < 100:
                        fuzzy_vars[col] = cond
                    else:
                        raise ValueError(f"Unsupported label_var condition: {lv}")
                elif isinstance(lv, str):
                    identical_vars.append(lv)


    matches = []

    df1_plain = df1.reset_index(drop=False).copy()
    df1_plain["_i"] = np.arange(len(df1_plain), dtype=np.int64)
    df2_plain = df2.reset_index(drop=False).copy()
    df2_plain["_j"] = np.arange(len(df2_plain), dtype=np.int64)
    df1_reset = df1_plain.add_suffix("_df1")
    df2_reset = df2_plain.add_suffix("_df2")

    dedup_keys = [grouping_var, *(identical_vars or []), element_var]
    df1_plain = df1_plain.drop_duplicates(subset=[k for k in dedup_keys if k in df1_plain.columns])
    df2_plain = df2_plain.drop_duplicates(subset=[k for k in dedup_keys if k in df2_plain.columns])

    # --- START MATCHING ---

    # --- IDENTICAL FILTER ---
    if identical_vars:
        merged = pd.merge(df1_plain, df2_plain,
                          on=identical_vars,
                          suffixes=("_df1", "_df2"))
        # TODO needed?
        # merged = merged.groupby([f"{grouping_var}_df1", f"{grouping_var}_df2"]).filter(
        #     lambda g: set(g[f"{element_var}_df1"]) == set(g[f"{element_var}_df2"])
        # )

        if not merged.empty:
            arr1 = normalize_series(merged[f"{element_var}_df1"])
            arr2 = normalize_series(merged[f"{element_var}_df2"])
            idx_i = merged["_i"].to_numpy()
            idx_j = merged["_j"].to_numpy()

            if threshold >= 100: # How this possible?
                eq = (arr1 == arr2)
                eq &= np.fromiter((_non_empty_pair(a, b) for a, b in zip(arr1, arr2)), bool, len(arr1))
                if self_join:
                    eq &= (idx_i != idx_j)
                for gi, gj in zip(idx_i[eq], idx_j[eq]):
                    if self_join and unique_rows and gj < gi:
                        continue
                    matches.append((gi, gj, 100))
            else:  # How this possible?
                scores = process.cpdist(arr1, arr2,
                                        scorer=fuzz.ratio,
                                        score_cutoff=threshold,
                                        workers=-1).ravel()
                for gi, gj, sc in zip(idx_i, idx_j, scores):
                    if self_join and gi == gj:
                        continue
                    if sc >= threshold and not (self_join and unique_rows and gj < gi):
                        matches.append((gi, gj, sc))

    # --- GLOBAL MATCHING via cdist ---
    else:
        arr1 = normalize_series(df1[element_var])
        arr2 = normalize_series(df2[element_var])
        idx1 = df1_plain["_i"].to_numpy()
        idx2 = df2_plain["_j"].to_numpy()

        if threshold >= 100:

            s1 = pd.Series(arr1, name="val")
            s2 = pd.Series(arr2, name="val")

            s1 = s1[s1.str.strip() != ""].reset_index(names="i_rel")
            s2 = s2[s2.str.strip() != ""].reset_index(names="j_rel")

            hits = s1.merge(s2, on="val", how="inner")

            gi_idx = idx1[hits["i_rel"].to_numpy()]
            gj_idx = idx2[hits["j_rel"].to_numpy()]

            # TODO check: needed twice?
            if self_join:
                mask = gi_idx != gj_idx
                gi_idx, gj_idx = gi_idx[mask], gj_idx[mask]
            if self_join and unique_rows:
                mask = gj_idx > gi_idx
                gi_idx, gj_idx = gi_idx[mask], gj_idx[mask]

            matches.extend(zip(gi_idx, gj_idx, np.full(len(gi_idx), 100.0)))

        else:
            batch_size = 2000
            for start in range(0, len(arr1), batch_size):
                sub1 = arr1[start:start+batch_size]
                sub_idx1 = idx1[start:start+batch_size]

                scores = process.cdist(sub1, arr2,
                                       scorer=fuzz.ratio,
                                       workers=-1,
                                       score_cutoff=threshold)

                gi_rel, gj_rel = np.nonzero(scores)
                gi_idx = sub_idx1[gi_rel]
                gj_idx = idx2[gj_rel]

                # TODO check: needed twice?
                if self_join:
                    mask = gi_idx != gj_idx
                    gi_rel, gj_rel, gi_idx, gj_idx = gi_rel[mask], gj_rel[mask], gi_idx[mask], gj_idx[mask]

                if self_join and unique_rows:
                    mask = gj_idx > gi_idx
                    gi_rel, gj_rel, gi_idx, gj_idx = gi_rel[mask], gj_rel[mask], gi_idx[mask], gj_idx[mask]

                sc = scores[gi_rel, gj_rel].astype(float)
                matches.extend(zip(gi_idx, gj_idx, sc))

    # --- END MATCHING ---
    if not matches:
        if verbose: print("No matches found (after self-join filtering and threshold).")
        return pd.DataFrame()

    # --- Build matches_df ---
    matches_df = pd.DataFrame(matches, columns=["i", "j", "score"])
    matches_df = matches_df.merge(df1_reset, left_on="i", right_on="_i_df1")
    matches_df = matches_df.merge(df2_reset, left_on="j", right_on="_j_df2")

    # if grouping_var and element_var in df1.columns and element_var in df2.columns:
    #     matches_df = matches_df.groupby([f"{grouping_var}_df1", f"{grouping_var}_df2"]).filter(
    #         lambda g: set(g[f"{element_var}_df1"]) == set(g[f"{element_var}_df2"])
    #     )

    if different_vars:
        ok_mask = np.ones(len(matches_df), dtype=bool)
        for col in different_vars:
            s1 = matches_df[f"{col}_df1"].fillna("").astype(str).str.lower().to_numpy()
            s2 = matches_df[f"{col}_df2"].fillna("").astype(str).str.lower().to_numpy()
            # TODO check
            ok_mask &= [(a.strip() != "" and b.strip() != "" and a != b) for a, b in zip(s1, s2)]

        matches_df = matches_df[ok_mask]

    # --- Fuzzy-label filter (post-match) ---
    if fuzzy_vars:
        # TODO check
        ok_mask =   (matches_df[element_var + "_df1"].astype(str).str.strip() != "") & \
                    (matches_df[element_var + "_df2"].astype(str).str.strip() != "")
        #  ok_mask = np.ones(len(matches_df), dtype=bool)

        for col, th in fuzzy_vars.items():
            col1 = f"{col}_df1"
            col2 = f"{col}_df2"
            if col1 not in matches_df.columns or col2 not in matches_df.columns:
                print(f"Fuzzy label column(s) missing after merge: {col1}/{col2}")
                continue

            s1 = matches_df[col1].fillna("").astype(str).str.lower().to_numpy()
            s2 = matches_df[col2].fillna("").astype(str).str.lower().to_numpy()

            sims = process.cpdist(s1, s2, scorer=fuzz.ratio, workers=-1, score_cutoff=th)
            ok_mask &= (sims >= th)

        matches_df = matches_df[ok_mask]

    if matches_df.empty:
        if verbose: print("No matches after fuzzy-label filter.")
        return pd.DataFrame()

    # --- Grouping ---
    if grouping_var and grouping_var in df1.columns and grouping_var in df2.columns:
        matches_df["group1"] = matches_df[f"{grouping_var}_df1"]
        matches_df["group2"] = matches_df[f"{grouping_var}_df2"]
    else:
        matches_df["group1"] = "all"
        matches_df["group2"] = "all"

    if self_join and grouping_var:
        matches_df = matches_df[matches_df["group1"] != matches_df["group2"]]

    # --- Aggregation ---
    agg_dict = {
        'df1_Elements': (f"{element_var}_df1", _uniq_join),
        'df2_Elements': (f"{element_var}_df2", _uniq_join),
        'Num_Matches': ('score', 'count'),
        'Average_Score': ('score', 'mean'),
        'Min_Score': ('score', 'min'),
        'Max_Score': ('score', 'max'),
    }
    for col in additional_vars_df1:
        agg_dict[f'df1_{col}'] = (f"{col}_df1", _uniq_join)
    for col in additional_vars_df2:
        agg_dict[f'df2_{col}'] = (f"{col}_df2", _uniq_join)

    aggregated = matches_df.groupby(['group1', 'group2']).agg(**agg_dict).reset_index()

    if self_join and unique_rows and grouping_var: #needed again?
        aggregated = aggregated[aggregated['group1'].astype(str) < aggregated['group2'].astype(str)]

    if match_all:
        aggregated = aggregated[aggregated['Min_Score'] >= threshold]

    aggregated = aggregated[aggregated['Max_Score'] >= threshold]

    if csv_filename:
        try:
            aggregated.to_csv(csv_filename, index=False)
        except Exception as e:
            print(f"Failed to save CSV file '{csv_filename}': {e}")

    if verbose:
        print(aggregated.info())
        print(aggregated.describe())

    return aggregated


def fuzzy_compare_legacy(df1=None,df2=None,
                    additional_vars_df1=None, additional_vars_df2=None,
                    endpoint_url=None,
                    query=None,
                    grouping_var=None, label_var=None, element_var=None, threshold=95, match_all=False, unique_rows=False, csv_filename="comparison.csv", verbose= True):
    """
    !! Left here, as new function not tested much....
    Fuzzy string matching between two DataFrames (or SPARQL query results) based on a common element column.

    Supports optional grouping, label filtering, and aggregation of match statistics.

    Args:
        df1 (pd.DataFrame, optional): First DataFrame.
        df2 (pd.DataFrame, optional): Second DataFrame. If not provided, df1 is used.
        additional_vars_df1 (list, optional): List of columns that will be aggregated in the result (using first per group).
        additional_vars_df2 (list, optional): List of columns that will be aggregated in the result (using first per group).
        endpoint_url (str, optional): SPARQL endpoint.
        query (str, optional): SPARQL query.
        grouping_var (str, optional): Column name used for grouping.
        label_var (str, optional): Optional label for filtering matches.
        element_var (str): Column containing the string values to compare.
        threshold (int): Fuzzy matching threshold (0–100). Default: 95.
        match_all (bool): If True, only include groups where all scores exceed the threshold.
        unique_rows (bool): If True, suppress duplicate pairings.
        csv_filename (str): File path to save the results.
        verbose (bool, optional): Whether to print insights into the dataframe.

    Returns:
        pd.DataFrame: Aggregated match statistics between df1 and df2 (or within df1).
    """



    if df1 is None and endpoint_url and query:
        try:
            # Fetch data and create DataFrame
            df1 = sparql_to_dataframe(endpoint_url, query, csv_filename=f"query_{csv_filename}" if csv_filename is not None else None)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")

    if df2 is None:
        df2 = df1


    if additional_vars_df1 is None:
        additional_vars_df1 = []
    else:
        additional_vars_df1 = [col for col in additional_vars_df1 if col in df1.columns]

    if additional_vars_df2 is None:
        additional_vars_df2 = []
    else:
        additional_vars_df2 = [col for col in additional_vars_df2 if col in df2.columns]

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

                    match_entry = {
                        'group2': group_key2,
                        'group1': group_key1,
                        'label': row2[label_var] if check_label else None,
                        'element2': row2[element_var],
                        'element1': row1[element_var],
                        'score': score
                    }

                    for col in additional_vars_df1:                        
                            match_entry[f'df1_{col}'] = row1[col]

                    for col in additional_vars_df2:                        
                            match_entry[f'df2_{col}'] = row2[col]

                    matches.append(match_entry)
    
    matches_df = pd.DataFrame(matches)
    aggregated = pd.DataFrame()
    if verbose:
        print(matches_df.info())
        matches_df.describe()

    if not match_all:
        matches_df = matches_df[matches_df['score'] >= threshold]

    if not matches_df.empty:
        agg_dict = {
            'Labels': ('label', lambda x: ", ".join(sorted(set(x)))),
            'df1_Elements': ('element1', lambda x: ", ".join(sorted(set(x)))),
            'df2_Elements': ('element2', lambda x: ", ".join(sorted(set(x)))),
            'Num_Matches': ('score', 'count'),
            'Average_Score': ('score', 'mean'),
            'Min_Score': ('score', 'min'),
            'Max_Score': ('score', 'max')
        }

        for col in additional_vars_df1:
            agg_dict[f'df1_{col}'] = (f'df1_{col}', 'first')
        for col in additional_vars_df2:
            agg_dict[f'df2_{col}'] = (f'df2_{col}', 'first')

        aggregated = matches_df.groupby(['group1', 'group2']).agg(**agg_dict).reset_index()

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