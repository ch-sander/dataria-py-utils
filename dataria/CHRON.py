"""
Visualization and aggregation of time-based data.

This module provides functionality to generate rolling time series plots and calendar heatmaps based on a date column. Input data can be provided via a DataFrame or fetched dynamically using a SPARQL query. Outputs include CSV summaries, Matplotlib and Plotly visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import calendar
import os
from .DATA import sparql_to_dataframe

def date_aggregation(
    df=None,
    endpoint_url=None,
    query=None,
    date_var='date',
    plot_type='rolling',
    mode='count',
    num_var=None,
    window=7,
    csv_filename='time_aggregated_data.csv',
    png_filename='time_plot.png',
    html_filename='time_render.html'
):
    """
    Aggregates and visualizes event data based on a date column.

    The function generates either a rolling time series plot or a heatmap grouped by weekday and month. Input data can be passed as a DataFrame or fetched from a SPARQL endpoint. Aggregated data and plots can be saved as CSV, PNG, and HTML files.

    Args:
        df (pd.DataFrame, optional): Input data as a DataFrame. Ignored if `endpoint_url` and `query` are provided.
        endpoint_url (str, optional): SPARQL endpoint URL.
        query (str, optional): SPARQL query to fetch data.
        date_var (str): Name of the column containing date values (default: 'date').
        plot_type (str): Type of plot to generate – either 'rolling' or 'heatmap'.
        mode (str): Aggregation mode – 'count', 'sum', or 'mean'.
        num_var (str, optional): Name of the numeric column for 'sum' or 'mean' mode.
        window (int): Window size in days for rolling aggregation.
        csv_filename (str): File path to save the aggregated data (CSV).
        png_filename (str): File path to save the plot (PNG).
        html_filename (str): File path to save the Plotly plot (HTML).

    Returns:
        None
    """
    
    if df is None and endpoint_url and query:
        try:
            # Fetch data and create DataFrame
            df = sparql_to_dataframe(endpoint_url, query, csv_filename=f"query_{csv_filename}" if csv_filename is not None else None)
        except Exception as e:
            raise ValueError(f"Failed to fetch or process SPARQL query results. Error: {e}")
    
    # Drop rows with invalid dates
    initial_count = len(df)
    df[date_var] = df[date_var].apply(
        lambda x: x if isinstance(x, (pd.Period, pd.Timestamp)) else pd.NaT
    )


    df = df.dropna(subset=[date_var])
    final_count = len(df)
    if final_count < initial_count:
        print(f"Dropped {initial_count - final_count} rows due to invalid dates in '{date_var}'.")
    
    # Set the date column as the index and sort
    df.set_index(date_var, inplace=True)
    
    # Homogenic index as PeriodIndex
    if not isinstance(df.index, pd.PeriodIndex):
        df.index = pd.PeriodIndex(df.index, freq='D')

    if plot_type.lower() == 'rolling':
        if mode == 'count':
            daily_values = df.groupby(df.index).size()
            rolling_window = daily_values.rolling(window=window, min_periods=1).sum()
        elif mode == 'sum':
            if num_var is None:
                raise ValueError("The column name must be specified for the 'sum' mode!")
            daily_values = df.groupby(df.index)[num_var].sum()
            rolling_window = daily_values.rolling(window=window, min_periods=1).sum()
        elif mode == 'mean':
            if num_var is None:
                raise ValueError("The column name must be specified for the 'mean' mode!")
            daily_values = df.groupby(df.index)[num_var].mean()
            rolling_window = daily_values.rolling(window=window, min_periods=1).mean()
        else:
            raise ValueError("Invalid mode. Use 'count', 'sum' or 'mean'.")
        
        # Convert PeriodIndex to strings for plotting
        rolling_window_str = rolling_window.index.strftime('%Y-%m-%d')
        
        # Create an aggregated DataFrame
        aggregated_df = pd.DataFrame({
            'Date': rolling_window_str,
            'Event_Count': rolling_window.values
        })
        
        # Save the aggregated DataFrame to CSV
        aggregated_df_sorted = aggregated_df.sort_values('Date')
        if len(csv_filename) > 0 and csv_filename is not None:
            try:
                aggregated_df_sorted.to_csv(csv_filename, index=False)
                print(f"Aggregated data saved to {csv_filename}")
            except Exception as e:
                print(f"Failed to save CSV file '{csv_filename}': {e}")
        
        ### Matplotlib Plot ###
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_window_str, rolling_window.values, marker='o', linestyle='-')
        plt.title(f'Distribution with Rolling Window of {window} Days')
        plt.xlabel('Date')
        plt.grid(True)
        
        # Customize xticks to show approximately monthly ticks
        num_ticks = 12  # Approx. one tick per month
        step = max(1, len(rolling_window_str) // num_ticks)
        xticks = rolling_window_str[::step]
        plt.xticks(ticks=range(0, len(rolling_window_str), step), labels=xticks, rotation=45)
        
        plt.legend(['Counter'])
        plt.tight_layout()
        plt.show()
        if len(png_filename) > 0 and png_filename is not None:
            try:
                plt.savefig(f"{plot_type.lower()}_{png_filename}", dpi=300, format='png')
                print(f"Matplotlib plot saved to {plot_type.lower()}_{png_filename}")
            except Exception as e:
                print(f"Failed to save file '{plot_type.lower()}_{png_filename}': {e}")
        
        ### Plotly Plot ###
        fig = px.line(
            x=rolling_window_str,
            y=rolling_window.values,
            labels={'x': 'Date', 'y': 'Counter'},
            title=f'Distribution with Rolling Window of {window} Days'
        )
        
        # Adjust x-axis to show ticks per month
        fig.update_layout(
            xaxis=dict(
                tickformat="%Y-%m",
                dtick="M1"  # Monthly ticks
            )
        )
        
        fig.update_traces(mode='lines+markers')
        if len(html_filename) > 0 and html_filename is not None:
            try:
                fig.write_html(html_filename)
                print(f"Plotly plot saved to {html_filename}")
            except Exception as e:
                print(f"Failed to save file '{html_filename}': {e}")
        
        # fig.show()

    
    elif plot_type.lower() == 'heatmap':
        # Add 'weekday' and 'month' columns
        df_sorted = df.copy()
        df_sorted['weekday'] = df_sorted.index.weekday.map(lambda x: calendar.day_name[x])
        df_sorted['month'] = df_sorted.index.month.map(lambda x: calendar.month_name[x])
        
        # Group by 'month' and 'weekday' and count events
        heatmap_data = df_sorted.groupby(['month', 'weekday']).size().unstack(fill_value=0)
        
        # Define the correct order for months and weekdays
        months_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        weekdays_order = [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday'
        ]
        
        # Reindex to ensure the correct order
        heatmap_data = heatmap_data.reindex(index=months_order, fill_value=0)
        heatmap_data = heatmap_data.reindex(columns=weekdays_order, fill_value=0)
        
        # Save the aggregated DataFrame to CSV
        aggregated_df = heatmap_data.reset_index()

        if len(csv_filename) > 0 and csv_filename is not None:
            try:
                aggregated_df.to_csv(csv_filename, index=False)
                print(f"Heatmap data saved to {csv_filename}")
            except Exception as e:
                print(f"Failed to save CSV file '{csv_filename}': {e}")

        
        ### Seaborn Heatmap ###
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            cbar_kws={'label': 'Number of Events'}
        )
        plt.title('Heatmap of Events by Month and Day of the Week', fontsize=16)
        plt.xlabel('Weekday', fontsize=12)
        plt.ylabel('Month', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.show()

        if len(png_filename) > 0 and png_filename is not None:
            try:
                plt.savefig(f"{plot_type.lower()}_{png_filename}", dpi=300, format='png')
                print(f"Matplotlib plot saved to {plot_type.lower()}_{png_filename}")
            except Exception as e:
                print(f"Failed to save file '{plot_type.lower()}_{png_filename}': {e}")
    
    else:
        raise ValueError("Invalid plot_type. Choose 'rolling' or 'heatmap'.")