import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import calendar
import os
from DATA import sparql_to_dataframe

def date_aggregation(
    df=None,
    endpoint_url=None,
    query=None,
    date_var='date',
    plot_type='rolling',
    window=7,
    output_csv_path='aggregated_data.csv',
    output_plot_path='plot'
):
    """
    Generates aggregated plots based on a date column in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        endpoint_url (str): The SPARQL endpoint URL to query. Ignored if df or gdf are defined.
        query (str): The SPARQL query to be executed. Ignored if df or gdf are defined.
        date_var (str): The name of the column containing date information (default 'date').
        plot_type (str): Type of plot to generate ('rolling' or 'heatmap').
        window (int): Rolling window size in days (used if plot_type='rolling').
        output_csv_path (str): File path to save the aggregated DataFrame as CSV.
        output_plot_path (str): Base file path to save the plot images (without extension).
        
    Returns:
        None
    """
    
    # Ensure the output directories exist
    csv_dir = os.path.dirname(output_csv_path)
    plot_dir = os.path.dirname(output_plot_path)
    
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    if df is None and endpoint_url and query:
        try:
            # Fetch data and create DataFrame
            df = sparql_to_dataframe(endpoint_url, query)
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
    
    if plot_type.lower() == 'rolling':
        # Aggregation: Count the number of events per day
        event_counts = df.groupby(df.index).size()
        
        # Apply rolling window
        rolling_window = event_counts.rolling(window=window, min_periods=1).sum()
        
        # Convert PeriodIndex to strings for plotting
        rolling_window_str = rolling_window.index.strftime('%Y-%m-%d')
        
        # Create an aggregated DataFrame
        aggregated_df = pd.DataFrame({
            'Date': rolling_window_str,
            'Event_Count': rolling_window.values
        })
        
        # Save the aggregated DataFrame to CSV
        aggregated_df_sorted = aggregated_df.sort_values('Date')
        aggregated_df_sorted.to_csv(output_csv_path, index=False)
        print(f"Aggregated data saved to {output_csv_path}")
        
        ### Matplotlib Plot ###
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_window_str, rolling_window.values, marker='o', linestyle='-')
        plt.title(f'Event Distribution with Rolling Window of {window} Days')
        plt.xlabel('Date')
        plt.ylabel('Number of Events')
        plt.grid(True)
        
        # Customize xticks to show approximately monthly ticks
        num_ticks = 12  # Approx. one tick per month
        step = max(1, len(rolling_window_str) // num_ticks)
        xticks = rolling_window_str[::step]
        plt.xticks(ticks=range(0, len(rolling_window_str), step), labels=xticks, rotation=45)
        
        plt.legend(['Number of Events'])
        plt.tight_layout()
        matplotlib_plot_path = f"{output_plot_path}_matplotlib.png"
        plt.savefig(matplotlib_plot_path, dpi=300, format='png')
        plt.show()
        print(f"Matplotlib plot saved to {matplotlib_plot_path}")
        
        ### Plotly Plot ###
        fig = px.line(
            x=rolling_window_str,
            y=rolling_window.values,
            labels={'x': 'Date', 'y': 'Number of Events'},
            title=f'Event Distribution with Rolling Window of {window} Days'
        )
        
        # Adjust x-axis to show ticks per month
        fig.update_layout(
            xaxis=dict(
                tickformat="%Y-%m",
                dtick="M1"  # Monthly ticks
            )
        )
        
        fig.update_traces(mode='lines+markers')
        plotly_html_path = f"{output_plot_path}_plotly.html"
        fig.write_html(plotly_html_path)
        fig.show()
        print(f"Plotly plot saved to {plotly_html_path}")
    
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
        aggregated_df.to_csv(output_csv_path, index=False)
        print(f"Aggregated heatmap data saved to {output_csv_path}")
        
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
        heatmap_plot_path = f"{output_plot_path}_heatmap.png"
        plt.savefig(heatmap_plot_path, dpi=300, format='png')
        plt.show()
        print(f"Heatmap plot saved to {heatmap_plot_path}")
    
    else:
        raise ValueError("Invalid plot_type. Choose 'rolling' or 'heatmap'.")