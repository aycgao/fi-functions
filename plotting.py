"""
This supplementary module uses AI generated code for simple plotting
Functions:
- plot_multiple_columns
"""

import pandas as pd
import numpy as np
from typing import Dict, Union
from pandas.core.series import Series
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dataframe_columns(df, x_col, y_col, plot_type="scatter", title=None, xlabel=None, ylabel=None, figsize=(8, 5)):
    """
    Creates a well-designed plot using two columns from a DataFrame, handling both column names and Series inputs.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_col (str or pd.Series): Column name (str) or Pandas Series for the x-axis.
        y_col (str or pd.Series): Column name (str) or Pandas Series for the y-axis.
        plot_type (str): Type of plot ('scatter', 'line', 'bar').
        title (str, optional): Custom title for the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        figsize (tuple, optional): Figure size (default (8,5)).

    Returns:
        None (Displays the plot).
    """
    # Handle direct Series input
    if isinstance(x_col, pd.Series) and isinstance(y_col, pd.Series):
        x_data = x_col
        y_data = y_col
        xlabel = xlabel if xlabel else x_col.name  # Get column name from Series
        ylabel = ylabel if ylabel else y_col.name
    else:
        x_data = df[x_col]
        y_data = df[y_col]
        xlabel = xlabel if xlabel else x_col
        ylabel = ylabel if ylabel else y_col

    # Suppress large DataFrame outputs
    _ = df.head()

    # Set up the figure size
    plt.figure(figsize=figsize)

    # Apply Seaborn style
    sns.set_style("whitegrid")

    # Plot selection
    if plot_type == "scatter":
        sns.scatterplot(x=x_data, y=y_data, color="blue", edgecolor="black")
    elif plot_type == "line":
        sns.lineplot(x=x_data, y=y_data, color="red", linewidth=2)
    elif plot_type == "bar":
        sns.barplot(x=x_data, y=y_data, palette="viridis")
    else:
        raise ValueError("Unsupported plot_type. Choose from: 'scatter', 'line', 'bar'.")

    # Titles and labels
    plt.title(title if title else f"{plot_type.capitalize()} Plot of {ylabel} vs {xlabel}", fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Show and clear figure
    plt.show()
    plt.close()

def plot_multiple_columns(df, columns=None, title="Multiple Columns vs Index", xlabel="Index", ylabel="Values", figsize=(10, 6)):
    """
    Plots multiple columns of a DataFrame against the DataFrame's index.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list, optional): List of column names to plot. If None, plots all numerical columns.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        figsize (tuple, optional): Figure size (default (10,6)).

    Returns:
        None (Displays the plot).
    """
    # Apply Seaborn style
    sns.set_style("whitegrid")

    # If columns not specified, select all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns

    # Create a new figure
    plt.figure(figsize=figsize)

    # Plot each column
    for col in columns:
        plt.plot(df.index, df[col], label=col, marker="o", linestyle="-")

    # Titles and labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Legend and grid
    plt.legend(title="Columns")
    plt.grid(True)

    # Show plot
    plt.show()
