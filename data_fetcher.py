from fredapi import Fred
from config import FRED_API_KEY
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
def get_economic_data(series_id, start_date='2000-01-01', end_date='2023-12-31'):
    """
    Fetches economic data for a given FRED series ID.
    
    Args:
    - series_id (str): The FRED series ID (e.g., 'GDP', 'CPIAUCSL').
    - start_date (str): The start date for fetching data.
    - end_date (str): The end date for fetching data.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the date and corresponding values.
    """
    fred = Fred(api_key=FRED_API_KEY)
    try:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        return pd.DataFrame(data, columns=['value']).reset_index().rename(columns={'index': 'Date'})
    except ValueError as e:
        print(f"Error fetching data for series ID {series_id}: {e}")
        return None

def calculate_monotonic_relationships(commodity_data, economic_series_ids):
    """
    Calculate the monotonic relationships between a commodity and multiple economic indicators.
    ...
    """
    correlation_results = {}

    for series_id in economic_series_ids:
        economic_data = get_economic_data(series_id, start_date='2000-01-01')

        if economic_data is None:
            continue  

        date_col = economic_data.columns[economic_data.columns.str.lower() == 'date'][0]
        economic_data[date_col] = pd.to_datetime(economic_data[date_col])

        merged_data = commodity_data.merge(economic_data, left_index=True, right_on=date_col, how='inner', suffixes=('_commodity', '_economic'))

        correlation, _ = spearmanr(merged_data.iloc[:, 0], merged_data['value'])
        correlation_results[series_id] = correlation

    correlation_series = pd.Series(correlation_results)

    top_increasing = correlation_series.nlargest(10)
    top_decreasing = correlation_series.nsmallest(10)

    return top_increasing, top_decreasing

def visualize_relationships(top_increasing, top_decreasing):
    sns.set_style("whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    def add_value_labels(ax, spacing=0.01):
        for rect in ax.patches:
            value = rect.get_width()
            text = f'{value:.2f}'
            y = rect.get_y() + rect.get_height() / 2
            x = rect.get_width()
            ha = 'left' if value >= 0 else 'right'
            ax.text(x + np.sign(x) * spacing, y, text, ha=ha, va='center')

    sns.barplot(x=top_increasing.values, y=top_increasing.index, ax=ax1, palette='YlOrRd')
    ax1.set_title('Top 10 Increasing Relationships', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Spearman Correlation Coefficient', fontsize=12)
    ax1.set_ylabel('')
    ax1.axvline(0, color='grey', lw=1, linestyle='--')
    add_value_labels(ax1)

    sns.barplot(x=top_decreasing.values, y=top_decreasing.index, ax=ax2, palette='YlGnBu_r')
    ax2.set_title('Top 10 Decreasing Relationships', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Spearman Correlation Coefficient', fontsize=12)
    ax2.set_ylabel('')
    ax2.axvline(0, color='grey', lw=1, linestyle='--')
    add_value_labels(ax2)

    plt.tight_layout()
    fig.suptitle('Top Increasing and Decreasing Relationships', fontsize=20, fontweight='bold', y=1.05)

    sm1 = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=1))
    sm2 = plt.cm.ScalarMappable(cmap='YlGnBu_r', norm=plt.Normalize(vmin=-1, vmax=0))
    cbar1 = fig.colorbar(sm1, ax=ax1, orientation='horizontal', pad=0.08, aspect=30)
    cbar2 = fig.colorbar(sm2, ax=ax2, orientation='horizontal', pad=0.08, aspect=30)
    cbar1.set_label('Strength of Positive Correlation', fontsize=10)
    cbar2.set_label('Strength of Negative Correlation', fontsize=10)

    plt.show()
