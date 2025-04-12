import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, Any, Tuple
# from sklearn.preprocessing import MinMaxScaler # No longer needed
import warnings
import io # For download button

# --- Style and Warnings Configuration ---
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None # type: ignore
plt.style.use('seaborn-v0_8-darkgrid') # Base style
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9})
# Define colors for dark theme visibility
DARK_BG_COLOR = '#0E1117' # Streamlit's dark theme background
LIGHT_TEXT_COLOR = '#FAFAFA' # A light color for text/ticks

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants & Config ---
DEFAULT_START_DATE = datetime.now() - timedelta(days=5*365)
DEFAULT_END_DATE = datetime.now()
FXEMPIRE_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9,tr-TR;q=0.8,tr;q=0.7",
    "priority": "u=1, i",
    "referer": "https://www.fxempire.com/commodities/silver", # Adjust referer maybe?
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "Windows",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "token": "null",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36" # Keep updated
}

# --- Helper function for styling plots ---
def style_plot_for_dark_theme(ax: plt.Axes, facecolor=DARK_BG_COLOR, textcolor=LIGHT_TEXT_COLOR):
    """Applies styling for better visibility on dark backgrounds."""
    ax.set_facecolor(facecolor)
    ax.title.set_color(textcolor)
    ax.xaxis.label.set_color(textcolor)
    ax.yaxis.label.set_color(textcolor)
    ax.tick_params(axis='x', colors=textcolor)
    ax.tick_params(axis='y', colors=textcolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(textcolor)
    try: # Style legend if it exists
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(facecolor)
            legend.get_frame().set_edgecolor(textcolor)
            for text in legend.get_texts():
                text.set_color(textcolor)
    except AttributeError: pass

# --- Data Fetching Function ---

@st.cache_data(ttl=3600)
def fetch_commodity_data_api(start_date_dt: datetime, commodity_type: str) -> pd.DataFrame:
    """Fetches commodity data from FXEmpire API, keeps OHLC if available."""
    start_date_unix = int(start_date_dt.timestamp())
    id_prefix = "" # Use prefix for column names
    if commodity_type.lower() == 'gold':
        instrument = "XAU/USD"; id_prefix = "Gold"
        api_url = "https://www.fxempire.com/api/v1/en/commodities/chart/candles"
    elif commodity_type.lower() == 'silver':
        instrument = "XAG/USD"; id_prefix = "Silver"
        api_url = "https://www.fxempire.com/api/v1/en/commodities/chart/candles"
    else:
        raise ValueError("Invalid commodity type.")

    logging.info(f"Fetching {commodity_type} data from Unix ts {start_date_unix}")
    querystring = {"instrument": instrument, "granularity": "D", "from": str(start_date_unix), "price": "M", "count": "5000"}

    try:
        response = requests.get(api_url, headers=FXEMPIRE_HEADERS, params=querystring, timeout=20)
        response.raise_for_status(); data = response.json()
    except requests.RequestException as e:
        logging.error(f"API Error fetching {commodity_type}: {e}")
        st.error(f"API Error fetching {commodity_type}: {e}.")
        return pd.DataFrame(columns=['Date']).set_index('Date') # Return empty DF with index

    if not data:
         st.warning(f"No {commodity_type} data from API.")
         return pd.DataFrame(columns=['Date']).set_index('Date')

    df = pd.DataFrame(data)
    if 'Date' not in df.columns:
         st.warning(f"API response for {commodity_type} missing 'Date' column.")
         return pd.DataFrame(columns=['Date']).set_index('Date')

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index('Date', inplace=True)

    # Standardize columns expected (OHLCV)
    column_mapping = {}
    potential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in potential_cols:
        # Check for exact match or case-insensitive match
        actual_col = next((c for c in df.columns if c.lower() == col.lower()), None)
        if actual_col:
            new_name = f"{id_prefix}_{col}" # Add prefix like Gold_Close
            column_mapping[actual_col] = new_name
            # Convert to numeric, coerce errors
            df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')

    # Check if 'Close' price exists after mapping
    close_col_name = f"{id_prefix}_Close"
    if close_col_name not in column_mapping.values():
        st.warning(f"API response for {commodity_type} missing essential 'Close' price data.")
        # Try to find *any* price column if 'Close' is absent (fallback)
        price_col = next((c for c in df.columns if 'price' in c.lower() or 'value' in c.lower()), None)
        if price_col:
             column_mapping[price_col] = close_col_name # Rename it to expected Close col name
             df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        else:
             return pd.DataFrame(columns=['Date']).set_index('Date') # Give up if no close/price

    # Select and rename columns
    df = df[list(column_mapping.keys())].rename(columns=column_mapping)

    # Ensure Close exists and drop rows where it's NaN
    df.dropna(subset=[close_col_name], inplace=True)

    df = df.sort_index()
    logging.info(f"Processed {len(df)} points for {commodity_type}, Cols: {list(df.columns)}")
    return df

# --- Helper to convert DF to CSV for download ---
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
   """Converts a Pandas DataFrame to CSV bytes."""
   # Use float_format to avoid excessive decimals in CSV
   output = io.StringIO()
   df.to_csv(output, index=True, float_format='%.4f')
   return output.getvalue().encode('utf-8')

# --- Commodity Investment Tracker Class ---

class CommodityInvestmentTracker:
    def __init__(self):
        pass

    def get_commodity_data(self, start_date_dt: datetime, commodity_type: str) -> pd.DataFrame:
        """Wrapper for fetching function."""
        return fetch_commodity_data_api(start_date_dt, commodity_type)

    def analyze_investment(self, df: pd.DataFrame, start_date_dt: datetime, end_date_dt: datetime,
                           initial_investment: float = 100.0) -> Tuple[Dict[str, Any], plt.Figure | None]:
        """Analyzes single investment with dark theme plots and Bollinger Bands."""
        if df.empty: return {"error": "No data."}, None
        df_filtered = df[(df.index >= start_date_dt) & (df.index <= end_date_dt)].copy()
        if df_filtered.empty or len(df_filtered) < 2: return {"error": f"No data in range."}, None

        # Identify the correct Close price column (e.g., 'Gold_Close' or 'Silver_Close')
        close_col = next((c for c in df_filtered.columns if c.endswith('_Close')), None)
        if not close_col: return {"error": "Close price column not found in DataFrame."}, None
        commodity_name = close_col.split('_')[0] # Extract Gold/Silver

        start_price = df_filtered[close_col].iloc[0]
        if pd.isna(start_price) or start_price <= 0: return {"error": "Invalid start price."}, None

        # --- Calculations ---
        units_bought = initial_investment / start_price
        df_filtered['Investment_Value'] = df_filtered[close_col] * units_bought
        final_value = df_filtered['Investment_Value'].iloc[-1]
        total_return = final_value - initial_investment
        days_held = (end_date_dt - start_date_dt).days
        years = days_held / 365.25
        annualized_return = (final_value / initial_investment) ** (1 / years) - 1 if (years > 0 and initial_investment > 0 and final_value > 0) else 0.0
        df_filtered['Daily_Returns_Pct'] = df_filtered[close_col].pct_change()
        trading_days_count = df_filtered['Daily_Returns_Pct'].count()
        trading_days_per_year = trading_days_count / years if years > 0 else 252
        annualized_volatility = df_filtered['Daily_Returns_Pct'].std() * np.sqrt(max(1, trading_days_per_year)) if trading_days_count > 1 else 0.0
        # Bollinger Bands calculation
        bb_window = 20
        bb_std = 2
        df_filtered['BB_SMA'] = df_filtered[close_col].rolling(window=bb_window).mean()
        df_filtered['BB_StdDev'] = df_filtered[close_col].rolling(window=bb_window).std()
        df_filtered['BB_Upper'] = df_filtered['BB_SMA'] + (bb_std * df_filtered['BB_StdDev'])
        df_filtered['BB_Lower'] = df_filtered['BB_SMA'] - (bb_std * df_filtered['BB_StdDev'])

        # --- Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor(DARK_BG_COLOR)
        axes = axes.flatten()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)

        # Plot 1: Price (unchanged)
        axes[0].plot(df_filtered.index, df_filtered[close_col], label='Close Price', color='deepskyblue')
        axes[0].set_title(f'{commodity_name} Close Price')
        axes[0].set_ylabel('Price (USD)')
        axes[0].xaxis.set_major_locator(locator); axes[0].xaxis.set_major_formatter(formatter)
        axes[0].grid(True, linestyle='--', alpha=0.3); axes[0].legend()
        style_plot_for_dark_theme(axes[0])

        # Plot 2: Investment Value (unchanged)
        axes[1].plot(df_filtered.index, df_filtered['Investment_Value'], label='Investment Value', color='limegreen')
        axes[1].axhline(initial_investment, color='tomato', linestyle='--', lw=1, label='Initial Inv.')
        axes[1].set_title('Investment Value Over Time')
        axes[1].set_ylabel('Value (USD)'); axes[1].ticklabel_format(style='plain', axis='y')
        axes[1].xaxis.set_major_locator(locator); axes[1].xaxis.set_major_formatter(formatter)
        axes[1].grid(True, linestyle='--', alpha=0.3); axes[1].legend()
        style_plot_for_dark_theme(axes[1])

        # Plot 3: Bollinger Bands
        axes[2].plot(df_filtered.index, df_filtered[close_col], label='Close Price', color='deepskyblue', lw=1.5)
        axes[2].plot(df_filtered.index, df_filtered['BB_SMA'], label=f'{bb_window}-Day SMA', color='orange', lw=1, linestyle='--')
        axes[2].plot(df_filtered.index, df_filtered['BB_Upper'], label='Upper Band', color='lightcoral', lw=1, linestyle=':')
        axes[2].plot(df_filtered.index, df_filtered['BB_Lower'], label='Lower Band', color='lightcoral', lw=1, linestyle=':')
        axes[2].fill_between(df_filtered.index, df_filtered['BB_Lower'], df_filtered['BB_Upper'], color='lightcoral', alpha=0.15, label=f'{bb_std} Std Dev Range')
        axes[2].set_title(f'Bollinger Bands ({bb_window}, {bb_std})')
        axes[2].set_ylabel('Price (USD)')
        axes[2].xaxis.set_major_locator(locator); axes[2].xaxis.set_major_formatter(formatter)
        axes[2].legend(loc='upper left')
        style_plot_for_dark_theme(axes[2]) # Apply style

        # Plot 4: Monthly Returns Heatmap (unchanged logic)
        # Ensure 'Daily_Returns' calculation uses the correct close_col
        df_filtered['Daily_Returns'] = df_filtered[close_col].pct_change() * 100
        df_filtered['Month'] = df_filtered.index.month; df_filtered['Year'] = df_filtered.index.year
        if len(df_filtered['Year'].unique()) > 0 and len(df_filtered['Month'].unique()) > 0 :
            heatmap_data = df_filtered.pivot_table(values='Daily_Returns', index='Year', columns='Month', aggfunc='mean')
            heatmap_data = heatmap_data.reindex(columns=range(1, 13)) # Ensure all months
            axes[3].set_facecolor(DARK_BG_COLOR)
            cbar_kws = {'label': 'Mean Daily Return (%)'} # Corrected label reference
            sns.heatmap(heatmap_data, cmap='RdYlGn', annot=True, fmt=".1f", linewidths=.5, ax=axes[3], cbar_kws=cbar_kws)
            axes[3].set_title('Heatmap of Mean Daily Returns (%)', color=LIGHT_TEXT_COLOR)
            axes[3].set_xlabel('Month', color=LIGHT_TEXT_COLOR); axes[3].set_ylabel('Year', color=LIGHT_TEXT_COLOR)
            axes[3].tick_params(axis='x', colors=LIGHT_TEXT_COLOR); axes[3].tick_params(axis='y', colors=LIGHT_TEXT_COLOR)
            # Access the color bar axis correctly to style its label and ticks
            cbar = axes[3].collections[0].colorbar
            cbar.ax.yaxis.label.set_color(LIGHT_TEXT_COLOR)
            cbar.ax.tick_params(axis='y', colors=LIGHT_TEXT_COLOR)
            axes[3].set_yticklabels(axes[3].get_yticklabels(), rotation=0)
            if not heatmap_data.empty:
                axes[3].set_xticks(np.arange(12) + 0.5); axes[3].set_xticklabels([datetime(2000, m, 1).strftime('%b') for m in range(1, 13)])
            plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
             axes[3].text(0.5, 0.5, 'Not enough data for heatmap', color=LIGHT_TEXT_COLOR, ha='center', va='center', transform=axes[3].transAxes, fontsize=10)
             axes[3].set_title('Heatmap of Mean Daily Returns (%)', color=LIGHT_TEXT_COLOR); axes[3].set_facecolor(DARK_BG_COLOR)
             axes[3].tick_params(axis='x', colors=LIGHT_TEXT_COLOR); axes[3].tick_params(axis='y', colors=LIGHT_TEXT_COLOR)

        fig.tight_layout(pad=2.0)

        results = {
            "initial_investment": initial_investment, "final_value": final_value, "total_return": total_return,
            "total_return_pct": (total_return / initial_investment) * 100 if initial_investment else 0,
            "annualized_return_pct": annualized_return * 100,
            "annualized_volatility_pct": annualized_volatility * 100,
            "days_held": days_held,
        }
        return results, fig

    def analyze_and_plot_periodic_investment(self, df: pd.DataFrame, start_date_dt: datetime, end_date_dt: datetime,
                                             interval_days: int, investment_amount: float, commodity_type: str
                                             ) -> Tuple[Dict[str, Any], plt.Figure | None]:
        """Analyzes periodic (DCA) investments with dark theme plots and Bollinger Bands."""
        if df.empty or investment_amount <= 0 or interval_days <= 0:
            err_msg = "No data."
            if investment_amount <= 0: err_msg = "Amount > 0 req."
            if interval_days <= 0: err_msg = "Interval > 0 req."
            return {"error": err_msg}, None

        df_filtered = df[(df.index >= start_date_dt) & (df.index <= end_date_dt)].copy()
        if df_filtered.empty or len(df_filtered) < 2: return {"error": f"No data in range."}, None
        # Find close price column (e.g., 'Gold_Close' or 'Silver_Close')
        close_col = next((c for c in df_filtered.columns if c.endswith('_Close')), None)
        if not close_col: return {"error": "Close price column not found."}, None
        commodity_name = commodity_type.capitalize() # Already passed, use it


        # --- Investment Simulation (uses close_col) ---
        investment_dates_potential = pd.date_range(start=start_date_dt, end=end_date_dt, freq=f'{interval_days}D')
        potential_df = pd.DataFrame(index=investment_dates_potential)
        df_filtered_reset = df_filtered.reset_index()
        # Make sure merge_asof uses the correct column
        investment_data_actual = pd.merge_asof(potential_df, df_filtered_reset, left_index=True, right_on='Date', direction='forward', tolerance=pd.Timedelta(days=max(1,interval_days//2)))
        investment_data_actual.dropna(subset=[close_col, 'Date'], inplace=True)
        investment_data_actual = investment_data_actual.drop_duplicates(subset=['Date'], keep='first')
        investment_data_actual.set_index('Date', inplace=True)

        total_invested = 0.0; total_units_bought = 0.0; investment_log = []
        for date, row in investment_data_actual.iterrows():
             price = row[close_col] # Use identified close column
             if pd.isna(price) or price <= 0: continue
             units_bought_this_period = investment_amount / price
             total_units_bought += units_bought_this_period; total_invested += investment_amount
             investment_log.append({'Date': date, 'Total_Invested': total_invested, 'Units_Held': total_units_bought, 'Current_Value': total_units_bought * price})

        if not investment_log: return {"error": "No valid investment periods found."}, None
        growth_df = pd.DataFrame(investment_log); growth_df.set_index('Date', inplace=True)

        # --- Final Calculations (uses close_col) ---
        last_available_price = df_filtered[close_col].iloc[-1]
        final_value = total_units_bought * last_available_price
        total_return = final_value - total_invested
        days_held = (end_date_dt - start_date_dt).days; years = days_held / 365.25
        annualized_return = (final_value / total_invested) ** (1 / years) - 1 if (years > 0 and total_invested > 0 and final_value > 0) else 0.0
        df_filtered['Daily_Returns_Pct'] = df_filtered[close_col].pct_change()
        trading_days_count = df_filtered['Daily_Returns_Pct'].count()
        trading_days_per_year = trading_days_count / years if years > 0 else 252
        annualized_volatility = df_filtered['Daily_Returns_Pct'].std() * np.sqrt(max(1, trading_days_per_year)) if trading_days_count > 1 else 0.0
        # Bollinger Bands calculation
        bb_window = 20; bb_std = 2
        df_filtered['BB_SMA'] = df_filtered[close_col].rolling(window=bb_window).mean()
        df_filtered['BB_StdDev'] = df_filtered[close_col].rolling(window=bb_window).std()
        df_filtered['BB_Upper'] = df_filtered['BB_SMA'] + (bb_std * df_filtered['BB_StdDev'])
        df_filtered['BB_Lower'] = df_filtered['BB_SMA'] - (bb_std * df_filtered['BB_StdDev'])

        # --- Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor(DARK_BG_COLOR)
        axes = axes.flatten()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7); formatter = mdates.ConciseDateFormatter(locator)

        # Plot 1: Price & Investment Points (uses close_col)
        axes[0].plot(df_filtered.index, df_filtered[close_col], label=f'{commodity_name} Price', color='deepskyblue', zorder=2)
        axes[0].scatter(investment_data_actual.index, investment_data_actual[close_col], color='tomato', marker='o', s=30, label='Investments', zorder=5)
        axes[0].set_title(f'{commodity_name} Price & Investment Dates'); axes[0].set_ylabel('Price (USD)')
        axes[0].xaxis.set_major_locator(locator); axes[0].xaxis.set_major_formatter(formatter)
        axes[0].grid(True, linestyle='--', alpha=0.3); axes[0].legend()
        style_plot_for_dark_theme(axes[0])

        # Plot 2: Investment Growth (unchanged)
        axes[1].plot(growth_df.index, growth_df['Current_Value'], label='Portfolio Value', color='limegreen', marker='.', markersize=5, linestyle='-')
        axes[1].plot(growth_df.index, growth_df['Total_Invested'], label='Total Invested', color='coral', linestyle='--')
        last_inv_date = growth_df.index[-1]
        axes[1].scatter(last_inv_date, final_value, marker='*', s=100, color='gold', label=f'Final Value', zorder=5)
        axes[1].set_title('Periodic Investment Growth'); axes[1].set_ylabel('Value (USD)')
        axes[1].ticklabel_format(style='plain', axis='y')
        axes[1].xaxis.set_major_locator(locator); axes[1].xaxis.set_major_formatter(formatter)
        axes[1].grid(True, linestyle='--', alpha=0.3); axes[1].legend()
        style_plot_for_dark_theme(axes[1])

        # Plot 3: Bollinger Bands (uses close_col)
        axes[2].plot(df_filtered.index, df_filtered[close_col], label='Close Price', color='deepskyblue', lw=1.5)
        axes[2].plot(df_filtered.index, df_filtered['BB_SMA'], label=f'{bb_window}-Day SMA', color='orange', lw=1, linestyle='--')
        axes[2].plot(df_filtered.index, df_filtered['BB_Upper'], label='Upper Band', color='lightcoral', lw=1, linestyle=':')
        axes[2].plot(df_filtered.index, df_filtered['BB_Lower'], label='Lower Band', color='lightcoral', lw=1, linestyle=':')
        axes[2].fill_between(df_filtered.index, df_filtered['BB_Lower'], df_filtered['BB_Upper'], color='lightcoral', alpha=0.15, label=f'{bb_std} Std Dev Range')
        axes[2].set_title(f'Bollinger Bands ({bb_window}, {bb_std})')
        axes[2].set_ylabel('Price (USD)')
        axes[2].xaxis.set_major_locator(locator); axes[2].xaxis.set_major_formatter(formatter)
        axes[2].legend(loc='upper left')
        style_plot_for_dark_theme(axes[2])

        # Plot 4: Monthly Returns Heatmap (uses close_col)
        # Identical logic to single investment plot 4, just ensure 'close_col' is used if logic copied/pasted
        df_filtered['Daily_Returns'] = df_filtered[close_col].pct_change() * 100
        df_filtered['Month'] = df_filtered.index.month; df_filtered['Year'] = df_filtered.index.year
        if len(df_filtered['Year'].unique()) > 0 and len(df_filtered['Month'].unique()) > 0 :
            heatmap_data = df_filtered.pivot_table(values='Daily_Returns', index='Year', columns='Month', aggfunc='mean')
            heatmap_data = heatmap_data.reindex(columns=range(1, 13))
            axes[3].set_facecolor(DARK_BG_COLOR)
            cbar_kws = {'label': 'Mean Daily Return (%)'}
            sns.heatmap(heatmap_data, cmap='RdYlGn', annot=True, fmt=".1f", linewidths=.5, ax=axes[3], cbar_kws=cbar_kws)
            axes[3].set_title(f'{commodity_name} Heatmap of Mean Daily Returns (%)', color=LIGHT_TEXT_COLOR)
            axes[3].set_xlabel('Month', color=LIGHT_TEXT_COLOR); axes[3].set_ylabel('Year', color=LIGHT_TEXT_COLOR)
            axes[3].tick_params(axis='x', colors=LIGHT_TEXT_COLOR); axes[3].tick_params(axis='y', colors=LIGHT_TEXT_COLOR)
            cbar = axes[3].collections[0].colorbar
            cbar.ax.yaxis.label.set_color(LIGHT_TEXT_COLOR)
            cbar.ax.tick_params(axis='y', colors=LIGHT_TEXT_COLOR)
            axes[3].set_yticklabels(axes[3].get_yticklabels(), rotation=0)
            if not heatmap_data.empty:
                axes[3].set_xticks(np.arange(12) + 0.5); axes[3].set_xticklabels([datetime(2000, m, 1).strftime('%b') for m in range(1, 13)])
            plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            axes[3].text(0.5, 0.5, 'Not enough data for heatmap', color=LIGHT_TEXT_COLOR, ha='center', va='center', transform=axes[3].transAxes, fontsize=10)
            axes[3].set_title(f'{commodity_name} Heatmap of Mean Daily Returns (%)', color=LIGHT_TEXT_COLOR); axes[3].set_facecolor(DARK_BG_COLOR)
            axes[3].tick_params(axis='x', colors=LIGHT_TEXT_COLOR); axes[3].tick_params(axis='y', colors=LIGHT_TEXT_COLOR)


        fig.tight_layout(pad=2.0)

        results = {
            "total_invested": total_invested, "final_value": final_value, "total_return": total_return,
            "total_return_pct": (total_return / total_invested) * 100 if total_invested else 0,
            "annualized_return_pct": annualized_return * 100,
            "annualized_volatility_pct": annualized_volatility * 100, # Vol of underlying
            "number_of_investments": len(growth_df),
            "average_cost_per_unit": total_invested / total_units_bought if total_units_bought else 0,
        }
        return results, fig

    def compare_commodities(self, gold_df: pd.DataFrame, silver_df: pd.DataFrame, start_date_dt: datetime, end_date_dt: datetime) -> Tuple[Dict[str, Any], plt.Figure | None]:
        """Compares Gold and Silver performance with dark theme plots."""
        if gold_df.empty or silver_df.empty: return {"error": "Missing data."}, None
        gold_close_col = next((c for c in gold_df.columns if c == 'Gold_Close'), None)
        silver_close_col = next((c for c in silver_df.columns if c == 'Silver_Close'), None)
        if not gold_close_col or not silver_close_col: return {"error":"Close price missing in one dataset."}, None

        # Use only the Close columns for comparison calculation
        gold_to_comp = gold_df[[gold_close_col]].copy()
        silver_to_comp = silver_df[[silver_close_col]].copy()

        common_index = gold_to_comp.index.intersection(silver_to_comp.index)
        gold_common = gold_to_comp.loc[common_index]
        silver_common = silver_to_comp.loc[common_index]

        mask = (gold_common.index >= start_date_dt) & (gold_common.index <= end_date_dt)
        gold_filtered = gold_common.loc[mask]
        silver_filtered = silver_common.loc[mask]

        if gold_filtered.empty or silver_filtered.empty or len(gold_filtered) < 2: return {"error": f"Insufficient overlap."}, None

        # --- Normalization & Perf Calc ---
        gold_start_price = gold_filtered[gold_close_col].iloc[0]
        silver_start_price = silver_filtered[silver_close_col].iloc[0]
        if pd.isna(gold_start_price) or gold_start_price <= 0 or pd.isna(silver_start_price) or silver_start_price <= 0: return {"error": "Invalid start price."}, None
        gold_normalized = (gold_filtered[gold_close_col] / gold_start_price) * 100
        silver_normalized = (silver_filtered[silver_close_col] / silver_start_price) * 100
        gold_end_price = gold_filtered[gold_close_col].iloc[-1]; silver_end_price = silver_filtered[silver_close_col].iloc[-1]
        gold_return_pct = (gold_end_price / gold_start_price - 1) * 100
        silver_return_pct = (silver_end_price / silver_start_price - 1) * 100

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(DARK_BG_COLOR)
        ax.plot(gold_normalized.index, gold_normalized, label='Gold (Norm)', color='gold', lw=1.5)
        ax.plot(silver_normalized.index, silver_normalized, label='Silver (Norm)', color='silver', lw=1.5)
        ax.set_title('Gold vs Silver Performance (Normalized)'); ax.set_xlabel('Date'); ax.set_ylabel('Normalized Price (Start=100)')
        locator=mdates.AutoDateLocator(minticks=4, maxticks=8); formatter=mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle='--', alpha=0.3); ax.legend()
        style_plot_for_dark_theme(ax)
        fig.tight_layout(pad=1.5)

        results = {"gold_return_pct": gold_return_pct, "silver_return_pct": silver_return_pct,
                   "start_date": gold_filtered.index.min().strftime('%Y-%m-%d'),
                   "end_date": gold_filtered.index.max().strftime('%Y-%m-%d')}
        return results, fig

    def analyze_gold_silver_ratio(self, gold_df: pd.DataFrame, silver_df: pd.DataFrame, start_date_dt: datetime, end_date_dt: datetime) -> Tuple[Dict[str, Any], plt.Figure | None]:
        """Calculates and plots the Gold/Silver ratio."""
        if gold_df.empty or silver_df.empty: return {"error": "Missing data."}, None
        gold_close_col = 'Gold_Close'
        silver_close_col = 'Silver_Close'
        if gold_close_col not in gold_df.columns or silver_close_col not in silver_df.columns:
            return {"error": "Close price columns (Gold_Close, Silver_Close) not found."}, None

        # Select only Close columns and align indexes
        merged_df = pd.merge(gold_df[[gold_close_col]], silver_df[[silver_close_col]],
                             left_index=True, right_index=True, how='inner')

        # Filter date range
        merged_df = merged_df[(merged_df.index >= start_date_dt) & (merged_df.index <= end_date_dt)]

        if merged_df.empty or len(merged_df) < 2: return {"error": f"No overlapping data in range."}, None

        # Calculate Ratio (ensure silver price isn't zero)
        if (merged_df[silver_close_col] <= 0).any():
            st.warning("Found non-positive Silver prices, ratio cannot be calculated for these dates.")
            merged_df = merged_df[merged_df[silver_close_col] > 0] # Filter out invalid silver prices
            if merged_df.empty: return {"error": "No valid data after filtering non-positive Silver prices."}, None

        merged_df['Ratio'] = merged_df[gold_close_col] / merged_df[silver_close_col]
        merged_df.dropna(subset=['Ratio'], inplace=True) # Drop any NaN ratios

        if merged_df.empty: return {"error": "Ratio calculation resulted in no valid data."}, None

        current_ratio = merged_df['Ratio'].iloc[-1]
        average_ratio = merged_df['Ratio'].mean()
        min_ratio = merged_df['Ratio'].min()
        max_ratio = merged_df['Ratio'].max()

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(DARK_BG_COLOR)

        ax.plot(merged_df.index, merged_df['Ratio'], label='Gold/Silver Ratio', color='skyblue', lw=1.5)
        ax.axhline(average_ratio, color='lightcoral', linestyle='--', lw=1, label=f'Period Avg: {average_ratio:.2f}')

        ax.set_title('Gold/Silver Ratio Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Ratio (Gold Price / Silver Price)')
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8); formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best')
        style_plot_for_dark_theme(ax)
        fig.tight_layout(pad=1.5)

        results = {
            "current_ratio": current_ratio,
            "average_ratio": average_ratio,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "start_date": merged_df.index.min().strftime('%Y-%m-%d'),
            "end_date": merged_df.index.max().strftime('%Y-%m-%d')
        }
        return results, fig


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Commodity Investment Analyzer")
st.title("🪙 Commodity Investment Analysis Tool")
st.markdown("Analyze Gold, Silver, compare performance, and explore the Gold/Silver Ratio.")

# --- Instantiate Analyzer ---
analyzer = CommodityInvestmentTracker()

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
commodity_options = ['Gold', 'Silver']
selected_commodity = st.sidebar.selectbox("Select Commodity for Main Analysis", commodity_options)
date_start = st.sidebar.date_input("Start Date", value=DEFAULT_START_DATE, max_value=datetime.now() - timedelta(days=1))
date_end = st.sidebar.date_input("End Date", value=DEFAULT_END_DATE, min_value=date_start + timedelta(days=1), max_value=datetime.now())
date_start_dt = datetime.combine(date_start, datetime.min.time())
date_end_dt = datetime.combine(date_end, datetime.max.time())

# --- Fetch Base Commodity Data (for selected commodity) ---
fetch_start_dt = date_start_dt - timedelta(days=90) # Padding for rolling calcs
try:
    fetch_key = f"{selected_commodity}_{fetch_start_dt.strftime('%Y%m%d')}"
    if 'fetched_data' not in st.session_state or st.session_state.get('fetch_key') != fetch_key:
        with st.spinner(f"Fetching {selected_commodity} data..."):
             st.session_state.commodity_df = analyzer.get_commodity_data(fetch_start_dt, selected_commodity)
             st.session_state.fetch_key = fetch_key
    commodity_df = st.session_state.commodity_df
    # Always filter to the precise user range for analysis display/download
    analysis_df = commodity_df[(commodity_df.index >= date_start_dt) & (commodity_df.index <= date_end_dt)].copy()
except Exception as e:
    st.error(f"Failed to fetch/process {selected_commodity} data: {e}")
    st.stop()

# Verify analysis_df after filtering
if analysis_df.empty:
     st.warning(f"No {selected_commodity} data found for the selected period: {date_start.strftime('%d-%m-%Y')} to {date_end.strftime('%d-%m-%Y')}.")
     # Optionally show the head of the *fetched* data if it wasn't empty before filtering
     if not commodity_df.empty:
         st.expander("View Head of Fetched Data (Wider Range)").dataframe(commodity_df.head())
     st.stop()


# --- Sidebar Analysis Selection & Download ---
st.sidebar.markdown("---")
st.sidebar.header("Analysis Options")
analysis_options = ['Single Investment', 'Periodic Investment (DCA)', 'Compare Gold & Silver', 'Gold/Silver Ratio']
analysis_type = st.sidebar.radio("Choose Analysis Type:", analysis_options)
st.sidebar.markdown("---")
st.sidebar.subheader("Download Data")
st.sidebar.download_button(label=f"Download {selected_commodity} Data (CSV)", data=convert_df_to_csv(analysis_df), file_name=f"{selected_commodity.lower()}_data_{date_start.strftime('%Y%m%d')}_{date_end.strftime('%Y%m%d')}.csv", mime='text/csv')


# --- Main Analysis Display ---

# Common function to fetch the 'other' commodity data if needed
def get_other_commodity_data(current_selection, start_fetch_dt):
    other_commodity = 'Silver' if current_selection == 'Gold' else 'Gold'
    other_fetch_key = f"{other_commodity}_{start_fetch_dt.strftime('%Y%m%d')}"
    try:
        if 'fetched_data_other' not in st.session_state or st.session_state.get('fetch_key_other') != other_fetch_key:
             with st.spinner(f"Fetching {other_commodity} data for comparison..."):
                st.session_state.other_commodity_df = analyzer.get_commodity_data(start_fetch_dt, other_commodity)
                st.session_state.fetch_key_other = other_fetch_key
        return st.session_state.other_commodity_df
    except Exception as e:
        st.error(f"Failed to fetch {other_commodity} data: {e}")
        return pd.DataFrame() # Return empty DF on error

# Display logic based on selection
if analysis_type == 'Single Investment':
    st.header(f"📈 Single Initial Investment Analysis: {selected_commodity}")
    initial_investment = st.number_input("Initial Investment Amount ($)", min_value=1.0, value=1000.0, step=100.0)
    if st.button("Analyze Single Investment", key="analyze_single"):
        with st.spinner("Analyzing..."):
            if not analysis_df.empty and initial_investment > 0:
                st.subheader(f"Results ({date_start.strftime('%d-%m-%Y')} to {date_end.strftime('%d-%m-%Y')})")
                results, fig = analyzer.analyze_investment(analysis_df, date_start_dt, date_end_dt, initial_investment)
                if "error" in results: st.error(results["error"])
                elif fig:
                    cols = st.columns(4); cols[0].metric("Initial Inv.", f"${results['initial_investment']:,.2f}"); cols[1].metric("Final Value", f"${results['final_value']:,.2f}"); cols[2].metric("Total Return", f"{results['total_return']:+,.2f}", f"{results['total_return_pct']:.2f}%")
                    vol_text = f"{results['annualized_volatility_pct']:.2f}%" if results.get('annualized_volatility_pct') is not None else "N/A"
                    cols[3].metric("Annualized Volatility", vol_text); st.metric("Annualized Return", f"{results['annualized_return_pct']:.2f}%")
                    st.pyplot(fig); plt.close(fig)
                else: st.error("Analysis failed.")
            else: st.warning("Ensure data loaded & investment > 0.")

elif analysis_type == 'Periodic Investment (DCA)':
    st.header(f"💰 Periodic (DCA) Investment Analysis: {selected_commodity}")
    col1, col2 = st.columns(2); periodic_amount = col1.number_input("Amount per Period ($)", min_value=1.0, value=100.0, step=10.0); interval_days = col2.number_input("Investment Interval (Days)", min_value=1, value=30, step=1)
    if st.button("Analyze Periodic Investment", key="analyze_periodic"):
        with st.spinner("Analyzing..."):
            if not analysis_df.empty and periodic_amount > 0 and interval_days > 0:
                st.subheader(f"Results ({date_start.strftime('%d-%m-%Y')} to {date_end.strftime('%d-%m-%Y')})")
                results, fig = analyzer.analyze_and_plot_periodic_investment(analysis_df, date_start_dt, date_end_dt, interval_days, periodic_amount, selected_commodity)
                if "error" in results: st.error(results["error"])
                elif fig:
                     st.markdown(f"Strategy: Invest **${periodic_amount:,.2f}** every **{interval_days}** days")
                     cols = st.columns(4); cols[0].metric("Total Invested", f"${results['total_invested']:,.2f}"); cols[1].metric("Final Value", f"${results['final_value']:,.2f}"); cols[2].metric("Total Return", f"{results['total_return']:+,.2f}", f"{results['total_return_pct']:.2f}%"); cols[3].metric("Avg. Cost / Unit", f"${results['average_cost_per_unit']:.2f}")
                     cols = st.columns(2); vol_text = f"{results['annualized_volatility_pct']:.2f}%" if results.get('annualized_volatility_pct') is not None else "N/A"; cols[0].metric("Annualized Return", f"{results['annualized_return_pct']:.2f}%"); cols[1].metric("Ann. Volatility (Underlying)", vol_text)
                     st.pyplot(fig); plt.close(fig)
                else: st.error("Analysis failed.")
            else: st.warning("Ensure data loaded & amount/interval > 0.")

elif analysis_type == 'Compare Gold & Silver':
    st.header("⚖️ Compare Gold vs. Silver Performance")
    st.markdown(f"Comparing performance: **{date_start.strftime('%d-%m-%Y')}** to **{date_end.strftime('%d-%m-%Y')}**")
    # Fetch other commodity data
    other_commodity_df = get_other_commodity_data(selected_commodity, fetch_start_dt)

    # Assign dfs correctly
    gold_df = commodity_df if selected_commodity == 'Gold' else other_commodity_df
    silver_df = commodity_df if selected_commodity == 'Silver' else other_commodity_df

    if not gold_df.empty and not silver_df.empty:
        with st.spinner("Generating comparison..."):
            results, fig = analyzer.compare_commodities(gold_df, silver_df, date_start_dt, date_end_dt)
            if "error" in results: st.error(results["error"])
            elif fig:
                 st.subheader(f"Normalized Performance ({results['start_date']} to {results['end_date']})")
                 col1, col2 = st.columns(2); col1.metric("Gold Total Return", f"{results.get('gold_return_pct', 0):.2f}%"); col2.metric("Silver Total Return", f"{results.get('silver_return_pct', 0):.2f}%")
                 st.pyplot(fig); plt.close(fig)
            else: st.error("Comparison plot failed.")
    else: st.warning("Data needed for both Gold & Silver.")

elif analysis_type == 'Gold/Silver Ratio':
    st.header("➗ Gold/Silver Ratio Analysis")
    st.markdown(f"Analyzing ratio: **{date_start.strftime('%d-%m-%Y')}** to **{date_end.strftime('%d-%m-%Y')}**")
    # Fetch other commodity data
    other_commodity_df = get_other_commodity_data(selected_commodity, fetch_start_dt)

    # Assign dfs correctly
    gold_df_ratio = commodity_df if selected_commodity == 'Gold' else other_commodity_df
    silver_df_ratio = commodity_df if selected_commodity == 'Silver' else other_commodity_df

    if not gold_df_ratio.empty and not silver_df_ratio.empty:
         with st.spinner("Calculating Ratio..."):
            results, fig = analyzer.analyze_gold_silver_ratio(gold_df_ratio, silver_df_ratio, date_start_dt, date_end_dt)
            if "error" in results: st.error(results["error"])
            elif fig:
                st.subheader(f"Ratio Analysis ({results['start_date']} to {results['end_date']})")
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Ratio", f"{results.get('current_ratio', 0):.2f}")
                col2.metric("Period Average Ratio", f"{results.get('average_ratio', 0):.2f}")
                col3.metric("Period Range", f"{results.get('min_ratio', 0):.2f} - {results.get('max_ratio', 0):.2f}")
                st.pyplot(fig); plt.close(fig)
            else: st.error("Ratio analysis failed.")
    else:
        st.warning("Data required for both Gold and Silver to calculate ratio.")


# --- Raw Data Table ---
with st.expander(f"View Raw {selected_commodity} Data (Incl. OHLC if available)"):
    st.dataframe(analysis_df, use_container_width=True)

st.markdown("---")
# --- Disclaimer Section ---
st.markdown("---") # Adds a horizontal line above the disclaimer

disclaimer_en = """
**Disclaimer:** The information and analyses provided by this tool are for informational and educational purposes only.
They do not constitute investment advice, financial advice, trading advice, or any other sort of advice, nor do they recommend buying or selling any commodity.
Investing in financial markets, including commodities, carries significant risk, and past performance is not indicative of future results.
Always conduct your own research and consult with a qualified financial professional before making any investment decisions. Data provided may not always be accurate or timely.
"""

disclaimer_tr = """
**Yasal Uyarı:** Bu araç tarafından sunulan bilgiler ve analizler yalnızca bilgilendirme ve eğitim amaçlıdır.
Yatırım tavsiyesi, finansal tavsiye, alım satım tavsiyesi veya başka herhangi bir tavsiye niteliği taşımaz ve herhangi bir emtia (altın, gümüş vb.) alım satım önerisi olarak değerlendirilmemelidir.
Finansal piyasalara, emtialar dahil olmak üzere, yatırım yapmak önemli riskler içerir ve geçmiş performans gelecekteki sonuçların göstergesi değildir.
Herhangi bir yatırım kararı vermeden önce lütfen kendi araştırmanızı yapın ve nitelikli bir finans uzmanına danışın. Sunulan veriler her zaman tam olarak doğru veya güncel olmayabilir.
"""

st.caption(f"{disclaimer_en}\n\n{disclaimer_tr}")

# End of your app.py file
