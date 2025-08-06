"""
Data loading and processing functions for PyPSA energy system model.
"""

from datetime import datetime

import numpy as np
import pandas as pd


def get_data_capacities():
    """
    Get the maximum capacities from the generation data file.

    Returns:
        tuple: (solar_max_capacity, wind_max_capacity) in MW
    """
    try:
        # Try to read Excel file
        gen_data = pd.read_excel('data/generator-p_set.xlsx')
        print(f"Loaded generation data from Excel file")
        print(f"Columns: {list(gen_data.columns)}")

        # Find solar and wind columns
        solar_cols = [col for col in gen_data.columns if 'solar' in col.lower()]
        wind_cols = [col for col in gen_data.columns if 'wind' in col.lower()]

        if solar_cols:
            if len(solar_cols) == 1:
                solar_absolute = gen_data[solar_cols[0]].values
            else:
                # Sum multiple solar columns
                solar_absolute = gen_data[solar_cols].sum(axis=1).values
        else:
            print("Warning: No solar columns found, using default capacity")
            solar_absolute = np.array([100])  # Default value

        if wind_cols:
            if len(wind_cols) == 1:
                wind_absolute = gen_data[wind_cols[0]].values
            else:
                # Sum multiple wind columns
                wind_absolute = gen_data[wind_cols].sum(axis=1).values
        else:
            print("Warning: No wind columns found, using default capacity")
            wind_absolute = np.array([100])  # Default value

        # Get maximum values as installed capacity
        solar_max_capacity = solar_absolute.max() if len(solar_absolute) > 0 else 100
        wind_max_capacity = wind_absolute.max() if len(wind_absolute) > 0 else 100

        print(f"Data shows maximum capacities:")
        print(f"Solar: {solar_max_capacity:.2f} MW (from columns: {solar_cols})")
        print(f"Wind: {wind_max_capacity:.2f} MW (from columns: {wind_cols})")

        return solar_max_capacity, wind_max_capacity

    except Exception as e:
        print(f"Error reading generation data: {e}")
        print("Using default capacities")
        return 100, 100  # Default values


def load_generation_data():
    """
    Load and process generation data from Excel file.

    Returns:
        tuple: (gen_data_indexed, solar_absolute, wind_absolute)
    """
    try:
        gen_data = pd.read_excel('data/generator-p_set.xlsx')

        # Handle datetime creation based on available columns
        if 'date' in gen_data.columns and 'time' in gen_data.columns:
            # Combine separate date and time columns - date contains full datetime, time is hour offset
            gen_data['datetime'] = pd.to_datetime(gen_data['date']) + pd.to_timedelta(gen_data['time'], unit='h')
        elif 'date' in gen_data.columns:
            # Only date column, assume hourly data
            gen_data['datetime'] = pd.to_datetime(gen_data['date'], errors='coerce')
        else:
            # Find any datetime-like column
            datetime_cols = [col for col in gen_data.columns if any(word in col.lower() for word in ['date', 'time', 'snapshot'])]
            if datetime_cols:
                datetime_col = datetime_cols[0]
                gen_data['datetime'] = pd.to_datetime(gen_data[datetime_col], errors='coerce')
            else:
                # Create a default datetime index
                hours_in_year = len(gen_data)
                gen_data['datetime'] = pd.date_range('2024-01-01', periods=hours_in_year, freq='H')

        # Remove any rows with invalid datetimes
        gen_data = gen_data.dropna(subset=['datetime'])

        # Remove duplicate datetimes to avoid reindexing issues
        gen_data = gen_data.drop_duplicates(subset=['datetime'])

        gen_data.set_index('datetime', inplace=True)

        # Find solar and wind columns
        solar_cols = [col for col in gen_data.columns if 'solar' in col.lower()]
        wind_cols = [col for col in gen_data.columns if 'wind' in col.lower()]

        if solar_cols:
            if len(solar_cols) == 1:
                solar_absolute = gen_data[solar_cols[0]].values
            else:
                solar_absolute = gen_data[solar_cols].sum(axis=1).values
        else:
            # Create default solar profile
            solar_absolute = np.random.uniform(0, 1, len(gen_data)) * 100

        if wind_cols:
            if len(wind_cols) == 1:
                wind_absolute = gen_data[wind_cols[0]].values
            else:
                wind_absolute = gen_data[wind_cols].sum(axis=1).values
        else:
            # Create default wind profile
            wind_absolute = np.random.uniform(0, 1, len(gen_data)) * 100

        return gen_data, solar_absolute, wind_absolute

    except Exception as e:
        print(f"Error loading generation data: {e}")
        print("Creating default generation profiles...")
        # Create default data
        hours_in_year = 8760
        dates = pd.date_range('2024-01-01', periods=hours_in_year, freq='H')
        gen_data = pd.DataFrame(index=dates)
        solar_absolute = np.random.uniform(0, 1, hours_in_year) * 100
        wind_absolute = np.random.uniform(0, 1, hours_in_year) * 100

        return gen_data, solar_absolute, wind_absolute


def load_and_process_load_data_aligned(annual_load_twh, snapshots):
    """
    Load load data from data/loads_t.csv, scale to desired annual total, and align with snapshots.
    Data format: dates as rows (d/m/yyyy), hours (0-23) as columns.

    Args:
        annual_load_twh (float): Target annual load in TWh
        snapshots (pd.DatetimeIndex): Network snapshots to align with

    Returns:
        np.ndarray: Aligned load profile in MW
    """
    try:
        # Load the loads_t.csv file
        load_data = pd.read_csv('data/loads_t.csv', index_col=0)
        print(f"Loaded load data from: data/loads_t.csv")
        print(f"Data shape: {load_data.shape}")

        # Data has dates as rows and hours (0-23) as columns
        # Melt the data to long format
        load_data = load_data.reset_index()
        load_data = pd.melt(load_data, id_vars=[load_data.columns[0]],
                           var_name='hour', value_name='load_mw')

        # Parse dates and create datetime
        load_data['date'] = pd.to_datetime(load_data.iloc[:, 0], format='%d/%m/%Y', errors='coerce')
        load_data['hour'] = pd.to_numeric(load_data['hour'], errors='coerce')
        load_data['datetime'] = load_data['date'] + pd.to_timedelta(load_data['hour'], unit='h')

        # Remove any rows with invalid dates or hours
        load_data = load_data.dropna(subset=['datetime', 'load_mw'])

        # Remove duplicates and sort by datetime
        load_data = load_data.drop_duplicates(subset=['datetime']).sort_values('datetime')
        load_data.set_index('datetime', inplace=True)

        # Calculate scaling for the original data
        original_load_profile = load_data['load_mw'].values
        current_annual_twh = (original_load_profile.sum() * 1) / 1e6  # Assuming hourly data
        scaling_factor = annual_load_twh / current_annual_twh

        # Scale the load data
        load_data['load_mw'] = load_data['load_mw'] * scaling_factor

        # Reindex to match snapshots (this handles any missing timestamps)
        load_data_aligned = load_data.reindex(snapshots, method='nearest', fill_value=load_data['load_mw'].mean())

        print(f"Load data processing:")
        print(f"Original annual load: {current_annual_twh:.2f} TWh")
        print(f"Target annual load: {annual_load_twh:.2f} TWh")
        print(f"Scaling factor: {scaling_factor:.3f}")
        print(f"Load profile length after alignment: {len(load_data_aligned)} hours")

        return load_data_aligned['load_mw'].values

    except Exception as e:
        print(f"Error loading load data: {e}")
        print("Using default load profile...")
        # Return a simple default profile matching snapshots
        if len(snapshots) > 0:
            base_load = annual_load_twh * 1e6 / len(snapshots)  # Convert TWh to average MW
            return np.ones(len(snapshots)) * base_load
        else:
            return np.array([1000])  # Default 1 GW if no snapshots
