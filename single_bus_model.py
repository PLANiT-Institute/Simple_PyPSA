import pypsa
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def get_data_capacities():
    """
    Get the maximum capacities from the generation data file.
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

def load_and_process_load_data_aligned(annual_load_twh, snapshots):
    """
    Load load data from data/loads_t.csv, scale to desired annual total, and align with snapshots.
    Data format: dates as rows (d/m/yyyy), hours (0-23) as columns.
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

def create_single_bus_model(
    # Solar parameters
    solar_capacity_mw=None,  # If None, use data maximum
    # Wind parameters  
    wind_capacity_mw=None,   # If None, use data maximum
    # Nuclear parameters
    nuclear_capacity_mw=1000,
    nuclear_capacity_factor=0.85,
    nuclear_p_min_pu=0.3,  # Minimum nuclear power output as fraction of capacity
    # Load parameters
    annual_load_twh=50,  # Annual load in TWh
    # Storage parameters (aggregated ESS + PHS)
    storage_power_capacity_mw=150,
    storage_max_hours=8,  # Hours of storage at full power
    storage_efficiency=0.85,  # Round-trip efficiency (charge * discharge)
    storage_charge_efficiency=0.95,  # Charging efficiency
    storage_discharge_efficiency=0.95,  # Discharging efficiency
    # Extendable options
    solar_extendable=True,
    wind_extendable=True,
    nuclear_extendable=False,
    storage_extendable=True,
    # Maximum capacity limits as multiplier of p_nom_min
    max_capacity_multiplier=2.0,  # Maximum capacity as multiplier of p_nom_min (e.g., 2.0 = 2x p_nom_min)
    # Capital costs ($/kW or $/kWh for storage)
    solar_capital_cost=1000,     # $/kW
    wind_capital_cost=1500,      # $/kW  
    nuclear_capital_cost=6000,   # $/kW
    storage_capital_cost=500     # $/kW (power capacity)
):
    """
    Create a single bus PyPSA model with solar, wind, nuclear, and aggregated storage.
    
    Parameters:
    -----------
    solar_capacity_mw : float or None
        Solar capacity in MW. If None, uses maximum from data.
        If extendable=True, this becomes p_nom_min.
    wind_capacity_mw : float or None
        Wind capacity in MW. If None, uses maximum from data.
        If extendable=True, this becomes p_nom_min.
    nuclear_capacity_mw : float
        Nuclear capacity in MW. If extendable=True, this becomes p_nom_min.
    nuclear_capacity_factor : float
        Nuclear capacity factor (0-1)
    annual_load_twh : float
        Annual load in TWh. Used to scale the load profile.
    storage_power_capacity_mw : float
        Aggregated storage power capacity in MW. If extendable=True, this becomes p_nom_min.
    storage_max_hours : float
        Storage duration in hours at full power
    storage_efficiency : float
        Storage round-trip efficiency (0-1)
    solar_extendable : bool
        Whether solar can be expanded (capacity becomes p_nom_min if True)
    wind_extendable : bool
        Whether wind can be expanded (capacity becomes p_nom_min if True)
    nuclear_extendable : bool
        Whether nuclear can be expanded (capacity becomes p_nom_min if True)
    storage_extendable : bool
        Whether storage can be expanded (capacity becomes p_nom_min if True)
    max_capacity_multiplier : float
        Maximum capacity as multiplier of p_nom_min (e.g., 2.0 means 2x p_nom_min, 0.5 means 50%).
        Only applies to extendable components.
    solar_capital_cost : float
        Solar capital cost in $/kW
    wind_capital_cost : float
        Wind capital cost in $/kW
    nuclear_capital_cost : float
        Nuclear capital cost in $/kW
    storage_capital_cost : float
        Storage capital cost in $/kW (power capacity)
    """
    
    # Create network
    network = pypsa.Network()
    
    # Load and process generation data
    try:
        gen_data = pd.read_excel('data/generator-p_set.xlsx')
        
        # Handle datetime creation based on available columns
        if 'date' in gen_data.columns and 'time' in gen_data.columns:
            # Combine separate date and time columns
            gen_data['datetime'] = pd.to_datetime(gen_data['date'].astype(str) + ' ' + gen_data['time'].astype(str), 
                                                errors='coerce')
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
        
    except Exception as e:
        print(f"Error loading generation data: {e}")
        print("Creating default generation profiles...")
        # Create default data
        hours_in_year = 8760
        dates = pd.date_range('2024-01-01', periods=hours_in_year, freq='H')
        gen_data = pd.DataFrame(index=dates)
        solar_absolute = np.random.uniform(0, 1, hours_in_year) * 100
        wind_absolute = np.random.uniform(0, 1, hours_in_year) * 100
    
    # Create snapshots
    network.set_snapshots(gen_data.index)
    
    # Add single bus
    network.add("Bus", "bus", v_nom=230)  # 230 kV bus
    
    # Add carriers
    network.add("Carrier", "solar", color="yellow")
    network.add("Carrier", "wind", color="blue")
    network.add("Carrier", "nuclear", color="red")
    network.add("Carrier", "electricity")
    
    # Get maximum values as installed capacity
    solar_max_capacity = solar_absolute.max() if solar_absolute.max() > 0 else 1
    wind_max_capacity = wind_absolute.max() if wind_absolute.max() > 0 else 1
    
    # Convert to capacity factors (p_max_pu)
    solar_capacity_factor = solar_absolute / solar_max_capacity
    wind_capacity_factor = wind_absolute / wind_max_capacity
    
    # Add solar generator
    actual_solar_capacity = solar_capacity_mw if solar_capacity_mw is not None else solar_max_capacity
    
    solar_params = {
        "name": "solar",
        "bus": "bus",
        "carrier": "solar",
        "p_nom_extendable": solar_extendable,
        "marginal_cost": 0,
        "capital_cost": solar_capital_cost,
        "p_max_pu": solar_capacity_factor
    }
    
    if solar_extendable:
        solar_params["p_nom_min"] = actual_solar_capacity
        solar_params["p_nom_max"] = actual_solar_capacity * max_capacity_multiplier
        solar_params["p_nom"] = actual_solar_capacity  # Starting point
    else:
        solar_params["p_nom"] = actual_solar_capacity
    
    network.add("Generator", **solar_params)
    
    # Add wind generator
    actual_wind_capacity = wind_capacity_mw if wind_capacity_mw is not None else wind_max_capacity
    
    wind_params = {
        "name": "wind",
        "bus": "bus",
        "carrier": "wind",
        "p_nom_extendable": wind_extendable,
        "marginal_cost": 0,
        "capital_cost": wind_capital_cost,
        "p_max_pu": wind_capacity_factor
    }
    
    if wind_extendable:
        wind_params["p_nom_min"] = actual_wind_capacity
        wind_params["p_nom_max"] = actual_wind_capacity * max_capacity_multiplier
        wind_params["p_nom"] = actual_wind_capacity  # Starting point
    else:
        wind_params["p_nom"] = actual_wind_capacity
    
    network.add("Generator", **wind_params)
    
    # Add nuclear generator (fixed capacity factor)
    nuclear_profile = np.full(len(network.snapshots), nuclear_capacity_factor)
    
    nuclear_params = {
        "name": "nuclear",
        "bus": "bus",
        "carrier": "nuclear",
        "p_min_pu": nuclear_p_min_pu,
        "p_nom_extendable": nuclear_extendable,
        "marginal_cost": 0,  # Low but non-zero marginal cost
        "capital_cost": nuclear_capital_cost,
        "p_max_pu": nuclear_profile
    }
    
    if nuclear_extendable:
        nuclear_params["p_nom_min"] = nuclear_capacity_mw
        nuclear_params["p_nom_max"] = nuclear_capacity_mw * max_capacity_multiplier
        nuclear_params["p_nom"] = nuclear_capacity_mw  # Starting point
    else:
        nuclear_params["p_nom"] = nuclear_capacity_mw
    
    network.add("Generator", **nuclear_params)
    
    # Add aggregated storage unit (ESS + PHS combined)
    network.add("Carrier", "storage")
    
    storage_params = {
        "name": "storage",
        "bus": "bus",
        "carrier": "storage",
        "p_nom_extendable": storage_extendable,
        "max_hours": storage_max_hours,
        "efficiency_store": storage_charge_efficiency,
        "efficiency_dispatch": storage_discharge_efficiency,
        "marginal_cost": 0,
        "capital_cost": storage_capital_cost,
        "cyclic_state_of_charge": True
    }
    
    if storage_extendable:
        storage_params["p_nom_min"] = storage_power_capacity_mw
        storage_params["p_nom_max"] = storage_power_capacity_mw * max_capacity_multiplier
        storage_params["p_nom"] = storage_power_capacity_mw  # Starting point
    else:
        storage_params["p_nom"] = storage_power_capacity_mw
    
    network.add("StorageUnit", **storage_params)
    
    # Load and process actual load data with proper time alignment
    load_profile = load_and_process_load_data_aligned(annual_load_twh, network.snapshots)
    
    network.add("Load",
                "load",
                bus="bus",
                p_set=load_profile)
    
    return network

def run_model_optimization(network, solver='highs'):
    """
    Run the optimization for the network model.
    """
    try:
        network.optimize(solver_name=solver)
        return True
    except Exception as e:
        print(f"Optimization failed: {e}")
        return False

def print_results(network):
    """
    Print optimization results.
    """
    if not hasattr(network, 'objective') or network.objective is None:
        print("No optimization results available.")
        return
        
    print(f"\nOptimization Results:")
    print(f"Objective Value: {network.objective:.2f}")
    print(f"\nInstalled Capacities:")
    
    for gen in network.generators.index:
        if network.generators.loc[gen, 'p_nom_extendable']:
            capacity = network.generators.loc[gen, 'p_nom_opt']
        else:
            capacity = network.generators.loc[gen, 'p_nom']
        print(f"{gen}: {capacity:.2f} MW")
    
    for storage in network.storage_units.index:
        if network.storage_units.loc[storage, 'p_nom_extendable']:
            power_capacity = network.storage_units.loc[storage, 'p_nom_opt']
        else:
            power_capacity = network.storage_units.loc[storage, 'p_nom']
        
        max_hours = network.storage_units.loc[storage, 'max_hours']
        energy_capacity = power_capacity * max_hours
        print(f"{storage} (power): {power_capacity:.2f} MW")
        print(f"{storage} (energy): {energy_capacity:.2f} MWh ({max_hours:.1f} hours)")

def create_interactive_plots(network):
    """
    Create interactive plots showing:
    1. Hourly generation by source with demand overlay
    2. Storage charge/discharge patterns
    """
    if not hasattr(network, 'objective') or network.objective is None:
        print("No optimization results available for plotting.")
        return
    
    # Extract time series data
    snapshots = network.snapshots
    
    # Get generation data
    gen_data = {}
    for gen in network.generators.index:
        if network.generators.loc[gen, 'p_nom_extendable']:
            capacity = network.generators_t.p.loc[:, gen].values
        else:
            capacity = network.generators_t.p.loc[:, gen].values
        gen_data[gen] = capacity
    
    # Get storage data
    storage_charge = network.storage_units_t.p_store.loc[:, 'storage'].values  # Positive = charging
    storage_discharge = -network.storage_units_t.p_dispatch.loc[:, 'storage'].values  # Negative = discharging
    storage_soc = network.storage_units_t.state_of_charge.loc[:, 'storage'].values
    
    # Get load data
    load_data = network.loads_t.p_set.loc[:, 'load'].values
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Hourly Generation vs Demand', 
            'Storage Charge/Discharge', 
            'Storage State of Charge'
        ),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]],
        vertical_spacing=0.1
    )
    
    # Plot 1: Generation by source with demand overlay
    colors = {'solar': '#FFA500', 'wind': '#87CEEB', 'nuclear': '#FF6B6B', 'storage': '#32CD32'}
    
    # Add generation traces
    for gen, data in gen_data.items():
        fig.add_trace(
            go.Scatter(
                x=snapshots, 
                y=data,
                name=f'{gen.title()} Generation',
                line=dict(color=colors.get(gen, '#000000')),
                stackgroup='generation',
                hovertemplate=f'{gen.title()}: %{{y:.0f}} MW<br>Time: %{{x}}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add demand line
    fig.add_trace(
        go.Scatter(
            x=snapshots,
            y=load_data,
            name='Demand',
            line=dict(color='black', width=2, dash='dash'),
            hovertemplate='Demand: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: Storage charge/discharge
    fig.add_trace(
        go.Scatter(
            x=snapshots,
            y=storage_charge,
            name='Storage Charging',
            line=dict(color='green'),
            fill='tozeroy',
            hovertemplate='Charging: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=snapshots,
            y=storage_discharge,
            name='Storage Discharging',
            line=dict(color='red'),
            fill='tozeroy',
            hovertemplate='Discharging: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot 3: Storage state of charge
    fig.add_trace(
        go.Scatter(
            x=snapshots,
            y=storage_soc,
            name='State of Charge',
            line=dict(color='blue'),
            fill='tozeroy',
            hovertemplate='SoC: %{y:.0f} MWh<br>Time: %{x}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Energy System Operation Analysis",
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Energy (MWh)", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    # Save the plot to HTML file
    filename = "energy_system_analysis.html"
    fig.write_html(filename)
    print(f"Interactive plot saved to: {filename}")
    print("Open this file in a web browser to view the interactive plots.")
    
    return fig

if __name__ == "__main__":
    # Show data capacities first
    print("Checking generation data...")
    get_data_capacities()
    
    print("\nCreating single bus PyPSA model...")
    
    # ==================== USER-DEFINED PARAMETERS ====================
    # Solar parameters
    solar_capacity_mw = 200000      # MW, set to None to use data maximum
    solar_extendable = True        # Whether solar can be expanded (only if solar_capacity_mw is None)
    
    # Wind parameters
    wind_capacity_mw = 100000       # MW, set to None to use data maximum
    wind_extendable = True         # Whether wind can be expanded (only if wind_capacity_mw is None)
    
    # Nuclear parameters
    nuclear_capacity_mw = 1200     # MW
    nuclear_capacity_factor = 0.90 # Capacity factor (0-1)
    nuclear_extendable = False     # Whether nuclear can be expanded
    
    # Load parameters
    annual_load_twh = 700           # Annual load in TWh
    
    # Storage parameters (aggregated ESS + PHS)
    storage_power_capacity_mw = 10000    # MW
    storage_max_hours = 6              # Hours of storage at full power
    storage_efficiency = 0.85          # Round-trip efficiency (0-1)
    storage_extendable = True          # Whether storage can be expanded
    
    # Optimization solver
    solver = 'highs'                   # Solver to use ('highs', 'gurobi', 'cplex', etc.)
    # ================================================================
    
    # Maximum capacity multiplier for extendable components
    max_capacity_multiplier = 20.0         # Multiplier for maximum capacity (2.0 = 200%, 0.5 = 50%)
    
    # Capital costs ($/kW)
    solar_capital_cost = 1000      # $/kW
    wind_capital_cost = 1500       # $/kW
    nuclear_capital_cost = 6000    # $/kW
    storage_capital_cost = 500     # $/kW (power capacity)
    
    # Create model with user-defined parameters
    model = create_single_bus_model(
        solar_capacity_mw=solar_capacity_mw,
        wind_capacity_mw=wind_capacity_mw,
        nuclear_capacity_mw=nuclear_capacity_mw,
        nuclear_capacity_factor=nuclear_capacity_factor,
        annual_load_twh=annual_load_twh,
        storage_power_capacity_mw=storage_power_capacity_mw,
        storage_max_hours=storage_max_hours,
        storage_efficiency=storage_efficiency,
        solar_extendable=solar_extendable,
        wind_extendable=wind_extendable,
        nuclear_extendable=nuclear_extendable,
        storage_extendable=storage_extendable,
        max_capacity_multiplier=max_capacity_multiplier,
        solar_capital_cost=solar_capital_cost,
        wind_capital_cost=wind_capital_cost,
        nuclear_capital_cost=nuclear_capital_cost,
        storage_capital_cost=storage_capital_cost
    )
    
    print("Model created successfully!")
    print(f"Number of snapshots: {len(model.snapshots)}")
    print(f"Generators: {list(model.generators.index)}")
    print(f"Storage systems: {list(model.storage_units.index)}")
    
    # Run optimization
    print("\nRunning optimization...")
    if run_model_optimization(model, solver=solver):
        print_results(model)
        
        # Create interactive plots
        print("\nCreating interactive plots...")
        create_interactive_plots(model)
    else:
        print("Optimization failed.")