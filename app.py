import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pypsa
from single_bus_model import create_single_bus_model, run_model_optimization

# Configure Streamlit page
st.set_page_config(
    page_title="PyPSA Energy System Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title
st.title("⚡ PyPSA Energy System Optimizer")

# Create main layout with left panel for inputs and right panel for outputs
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("⚙️ System Parameters")
    
    # System-wide parameters
    st.subheader("🏭 System Settings")
    annual_load = st.number_input(
        "Annual Load (TWh)", 
        value=700.0, 
        min_value=10.0, 
        max_value=10000.0, 
        step=50.0,
        help="Total annual electricity demand"
    )
    
    max_capacity_multiplier = st.number_input(
        "Max Capacity Multiplier",
        min_value=0.5,
        max_value=50.0,
        value=20.0,
        step=0.5,
        help="Maximum capacity as multiplier of minimum (e.g., 2.0 = 200% of min)"
    )
    
    solver = st.selectbox(
        "Optimization Solver",
        options=["highs", "cbc", "gurobi", "cplex", "glpk"],
        index=0,
        help="Choose optimization solver (CBC recommended for storage constraints)"
    )

    # Solar parameters
    st.subheader("🌞 Solar Generation")
    solar_capacity = st.number_input(
        "Solar Capacity (MW)", 
        value=200000, 
        min_value=1000, 
        max_value=1000000, 
        step=10000,
        help="Minimum solar capacity (p_nom_min if extendable)"
    )
    
    solar_cost = st.number_input(
        "Solar Capital Cost ($/kW)", 
        value=1000, 
        min_value=100, 
        max_value=10000, 
        step=100
    )
    
    solar_extendable = st.checkbox("Solar Extendable", value=True)

    # Wind parameters
    st.subheader("💨 Wind Generation")
    wind_capacity = st.number_input(
        "Wind Capacity (MW)", 
        value=100000, 
        min_value=1000, 
        max_value=1000000, 
        step=10000,
        help="Minimum wind capacity (p_nom_min if extendable)"
    )
    
    wind_cost = st.number_input(
        "Wind Capital Cost ($/kW)", 
        value=1500, 
        min_value=100, 
        max_value=10000, 
        step=100
    )
    
    wind_extendable = st.checkbox("Wind Extendable", value=True)

    # Nuclear parameters
    st.subheader("⚛️ Nuclear Generation")
    nuclear_capacity = st.number_input(
        "Nuclear Capacity (MW)", 
        value=24000, 
        min_value=0, 
        max_value=50000, 
        step=100,
        help="Nuclear plant capacity"
    )
    
    nuclear_p_min_pu = st.number_input(
        "Nuclear p_min_pu", 
        value=0.8, 
        min_value=0.0, 
        max_value=1.0, 
        step=0.05,
        help="Minimum nuclear power output as fraction of capacity (0-1)"
    )
    
    nuclear_p_max_pu = st.number_input(
        "Nuclear p_max_pu", 
        value=1.0, 
        min_value=0.0, 
        max_value=1.0, 
        step=0.05,
        help="Maximum nuclear power output as fraction of capacity (0-1)"
    )
    
    nuclear_cost = st.number_input(
        "Nuclear Capital Cost ($/kW)", 
        value=6000, 
        min_value=1000, 
        max_value=20000, 
        step=500
    )
    
    nuclear_extendable = st.checkbox("Nuclear Extendable", value=False)

    # Storage parameters
    st.subheader("🔋 Energy Storage")
    storage_capacity = st.number_input(
        "Storage Power (MW)", 
        value=10000, 
        min_value=100, 
        max_value=500000, 
        step=1000,
        help="Storage power capacity"
    )
    
    storage_hours = st.number_input(
        "Storage Hours", 
        value=6.0, 
        min_value=1.0, 
        max_value=24.0, 
        step=0.5
    )
    
    storage_charge_efficiency = st.number_input(
        "Charge Efficiency", 
        value=0.95, 
        min_value=0.5, 
        max_value=1.0, 
        step=0.05,
        help="Efficiency when charging storage (0-1)"
    )
    
    storage_discharge_efficiency = st.number_input(
        "Discharge Efficiency", 
        value=0.95, 
        min_value=0.5, 
        max_value=1.0, 
        step=0.05,
        help="Efficiency when discharging storage (0-1)"
    )
    
    storage_cost = st.number_input(
        "Storage Capital Cost ($/kW)", 
        value=500, 
        min_value=100, 
        max_value=5000, 
        step=100
    )
    
    storage_extendable = st.checkbox("Storage Extendable", value=True)
    
    storage_initial_soc = st.number_input(
        "Initial State of Charge",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Initial state of charge as fraction of maximum energy capacity (0-1)"
    )

    # Run optimization button
    st.markdown("---")
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        with st.spinner("Running optimization... This may take a few minutes."):
            try:
                # Calculate round-trip efficiency
                storage_efficiency = storage_charge_efficiency * storage_discharge_efficiency
                
                # Create model
                model = create_single_bus_model(
                    solar_capacity_mw=solar_capacity,
                    wind_capacity_mw=wind_capacity,
                    nuclear_capacity_mw=nuclear_capacity,
                    nuclear_p_min_pu=nuclear_p_min_pu,
                    nuclear_p_max_pu=nuclear_p_max_pu,
                    annual_load_twh=annual_load,
                    storage_power_capacity_mw=storage_capacity,
                    storage_max_hours=storage_hours,
                    storage_efficiency=storage_efficiency,
                    storage_charge_efficiency=storage_charge_efficiency,
                    storage_discharge_efficiency=storage_discharge_efficiency,
                    storage_initial_soc=storage_initial_soc,
                    solar_extendable=solar_extendable,
                    wind_extendable=wind_extendable,
                    nuclear_extendable=nuclear_extendable,
                    storage_extendable=storage_extendable,
                    max_capacity_multiplier=max_capacity_multiplier,
                    solar_capital_cost=solar_cost,
                    wind_capital_cost=wind_cost,
                    nuclear_capital_cost=nuclear_cost,
                    storage_capital_cost=storage_cost
                )
                
                # Run optimization
                success = run_model_optimization(model, solver=solver)
                
                if success:
                    # Check if optimization actually produced results
                    if hasattr(model, 'generators_t') and hasattr(model.generators_t, 'p'):
                        st.session_state.model = model
                        st.session_state.optimization_success = True
                        st.success("✅ Optimization completed successfully!")
                    else:
                        st.error("❌ Optimization completed but produced no results. Try a different solver.")
                        st.session_state.optimization_success = False
                        # Store the model anyway for debugging
                        st.session_state.model = model
                        st.session_state.partial_results = True
                else:
                    st.error("❌ Optimization failed. Please check your parameters or try a different solver.")
                    st.session_state.optimization_success = False
                    
            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                st.session_state.optimization_success = False

with right_col:
    # Output area
    if 'optimization_success' in st.session_state and st.session_state.optimization_success:
        model = st.session_state.model
        
        # Results overview
        st.header("📊 Optimization Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if hasattr(model, 'objective') and model.objective is not None:
                st.metric("Total Cost", f"${model.objective/1e6:.1f}M")
            else:
                st.metric("Total Cost", "N/A")
        
        with col2:
            total_gen_capacity = 0
            for gen in model.generators.index:
                if model.generators.loc[gen, 'p_nom_extendable']:
                    capacity = model.generators.loc[gen, 'p_nom_opt']
                else:
                    capacity = model.generators.loc[gen, 'p_nom']
                total_gen_capacity += capacity
            st.metric("Total Generation", f"{total_gen_capacity/1000:.1f} GW")
        
        with col3:
            if 'storage' in model.storage_units.index:
                if model.storage_units.loc['storage', 'p_nom_extendable']:
                    storage_power = model.storage_units.loc['storage', 'p_nom_opt']
                else:
                    storage_power = model.storage_units.loc['storage', 'p_nom']
                st.metric("Storage Power", f"{storage_power/1000:.1f} GW")
            else:
                st.metric("Storage Power", "0 GW")
        
        with col4:
            num_snapshots = len(model.snapshots)
            st.metric("Time Snapshots", f"{num_snapshots:,}")
        
        # Output breakdown
        st.subheader("📊 Output")
        
        output_data = []
        # Define order: nuclear, solar, wind
        generator_order = ['nuclear', 'solar', 'wind']
        for gen in generator_order:
            if gen in model.generators.index:
                try:
                    # Get installed capacity
                    if model.generators.loc[gen, 'p_nom_extendable']:
                        capacity = model.generators.loc[gen, 'p_nom_opt']
                    else:
                        capacity = model.generators.loc[gen, 'p_nom']
                    
                    # Get actual generation (sum over all time periods) - check if results exist
                    if hasattr(model, 'generators_t') and hasattr(model.generators_t, 'p') and gen in model.generators_t.p.columns:
                        actual_generation = model.generators_t.p.loc[:, gen].sum()  # MWh
                        generation_gwh = actual_generation / 1000  # Convert to GWh
                        
                        # Calculate capacity factor
                        if capacity > 0:
                            max_possible_generation = capacity * len(model.snapshots)  # MW * hours = MWh
                            capacity_factor = (actual_generation / max_possible_generation) * 100  # Percentage
                        else:
                            capacity_factor = 0
                    else:
                        # No optimization results available
                        generation_gwh = 0
                        capacity_factor = 0
                    
                    output_data.append({
                        'Technology': gen.title(),
                        'Installed Capacity (MW)': capacity,
                        'Generation (GWh)': generation_gwh,
                        'Capacity Factor (%)': capacity_factor
                    })
                except Exception as e:
                    st.warning(f"Could not process {gen} generator data: {e}")
                    output_data.append({
                        'Technology': gen.title(),
                        'Installed Capacity (MW)': 0,
                        'Generation (GWh)': 0,
                        'Capacity Factor (%)': 0
                    })
        
        for storage in model.storage_units.index:
            if model.storage_units.loc[storage, 'p_nom_extendable']:
                power_capacity = model.storage_units.loc[storage, 'p_nom_opt']
            else:
                power_capacity = model.storage_units.loc[storage, 'p_nom']
            
            max_hours = model.storage_units.loc[storage, 'max_hours']
            energy_capacity = power_capacity * max_hours
            
            # Get storage discharge (energy delivered)
            storage_discharge = model.storage_units_t.p_dispatch.loc[:, storage].sum()  # MWh
            discharge_gwh = storage_discharge / 1000  # Convert to GWh
            
            # Calculate capacity factor for storage (based on discharge)
            if power_capacity > 0:
                max_possible_discharge = power_capacity * len(model.snapshots)  # MW * hours = MWh
                storage_capacity_factor = (storage_discharge / max_possible_discharge) * 100  # Percentage
            else:
                storage_capacity_factor = 0
            
            # Use "ESS" instead of storage name
            storage_name = "ESS" if storage == "storage" else storage.title()
            output_data.append({
                'Technology': f'{storage_name} (Power)',
                'Installed Capacity (MW)': power_capacity,
                'Generation (GWh)': discharge_gwh,
                'Capacity Factor (%)': storage_capacity_factor
            })
            output_data.append({
                'Technology': f'{storage_name} (Energy)',
                'Installed Capacity (MW)': energy_capacity,
                'Generation (GWh)': discharge_gwh,
                'Capacity Factor (%)': 'N/A'
            })
        
        df_output = pd.DataFrame(output_data)
        # Format the dataframe for better display
        df_output['Installed Capacity (MW)'] = df_output['Installed Capacity (MW)'].round(1)
        df_output['Generation (GWh)'] = df_output['Generation (GWh)'].round(2)
        df_output['Capacity Factor (%)'] = df_output['Capacity Factor (%)'].apply(
            lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
        )
        st.dataframe(df_output, use_container_width=True)
    
        # Interactive plots
        st.header("📈 System Operation Analysis")
    
        # Create plots
        snapshots = model.snapshots
    
        # Get generation data in order: nuclear, solar, wind
        gen_data = {}
        gen_available = {}  # Available generation (including curtailed)
        generator_order = ['nuclear', 'solar', 'wind']
        for gen in generator_order:
            if gen in model.generators.index:
                try:
                    # Check if optimization results exist
                    if hasattr(model, 'generators_t') and hasattr(model.generators_t, 'p') and gen in model.generators_t.p.columns:
                        # Actual generation
                        actual_generation = model.generators_t.p.loc[:, gen].values
                        gen_data[gen] = actual_generation
                        
                        # Available generation (capacity * p_max_pu)
                        if model.generators.loc[gen, 'p_nom_extendable']:
                            capacity = model.generators.loc[gen, 'p_nom_opt']
                        else:
                            capacity = model.generators.loc[gen, 'p_nom']
                        
                        # Get p_max_pu profile
                        if hasattr(model.generators_t, 'p_max_pu') and gen in model.generators_t.p_max_pu.columns:
                            p_max_pu = model.generators_t.p_max_pu.loc[:, gen].values
                        else:
                            p_max_pu = np.ones(len(snapshots)) * model.generators.loc[gen, 'p_max_pu']
                        
                        available_generation = capacity * p_max_pu
                        gen_available[gen] = available_generation
                    else:
                        # No optimization results - create zero arrays
                        gen_data[gen] = np.zeros(len(snapshots))
                        gen_available[gen] = np.zeros(len(snapshots))
                        
                except Exception as e:
                    st.warning(f"Could not process {gen} generation data: {e}")
                    gen_data[gen] = np.zeros(len(snapshots))
                    gen_available[gen] = np.zeros(len(snapshots))
    
        # Get storage data
        if 'storage' in model.storage_units.index:
            try:
                # Check if optimization results exist for storage
                if (hasattr(model, 'storage_units_t') and 
                    hasattr(model.storage_units_t, 'p_store') and 
                    'storage' in model.storage_units_t.p_store.columns):
                    # Get the corrected storage data from the model (after _fix_simultaneous_storage_operation)
                    p_store_raw = model.storage_units_t.p_store.loc[:, 'storage'].values
                    p_dispatch_raw = model.storage_units_t.p_dispatch.loc[:, 'storage'].values
                    
                    # Check for simultaneous operations in the data
                    threshold = 0.0001
                    simultaneous = (p_store_raw > threshold) & (p_dispatch_raw > threshold)
                    
                    if simultaneous.any():
                        count = simultaneous.sum()
                        st.warning(f"⚠️ Found {count} periods with simultaneous charge/discharge in the display data. This suggests the post-processing fix didn't work properly.")
                        
                        # Apply the fix directly in the app for display purposes
                        p_store_fixed = p_store_raw.copy()
                        p_dispatch_fixed = p_dispatch_raw.copy()
                        
                        for i, is_simultaneous in enumerate(simultaneous):
                            if is_simultaneous:
                                if p_store_raw[i] > p_dispatch_raw[i]:
                                    # Keep charging, remove discharging
                                    p_dispatch_fixed[i] = 0.0
                                else:
                                    # Keep discharging, remove charging
                                    p_store_fixed[i] = 0.0
                        
                        # Use the fixed values
                        storage_charge = -p_store_fixed  # Charging should be negative
                        storage_discharge = p_dispatch_fixed  # Discharging should be positive
                    else:
                        # No simultaneous operations detected, use original data
                        storage_charge = -p_store_raw  # Charging should be negative
                        storage_discharge = p_dispatch_raw  # Discharging should be positive
                        st.success("✅ No simultaneous charge/discharge detected in storage operation.")
                    
                    storage_soc = model.storage_units_t.state_of_charge.loc[:, 'storage'].values
                else:
                    # No optimization results for storage
                    storage_charge = np.zeros(len(snapshots))
                    storage_discharge = np.zeros(len(snapshots))
                    storage_soc = np.zeros(len(snapshots))
            except Exception as e:
                st.warning(f"Could not process storage data: {e}")
                storage_charge = np.zeros(len(snapshots))
                storage_discharge = np.zeros(len(snapshots))
                storage_soc = np.zeros(len(snapshots))
        else:
            storage_charge = np.zeros(len(snapshots))
            storage_discharge = np.zeros(len(snapshots))
            storage_soc = np.zeros(len(snapshots))
    
        # Get load data
        load_data = model.loads_t.p_set.loc[:, 'load'].values
    
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Hourly Generation vs Demand', 
                'ESS Charge/Discharge', 
                'ESS State of Charge'
            ),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
    
        # Colors for different technologies (nuclear, solar, wind, ESS)
        colors = {'nuclear': 'rgba(255, 107, 107, 0.7)', 'solar': '#FFA500', 'wind': '#87CEEB', 'storage': '#32CD32'}
        curtailed_colors = {'solar': 'rgba(255, 220, 150, 0.6)', 'wind': 'rgba(173, 216, 230, 0.6)'}  # Much lighter colors for curtailed
    
        # Plot 1: Generation by source with demand overlay
        
        # 1. Add nuclear generation (semi-transparent base)
        if 'nuclear' in gen_data:
            fig.add_trace(
                go.Scatter(
                    x=snapshots, 
                    y=gen_data['nuclear'],
                    name='Nuclear Generation',
                    line=dict(color=colors['nuclear']),
                    fill='tozeroy',
                    stackgroup='generation',
                    hovertemplate='Nuclear: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Add actual solar generation (stacked on nuclear)
        if 'solar' in gen_data:
            fig.add_trace(
                go.Scatter(
                    x=snapshots, 
                    y=gen_data['solar'],
                    name='Solar Generation',
                    line=dict(color=colors['solar']),
                    stackgroup='generation',
                    hovertemplate='Solar: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 3. Add actual wind generation (stacked on solar)
        if 'wind' in gen_data:
            fig.add_trace(
                go.Scatter(
                    x=snapshots, 
                    y=gen_data['wind'],
                    name='Wind Generation',
                    line=dict(color=colors['wind']),
                    stackgroup='generation',
                    hovertemplate='Wind: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 4. Add ESS discharge (positive contribution to generation)
        if 'storage' in model.storage_units.index:
            fig.add_trace(
                go.Scatter(
                    x=snapshots,
                    y=storage_discharge,  # Use the correctly processed positive discharge values
                    name='ESS Discharging',
                    line=dict(color='red'),
                    stackgroup='generation',
                    hovertemplate='ESS Discharging: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 5. Add ESS charge (negative values - consuming power)
        if 'storage' in model.storage_units.index:
            fig.add_trace(
                go.Scatter(
                    x=snapshots,
                    y=storage_charge,  # Use the correctly processed negative charge values
                    name='ESS Charging',
                    line=dict(color='green'),
                    fill='tozeroy',
                    hovertemplate='ESS Charging: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
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
    
        # Plot 2: Storage charge/discharge as bar chart (mutually exclusive)
        # Combine charge and discharge into a single series where:
        # - Negative values = charging (green bars going down)
        # - Positive values = discharging (red bars going up)
        
        # Create combined storage operation data
        storage_combined = np.where(
            np.abs(storage_charge) > np.abs(storage_discharge),
            storage_charge,  # Use charging (negative values)
            storage_discharge  # Use discharging (positive values)
        )
        
        # Create colors array - green for negative (charging), red for positive (discharging)
        colors_array = ['green' if val < 0 else 'red' for val in storage_combined]
        
        fig.add_trace(
            go.Bar(
                x=snapshots,
                y=storage_combined,
                name='ESS Operation',
                marker=dict(color=colors_array),
                hovertemplate='ESS: %{y:.0f} MW<br>Time: %{x}<br>' +
                             '<i>Negative = Charging, Positive = Discharging</i><extra></extra>'
            ),
            row=2, col=1
        )
    
        # Plot 3: ESS state of charge as bar chart
        fig.add_trace(
            go.Line(
                x=snapshots,
                y=storage_soc,
                name='ESS State of Charge',
                marker=dict(color='blue'),
                hovertemplate='SoC: %{y:.0f} MWh<br>Time: %{x}<extra></extra>'
            ),
            row=3, col=1
        )
    
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Energy System Operation Analysis",
            showlegend=True,
            hovermode='x unified'
        )
    
        # Update y-axis labels
        fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
        fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
        fig.update_yaxes(title_text="Energy (MWh)", row=3, col=1)
        
        # Update x-axis labels and synchronize x-axes
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        # Synchronize x-axes across all three subplots
        fig.update_xaxes(matches='x', row=1, col=1)
        fig.update_xaxes(matches='x', row=2, col=1)
        fig.update_xaxes(matches='x', row=3, col=1)
    
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
    
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📅 Time Series Data", "📊 Summary Statistics", "💾 Export Data"])
    
        with tab1:
            st.subheader("Raw Time Series Data")
            
            # Create comprehensive dataframe
            df_timeseries = pd.DataFrame(index=snapshots)
            
            # Add generation data in order: nuclear, solar, wind
            for gen, data in gen_data.items():
                df_timeseries[f'{gen.title()} Generation (MW)'] = data
            
            # Add storage data
            df_timeseries['ESS Charging (MW)'] = storage_charge
            df_timeseries['ESS Discharging (MW)'] = storage_discharge
            df_timeseries['ESS SoC (MWh)'] = storage_soc
            
            # Add load data
            df_timeseries['Demand (MW)'] = load_data
            
            st.dataframe(df_timeseries, use_container_width=True)
    
        with tab2:
            st.subheader("Summary Statistics")
            
            # Calculate statistics
            stats_data = []
            
            for gen, data in gen_data.items():
                stats_data.append({
                    'Component': f'{gen.title()} Generation',
                    'Mean (MW)': np.mean(data),
                    'Max (MW)': np.max(data),
                    'Min (MW)': np.min(data),
                    'Std Dev (MW)': np.std(data),
                    'Total Energy (MWh)': np.sum(data)
                })
            
            stats_data.extend([
                {
                    'Component': 'ESS Charging',
                    'Mean (MW)': np.mean(storage_charge),
                    'Max (MW)': np.max(storage_charge),
                    'Min (MW)': np.min(storage_charge),
                    'Std Dev (MW)': np.std(storage_charge),
                    'Total Energy (MWh)': np.sum(storage_charge)
                },
                {
                    'Component': 'ESS Discharging',
                    'Mean (MW)': np.mean(np.abs(storage_discharge)),
                    'Max (MW)': np.max(np.abs(storage_discharge)),
                    'Min (MW)': np.min(np.abs(storage_discharge)),
                    'Std Dev (MW)': np.std(storage_discharge),
                    'Total Energy (MWh)': np.sum(np.abs(storage_discharge))
                },
                {
                    'Component': 'Demand',
                    'Mean (MW)': np.mean(load_data),
                    'Max (MW)': np.max(load_data),
                    'Min (MW)': np.min(load_data),
                    'Std Dev (MW)': np.std(load_data),
                    'Total Energy (MWh)': np.sum(load_data)
                }
            ])
            
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True)
    
        with tab3:
            st.subheader("Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download time series data
                csv_timeseries = df_timeseries.to_csv()
                st.download_button(
                    label="📄 Download Time Series CSV",
                    data=csv_timeseries,
                    file_name="energy_system_timeseries.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download capacity data
                csv_output = df_output.to_csv(index=False)
                st.download_button(
                    label="📄 Download Output CSV", 
                    data=csv_output,
                    file_name="energy_system_output.csv",
                    mime="text/csv"
                )

    else:
        # Initial state - show information
        st.info("👈 Configure your energy system parameters on the left and click 'Run Optimization' to get started!")
        
        st.markdown("""
        ## 🔋 About this Tool
        
        This interactive energy system optimizer helps you design and analyze power systems with:
        
        - **🌞 Solar Generation**: Photovoltaic power with time-varying output
        - **💨 Wind Generation**: Wind turbines with capacity factor profiles  
        - **⚛️ Nuclear Power**: Baseload generation with high capacity factors
        - **🔋 Energy Storage**: Battery/pumped hydro for grid balancing
        
        ### 📊 Key Features:
        - Interactive parameter adjustment
        - Real-time optimization with PyPSA
        - Detailed time-series visualization
        - Capital cost optimization
        - Capacity expansion planning
        - Export capabilities for further analysis
        
        ### 🚀 Getting Started:
        1. Adjust system parameters on the left
        2. Set capital costs for each technology
        3. Configure capacity limits and extendable options
        4. Click 'Run Optimization' to solve the model
        5. Explore results in interactive charts and tables
        """)