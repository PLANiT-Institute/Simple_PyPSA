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
    
    max_capacity_multiplier = st.slider(
        "Max Capacity Multiplier",
        min_value=0.5,
        max_value=50.0,
        value=20.0,
        step=0.5,
        help="Maximum capacity as multiplier of minimum (e.g., 2.0 = 200% of min)"
    )
    
    solver = st.selectbox(
        "Optimization Solver",
        options=["highs", "gurobi", "cplex", "glpk"],
        index=0,
        help="Choose optimization solver"
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
        value=1200, 
        min_value=0, 
        max_value=50000, 
        step=100,
        help="Nuclear plant capacity"
    )
    
    nuclear_cf = st.number_input(
        "Nuclear Capacity Factor",
        value=0.90,
        min_value=0.1,
        max_value=1.0,
        step=0.05,
        help="Nuclear plant capacity factor (0-1)"
    )
    
    nuclear_p_min_pu = st.number_input(
        "Nuclear p_min_pu", 
        value=0.3, 
        min_value=0.0, 
        max_value=1.0, 
        step=0.05,
        help="Minimum nuclear power output as fraction of capacity (0-1)"
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
                    nuclear_capacity_factor=nuclear_cf,
                    nuclear_p_min_pu=nuclear_p_min_pu,
                    annual_load_twh=annual_load,
                    storage_power_capacity_mw=storage_capacity,
                    storage_max_hours=storage_hours,
                    storage_efficiency=storage_efficiency,
                    storage_charge_efficiency=storage_charge_efficiency,
                    storage_discharge_efficiency=storage_discharge_efficiency,
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
                    st.session_state.model = model
                    st.session_state.optimization_success = True
                    st.success("✅ Optimization completed successfully!")
                else:
                    st.error("❌ Optimization failed. Please check your parameters.")
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
        
        # Capacity breakdown
        st.subheader("🔋 Installed Capacities")
        
        capacity_data = []
        for gen in model.generators.index:
            if model.generators.loc[gen, 'p_nom_extendable']:
                capacity = model.generators.loc[gen, 'p_nom_opt']
            else:
                capacity = model.generators.loc[gen, 'p_nom']
            capacity_data.append({
                'Technology': gen.title(),
                'Capacity (MW)': capacity,
                'Capacity (GW)': capacity/1000
            })
        
        for storage in model.storage_units.index:
            if model.storage_units.loc[storage, 'p_nom_extendable']:
                power_capacity = model.storage_units.loc[storage, 'p_nom_opt']
            else:
                power_capacity = model.storage_units.loc[storage, 'p_nom']
            
            max_hours = model.storage_units.loc[storage, 'max_hours']
            energy_capacity = power_capacity * max_hours
            
            capacity_data.append({
                'Technology': f'{storage.title()} (Power)',
                'Capacity (MW)': power_capacity,
                'Capacity (GW)': power_capacity/1000
            })
            capacity_data.append({
                'Technology': f'{storage.title()} (Energy)',
                'Capacity (MW)': energy_capacity,
                'Capacity (GW)': energy_capacity/1000
            })
        
        df_capacity = pd.DataFrame(capacity_data)
        st.dataframe(df_capacity, use_container_width=True)
    
        # Interactive plots
        st.header("📈 System Operation Analysis")
    
        # Create plots
        snapshots = model.snapshots
    
        # Get generation data
        gen_data = {}
        for gen in model.generators.index:
            capacity = model.generators_t.p.loc[:, gen].values
            gen_data[gen] = capacity
    
        # Get storage data
        if 'storage' in model.storage_units.index:
            storage_charge = model.storage_units_t.p_store.loc[:, 'storage'].values
            storage_discharge = -model.storage_units_t.p_dispatch.loc[:, 'storage'].values
            storage_soc = model.storage_units_t.state_of_charge.loc[:, 'storage'].values
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
                'Storage Charge/Discharge', 
                'Storage State of Charge'
            ),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
    
        # Colors for different technologies
        colors = {'solar': '#FFA500', 'wind': '#87CEEB', 'nuclear': '#FF6B6B', 'storage': '#32CD32'}
    
        # Plot 1: Generation by source with demand overlay
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
            height=1000,
            title_text="Energy System Operation Analysis",
            showlegend=True,
            hovermode='x unified'
        )
    
        # Update y-axis labels
        fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
        fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
        fig.update_yaxes(title_text="Energy (MWh)", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
    
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
    
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📅 Time Series Data", "📊 Summary Statistics", "💾 Export Data"])
    
        with tab1:
            st.subheader("Raw Time Series Data")
            
            # Create comprehensive dataframe
            df_timeseries = pd.DataFrame(index=snapshots)
            
            # Add generation data
            for gen, data in gen_data.items():
                df_timeseries[f'{gen.title()} Generation (MW)'] = data
            
            # Add storage data
            df_timeseries['Storage Charging (MW)'] = storage_charge
            df_timeseries['Storage Discharging (MW)'] = storage_discharge
            df_timeseries['Storage SoC (MWh)'] = storage_soc
            
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
                    'Component': 'Storage Charging',
                    'Mean (MW)': np.mean(storage_charge),
                    'Max (MW)': np.max(storage_charge),
                    'Min (MW)': np.min(storage_charge),
                    'Std Dev (MW)': np.std(storage_charge),
                    'Total Energy (MWh)': np.sum(storage_charge)
                },
                {
                    'Component': 'Storage Discharging',
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
                csv_capacity = df_capacity.to_csv(index=False)
                st.download_button(
                    label="📄 Download Capacity CSV", 
                    data=csv_capacity,
                    file_name="energy_system_capacities.csv",
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