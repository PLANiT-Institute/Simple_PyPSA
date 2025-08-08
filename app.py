import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from libs.results import create_generation_mix_chart, get_generation_mix_data
from single_bus_model import run_energy_system_optimization

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
        step=50.0,
        help="Total annual electricity demand"
    )


    solver = st.selectbox(
        "Optimization Solver",
        options=["highs", "cbc", "gurobi", "cplex", "glpk"],
        index=0,
        help="Choose optimization solver (CBC recommended for storage constraints)"
    )

    # Solar parameters
    st.subheader("🌞 Solar Generation")
    
    solar_extendable = st.checkbox("Solar Extendable", value=False)
    
    if solar_extendable:
        solar_p_nom_min = st.number_input(
            "Minimum Capacity (when extendable) (MW)",
            value=200000,
            min_value=0,
            step=10000,
            help="Minimum solar capacity for optimization"
        )
        
        solar_p_nom_max = st.number_input(
            "Maximum Capacity (when extendable) (MW)",
            value=500000,
            min_value=0,
            step=10000,
            help="Maximum solar capacity for optimization"
        )
        
        solar_capacity = None  # Not used when extendable
    else:
        solar_capacity = st.number_input(
            "Fixed Capacity (MW)",
            value=200000,
            min_value=0,
            step=10000,
            help="Fixed solar capacity (not optimized)"
        )
        
        solar_p_nom_min = None  # Not used when not extendable
        solar_p_nom_max = None  # Not used when not extendable

    solar_cost = st.number_input(
        "Solar Capital Cost ($/kW)",
        value=1000,
        min_value=0,
        step=100
    )
    
    solar_marginal_cost = st.number_input(
        "Solar Marginal Cost ($/MWh)",
        value=0.1,
        min_value=0.0,
        step=0.1,
        help="Operational cost per MWh of solar generation"
    )

    # Wind parameters
    st.subheader("💨 Wind Generation")
    
    wind_extendable = st.checkbox("Wind Extendable", value=False)
    
    if wind_extendable:
        wind_p_nom_min = st.number_input(
            "Minimum Capacity (when extendable) (MW)",
            value=100000,
            min_value=0,
            step=10000,
            help="Minimum wind capacity for optimization"
        )
        
        wind_p_nom_max = st.number_input(
            "Maximum Capacity (when extendable) (MW)",
            value=300000,
            min_value=0,
            step=10000,
            help="Maximum wind capacity for optimization"
        )
        
        wind_capacity = None  # Not used when extendable
    else:
        wind_capacity = st.number_input(
            "Fixed Capacity (MW)",
            value=100000,
            min_value=0,
            step=10000,
            help="Fixed wind capacity (not optimized)"
        )
        
        wind_p_nom_min = None  # Not used when not extendable
        wind_p_nom_max = None  # Not used when not extendable

    wind_cost = st.number_input(
        "Wind Capital Cost ($/kW)",
        value=1500,
        min_value=0,
        step=100
    )
    
    wind_marginal_cost = st.number_input(
        "Wind Marginal Cost ($/MWh)",
        value=0.1,
        min_value=0.0,
        step=0.1,
        help="Operational cost per MWh of wind generation"
    )

    # Nuclear parameters
    st.subheader("⚛️ Nuclear Generation")
    
    nuclear_extendable = st.checkbox("Nuclear Extendable", value=False)
    
    if nuclear_extendable:
        nuclear_p_nom_min = st.number_input(
            "Minimum Capacity (when extendable) (MW)",
            value=24000,
            min_value=0,
            step=100,
            help="Minimum nuclear capacity for optimization"
        )
        
        nuclear_p_nom_max = st.number_input(
            "Maximum Capacity (when extendable) (MW)",
            value=50000,
            min_value=0,
            step=100,
            help="Maximum nuclear capacity for optimization"
        )
        
        nuclear_capacity = None  # Not used when extendable
    else:
        nuclear_capacity = st.number_input(
            "Fixed Capacity (MW)",
            value=24000,
            min_value=0,
            step=100,
            help="Fixed nuclear capacity (not optimized)"
        )
        
        nuclear_p_nom_min = None  # Not used when not extendable
        nuclear_p_nom_max = None  # Not used when not extendable

    nuclear_p_min_pu = st.number_input(
        "Minimum Power Output Limit",
        value=0.8,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        help="Minimum nuclear power output as fraction of capacity (0-1)"
    )

    nuclear_p_max_pu = st.number_input(
        "Maximum Power Output Limit",
        value=1.0,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        help="Maximum nuclear power output as fraction of capacity (0-1)"
    )

    nuclear_cost = st.number_input(
        "Nuclear Capital Cost ($/kW)",
        value=6000,
        min_value=0,
        step=500
    )
    
    nuclear_marginal_cost = st.number_input(
        "Nuclear Marginal Cost ($/MWh)",
        value=0.1,
        min_value=0.0,
        step=0.1,
        help="Operational cost per MWh of nuclear generation"
    )

    # Hydrogen parameters
    st.subheader("🔋 Hydrogen Generation")
    
    hydrogen_extendable = st.checkbox("Hydrogen Extendable", value=False)
    
    if hydrogen_extendable:
        hydrogen_p_nom_min = st.number_input(
            "Minimum Capacity (when extendable) (MW)",
            value=0,
            min_value=0,
            step=500,
            help="Minimum hydrogen capacity for optimization"
        )
        
        hydrogen_p_nom_max = st.number_input(
            "Maximum Capacity (when extendable) (MW)",
            value=50000,
            min_value=0,
            step=500,
            help="Maximum hydrogen capacity for optimization"
        )
        
        hydrogen_capacity = None  # Not used when extendable
    else:
        hydrogen_capacity = st.number_input(
            "Fixed Capacity (MW)",
            value=0,
            min_value=0,
            step=500,
            help="Fixed hydrogen capacity (0 = disabled, unlimited fuel source)"
        )
        
        hydrogen_p_nom_min = None  # Not used when not extendable
        hydrogen_p_nom_max = None  # Not used when not extendable

    hydrogen_p_min_pu = st.number_input(
        "Minimum Power Output Limit",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        help="Minimum hydrogen power output as fraction of capacity (0-1)"
    )

    hydrogen_p_max_pu = st.number_input(
        "Maximum Power Output Limit",
        value=1.0,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        help="Maximum hydrogen power output as fraction of capacity (0-1)"
    )

    hydrogen_marginal_cost = st.number_input(
        "Hydrogen Marginal Cost ($/MWh)",
        value=50.0,
        min_value=0.0,
        step=5.0,
        help="Fuel cost for hydrogen generation (higher cost = less preferred)"
    )

    hydrogen_cost = st.number_input(
        "Hydrogen Capital Cost ($/kW)",
        value=1200,
        min_value=0,
        step=100
    )

    # Storage parameters
    st.subheader("🔋 Energy Storage")
    
    storage_extendable = st.checkbox("Storage Extendable", value=False)
    
    if storage_extendable:
        storage_p_nom_min = st.number_input(
            "Minimum Capacity (when extendable) (MW)",
            value=10000,
            min_value=0,
            step=1000,
            help="Minimum storage power capacity for optimization"
        )
        
        storage_p_nom_max = st.number_input(
            "Maximum Capacity (when extendable) (MW)",
            value=100000,
            min_value=0,
            step=1000,
            help="Maximum storage power capacity for optimization"
        )
        
        storage_capacity = None  # Not used when extendable
    else:
        storage_capacity = st.number_input(
            "Fixed Capacity (MW)",
            value=10000,
            min_value=0,
            step=1000,
            help="Fixed storage power capacity (not optimized)"
        )
        
        storage_p_nom_min = None  # Not used when not extendable
        storage_p_nom_max = None  # Not used when not extendable

    storage_hours = st.number_input(
        "Storage Hours",
        value=6.0,
        min_value=0.0,
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
        min_value=0,
        step=100
    )
    
    storage_marginal_cost = st.number_input(
        "Storage Marginal Cost ($/MWh)",
        value=0.1,
        min_value=0.0,
        step=0.1,
        help="Operational cost per MWh of storage operation (charge/discharge)"
    )

    storage_initial_soc = st.number_input(
        "Initial State of Charge",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Initial state of charge as fraction of maximum energy capacity (0-1)"
    )


    st.markdown("---")
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        with st.spinner("Running optimization... This may take a few minutes."):
            try:
                # Prepare parameters, only pass non-None values
                optimization_params = {
                    'annual_load_twh': annual_load,
                    'solar_capacity_mw': solar_capacity,
                    'wind_capacity_mw': wind_capacity,
                    'nuclear_capacity_mw': nuclear_capacity,
                    'hydrogen_capacity_mw': hydrogen_capacity,
                    'storage_power_capacity_mw': storage_capacity,
                    'storage_max_hours': storage_hours,
                    'nuclear_p_min_pu': nuclear_p_min_pu,
                    'nuclear_p_max_pu': nuclear_p_max_pu,
                    'hydrogen_p_min_pu': hydrogen_p_min_pu,
                    'hydrogen_p_max_pu': hydrogen_p_max_pu,
                    'hydrogen_marginal_cost': hydrogen_marginal_cost,
                    'storage_efficiency': storage_charge_efficiency * storage_discharge_efficiency,
                    'storage_charge_efficiency': storage_charge_efficiency,
                    'storage_discharge_efficiency': storage_discharge_efficiency,
                    'storage_initial_soc': storage_initial_soc,
                    'solar_extendable': solar_extendable,
                    'wind_extendable': wind_extendable,
                    'nuclear_extendable': nuclear_extendable,
                    'hydrogen_extendable': hydrogen_extendable,
                    'storage_extendable': storage_extendable,
                    'solar_capital_cost': solar_cost,
                    'wind_capital_cost': wind_cost,
                    'nuclear_capital_cost': nuclear_cost,
                    'hydrogen_capital_cost': hydrogen_cost,
                    'storage_capital_cost': storage_cost,
                    'solar_marginal_cost': solar_marginal_cost,
                    'wind_marginal_cost': wind_marginal_cost,
                    'nuclear_marginal_cost': nuclear_marginal_cost,
                    'storage_marginal_cost': storage_marginal_cost,
                    'solver': solver,
                    'verbose': False,
                    'create_plots': False,
                }
                
                # Add individual p_nom_min/max parameters if they are not None
                if solar_p_nom_min is not None:
                    optimization_params['solar_p_nom_min'] = solar_p_nom_min
                if solar_p_nom_max is not None:
                    optimization_params['solar_p_nom_max'] = solar_p_nom_max
                if wind_p_nom_min is not None:
                    optimization_params['wind_p_nom_min'] = wind_p_nom_min
                if wind_p_nom_max is not None:
                    optimization_params['wind_p_nom_max'] = wind_p_nom_max
                if nuclear_p_nom_min is not None:
                    optimization_params['nuclear_p_nom_min'] = nuclear_p_nom_min
                if nuclear_p_nom_max is not None:
                    optimization_params['nuclear_p_nom_max'] = nuclear_p_nom_max
                if hydrogen_p_nom_min is not None:
                    optimization_params['hydrogen_p_nom_min'] = hydrogen_p_nom_min
                if hydrogen_p_nom_max is not None:
                    optimization_params['hydrogen_p_nom_max'] = hydrogen_p_nom_max
                if storage_p_nom_min is not None:
                    optimization_params['storage_p_nom_min'] = storage_p_nom_min
                if storage_p_nom_max is not None:
                    optimization_params['storage_p_nom_max'] = storage_p_nom_max
                
                optimization_result = run_energy_system_optimization(**optimization_params)

                if optimization_result['status'] == 'success':
                    st.session_state.optimization_result = optimization_result
                    st.session_state.model = optimization_result['network']
                    st.session_state.optimization_success = True
                    st.success("✅ Optimization completed successfully!")
                elif optimization_result['status'] == 'optimization_failed':
                    st.error("❌ Optimization failed to converge. Try adjusting parameters or using a different solver.")
                    st.session_state.optimization_success = False
                else:
                    st.error(f"❌ Error: {optimization_result.get('message', 'Unknown error occurred')}")
                    st.session_state.optimization_success = False

            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                st.session_state.optimization_success = False

with right_col:
    # Output area
    if 'optimization_success' in st.session_state and st.session_state.optimization_success:
        model = st.session_state.model
        optimization_result = st.session_state.get('optimization_result', {})

        # Results overview
        st.header("📊 Optimization Results")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        results = optimization_result.get('results', {})
        model_summary = optimization_result.get('model_summary', {})

        with col1:
            objective_value = results.get('objective_value', 0)
            st.metric("Total Cost", f"${objective_value/1e6:.1f}M")

        with col2:
            total_gen_capacity = sum(
                gen_info['capacity_mw']
                for gen_info in results.get('generators', {}).values()
            )
            st.metric("Total Generation", f"{total_gen_capacity/1000:.1f} GW")

        with col3:
            storage_info = results.get('storage_units', {}).get('storage', {})
            storage_power = storage_info.get('power_capacity_mw', 0)
            st.metric("Storage Power", f"{storage_power/1000:.1f} GW")

        with col4:
            num_snapshots = model_summary.get('snapshots', 0)
            st.metric("Time Snapshots", f"{num_snapshots:,}")


        # Output breakdown
        st.subheader("📊 Output")

        output_data = []

        # Add generator results
        generator_order = ['nuclear', 'solar', 'wind', 'hydrogen']
        for gen in generator_order:
            gen_info = results.get('generators', {}).get(gen, {})
            if gen_info:
                output_data.append({
                    'Technology': gen.title(),
                    'Installed Capacity (MW)': gen_info.get('capacity_mw', 0),
                    'Generation (GWh)': gen_info.get('generation_gwh', 0),
                    'Capacity Factor (%)': gen_info.get('capacity_factor_pct', 0)
                })

        # Add storage results
        for storage_name, storage_info in results.get('storage_units', {}).items():
            display_name = "ESS" if storage_name == "storage" else storage_name.title()

            output_data.append({
                'Technology': f'{display_name} (Power)',
                'Installed Capacity (MW)': storage_info.get('power_capacity_mw', 0),
                'Generation (GWh)': storage_info.get('discharge_gwh', 0),
                'Capacity Factor (%)': storage_info.get('capacity_factor_pct', 0)
            })

            output_data.append({
                'Technology': f'{display_name} (Energy)',
                'Installed Capacity (MW)': storage_info.get('energy_capacity_mwh', 0),
                'Generation (GWh)': storage_info.get('discharge_gwh', 0),
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

        # Generation mix chart
        st.subheader("🥧 Generation Mix")

        try:
            mix_chart = create_generation_mix_chart(model)
            if mix_chart:
                st.plotly_chart(mix_chart, use_container_width=True)
            else:
                st.info("Generation mix chart not available")
        except Exception as e:
            st.warning(f"Could not create generation mix chart: {e}")

        # Interactive plots
        st.header("📈 System Operation Analysis")

        # Create plots
        snapshots = model.snapshots

        gen_data = {}
        gen_available = {}
        generator_order = ['nuclear', 'solar', 'wind', 'hydrogen']
        for gen in generator_order:
            if gen in model.generators.index:
                try:
                    if hasattr(model, 'generators_t') and hasattr(model.generators_t, 'p') and gen in model.generators_t.p.columns:
                        actual_generation = model.generators_t.p.loc[:, gen].values
                        gen_data[gen] = actual_generation

                        if model.generators.loc[gen, 'p_nom_extendable']:
                            capacity = model.generators.loc[gen, 'p_nom_opt']
                        else:
                            capacity = model.generators.loc[gen, 'p_nom']

                        if hasattr(model.generators_t, 'p_max_pu') and gen in model.generators_t.p_max_pu.columns:
                            p_max_pu = model.generators_t.p_max_pu.loc[:, gen].values
                        else:
                            p_max_pu = np.ones(len(snapshots)) * model.generators.loc[gen, 'p_max_pu']

                        available_generation = capacity * p_max_pu
                        gen_available[gen] = available_generation
                    else:
                        gen_data[gen] = np.zeros(len(snapshots))
                        gen_available[gen] = np.zeros(len(snapshots))

                except Exception as e:
                    st.warning(f"Could not process {gen} generation data: {e}")
                    gen_data[gen] = np.zeros(len(snapshots))
                    gen_available[gen] = np.zeros(len(snapshots))

        if 'storage' in model.storage_units.index:
            try:
                if (hasattr(model, 'storage_units_t') and
                    hasattr(model.storage_units_t, 'p_store') and
                    'storage' in model.storage_units_t.p_store.columns):
                    p_store_raw = model.storage_units_t.p_store.loc[:, 'storage'].values
                    p_dispatch_raw = model.storage_units_t.p_dispatch.loc[:, 'storage'].values

                    threshold = 0.0001
                    simultaneous = (p_store_raw > threshold) & (p_dispatch_raw > threshold)

                    if simultaneous.any():
                        count = simultaneous.sum()
                        st.warning(f"⚠️ Found {count} periods with simultaneous charge/discharge")

                        p_store_fixed = p_store_raw.copy()
                        p_dispatch_fixed = p_dispatch_raw.copy()

                        for i, is_simultaneous in enumerate(simultaneous):
                            if is_simultaneous:
                                if p_store_raw[i] > p_dispatch_raw[i]:
                                    p_dispatch_fixed[i] = 0.0
                                else:
                                    p_store_fixed[i] = 0.0

                        storage_charge = -p_store_fixed
                        storage_discharge = p_dispatch_fixed
                    else:
                        storage_charge = -p_store_raw
                        storage_discharge = p_dispatch_raw
                        st.success("✅ No simultaneous charge/discharge detected")

                    storage_soc = model.storage_units_t.state_of_charge.loc[:, 'storage'].values
                else:
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

        # Colors for different technologies (nuclear, solar, wind, hydrogen, ESS)
        colors = {'nuclear': 'rgba(255, 107, 107, 0.7)', 'solar': '#FFA500', 'wind': '#87CEEB', 'hydrogen': '#9370DB', 'storage': '#32CD32'}
        curtailed_colors = {'solar': 'rgba(255, 220, 150, 0.6)', 'wind': 'rgba(173, 216, 230, 0.6)'}  # Much lighter colors for curtailed

        # Plot 1: Generation by source with demand overlay

        # Nuclear generation
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

        # Solar generation
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

        # Wind generation
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

        # Hydrogen generation
        if 'hydrogen' in gen_data:
            fig.add_trace(
                go.Scatter(
                    x=snapshots,
                    y=gen_data['hydrogen'],
                    name='Hydrogen Generation',
                    line=dict(color=colors['hydrogen']),
                    stackgroup='generation',
                    hovertemplate='Hydrogen: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )

        # ESS discharge
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

        # ESS charge
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
