"""
Results analysis and visualization functions for PyPSA energy system model.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def print_results(network):
    """
    Print optimization results to console.

    Args:
        network: PyPSA network object with optimization results
    """
    if not hasattr(network, 'objective') or network.objective is None:
        print("No optimization results available.")
        return

    print(f"\nOptimization Results:")
    print(f"Objective Value: {network.objective:.2f}")
    print(f"\nOutput:")
    print(f"{'Technology':<15} {'Capacity (MW)':<15} {'Generation (GWh)':<18} {'Capacity Factor (%)':<18}")
    print("-" * 70)

    generator_order = ['nuclear', 'solar', 'wind', 'hydrogen']
    for gen in generator_order:
        if gen in network.generators.index:
            if network.generators.loc[gen, 'p_nom_extendable']:
                capacity = network.generators.loc[gen, 'p_nom_opt']
            else:
                capacity = network.generators.loc[gen, 'p_nom']

            actual_generation = network.generators_t.p.loc[:, gen].sum()
            generation_gwh = actual_generation / 1000

            if capacity > 0:
                max_possible_generation = capacity * len(network.snapshots)
                capacity_factor = (actual_generation / max_possible_generation) * 100
            else:
                capacity_factor = 0

            print(f"{gen.title():<15} {capacity:<15.1f} {generation_gwh:<18.2f} {capacity_factor:<18.1f}")

    for storage in network.storage_units.index:
        if network.storage_units.loc[storage, 'p_nom_extendable']:
            power_capacity = network.storage_units.loc[storage, 'p_nom_opt']
        else:
            power_capacity = network.storage_units.loc[storage, 'p_nom']

        max_hours = network.storage_units.loc[storage, 'max_hours']
        energy_capacity = power_capacity * max_hours

        storage_discharge = network.storage_units_t.p_dispatch.loc[:, storage].sum()
        discharge_gwh = storage_discharge / 1000

        if power_capacity > 0:
            max_possible_discharge = power_capacity * len(network.snapshots)
            storage_capacity_factor = (storage_discharge / max_possible_discharge) * 100
        else:
            storage_capacity_factor = 0

        storage_name = "ESS" if storage == "storage" else storage
        print(f"{storage_name + ' (power)':<15} {power_capacity:<15.1f} {discharge_gwh:<18.2f} {storage_capacity_factor:<18.1f}")
        print(f"{storage_name + ' (energy)':<15} {energy_capacity:<15.1f} {discharge_gwh:<18.2f} {'N/A':<18}")


def get_results_summary(network):
    """
    Get optimization results as structured data.

    Args:
        network: PyPSA network object with optimization results

    Returns:
        dict: Results summary
    """
    if not hasattr(network, 'objective') or network.objective is None:
        return {'status': 'no_results', 'message': 'No optimization results available'}

    results = {
        'status': 'success',
        'objective_value': network.objective,
        'generators': {},
        'storage_units': {},
        'energy_balance': {},
        'costs': {}
    }

    try:
        # Generator results
        for gen in network.generators.index:
            if network.generators.loc[gen, 'p_nom_extendable']:
                capacity = network.generators.loc[gen, 'p_nom_opt']
            else:
                capacity = network.generators.loc[gen, 'p_nom']

            actual_generation = network.generators_t.p[gen].sum()  # MWh
            generation_gwh = actual_generation / 1000  # GWh

            if capacity > 0:
                max_possible_generation = capacity * len(network.snapshots)
                capacity_factor = (actual_generation / max_possible_generation) * 100
            else:
                capacity_factor = 0

            results['generators'][gen] = {
                'capacity_mw': capacity,
                'generation_gwh': generation_gwh,
                'capacity_factor_pct': capacity_factor,
                'capital_cost': network.generators.loc[gen, 'capital_cost'],
                'marginal_cost': network.generators.loc[gen, 'marginal_cost']
            }

        # Storage results
        for storage in network.storage_units.index:
            if network.storage_units.loc[storage, 'p_nom_extendable']:
                power_capacity = network.storage_units.loc[storage, 'p_nom_opt']
            else:
                power_capacity = network.storage_units.loc[storage, 'p_nom']

            max_hours = network.storage_units.loc[storage, 'max_hours']
            energy_capacity = power_capacity * max_hours

            storage_discharge = network.storage_units_t.p_dispatch[storage].sum()
            storage_charge = network.storage_units_t.p_store[storage].sum()
            discharge_gwh = storage_discharge / 1000
            charge_gwh = storage_charge / 1000

            if power_capacity > 0:
                max_possible_discharge = power_capacity * len(network.snapshots)
                storage_capacity_factor = (storage_discharge / max_possible_discharge) * 100
            else:
                storage_capacity_factor = 0

            results['storage_units'][storage] = {
                'power_capacity_mw': power_capacity,
                'energy_capacity_mwh': energy_capacity,
                'discharge_gwh': discharge_gwh,
                'charge_gwh': charge_gwh,
                'capacity_factor_pct': storage_capacity_factor,
                'round_trip_efficiency': storage_discharge / storage_charge if storage_charge > 0 else 0
            }

        # Energy balance
        total_generation = sum(results['generators'][gen]['generation_gwh'] for gen in results['generators'])
        total_storage_discharge = sum(results['storage_units'][storage]['discharge_gwh'] for storage in results['storage_units'])
        total_load = sum(network.loads_t.p_set[load].sum() for load in network.loads.index) / 1000  # GWh
        total_storage_charge = sum(results['storage_units'][storage]['charge_gwh'] for storage in results['storage_units'])

        results['energy_balance'] = {
            'total_generation_gwh': total_generation,
            'total_storage_discharge_gwh': total_storage_discharge,
            'total_supply_gwh': total_generation + total_storage_discharge,
            'total_load_gwh': total_load,
            'total_storage_charge_gwh': total_storage_charge,
            'total_demand_gwh': total_load + total_storage_charge,
            'balance_error_gwh': (total_generation + total_storage_discharge) - (total_load + total_storage_charge)
        }

        return results

    except Exception as e:
        return {'status': 'error', 'message': f"Error processing results: {e}"}


def create_interactive_plots(network, save_to_file=True, filename="energy_system_analysis.html"):
    """
    Create interactive plots showing:
    1. Hourly generation by source with demand overlay
    2. Storage charge/discharge patterns
    3. Storage state of charge

    Args:
        network: PyPSA network object with optimization results
        save_to_file (bool): Whether to save plot to HTML file
        filename (str): Output filename for HTML plot

    Returns:
        plotly.graph_objects.Figure: Interactive plot figure
    """
    if not hasattr(network, 'objective') or network.objective is None:
        print("No optimization results available for plotting.")
        return None

    snapshots = network.snapshots

    gen_data = {}
    gen_available = {}
    generator_order = ['nuclear', 'solar', 'wind', 'hydrogen']
    for gen in generator_order:
        if gen in network.generators.index:
            actual_generation = network.generators_t.p.loc[:, gen].values
            gen_data[gen] = actual_generation
            if network.generators.loc[gen, 'p_nom_extendable']:
                capacity = network.generators.loc[gen, 'p_nom_opt']
            else:
                capacity = network.generators.loc[gen, 'p_nom']

            if hasattr(network.generators_t, 'p_max_pu') and gen in network.generators_t.p_max_pu.columns:
                p_max_pu = network.generators_t.p_max_pu.loc[:, gen].values
            else:
                p_max_pu = np.ones(len(snapshots)) * network.generators.loc[gen, 'p_max_pu']

            available_generation = capacity * p_max_pu
            gen_available[gen] = available_generation

    storage_charge = -network.storage_units_t.p_store.loc[:, 'storage'].values
    storage_discharge = network.storage_units_t.p_dispatch.loc[:, 'storage'].values
    storage_soc = network.storage_units_t.state_of_charge.loc[:, 'storage'].values

    load_data = network.loads_t.p_set.loc[:, 'load'].values

    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Hourly Generation vs Demand',
            'ESS Charge/Discharge',
            'ESS State of Charge'
        ),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]],
        vertical_spacing=0.1
    )

    colors = {'nuclear': 'rgba(255, 107, 107, 0.7)', 'solar': '#FFA500', 'wind': '#87CEEB', 'hydrogen': '#9370DB', 'storage': '#32CD32'}
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

    ess_discharge = network.storage_units_t.p_dispatch.loc[:, 'storage'].values
    fig.add_trace(
        go.Scatter(
            x=snapshots,
            y=ess_discharge,
            name='ESS Discharging',
            line=dict(color='red'),
            stackgroup='generation',
            hovertemplate='ESS Discharging: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
        ),
        row=1, col=1
    )

    ess_charge = -network.storage_units_t.p_store.loc[:, 'storage'].values
    fig.add_trace(
        go.Scatter(
            x=snapshots,
            y=ess_charge,
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

    fig.add_trace(
        go.Scatter(
            x=snapshots,
            y=storage_charge,
            name='ESS Charging',
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
            name='ESS Discharging',
            line=dict(color='red'),
            fill='tozeroy',
            hovertemplate='Discharging: %{y:.0f} MW<br>Time: %{x}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=snapshots,
            y=storage_soc,
            name='ESS State of Charge',
            line=dict(color='blue'),
            fill='tozeroy',
            hovertemplate='SoC: %{y:.0f} MWh<br>Time: %{x}<extra></extra>'
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=1200,
        title_text="Energy System Operation Analysis",
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Energy (MWh)", row=3, col=1)

    fig.update_xaxes(title_text="Time", row=3, col=1)

    fig.update_xaxes(matches='x', row=1, col=1)
    fig.update_xaxes(matches='x', row=2, col=1)
    fig.update_xaxes(matches='x', row=3, col=1)

    if save_to_file:
        fig.write_html(filename)
        print(f"Interactive plot saved to: {filename}")
        print("Open this file in a web browser to view the interactive plots.")

    return fig


def get_generation_mix_data(network):
    """
    Get generation mix data for visualization.

    Args:
        network: PyPSA network object with optimization results

    Returns:
        pd.DataFrame: Generation mix data
    """
    if not hasattr(network, 'objective') or network.objective is None:
        return pd.DataFrame()

    data = []

    for gen in network.generators.index:
        if network.generators.loc[gen, 'p_nom_extendable']:
            capacity = network.generators.loc[gen, 'p_nom_opt']
        else:
            capacity = network.generators.loc[gen, 'p_nom']

        actual_generation = network.generators_t.p[gen].sum() / 1000  # GWh

        data.append({
            'Technology': gen.title(),
            'Capacity_MW': capacity,
            'Generation_GWh': actual_generation,
            'Carrier': network.generators.loc[gen, 'carrier']
        })

    return pd.DataFrame(data)


def create_generation_mix_chart(network):
    """
    Create a pie chart showing generation mix.

    Args:
        network: PyPSA network object with optimization results

    Returns:
        plotly.graph_objects.Figure: Pie chart figure
    """
    df = get_generation_mix_data(network)

    if df.empty:
        return None

    fig = px.pie(df, values='Generation_GWh', names='Technology',
                 title='Generation Mix (Annual Energy)',
                 color_discrete_map={
                     'Nuclear': '#FF6B6B',
                     'Solar': '#FFA500',
                     'Wind': '#87CEEB',
                     'Hydrogen': '#9370DB'
                 })

    fig.update_traces(textposition='inside', textinfo='percent+label')

    return fig
