"""
Model building functions for PyPSA energy system model.
"""

import numpy as np
import pandas as pd
import pypsa

from .data_loader import (load_and_process_load_data_aligned,
                          load_generation_data)


def create_single_bus_model(
    solar_capacity_mw=None,
    wind_capacity_mw=None,
    nuclear_capacity_mw=24000,
    nuclear_p_min_pu=0.8,
    nuclear_p_max_pu=1.0,
    hydrogen_capacity_mw=0,
    hydrogen_p_min_pu=0.0,
    hydrogen_p_max_pu=1.0,
    hydrogen_marginal_cost=50,
    annual_load_twh=50,
    storage_power_capacity_mw=150,
    storage_max_hours=8,
    storage_efficiency=0.85,
    storage_charge_efficiency=0.95,
    storage_discharge_efficiency=0.95,
    storage_initial_soc=0.5,
    # Individual capacity bounds (p_nom_min/max for extendable generators)
    solar_p_nom_min=None,
    solar_p_nom_max=None,
    wind_p_nom_min=None,
    wind_p_nom_max=None,
    nuclear_p_nom_min=None,
    nuclear_p_nom_max=None,
    hydrogen_p_nom_min=None,
    hydrogen_p_nom_max=None,
    storage_p_nom_min=None,
    storage_p_nom_max=None,
    solar_extendable=True,
    wind_extendable=True,
    nuclear_extendable=False,
    hydrogen_extendable=False,
    storage_extendable=True,
    solar_capital_cost=1000,
    wind_capital_cost=1500,
    nuclear_capital_cost=6000,
    hydrogen_capital_cost=1200,
    storage_capital_cost=500,
    solar_marginal_cost=0.1,
    wind_marginal_cost=0.1,
    nuclear_marginal_cost=0.1,
    storage_marginal_cost=0.1
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
    nuclear_p_min_pu : float
        Minimum nuclear power output as fraction of capacity (0-1)
    nuclear_p_max_pu : float
        Maximum nuclear power output as fraction of capacity (0-1)
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

    Returns:
    --------
    pypsa.Network: Configured PyPSA network model
    """

    network = pypsa.Network()
    gen_data, solar_absolute, wind_absolute = load_generation_data()
    network.set_snapshots(gen_data.index)

    network.add("Bus", "bus", v_nom=230, carrier="electricity")

    network.add("Carrier", "solar", color="yellow")
    network.add("Carrier", "wind", color="blue")
    network.add("Carrier", "nuclear", color="red")
    network.add("Carrier", "hydrogen", color="purple")
    network.add("Carrier", "electricity")

    solar_max_capacity = solar_absolute.max() if solar_absolute.max() > 0 else 1
    wind_max_capacity = wind_absolute.max() if wind_absolute.max() > 0 else 1

    solar_capacity_factor = solar_absolute / solar_max_capacity
    wind_capacity_factor = wind_absolute / wind_max_capacity

    actual_solar_capacity = solar_capacity_mw if solar_capacity_mw is not None else solar_max_capacity
    actual_solar_capacity = actual_solar_capacity if actual_solar_capacity is not None else 0  # Ensure not None

    solar_params = {
        "name": "solar",
        "bus": "bus",
        "carrier": "solar",
        "p_nom_extendable": solar_extendable,
        "marginal_cost": solar_marginal_cost,
        "capital_cost": solar_capital_cost,
        "p_max_pu": solar_capacity_factor
    }

    if solar_extendable:
        # Use individual bounds if provided, otherwise use capacity as min and fallback max
        p_nom_min = solar_p_nom_min if solar_p_nom_min is not None else actual_solar_capacity
        # For default max: if capacity is 0, use a reasonable default, otherwise use capacity * 10
        if solar_p_nom_max is not None:
            p_nom_max = solar_p_nom_max
        elif actual_solar_capacity and actual_solar_capacity > 0:
            p_nom_max = actual_solar_capacity * 10
        else:
            p_nom_max = 1000000  # Default max when starting from 0 capacity
        
        solar_params["p_nom_min"] = p_nom_min
        solar_params["p_nom_max"] = p_nom_max
        solar_params["p_nom"] = p_nom_min  # Starting point at minimum
    else:
        solar_params["p_nom"] = actual_solar_capacity

    # Only add solar if it has capacity > 0 or is extendable
    if actual_solar_capacity > 0 or solar_extendable:
        network.add("Generator", **solar_params)
    else:
        print("Skipping solar generator (0 capacity and not extendable)")

    actual_wind_capacity = wind_capacity_mw if wind_capacity_mw is not None else wind_max_capacity
    actual_wind_capacity = actual_wind_capacity if actual_wind_capacity is not None else 0  # Ensure not None

    wind_params = {
        "name": "wind",
        "bus": "bus",
        "carrier": "wind",
        "p_nom_extendable": wind_extendable,
        "marginal_cost": wind_marginal_cost,
        "capital_cost": wind_capital_cost,
        "p_max_pu": wind_capacity_factor
    }

    if wind_extendable:
        # Use individual bounds if provided, otherwise use capacity as min and fallback max
        p_nom_min = wind_p_nom_min if wind_p_nom_min is not None else actual_wind_capacity
        # For default max: if capacity is 0, use a reasonable default, otherwise use capacity * 10
        if wind_p_nom_max is not None:
            p_nom_max = wind_p_nom_max
        elif actual_wind_capacity and actual_wind_capacity > 0:
            p_nom_max = actual_wind_capacity * 10
        else:
            p_nom_max = 500000  # Default max when starting from 0 capacity
        
        wind_params["p_nom_min"] = p_nom_min
        wind_params["p_nom_max"] = p_nom_max
        wind_params["p_nom"] = p_nom_min  # Starting point at minimum
    else:
        wind_params["p_nom"] = actual_wind_capacity

    # Only add wind if it has capacity > 0 or is extendable
    if actual_wind_capacity > 0 or wind_extendable:
        network.add("Generator", **wind_params)
    else:
        print("Skipping wind generator (0 capacity and not extendable)")

    # Ensure nuclear capacity is not None for safe comparisons
    nuclear_capacity_safe = nuclear_capacity_mw if nuclear_capacity_mw is not None else 0
    
    nuclear_params = {
        "name": "nuclear",
        "bus": "bus",
        "carrier": "nuclear",
        "p_min_pu": nuclear_p_min_pu,
        "p_max_pu": nuclear_p_max_pu,
        "p_nom_extendable": nuclear_extendable,
        "marginal_cost": nuclear_marginal_cost,
        "capital_cost": nuclear_capital_cost
    }

    if nuclear_extendable:
        # Use individual bounds if provided, otherwise use capacity as min and fallback max
        p_nom_min = nuclear_p_nom_min if nuclear_p_nom_min is not None else nuclear_capacity_safe
        # For default max: if capacity is 0, use a reasonable default, otherwise use capacity * 10
        if nuclear_p_nom_max is not None:
            p_nom_max = nuclear_p_nom_max
        elif nuclear_capacity_mw and nuclear_capacity_mw > 0:
            p_nom_max = nuclear_capacity_mw * 10
        else:
            p_nom_max = 100000  # Default max when starting from 0 capacity
        
        nuclear_params["p_nom_min"] = p_nom_min
        nuclear_params["p_nom_max"] = p_nom_max
        nuclear_params["p_nom"] = p_nom_min  # Starting point at minimum
    else:
        nuclear_params["p_nom"] = nuclear_capacity_safe

    # Only add nuclear if it has capacity > 0 or is extendable
    if nuclear_capacity_safe > 0 or nuclear_extendable:
        network.add("Generator", **nuclear_params)
    else:
        print("Skipping nuclear generator (0 capacity and not extendable)")

    # Add hydrogen generator if capacity > 0 OR if it's extendable (can start from 0 and expand)
    hydrogen_capacity_safe = hydrogen_capacity_mw if hydrogen_capacity_mw is not None else 0
    if hydrogen_capacity_safe > 0 or hydrogen_extendable:
        hydrogen_params = {
            "name": "hydrogen",
            "bus": "bus",
            "carrier": "hydrogen",
            "p_min_pu": hydrogen_p_min_pu,
            "p_max_pu": hydrogen_p_max_pu,
            "p_nom_extendable": hydrogen_extendable,
            "marginal_cost": hydrogen_marginal_cost,  # Fuel cost ($/MWh)
            "capital_cost": hydrogen_capital_cost
        }

        if hydrogen_extendable:
            # Use individual bounds if provided, otherwise use capacity as min and fallback max
            p_nom_min = hydrogen_p_nom_min if hydrogen_p_nom_min is not None else hydrogen_capacity_safe
            # For default max: if capacity is 0, use a reasonable default, otherwise use capacity * 10
            if hydrogen_p_nom_max is not None:
                p_nom_max = hydrogen_p_nom_max
            elif hydrogen_capacity_mw and hydrogen_capacity_mw > 0:
                p_nom_max = hydrogen_capacity_mw * 10
            else:
                p_nom_max = 50000  # Default max when starting from 0 capacity
            
            hydrogen_params["p_nom_min"] = p_nom_min
            hydrogen_params["p_nom_max"] = p_nom_max
            hydrogen_params["p_nom"] = p_nom_min  # Starting point at minimum
        else:
            hydrogen_params["p_nom"] = hydrogen_capacity_safe

        network.add("Generator", **hydrogen_params)
        print(f"Added hydrogen generator: {hydrogen_capacity_mw} MW, marginal cost: ${hydrogen_marginal_cost}/MWh")

    network.add("Carrier", "storage")

    # Ensure storage capacity is not None for safe comparisons
    storage_capacity_safe = storage_power_capacity_mw if storage_power_capacity_mw is not None else 0

    storage_params = {
        "name": "storage",
        "bus": "bus",
        "carrier": "storage",
        "p_nom_extendable": storage_extendable,
        "max_hours": storage_max_hours,
        "efficiency_store": storage_charge_efficiency,
        "efficiency_dispatch": storage_discharge_efficiency,
        "marginal_cost": storage_marginal_cost,
        "capital_cost": storage_capital_cost,
        "cyclic_state_of_charge": True,
        "state_of_charge_initial": storage_initial_soc  # Set initial SOC
    }

    if storage_extendable:
        # Use individual bounds if provided, otherwise use capacity as min and fallback max
        p_nom_min = storage_p_nom_min if storage_p_nom_min is not None else storage_capacity_safe
        # For default max: if capacity is 0, use a reasonable default, otherwise use capacity * 10
        if storage_p_nom_max is not None:
            p_nom_max = storage_p_nom_max
        elif storage_power_capacity_mw and storage_power_capacity_mw > 0:
            p_nom_max = storage_power_capacity_mw * 10
        else:
            p_nom_max = 200000  # Default max when starting from 0 capacity
        
        storage_params["p_nom_min"] = p_nom_min
        storage_params["p_nom_max"] = p_nom_max
        storage_params["p_nom"] = p_nom_min  # Starting point at minimum
    else:
        storage_params["p_nom"] = storage_capacity_safe

    # Only add storage if it has capacity > 0 or is extendable
    if storage_capacity_safe > 0 or storage_extendable:
        network.add("StorageUnit", **storage_params)
    else:
        print("Skipping storage unit (0 capacity and not extendable)")

    load_profile = load_and_process_load_data_aligned(annual_load_twh, network.snapshots)

    network.add("Load",
                "load",
                bus="bus",
                p_set=load_profile)

    return network


def get_model_summary(network):
    """
    Get a summary of the model configuration.

    Args:
        network: PyPSA network object

    Returns:
        dict: Model summary information
    """
    summary = {
        'snapshots': len(network.snapshots),
        'time_range': (network.snapshots[0], network.snapshots[-1]) if len(network.snapshots) > 0 else None,
        'buses': len(network.buses),
        'generators': {},
        'storage_units': {},
        'loads': len(network.loads),
        'total_load': 0
    }

    try:
        # Generator summary
        for gen in network.generators.index:
            gen_info = {
                'carrier': network.generators.loc[gen, 'carrier'],
                'capacity_mw': network.generators.loc[gen, 'p_nom'],
                'extendable': network.generators.loc[gen, 'p_nom_extendable'],
                'capital_cost': network.generators.loc[gen, 'capital_cost'],
                'marginal_cost': network.generators.loc[gen, 'marginal_cost']
            }
            summary['generators'][gen] = gen_info

        # Storage summary
        for storage in network.storage_units.index:
            storage_info = {
                'power_capacity_mw': network.storage_units.loc[storage, 'p_nom'],
                'max_hours': network.storage_units.loc[storage, 'max_hours'],
                'extendable': network.storage_units.loc[storage, 'p_nom_extendable'],
                'efficiency_store': network.storage_units.loc[storage, 'efficiency_store'],
                'efficiency_dispatch': network.storage_units.loc[storage, 'efficiency_dispatch']
            }
            storage_info['energy_capacity_mwh'] = storage_info['power_capacity_mw'] * storage_info['max_hours']
            summary['storage_units'][storage] = storage_info

        # Load summary
        for load in network.loads.index:
            if hasattr(network.loads_t, 'p_set') and load in network.loads_t.p_set.columns:
                load_total = network.loads_t.p_set[load].sum() / 1000  # Convert to GWh
                summary['total_load'] += load_total

        return summary

    except Exception as e:
        print(f"Error creating model summary: {e}")
        return summary
