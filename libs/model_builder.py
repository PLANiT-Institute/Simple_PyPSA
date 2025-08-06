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
    solar_extendable=True,
    wind_extendable=True,
    nuclear_extendable=False,
    hydrogen_extendable=False,
    storage_extendable=True,
    max_capacity_multiplier=2.0,
    solar_capital_cost=1000,
    wind_capital_cost=1500,
    nuclear_capital_cost=6000,
    hydrogen_capital_cost=1200,
    storage_capital_cost=500
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

    nuclear_params = {
        "name": "nuclear",
        "bus": "bus",
        "carrier": "nuclear",
        "p_min_pu": nuclear_p_min_pu,
        "p_max_pu": nuclear_p_max_pu,
        "p_nom_extendable": nuclear_extendable,
        "marginal_cost": 0,  # Low but non-zero marginal cost
        "capital_cost": nuclear_capital_cost
    }

    if nuclear_extendable:
        nuclear_params["p_nom_min"] = nuclear_capacity_mw
        nuclear_params["p_nom_max"] = nuclear_capacity_mw * max_capacity_multiplier
        nuclear_params["p_nom"] = nuclear_capacity_mw  # Starting point
    else:
        nuclear_params["p_nom"] = nuclear_capacity_mw

    network.add("Generator", **nuclear_params)

    if hydrogen_capacity_mw > 0:
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
            hydrogen_params["p_nom_min"] = hydrogen_capacity_mw
            hydrogen_params["p_nom_max"] = hydrogen_capacity_mw * max_capacity_multiplier
            hydrogen_params["p_nom"] = hydrogen_capacity_mw  # Starting point
        else:
            hydrogen_params["p_nom"] = hydrogen_capacity_mw

        network.add("Generator", **hydrogen_params)
        print(f"Added hydrogen generator: {hydrogen_capacity_mw} MW, marginal cost: ${hydrogen_marginal_cost}/MWh")

    network.add("Carrier", "storage")

    storage_params = {
        "name": "storage",
        "bus": "bus",
        "carrier": "storage",
        "p_nom_extendable": storage_extendable,
        "max_hours": storage_max_hours,
        "efficiency_store": storage_charge_efficiency,
        "efficiency_dispatch": storage_discharge_efficiency,
        "marginal_cost": 0,  # Higher cost to discourage unnecessary cycling
        "capital_cost": storage_capital_cost,
        "cyclic_state_of_charge": True,
        "state_of_charge_initial": storage_initial_soc  # Set initial SOC
    }

    if storage_extendable:
        storage_params["p_nom_min"] = storage_power_capacity_mw
        storage_params["p_nom_max"] = storage_power_capacity_mw * max_capacity_multiplier
        storage_params["p_nom"] = storage_power_capacity_mw  # Starting point
    else:
        storage_params["p_nom"] = storage_power_capacity_mw

    network.add("StorageUnit", **storage_params)

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
