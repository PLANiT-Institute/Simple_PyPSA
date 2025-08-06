"""
Custom constraints for PyPSA energy system model.
"""

import numpy as np


def add_custom_constraints(network, constraint_config=None):
    """
    Add custom constraints to the network model.

    Args:
        network: PyPSA network object
        constraint_config (dict): Configuration for constraints

    Returns:
        bool: True if constraints added successfully
    """
    if constraint_config is None:
        # No custom constraints to add
        return True

    try:
        print("Adding custom constraints...")

        # Future custom constraints can be added here
        # Currently no constraints are implemented

        print("✓ No custom constraints to configure")
        return True

    except Exception as e:
        print(f"Error adding custom constraints: {e}")
        return False


# Generation share constraints have been removed


def validate_network_constraints(network):
    """
    Validate that network satisfies all constraints after optimization.

    Args:
        network: PyPSA network object

    Returns:
        dict: Validation results
    """
    validation_results = {
        'energy_balance': False,
        'capacity_limits': False,
        'storage_operation': False,
        'custom_constraints': False
    }

    try:
        # Check energy balance
        if _check_energy_balance(network):
            validation_results['energy_balance'] = True

        # Check capacity limits
        if _check_capacity_limits(network):
            validation_results['capacity_limits'] = True

        # Check storage operation
        if _check_storage_constraints(network):
            validation_results['storage_operation'] = True

        # Check custom constraints (currently none implemented)
        validation_results['custom_constraints'] = True
        print("✓ No custom constraints to validate")

        return validation_results

    except Exception as e:
        print(f"Error validating constraints: {e}")
        return validation_results


def _check_energy_balance(network):
    """Check if energy supply equals demand across all timesteps."""
    try:
        # Get total generation
        total_generation = 0
        for gen in network.generators.index:
            total_generation += network.generators_t.p[gen].sum()

        # Add storage discharge
        for storage in network.storage_units.index:
            total_generation += network.storage_units_t.p_dispatch[storage].sum()

        # Get total demand
        total_demand = 0
        for load in network.loads.index:
            total_demand += network.loads_t.p_set[load].sum()

        # Add storage charging
        for storage in network.storage_units.index:
            total_demand += network.storage_units_t.p_store[storage].sum()

        # Check balance (allow small numerical tolerance)
        balance_error = abs(total_generation - total_demand)
        tolerance = 1e-6 * max(total_generation, total_demand)

        if balance_error < tolerance:
            print("✓ Energy balance validated")
            return True
        else:
            print(f"❌ Energy balance error: {balance_error:.2f} MWh")
            return False

    except Exception as e:
        print(f"Error checking energy balance: {e}")
        return False


def _check_capacity_limits(network):
    """Check if generation/storage stays within capacity limits."""
    try:
        violations = 0

        # Check generator capacity limits
        for gen in network.generators.index:
            capacity = network.generators.loc[gen, 'p_nom_opt'] if network.generators.loc[gen, 'p_nom_extendable'] else network.generators.loc[gen, 'p_nom']
            generation = network.generators_t.p[gen]

            if (generation > capacity * 1.001).any():  # Small tolerance for numerical precision
                violations += 1
                print(f"❌ Generator {gen} exceeds capacity limit")

        # Check storage capacity limits
        for storage in network.storage_units.index:
            power_cap = network.storage_units.loc[storage, 'p_nom_opt'] if network.storage_units.loc[storage, 'p_nom_extendable'] else network.storage_units.loc[storage, 'p_nom']

            p_store = network.storage_units_t.p_store[storage]
            p_dispatch = network.storage_units_t.p_dispatch[storage]

            if (p_store > power_cap * 1.001).any() or (p_dispatch > power_cap * 1.001).any():
                violations += 1
                print(f"❌ Storage {storage} exceeds power capacity limit")

        if violations == 0:
            print("✓ Capacity limits validated")
            return True
        else:
            print(f"❌ Found {violations} capacity limit violations")
            return False

    except Exception as e:
        print(f"Error checking capacity limits: {e}")
        return False


def _check_storage_constraints(network):
    """Check storage-specific constraints."""
    try:
        violations = 0

        for storage in network.storage_units.index:
            # Check energy capacity limits
            max_hours = network.storage_units.loc[storage, 'max_hours']
            power_cap = network.storage_units.loc[storage, 'p_nom_opt'] if network.storage_units.loc[storage, 'p_nom_extendable'] else network.storage_units.loc[storage, 'p_nom']
            energy_cap = power_cap * max_hours

            soc = network.storage_units_t.state_of_charge[storage]

            if (soc > energy_cap * 1.001).any() or (soc < -0.001).any():
                violations += 1
                print(f"❌ Storage {storage} violates energy capacity limits")

            # Check for simultaneous charge/discharge
            p_store = network.storage_units_t.p_store[storage]
            p_dispatch = network.storage_units_t.p_dispatch[storage]
            threshold = 0.001

            simultaneous = (p_store > threshold) & (p_dispatch > threshold)
            if simultaneous.any():
                violations += 1
                count = simultaneous.sum()
                print(f"❌ Storage {storage} has {count} periods with simultaneous charge/discharge")

        if violations == 0:
            print("✓ Storage constraints validated")
            return True
        else:
            print(f"❌ Found {violations} storage constraint violations")
            return False

    except Exception as e:
        print(f"Error checking storage constraints: {e}")
        return False
