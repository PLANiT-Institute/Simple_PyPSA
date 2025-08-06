"""
PyPSA Single Bus Energy System Model - Clean Backend

This module provides a clean backend interface for PyPSA energy system optimization.
It acts as a coordinator between the frontend (app.py) and the modular libraries.
"""

from libs.constraints import (add_custom_constraints,
                              validate_network_constraints)
# Import the modular libraries
from libs.data_loader import get_data_capacities
from libs.model_builder import create_single_bus_model, get_model_summary
from libs.optimization import run_model_optimization
from libs.results import (create_interactive_plots, get_results_summary,
                          print_results)


def run_energy_system_optimization(
    # Model configuration parameters
    annual_load_twh=700,
    solar_capacity_mw=200000,
    wind_capacity_mw=100000,
    nuclear_capacity_mw=24000,
    hydrogen_capacity_mw=0,
    storage_power_capacity_mw=10000,
    storage_max_hours=6,
    # Operational parameters
    nuclear_p_min_pu=0.8,
    nuclear_p_max_pu=1.0,
    hydrogen_p_min_pu=0.0,
    hydrogen_p_max_pu=1.0,
    hydrogen_marginal_cost=50,
    storage_efficiency=0.85,
    storage_charge_efficiency=0.95,
    storage_discharge_efficiency=0.95,
    storage_initial_soc=0.5,
    # Extendable options
    solar_extendable=True,
    wind_extendable=True,
    nuclear_extendable=False,
    hydrogen_extendable=False,
    storage_extendable=True,
    max_capacity_multiplier=20.0,
    # Cost parameters ($/kW)
    solar_capital_cost=1000,
    wind_capital_cost=1500,
    nuclear_capital_cost=6000,
    hydrogen_capital_cost=1200,
    storage_capital_cost=500,
    # Optimization settings
    solver='highs',
    custom_constraints=None,
    # Output settings
    verbose=True,
    create_plots=True,
    plot_filename="energy_system_analysis.html"
):
    """
    Run complete energy system optimization workflow.

    This function coordinates the entire optimization process:
    1. Data loading and validation
    2. Model building and configuration
    3. Constraint application
    4. Optimization execution
    5. Results processing and visualization

    Args:
        annual_load_twh (float): Annual electricity demand in TWh
        solar_capacity_mw (float): Solar capacity in MW
        wind_capacity_mw (float): Wind capacity in MW
        nuclear_capacity_mw (float): Nuclear capacity in MW
        hydrogen_capacity_mw (float): Hydrogen capacity in MW (0 = disabled)
        storage_power_capacity_mw (float): Storage power capacity in MW
        storage_max_hours (float): Storage duration in hours
        nuclear_p_min_pu (float): Minimum nuclear output fraction (0-1)
        nuclear_p_max_pu (float): Maximum nuclear output fraction (0-1)
        hydrogen_p_min_pu (float): Minimum hydrogen output fraction (0-1)
        hydrogen_p_max_pu (float): Maximum hydrogen output fraction (0-1)
        hydrogen_marginal_cost (float): Hydrogen fuel cost ($/MWh)
        storage_efficiency (float): Storage round-trip efficiency (0-1)
        storage_charge_efficiency (float): Storage charging efficiency (0-1)
        storage_discharge_efficiency (float): Storage discharging efficiency (0-1)
        storage_initial_soc (float): Initial state of charge fraction (0-1)
        solar_extendable (bool): Whether solar capacity can be expanded
        wind_extendable (bool): Whether wind capacity can be expanded
        nuclear_extendable (bool): Whether nuclear capacity can be expanded
        hydrogen_extendable (bool): Whether hydrogen capacity can be expanded
        storage_extendable (bool): Whether storage capacity can be expanded
        max_capacity_multiplier (float): Maximum capacity as multiplier of minimum
        solar_capital_cost (float): Solar capital cost ($/kW)
        wind_capital_cost (float): Wind capital cost ($/kW)
        nuclear_capital_cost (float): Nuclear capital cost ($/kW)
        hydrogen_capital_cost (float): Hydrogen capital cost ($/kW)
        storage_capital_cost (float): Storage capital cost ($/kW)
        solver (str): Optimization solver to use
        custom_constraints (dict): Custom constraint configuration
        verbose (bool): Whether to print detailed output
        create_plots (bool): Whether to create visualization plots
        plot_filename (str): Output filename for plots

    Returns:
        dict: Complete optimization results including network object and summary
    """

    if verbose:
        print("=" * 60)
        print("PyPSA Energy System Optimization")
        print("=" * 60)

    try:
        # Step 1: Data loading and validation
        if verbose:
            print("\n1. Loading and validating input data...")

        # Check data availability and capacities
        get_data_capacities()

        # Step 2: Model building
        if verbose:
            print("\n2. Building energy system model...")

        network = create_single_bus_model(
            solar_capacity_mw=solar_capacity_mw,
            wind_capacity_mw=wind_capacity_mw,
            nuclear_capacity_mw=nuclear_capacity_mw,
            nuclear_p_min_pu=nuclear_p_min_pu,
            nuclear_p_max_pu=nuclear_p_max_pu,
            hydrogen_capacity_mw=hydrogen_capacity_mw,
            hydrogen_p_min_pu=hydrogen_p_min_pu,
            hydrogen_p_max_pu=hydrogen_p_max_pu,
            hydrogen_marginal_cost=hydrogen_marginal_cost,
            annual_load_twh=annual_load_twh,
            storage_power_capacity_mw=storage_power_capacity_mw,
            storage_max_hours=storage_max_hours,
            storage_efficiency=storage_efficiency,
            storage_charge_efficiency=storage_charge_efficiency,
            storage_discharge_efficiency=storage_discharge_efficiency,
            storage_initial_soc=storage_initial_soc,
            solar_extendable=solar_extendable,
            wind_extendable=wind_extendable,
            nuclear_extendable=nuclear_extendable,
            hydrogen_extendable=hydrogen_extendable,
            storage_extendable=storage_extendable,
            max_capacity_multiplier=max_capacity_multiplier,
            solar_capital_cost=solar_capital_cost,
            wind_capital_cost=wind_capital_cost,
            nuclear_capital_cost=nuclear_capital_cost,
            hydrogen_capital_cost=hydrogen_capital_cost,
            storage_capital_cost=storage_capital_cost
        )

        if verbose:
            model_summary = get_model_summary(network)
            print(f"Model created successfully!")
            print(f"  - Snapshots: {model_summary['snapshots']}")
            print(f"  - Generators: {list(model_summary['generators'].keys())}")
            print(f"  - Storage units: {list(model_summary['storage_units'].keys())}")
            print(f"  - Total load: {model_summary['total_load']:.1f} GWh")

        if custom_constraints is not None:
            if verbose:
                print("\n3. Applying custom constraints...")
            add_custom_constraints(network, custom_constraints)
        elif verbose:
            print("\n3. No custom constraints to apply...")

        if verbose:
            print("\n4. Running optimization...")

        optimization_success = run_model_optimization(network, solver=solver)

        if not optimization_success:
            return {
                'status': 'optimization_failed',
                'message': 'Optimization failed to converge',
                'network': network,
                'results': None
            }

        if verbose:
            print("\n5. Validating optimization results...")
        validation_results = validate_network_constraints(network)

        if verbose:
            print("\n6. Processing results...")

        if verbose:
            print_results(network)

        plot_figure = None
        if create_plots:
            if verbose:
                print("\n7. Creating visualization...")
            plot_figure = create_interactive_plots(network, save_to_file=True, filename=plot_filename)

        results_summary = get_results_summary(network)
        results_summary['validation'] = validation_results

        if verbose:
            print("\n" + "=" * 60)
            print("Optimization completed successfully!")
            print("=" * 60)

        return {
            'status': 'success',
            'network': network,
            'results': results_summary,
            'plot_figure': plot_figure,
            'validation': validation_results,
            'model_summary': get_model_summary(network)
        }

    except Exception as e:
        error_msg = f"Error during optimization workflow: {e}"
        if verbose:
            print(f"\n❌ {error_msg}")

        return {
            'status': 'error',
            'message': error_msg,
            'network': None,
            'results': None
        }


# Maintain backward compatibility with old interface
def create_single_bus_model_legacy(*args, **kwargs):
    """Legacy interface for backward compatibility."""
    return create_single_bus_model(*args, **kwargs)

def run_model_optimization_legacy(*args, **kwargs):
    """Legacy interface for backward compatibility."""
    return run_model_optimization(*args, **kwargs)


if __name__ == "__main__":
    """
    Main execution when run as script.
    Demonstrates the new modular backend with a sample configuration.
    """

    print("PyPSA Energy System Optimization - Modular Backend")
    print("Running sample optimization...")

    # Sample configuration
    config = {
        'annual_load_twh': 700,
        'solar_capacity_mw': 200000,
        'wind_capacity_mw': 100000,
        'nuclear_capacity_mw': 24000,
        'storage_power_capacity_mw': 10000,
        'storage_max_hours': 6,
        'solar_extendable': True,
        'wind_extendable': True,
        'nuclear_extendable': False,
        'storage_extendable': True,
        'max_capacity_multiplier': 20.0,
        'solver': 'highs',
        'verbose': True,
        'create_plots': True
    }

    # Run optimization
    result = run_energy_system_optimization(**config)

    if result['status'] == 'success':
        print("\n✅ Optimization completed successfully!")
        print(f"📊 Objective value: ${result['results']['objective_value']:.2f}")
        print(f"📈 Interactive plots saved to: energy_system_analysis.html")
    else:
        print(f"\n❌ Optimization failed: {result.get('message', 'Unknown error')}")
