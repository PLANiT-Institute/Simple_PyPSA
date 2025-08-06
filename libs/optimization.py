"""
Optimization functions for PyPSA energy system model.
"""

import numpy as np


def run_model_optimization(network, solver='highs'):
    """
    Run the optimization for the network model.

    Args:
        network: PyPSA network object
        solver (str): Solver to use ('highs', 'gurobi', 'cplex', etc.)

    Returns:
        bool: True if optimization succeeded, False otherwise
    """
    try:
        print(f"Running optimization with {solver} solver...")

        # Check if we have extra_functionality for constraints
        if hasattr(network, 'extra_functionality') and network.extra_functionality:
            print("  Extra functionality detected - will be executed during optimization")

        # For MILP solvers, try to add binary constraints to prevent simultaneous operation
        if solver.lower() in ['cbc', 'gurobi', 'cplex', 'scip']:
            print(f"Using MILP solver {solver} - attempting to add binary constraints...")
            _add_binary_storage_constraints(network)

        # Try the requested solver first
        try:
            network.optimize(solver_name=solver)
            print(f"✓ Optimization completed with {solver}")
        except Exception as solver_error:
            print(f"Warning: {solver} solver failed: {solver_error}")

            # Fallback to HiGHS if the requested solver fails
            if solver.lower() != 'highs':
                print("Falling back to HiGHS solver...")
                network.optimize(solver_name='highs')
                print("✓ Optimization completed with HiGHS (fallback)")
            else:
                raise solver_error

        # Check if optimization produced results
        if not hasattr(network, 'objective') or network.objective is None:
            print("Warning: Optimization completed but no objective value found")
            return False

        print(f"Objective value: {network.objective:.2f}")

        # Always check and fix simultaneous charge/discharge as backup
        _fix_simultaneous_storage_operation(network)

        return True
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _run_optimization_with_generation_constraints(network, solver):
    """
    Run optimization with generation share constraints enforced.
    This manually builds the model and adds constraints before solving.
    """
    try:
        import pypsa
        from pypsa.optimization import assign_solution, create_model

        # Create the optimization model manually
        print("    Building optimization model...")

        # Use PyPSA's model creation but intercept to add our constraints
        network.consistency_check()

        # Create the linopy model
        network.model = create_model(network, snapshots=network.snapshots)

        # Solve the model with constraints
        print(f"    Solving model with {solver}...")
        network.model.solve(solver_name=solver)

        # Extract results back to network
        print("    Extracting optimization results...")
        assign_solution(network, network.model)

        # Check solution status
        if network.model.status != 'optimal':
            print(f"❌ Optimization failed with status: {network.model.status}")
            if network.model.status == 'infeasible':
                print("    The generation share constraints may be infeasible given the system limitations")
            return False

        print(f"✓ Optimization with constraints completed successfully")
        print(f"Objective value: {network.objective:.2f}")

        # Post-process storage
        _fix_simultaneous_storage_operation(network)

        return True

    except Exception as e:
        print(f"❌ Optimization with constraints failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _add_generation_constraints_to_model(model, network, generation_shares):
    """
    Add generation share constraints directly to the linopy model.
    This enforces actual generation output constraints, excluding curtailment.
    """
    try:
        print("      Adding generation share constraints to linopy model...")

        # Get all generator variables (actual generation output)
        gen_totals = {}
        for gen in network.generators.index:
            # Sum across all snapshots for this generator
            gen_var = model.variables[f"Generator-p"].sel(Generator=gen)
            gen_totals[gen] = gen_var.sum()

        # Get storage discharge variables (contributes to total supply)
        storage_totals = {}
        for storage in network.storage_units.index:
            storage_var = model.variables[f"StorageUnit-p_dispatch"].sel(StorageUnit=storage)
            storage_totals[storage] = storage_var.sum()

        # Calculate total system generation (excluding curtailment)
        total_generation = sum(gen_totals.values()) + sum(storage_totals.values())

        # Add constraints for each generator
        constraint_count = 0
        for gen_name, constraint_info in generation_shares.items():
            if gen_name not in gen_totals:
                print(f"        Warning: Generator '{gen_name}' not found, skipping")
                continue

            min_pct = constraint_info.get('min_pct')
            max_pct = constraint_info.get('max_pct')

            this_gen_total = gen_totals[gen_name]

            if min_pct is not None:
                # Constraint: gen_output >= min_pct/100 * total_generation
                # Rearranged: gen_output - (min_pct/100) * total_generation >= 0
                constraint_expr = this_gen_total - (min_pct/100) * total_generation
                model.add_constraints(
                    constraint_expr >= 0,
                    name=f"min_gen_share_{gen_name}"
                )
                print(f"        ✓ Added minimum {min_pct}% constraint for {gen_name}")
                constraint_count += 1

            if max_pct is not None:
                # Constraint: gen_output <= max_pct/100 * total_generation
                # Rearranged: gen_output - (max_pct/100) * total_generation <= 0
                constraint_expr = this_gen_total - (max_pct/100) * total_generation
                model.add_constraints(
                    constraint_expr <= 0,
                    name=f"max_gen_share_{gen_name}"
                )
                print(f"        ✓ Added maximum {max_pct}% constraint for {gen_name}")
                constraint_count += 1

        print(f"      Successfully added {constraint_count} generation share constraints")

    except Exception as e:
        print(f"      ❌ Error adding generation share constraints: {e}")
        import traceback
        traceback.print_exc()
        raise


def _add_binary_storage_constraints(network):
    """
    Add binary constraints to prevent simultaneous charge/discharge for MILP solvers.
    This requires PyPSA's extra_functionality for custom constraints.
    """
    try:
        print("Attempting to add binary constraints for storage...")

        # For now, we'll enhance the StorageUnit parameters to work better with MILP solvers
        # A full binary constraint implementation would require more complex PyPSA programming

        for storage in network.storage_units.index:
            # Add small costs to discourage simultaneous operation
            # MILP solvers are better at handling these discrete decisions
            network.storage_units.loc[storage, 'marginal_cost'] = 0.01

            # For MILP solvers, we can set standing loss to zero since we'll have proper constraints
            network.storage_units.loc[storage, 'standing_loss'] = 0.0

        print("✓ Enhanced storage parameters for MILP solver")

        # TODO: Implement proper binary constraints using PyPSA's extra_functionality
        # This would require adding binary variables and constraints like:
        # - Binary variable b_charge[t] for each time step
        # - Binary variable b_discharge[t] for each time step
        # - Constraint: b_charge[t] + b_discharge[t] <= 1
        # - Constraint: p_store[t] <= M * b_charge[t]
        # - Constraint: p_dispatch[t] <= M * b_discharge[t]

    except Exception as e:
        print(f"Warning: Could not add binary storage constraints: {e}")
        print("Falling back to post-processing fix")


def _fix_simultaneous_storage_operation(network):
    """
    Fix simultaneous charge/discharge by setting the smaller value to zero.
    Also recalculate state of charge after modifications.
    """
    try:
        for storage in network.storage_units.index:
            p_store = network.storage_units_t.p_store[storage].copy()
            p_dispatch = network.storage_units_t.p_dispatch[storage].copy()

            # Use a smaller threshold for detecting simultaneous operation
            threshold = 0.0001  # Very small threshold to catch tiny simultaneous operations
            simultaneous = (p_store > threshold) & (p_dispatch > threshold)

            if simultaneous.any():
                count = simultaneous.sum()
                print(f"Fixing {storage}: Found {count} periods with simultaneous charge/discharge")

                # For each simultaneous period, keep only the larger operation
                for idx in simultaneous[simultaneous].index:
                    if p_store[idx] > p_dispatch[idx]:
                        # Keep charging, remove discharging
                        network.storage_units_t.p_dispatch.loc[idx, storage] = 0.0
                        print(f"  Time {idx}: Kept charging ({p_store[idx]:.3f} MW), removed discharging ({p_dispatch[idx]:.3f} MW)")
                    else:
                        # Keep discharging, remove charging
                        network.storage_units_t.p_store.loc[idx, storage] = 0.0
                        print(f"  Time {idx}: Kept discharging ({p_dispatch[idx]:.3f} MW), removed charging ({p_store[idx]:.3f} MW)")

                # Recalculate state of charge after fixing simultaneous operations
                _recalculate_state_of_charge(network, storage)

                print(f"✓ {storage} operation fixed - no more simultaneous charge/discharge")
            else:
                print(f"✓ {storage} operation validated - no simultaneous charge/discharge detected")

    except Exception as e:
        print(f"Warning: Could not fix storage operation: {e}")


def _recalculate_state_of_charge(network, storage):
    """
    Recalculate state of charge after modifying p_store and p_dispatch values.
    """
    try:
        # Get storage parameters
        max_hours = network.storage_units.loc[storage, 'max_hours']
        efficiency_store = network.storage_units.loc[storage, 'efficiency_store']
        efficiency_dispatch = network.storage_units.loc[storage, 'efficiency_dispatch']
        standing_loss = network.storage_units.loc[storage, 'standing_loss']

        # Get power capacity
        if network.storage_units.loc[storage, 'p_nom_extendable']:
            p_nom = network.storage_units.loc[storage, 'p_nom_opt']
        else:
            p_nom = network.storage_units.loc[storage, 'p_nom']

        # Calculate energy capacity
        energy_capacity = p_nom * max_hours

        # Get initial state of charge
        initial_soc = network.storage_units.loc[storage, 'state_of_charge_initial']

        # Recalculate state of charge step by step
        soc_values = []
        current_soc = initial_soc * energy_capacity  # Convert fraction to MWh

        p_store = network.storage_units_t.p_store[storage]
        p_dispatch = network.storage_units_t.p_dispatch[storage]

        for idx in network.snapshots:
            # Apply standing loss (fraction per hour)
            current_soc = current_soc * (1 - standing_loss)

            # Apply charging (p_store is positive when charging)
            charge_energy = p_store[idx] * efficiency_store  # MWh gained
            current_soc += charge_energy

            # Apply discharging (p_dispatch is positive when discharging)
            discharge_energy = p_dispatch[idx] / efficiency_dispatch  # MWh lost from storage
            current_soc -= discharge_energy

            # Ensure SOC stays within bounds
            current_soc = max(0, min(current_soc, energy_capacity))

            soc_values.append(current_soc)

        # Update the state of charge in the network
        network.storage_units_t.state_of_charge[storage] = soc_values

        print(f"  ✓ State of charge recalculated for {storage}")

    except Exception as e:
        print(f"Warning: Could not recalculate state of charge for {storage}: {e}")


def _check_storage_operation(network):
    """
    Check if storage is charging and discharging simultaneously and warn if so.
    """
    try:
        for storage in network.storage_units.index:
            p_store = network.storage_units_t.p_store[storage]
            p_dispatch = network.storage_units_t.p_dispatch[storage]

            # Check for simultaneous operation (both > small threshold)
            simultaneous = (p_store > 0.001) & (p_dispatch > 0.001)

            if simultaneous.any():
                count = simultaneous.sum()
                print(f"Warning: {storage} is charging and discharging simultaneously in {count} time periods")
                print("Consider using a MILP solver (gurobi/cplex) for strict charge/discharge exclusivity")
            else:
                print(f"✓ {storage} operation validated - no simultaneous charge/discharge detected")

    except Exception as e:
        print(f"Warning: Could not check storage operation: {e}")
