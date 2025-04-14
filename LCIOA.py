import numpy as np
import time
from solution import solution

def LCIOA(cost_function, min_val, max_val, d, n_pop, max_iter):
    # Initialize population
    pop_positions = np.random.uniform(min_val, max_val, (n_pop, d))  # Create an n_pop x d matrix
    pop_costs = np.apply_along_axis(cost_function, 1, pop_positions)  # Apply cost function on each individual
    
    # Best solution initialization
    best_idx = np.argmin(pop_costs)
    best_cost = pop_costs[best_idx]
    best_position = pop_positions[best_idx]
    
    # Parameters initialization
    gamma = 0.1  # Initial exploration factor
    mu = 0.2  # Mutation factor
    eta = 0.5  # Migration factor
    delta = 0.1  # Exploration/exploitation balance factor
    mutation_rate = 0.1  # Initial mutation rate

    # Convergence curve to track progress
    conv_curve = np.zeros(max_iter)

    # Solution object to store results
    s = solution()
    timer_start = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    stagnation_threshold = 10  # Number of iterations without improvement before reintroducing diversity
    no_improvement_counter = 0  # Counter for stagnation
    exploration_increase_period = 10  # After 50 iterations, increase exploration

    for it in range(max_iter):
        # Dynamically adjust exploration and exploitation balance
        decay_factor = 1 - (it / max_iter)  # Linear decay for exploration
        gamma *= decay_factor  # Reduce exploration over time
        mu *= decay_factor
        eta *= decay_factor
        local_search_factor = (1 - decay_factor) * 0.5  # As we converge, more focus on exploitation

        # **Enhanced Exploitation Phase**
        gradients = np.random.randn(n_pop, d)  # Generate random gradients for all individuals
        local_steps = gamma * delta * gradients  # Local exploration steps
        
        # Focusing search around the best solution
        best_invasion_step = local_search_factor * (best_position - pop_positions)  # Small step around best solution
        local_steps += best_invasion_step  # Add best solution influence
        new_positions = np.clip(pop_positions + local_steps, min_val, max_val)  # Update positions
        new_costs = np.apply_along_axis(cost_function, 1, new_positions)  # Calculate new costs for updated positions
        
        # Update positions and costs if improved
        improved_idx = new_costs < pop_costs
        pop_positions[improved_idx] = new_positions[improved_idx]
        pop_costs[improved_idx] = new_costs[improved_idx]
        
        # Update best solution
        best_idx = np.argmin(pop_costs)
        best_cost = pop_costs[best_idx]
        best_position = pop_positions[best_idx]

        # **Metastasis Phase (Parallel Exploration)**
        # Generate new solutions via mutation
        mutations = mu * np.random.randn(n_pop * 2, d)  # Generate mutations for offspring
        offspring_positions = np.clip(np.vstack([pop_positions, pop_positions]) + mutations, min_val, max_val)
        offspring_costs = np.apply_along_axis(cost_function, 1, offspring_positions)  # Calculate new costs for offspring
        
        # Combine offspring and parent populations, and keep the best n_pop individuals
        combined_positions = np.vstack([pop_positions, offspring_positions])
        combined_costs = np.hstack([pop_costs, offspring_costs])
        sorted_indices = np.argsort(combined_costs)
        
        # Keep the best n_pop individuals
        pop_positions = combined_positions[sorted_indices[:n_pop]]
        pop_costs = combined_costs[sorted_indices[:n_pop]]

        # **Global Migration Phase (Distant Exploration)**
        # Global migration for further exploration
        migration_terms = eta * (best_position - pop_positions)  # Global migration to best solution
        global_migrants = np.clip(pop_positions + migration_terms, min_val, max_val)  # Update positions
        migrant_costs = np.apply_along_axis(cost_function, 1, global_migrants)  # Calculate new costs for migrants
        
        # Replace population with global migrants if improved
        improved_idx = migrant_costs < pop_costs
        pop_positions[improved_idx] = global_migrants[improved_idx]
        pop_costs[improved_idx] = migrant_costs[improved_idx]

        # **Diversity Reintroduction to Avoid Local Optima**
        if np.min(pop_costs) == best_cost:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        # If stagnation occurs, introduce diversity by randomly reinitializing some solutions
        if no_improvement_counter >= stagnation_threshold:
            print("Stagnation detected, reintroducing diversity!")
            random_indices = np.random.choice(n_pop, size=int(n_pop * 0.2), replace=False)  # Reinitialize 20% of the population
            pop_positions[random_indices] = np.random.uniform(min_val, max_val, (len(random_indices), d))
            pop_costs[random_indices] = np.apply_along_axis(cost_function, 1, pop_positions[random_indices])
            no_improvement_counter = 0  # Reset stagnation counter

        # **Increase Exploration Periodically**
        if it % exploration_increase_period == 0:
            gamma = 0.2  # Increase exploration periodically
            eta = 0.6

        # Update best solution after all phases
        best_idx = np.argmin(pop_costs)
        best_cost = pop_costs[best_idx]
        best_position = pop_positions[best_idx]

        # Update convergence curve
        conv_curve[it] = best_cost

        # Display iteration progress
        print(f"Iteration {it + 1}: Best Cost = {conv_curve[it]}")

    # Final results
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = time.time() - timer_start
    s.convergence = conv_curve
    s.optimizer = "LCIOA"
    s.objfname = cost_function.__name__
    s.best = best_cost
    s.bestIndividual = best_position

    return s
