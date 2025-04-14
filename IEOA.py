# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:19:49 2016

@author: hossam
"""
import random
import numpy
import math
from solution import solution
import time

from scipy.spatial.distance import cdist, pdist, squareform

import numpy as np

def IEOA(objective_function,lb,ub, num_dimensions,population_size,max_iterations ):
    """
    Generic optimization algorithm.

    Parameters:
        population_size (int): Number of individuals in the population.
        num_dimensions (int): Dimensionality of the problem.
        max_iterations (int): Maximum number of iterations.
        lb (float or np.ndarray): Lower bound for decision variables.
        ub (float or np.ndarray): Upper bound for decision variables.
        objective_function (callable): Objective function to minimize.

    Returns:
        best_solution (np.ndarray): The best solution found.
        best_fitness (float): The fitness value of the best solution.
    """

    # Initialize population
    lb = np.array(lb) if isinstance(lb, (list, np.ndarray)) else np.full(num_dimensions, lb)
    ub = np.array(ub) if isinstance(ub, (list, np.ndarray)) else np.full(num_dimensions, ub)
    population = np.random.uniform(lb, ub, size=(population_size, num_dimensions))
    fitness = np.zeros(population_size)
    convergence = []

    # Main loop
    for iteration in range(max_iterations):
        # Evaluate fitness of the population
        for i in range(population_size):
            fitness[i] = objective_function(population[i, :])

        # Election process: Select the leader (best solution)
        leader_index = np.argmin(fitness)
        leader = population[leader_index, :]
        leader_fitness = fitness[leader_index]
        convergence.append(leader_fitness)
        s = solution()
        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        # Campaign: Update the population based on the leader
        for i in range(population_size):
            if i != leader_index:  # Exclude the leader
                # Perform crossover and mutation with the leader
                # Crossover
                crossover_point = np.random.randint(1, num_dimensions)
                new_solution = np.concatenate([population[i, :crossover_point], leader[crossover_point:]])

                # Mutation
                mutation_rate = 0.1
                mutation_mask = np.random.rand(num_dimensions) < mutation_rate
                new_solution += mutation_mask * np.random.randn(num_dimensions)

                # Ensure the new solution adheres to lb and ub
                new_solution = np.clip(new_solution, lb, ub)

                # Evaluate the fitness of the new solution
                new_fitness = objective_function(new_solution)

                # Replace the old solution if the new solution is better
                if new_fitness < fitness[i]:
                    population[i, :] = new_solution
                    fitness[i] = new_fitness


        # **DLH mechanism**
        distances = cdist(population, population, metric="euclidean")
        radius = distances[leader_index, :]
        new_population_dlh = np.zeros_like(population)

        for t in range(population_size):
            if t != leader_index:  # Exclude the leader
                # Find neighbors based on radius
                neighbor_indices = np.where(distances[t, :] <= radius[t])[0]
                if len(neighbor_indices) > 0:
                    random_neighbor_idx = np.random.choice(neighbor_indices)
                    random_neighbor = population[random_neighbor_idx, :]

                    # Generate new solution using DLH
                    new_dlh_solution = population[t, :] + np.random.rand(num_dimensions) * (
                        random_neighbor - population[t, :]
                    )

                    # Ensure the new DLH solution adheres to bounds
                    new_dlh_solution = np.clip(new_dlh_solution, lb, ub)

                    # Evaluate fitness of the new DLH solution
                    new_dlh_fitness = objective_function(new_dlh_solution)

                    # Replace solution if DLH solution is better
                    if new_dlh_fitness < fitness[t]:
                        new_population_dlh[t, :] = new_dlh_solution
                        fitness[t] = new_dlh_fitness
                    else:
                        new_population_dlh[t, :] = population[t, :]
                else:
                    # Retain current solution if no neighbors found
                    new_population_dlh[t, :] = population[t, :]
            else:
                # Leader remains unchanged
                new_population_dlh[t, :] = leader

        # Update population after DLH
        population = new_population_dlh

    # Find the best solution
    best_index = np.argmin(fitness)
    best_solution = population[best_index, :]
    best_fitness = fitness[best_index]

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "IEOA"
    s.objfname = objective_function.__name__
    s.best = best_fitness
    s.bestIndividual = best_solution

    return s
