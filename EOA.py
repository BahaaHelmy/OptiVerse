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


import numpy as np

def EOA(objective_function,lb,ub, num_dimensions,population_size,max_iterations ):
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

    # Find the best solution
    best_index = np.argmin(fitness)
    best_solution = population[best_index, :]
    best_fitness = fitness[best_index]

    # Display the result
    print('Optimization finished.')
    print('Best solution found:', best_solution)
    print('Best fitness:', best_fitness)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "EOA"
    s.objfname = objective_function.__name__
    s.best = best_fitness
    s.bestIndividual = best_solution

    return s
