# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:42:18 2016

@author: hossam
"""

import random
import numpy as np
import math
from solution import solution
import time

def initialization(N, dim, ub, lb):
    return np.random.uniform(lb, ub, (N, dim))

def SMA(fobj, lb, ub, dim, N, Max_iter):
    print('SMA is now tackling your problem')

    best_positions = np.zeros(dim)
    destination_fitness = float('inf')  # Change to -float('inf') for maximization problems
    all_fitness = np.full(N, float('inf'))
    weight = np.ones((N, dim))
    
    # Initialize the set of random solutions
    X = initialization(N, dim, ub, lb)
    convergence_curve = np.zeros(Max_iter)

    lb = np.full(dim, lb)
    ub = np.full(dim, ub)
    z = 0.03  # parameter

    it = 0  # Number of iterations
    s = solution()
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    while it < Max_iter:
        # Evaluate fitness for all agents
        for i in range(N):
            # Ensure solutions stay within bounds
            X[i, :] = np.clip(X[i, :], lb, ub)
            all_fitness[i] = fobj(X[i, :])

        # Sort fitness
        smell_order = np.argsort(all_fitness)
        smell_index = smell_order
        worst_fitness = all_fitness[smell_order[-1]]
        best_fitness = all_fitness[smell_order[0]]

        S = best_fitness - worst_fitness + np.finfo(float).eps  # Avoid division by zero

        # Calculate weights for each slime mold
        for i in range(N):
            for j in range(dim):
                if i <= (N / 2):  # Eq.(2.5)
                    weight[smell_index[i], j] = 1 + np.random.rand() * np.log10((best_fitness - all_fitness[smell_index[i]]) / S + 1)
                else:
                    weight[smell_index[i], j] = 1 - np.random.rand() * np.log10((best_fitness - all_fitness[smell_index[i]]) / S + 1)

        # Update the best fitness and positions
        if best_fitness < destination_fitness:
            best_positions = X[smell_index[0], :]
            destination_fitness = best_fitness

        # Update control parameters
        a = np.arctanh(np.clip(-(it / Max_iter) + 1, -0.999999, 0.999999))  # Eq.(2.4) with clipping
        b = 1 - it / Max_iter

        # Update positions of search agents
        for i in range(N):
            if np.random.rand() < z:  # Eq.(2.7)
                X[i, :] = np.random.uniform(lb, ub, dim)
            else:
                p = np.tanh(abs(all_fitness[i] - destination_fitness))  # Eq.(2.2)
                vb = np.random.uniform(-a, a, dim)  # Eq.(2.3)
                vc = np.random.uniform(-b, b, dim)
                for j in range(dim):
                    r = np.random.rand()
                    A = np.random.randint(0, N)  # Randomly select two positions from the population
                    B = np.random.randint(0, N)
                    if r < p:  # Eq.(2.1)
                        X[i, j] = best_positions[j] + vb[j] * (weight[i, j] * X[A, j] - X[B, j])
                    else:
                        X[i, j] = vc[j] * X[i, j]

        # Record convergence information
        convergence_curve[it] = destination_fitness
        it += 1

    timerEnd = time.time()
    s.best = destination_fitness
    s.bestIndividual = best_positions
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "SMA"
    s.objfname = fobj.__name__

    return s