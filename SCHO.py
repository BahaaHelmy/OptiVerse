#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from solution import solution
def SCHO(fobj=None,lb=None, ub=None,dim=None,N=None, Max_iteration=None):
    Destination_position = np.zeros((1, dim))
    Destination_fitness = float('inf')
    Destination_position_second = np.zeros((1, dim))
    Convergence_curve = np.zeros((1, Max_iteration))
    Position_sort = np.zeros((N, dim))

    # Initialize SCHO parameters
    u = 0.388
    m = 0.45
    n = 0.5
    p = 10
    q = 9
    Alpha = 4.6
    Beta = 1.55
    BS = int(np.floor(Max_iteration / Beta))
    ct = 3.6
    T = int(np.floor(Max_iteration / ct))
    BSi = 0
    BSi_temp = 0
    ub_2 = ub
    lb_2 = lb

    # Initialize the set of random solutions
    X = initialization(N, dim, ub, lb)
    Objective_values = np.zeros((1, X.shape[0]))

    # Calculate the fitness of the first set and find the best one
    for i in range(X.shape[0]):
        Objective_values[0, i] = fobj(X[i, :])
        if Objective_values[0, i] < Destination_fitness:
            Destination_position = X[i, :]
            Destination_fitness = Objective_values[0, i]

    Convergence_curve[0] = Destination_fitness
    t = 1
    s = solution()

    print('SCHO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    while t <= Max_iteration:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Update A by using Eq. (17)
                cosh2 = (np.exp(t / Max_iteration) + np.exp(-t / Max_iteration)) / 2
                sinh2 = (np.exp(t / Max_iteration) - np.exp(-t / Max_iteration)) / 2
                r1 = np.random.rand()
                A = (p - q * (t / Max_iteration) ** (cosh2 / sinh2)) * r1

                # Enter the bounded search strategy
                if t == BSi:
                    ub_2 = Destination_position[j] + (1 - t / Max_iteration) * np.abs(
                        Destination_position[j] - Destination_position_second[j])
                    lb_2 = Destination_position[j] - (1 - t / Max_iteration) * np.abs(
                        Destination_position[j] - Destination_position_second[j])

                    if ub_2 > ub:
                        ub_2 = ub
                    if lb_2 < lb:
                        lb_2 = lb

                    X = initialization(N, dim, ub_2, lb_2)
                    BSi_temp = BSi
                    BSi = 0

                # The first phase of exploration and exploitation
                if t <= T:
                    r2 = np.random.rand()
                    r3 = np.random.rand()
                    a1 = 3 * (-1.3 * t / Max_iteration + m)
                    r4 = np.random.rand()
                    r5 = np.random.rand()

                    if A > 1:
                        sinh = (np.exp(r3) - np.exp(-r3)) / 2
                        cosh = (np.exp(r3) + np.exp(-r3)) / 2
                        W1 = r2 * a1 * (cosh + u * sinh - 1)

                        if r5 <= 0.5:
                            X[i, j] = Destination_position[j] + r4 * W1 * X[i, j]
                        else:
                            X[i, j] = Destination_position[j] - r4 * W1 * X[i, j]
                    else:
                        sinh = (np.exp(r3) - np.exp(-r3)) / 2
                        cosh = (np.exp(r3) + np.exp(-r3)) / 2
                        W3 = r2 * a1 * (cosh + u * sinh)

                        if r5 <= 0.5:
                            X[i, j] = Destination_position[j] + r4 * W3 * X[i, j]
                        else:
                            X[i, j] = Destination_position[j] - r4 * W3 * X[i, j]
                else:
                    # The second phase of exploration and exploitation
                    r2 = np.random.rand()
                    r3 = np.random.rand()
                    a2 = 2 * (-t / Max_iteration + n)
                    W2 = r2 * a2
                    r4 = np.random.rand()
                    r5 = np.random.rand()

                    if A < 1:
                        sinh = (np.exp(r3) - np.exp(-r3)) / 2
                        cosh = (np.exp(r3) + np.exp(-r3)) / 2
                        X[i, j] = X[i, j] + (r5 * sinh / cosh * np.abs(W2 * Destination_position[j] - X[i, j]))
                    else:
                        if r4 <= 0.5:
                            X[i, j] = X[i, j] + (np.abs(0.003 * W2 * Destination_position[j] - X[i, j]))
                        else:
                            X[i, j] = X[i, j] + (-np.abs(0.003 * W2 * Destination_position[j] - X[i, j]))

            BSi = BSi_temp

        for i in range(X.shape[0]):
            # Check if solutions go outside the search space and bring them back
            Flag4ub = X[i, :] > ub_2
            Flag4lb = X[i, :] < lb_2
            X[i, :] = np.where(~(Flag4ub + Flag4lb), X[i, :], (ub_2 + lb_2) / 2.0 * Flag4ub + lb_2 * Flag4lb)
        
            # Calculate the objective values
            Objective_values[0, i] = fobj(X[i, :])

            # Update the destination if there is a better solution
            if Objective_values[0, i] < Destination_fitness:
                Destination_position = X[i, :]
                Destination_fitness = Objective_values[0, i]

        # Find the second solution
        if t == BS:
            BSi = BS + 1
            BS = BS + int(np.floor((Max_iteration - BS) / Alpha))
            temp = np.zeros((1, dim))
            temp2 = np.zeros((N, dim))

            # Sorting
            for i in range(X.shape[0] - 1):
                for j in range(X.shape[0] - 1 - i):
                    if Objective_values[0, j] > Objective_values[0, j + 1]:
                        temp[0, j] = Objective_values[0, j]
                        Objective_values[0, j] = Objective_values[0, j + 1]
                        Objective_values[0, j + 1] = temp[0, j]

                        temp2[j, :] = Position_sort[j, :]
                        Position_sort[j, :] = Position_sort[j + 1, :]
                        Position_sort[j + 1, :] = temp2[j, :]

            Destination_position_second = Position_sort[1, :]

        Convergence_curve[0, t - 1] = Destination_fitness
        t += 1
    
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "SCHO"
    s.objfname = objf.__name__

    return s

def initialization(N, dim, ub, lb):
    return (ub - lb) * np.random.rand(N, dim) + lb

# Example usage:
# Define your objective function fobj
# def your_objective_function(x):
#     return your_calculation(x)

# Call the SCHO function
# result_fitness, result_position, convergence_curve = SCHO(N=10, Max_iteration=100, lb=0, ub=1, dim=5, fobj=your_objective_function)

