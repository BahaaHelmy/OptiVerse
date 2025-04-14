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

def RUN(fobj, lb, ub, dim, nP, MaxIt):
    def initialization(nP, dim, ub, lb):
        return lb + np.random.rand(nP, dim) * (ub - lb)

    def unifrnd(a, b, c, dim):
        mu = (a + b) / 2
        sigma = (b - a) / 2
        return mu + sigma * (2 * np.random.rand(c, dim) - 1)

    def rndx(nP, i):
        Qi = np.random.permutation(nP).tolist()
        Qi.remove(i)
        return Qi[0], Qi[1], Qi[2]

    def runge_kutta(XB, XW, DelX):
        dim = XB.shape[0]
        C = np.random.randint(1, 3) * (1 - np.random.rand())
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)

        K1 = 0.5 * (np.random.rand() * XW - C * XB)
        K2 = 0.5 * (np.random.rand() * (XW + r2 * K1 * DelX / 2) - (C * XB + r1 * K1 * DelX / 2))
        K3 = 0.5 * (np.random.rand() * (XW + r2 * K2 * DelX / 2) - (C * XB + r1 * K2 * DelX / 2))
        K4 = 0.5 * (np.random.rand() * (XW + r2 * K3 * DelX) - (C * XB + r1 * K3 * DelX))

        XRK = (K1 + 2 * K2 + 2 * K3 + K4)
        return 1 / 6 * XRK

    # Initialization
    Cost = np.zeros(nP)
    X = initialization(nP, dim, ub, lb)
    Xnew2 = np.zeros(dim)
    Convergence_curve = np.zeros(MaxIt)

    for i in range(nP):
        Cost[i] = fobj(X[i, :])

    Best_Cost = np.min(Cost)
    ind = np.argmin(Cost)
    Best_X = X[ind, :]
    print('RUN is optimizing  "' + fobj.__name__ + '"')

    Convergence_curve[0] = Best_Cost
    s = solution()
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main Loop of RUN
    it = 0
    while it < MaxIt - 1:
        it += 1
        f = 20 * np.exp(-12 * (it / MaxIt))
        Xavg = np.mean(X, axis=0)
        SF = 2 * (0.5 - np.random.rand(nP)) * f

        for i in range(nP):
            ind_l = np.argmin(Cost)
            lBest = X[ind_l, :]

            A, B, C = rndx(nP, i)
            ind1 = np.argmin(Cost[[A, B, C]])

            # Calculate Delta X
            gamma = np.random.rand() * (X[i, :] - np.random.rand(dim) * (ub - lb)) * np.exp(-4 * it / MaxIt)
            Stp = np.random.rand(dim) * ((Best_X - np.random.rand() * Xavg) + gamma)
            DelX = 2 * np.random.rand(dim) * np.abs(Stp)

            if Cost[i] < Cost[ind1]:
                Xb = X[i, :]
                Xw = X[ind1, :]
            else:
                Xb = X[ind1, :]
                Xw = X[i, :]

            SM = runge_kutta(Xb, Xw, DelX)

            L = np.random.rand(dim) < 0.5
            Xc = L * X[i, :] + (1 - L) * X[A, :]
            Xm = L * Best_X + (1 - L) * lBest

            vec = [1, -1]
            r = np.random.choice(vec, size=dim)
            g = 2 * np.random.rand()
            mu = 0.5 + 0.1 * np.random.randn(dim)

            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * SM + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * SM + mu * (X[A, :] - X[B, :])

            Xnew = np.clip(Xnew, lb, ub)
            CostNew = fobj(Xnew)

            if CostNew < Cost[i]:
                X[i, :] = Xnew
                Cost[i] = CostNew

            # Enhanced Solution Quality (ESQ)
            if np.random.rand() < 0.5:
                EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
                r = np.random.randint(-1, 2)
                u = 2 * np.random.rand(dim)
                w = unifrnd(0, 2, 1, dim).flatten() * EXP  # Ensure w is 1D

                A, B, C = rndx(nP, i)
                Xavg = (X[A, :] + X[B, :] + X[C, :]) / 3

                beta = np.random.rand(dim)
                Xnew1 = beta * Best_X + (1 - beta) * Xavg

                for j in range(dim):
                    if w[j] < 1:  # Compare scalar w[j]
                        Xnew2[j] = Xnew1[j] + r * w[j] * abs((Xnew1[j] - Xavg[j]) + np.random.randn())
                    else:
                        Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[j] * abs((u[j] * Xnew1[j] - Xavg[j]) + np.random.randn())

                Xnew2 = np.clip(Xnew2, lb, ub)
                CostNew = fobj(Xnew2)

                if CostNew < Cost[i]:
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                elif np.random.rand() < w[np.random.randint(dim)]:
                    SM = runge_kutta(X[i, :], Xnew2, DelX)
                    Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (SM + (2 * np.random.rand(dim) * Best_X - Xnew2))

                    Xnew = np.clip(Xnew, lb, ub)
                    CostNew = fobj(Xnew)

                    if CostNew < Cost[i]:
                        X[i, :] = Xnew
                        Cost[i] = CostNew

            if Cost[i] < Best_Cost:
                Best_X = X[i, :]
                Best_Cost = Cost[i]

        Convergence_curve[it] = Best_Cost

    timerEnd = time.time()
    s.best = Best_Cost
    s.bestIndividual = Best_X
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "RUN"
    s.objfname = fobj.__name__

    return s