"""
Created on Thu Jan 27 2022

@author: Iman Ahmadianfar
"""
#---------------------------------------------------------------------------------------------------------------------------
# weIghted meaN oF vectOrs (INFO)
# INFO: An Efficient Optimization Algorithm based on Weighted Mean of Vectors
# Codes of INFO:http://imanahmadianfar.com/codes/
# Website and codes of INFO:http://www.aliasgharheidari.com/INFO.html

# Iman Ahmadianfar, Ali asghar Heidari, Saeed Noushadian, Huiling Chen, and Amir H. Gandomi 

#  Last update: 01-10-2022

#  e-Mail: im.ahmadian@gmail.com,i.ahmadianfar@bkatu.ac.ir.
#  e-Mail: as_heidari@ut.ac.ir, aliasghar68@gmail.com,
#  
#---------------------------------------------------------------------------------------------------------------------------
#  Authors: Ali Asghar Heidari(as_heidari@ut.ac.ir, aliasghar68@gmail.com),Saeed Noushadian, Huiling Chen(chenhuiling.jlu@gmail.com), and Amir H Gandomi, 
#---------------------------------------------------------------------------------------------------------------------------

# After use, please cite to the main paper:
# Iman Ahmadianfar, Ali asghar Heidari, Saeed Noushadian, Huiling Chen, Amir H. Gandomi  
# INFO: An Efficient Optimization Algorithm based on Weighted Mean of Vectors
# Expert Systems With Applications, 116516, 2022, doi: https://doi.org/10.1016/j.eswa.2022.116516 (Q1, 5-Year Impact Factor: 6.95)
#---------------------------------------------------------------------------------------------------------------------------
# You can also follow the paper for related updates in researchgate: 
# https://www.researchgate.net/profile/Iman_Ahmadianfar
# https://www.researchgate.net/profile/Ali_Asghar_Heidari.

#  Website and codes of INFO:%  http://www.aliasgharheidari.com/INFO.html

# You can also use and compare with our other new optimization methods:
                                                                       #(INFO)-2020-http://www.imanahmadianfar.com/codes.
                                                                       #(GBO)-2020-http://www.imanahmadianfar.com/codes.
                                                                       #(INFO)-2022- http://www.aliasgharheidari.com/INFO.html
																	   #(RUN)-2021- http://www.aliasgharheidari.com/RUN.html
                                                                       #(HGS)-2021- http://www.aliasgharheidari.com/HGS.html
                                                                       #(SMA)-2020- http://www.aliasgharheidari.com/SMA.html
                                                                       #(HHO)-2019- http://www.aliasgharheidari.com/HHO.html  

#---------------------------------------------------------------------------------------------------------------------------


import numpy
import numpy as np
import math
from solution import solution
import time
from random import randint

def INFO(objf,lb,ub,dim,nP,MaxIt):  
    
    Best_X=numpy.zeros(dim)
    Best_Cost=float("inf") 
    Convergence_curve=numpy.zeros(MaxIt)
    
    s=solution()
    print("INFO is optimizing  \""+objf.__name__+"\"")
    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")  
 
    Cost=numpy.zeros(nP) 
    M=numpy.zeros(nP) 
    W=numpy.zeros(3)
    MM=numpy.zeros(3)
    New_Cost=numpy.zeros((nP))
    New_X=numpy.zeros((nP,dim))
    X=numpy.random.uniform(0,1,(nP,dim)) *(ub-lb)+lb

    for i in range (0,nP):
        Cost[i]=objf(X[i,:])
        M[i]=Cost[i]
      
        #Cost=numpy.sort(Cost)
    Ind=numpy.argsort(Cost)
    Best_X = X[Ind[0],:]
    Best_Cost = Cost[Ind[0]]

    Worst_X = X[Ind[nP-1],:]
    Worst_Cost = Cost[Ind[nP-1]]

    I = randint(2,5)
    Better_X = X[Ind[I],:]
    Better_Cost = Cost[Ind[I]]

    for it in range(0,MaxIt):
        alpha=2*math.exp(-4*it/MaxIt)            # Eqs. (5.1) # % Eq. (9.1)                                        
    
        M_Best=Best_Cost;
        M_Better=Better_Cost;
        M_Worst=Worst_Cost;
    
        for i in range(0,nP):
            dl=2*np.random.rand()*alpha-alpha             # Eq. (5)                                              
            sigm=2*np.random.rand()*alpha-alpha           # Eq. (9) 
        
            a,b,c = np.random.choice(list(set(range(0, nP))-{i}),3,replace=False)

            e=1e-25
            epsi=e*np.random.rand()

            omg = max([M[a], M[b], M[c]])
            MM = [(M[a]-M[b]), (M[a]-M[c]), (M[b]-M[c])]
        
            W[0] = (math.cos(MM[0]+math.pi))*math.exp(-abs(MM[0]/omg))         # Eq. (4.2)
            W[1] = (math.cos(MM[1]+math.pi))*math.exp(-abs(MM[1]/omg))         # Eq. (4.3)
            W[2] = (math.cos(MM[2]+math.pi))*math.exp(-abs(MM[2]/omg))         # Eq. (4.4)
            Wt = sum(W)

            WM1 = dl*( W[0]*(X[a,:]-X[b,:])+W[1]*(X[a,:]-X[c,:])+ W[2]*(X[b,:]-X[c,:]))/(Wt+1)+epsi      # Eq. (4.1)
   
            omg = max([M_Best, M_Better, M_Worst])
            MM = [(M_Best-M_Better), (M_Best-M_Better), (M_Better-M_Worst)]

            W[0] = (math.cos(MM[0]+math.pi))*math.exp(-abs(MM[0]/omg))                                   # Eq. (4.7)
            W[1] = (math.cos(MM[1]+math.pi))*math.exp(-abs(MM[1]/omg))                                   # Eq. (4.8)
            W[2] = (math.cos(MM[2]+math.pi))*math.exp(-abs(MM[2]/omg))                                   # Eq. (4.9)
            Wt = sum(W)

            WM2 = dl*(W[0]*(Best_X-Better_X)+W[1]*(Best_X-Worst_X)+ W[2]*(Better_X-Worst_X))/(Wt+1)+epsi    # Eq. (4.6)
            
            # Determine MeanRule        
            r = numpy.random.uniform(0.1,0.5)
            MeanRule = r*WM1+(1-r)*WM2                                                                      # Eq. (4)

            if (np.random.rand()<0.5):
               z1 = X[i,:]+sigm*(np.random.rand()*MeanRule)+numpy.random.randn()*(Best_X-X[a,:])/(M_Best-M[a]+1)
               z2 = Best_X+sigm*(np.random.rand()*MeanRule)+numpy.random.randn()*(X[a,:]-X[b,:])/(M[a]-M[b]+1)            
            else:                                                                                                 # Eq. (8)
               z1 = X[a,:]+sigm*(np.random.rand()*MeanRule)+numpy.random.randn()*(X[b,:]-X[c,:])/(M[b]-M[c]+1)
               z2 = Better_X+sigm*(np.random.rand()*MeanRule)+numpy.random.randn()*(X[a,:]-X[b,:])/(M[a]-M[b]+1) 
           
            # Vector combining stage
            u=numpy.zeros(dim)
            for j in range(0,dim):
               mu = 0.05*numpy.random.randn()
               if (np.random.rand() <0.5): 
                  if (np.random.rand()<0.5):
                     u[j] = z1[j] + mu*abs(z1[j]-z2[j])                                   # Eq. (10.1)
                  else:
                     u[j] = z2[j] + mu*abs(z1[j]-z2[j])                                   # Eq. (10.2)
            
               else:
                  u[j] = X[i,j]                                                           # Eq. (10.3)
         
            
            # Local search stage
            if (np.random.rand()<0.5):
                    L=np.random.rand()<0.5;v1=(1-L)*2*(np.random.rand())+L                # Eqs. (11.5) & % Eq. (11.6)
                    v2=np.random.rand()*L+(1-L)
                                                                                
                    Xavg=(X[a,:]+X[b,:]+X[c,:])/3;                                        # Eq. (11.4)
                    phi=np.random.rand();
                    Xrnd = phi*(Xavg)+(1-phi)*(phi*Better_X+(1-phi)*Best_X);              # Eq. (11.3)
                    Randn = L*numpy.random.randn(1,dim)+(1-L)*numpy.random.randn()
                    if (np.random.rand()<0.5):
                        u = Best_X + Randn*(MeanRule+numpy.random.randn()*(Best_X-X[a,:]));        # Eq. (11.1)
                    else:
                        u = Xrnd + Randn*(MeanRule+numpy.random.randn()*(v1*Best_X-v2*Xrnd));      # Eq. (11.2)
                    
            # Check if new solution go outside the search space and bring them back
            New_X[i,:]= numpy.clip(u, lb, ub)
            New_Cost[i] = objf(New_X[i,:])                    
                         
            if (New_Cost[i]<Cost[i]):
               X[i,:]=New_X[i,:]
               Cost[i]=New_Cost[i];
               M[i]=Cost[i];
               if (Cost[i]<Best_Cost):
                  Best_X=X[i,:]
                  Best_Cost = Cost[i]
          
          
        Ind=numpy.argsort(Cost)
        # Determine the worst solution
        Worst_X = X[Ind[nP-1],:]
        Worst_Cost = Cost[Ind[nP-1]]
        
        # Determine the better solution
        I = randint(2,5)
        Better_X = X[Ind[I],:]
        Better_Cost = Cost[Ind[I]]
        
        # Update Convergence_curve
        Convergence_curve[it]=Best_Cost;
        
        # Show Iteration Information
            
    timerEnd=time.time()
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="INFO"   
    s.objfname=objf.__name__
    s.best=Best_Cost
    s.bestIndividual=Better_X
    return s

















