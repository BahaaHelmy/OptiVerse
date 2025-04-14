#  Sinh Cosh Optimizer (SCHO)
#
#  Developed in MATLAB R2022a
#
#  programming: Jianfu Bai
#
#  e-Mail: Jianfu.Bai@UGent.be, magd.abdelwahab@ugent.be
#  Soete Laboratory, Department of Electrical Energy, Metals, Mechanical Constructions, and Systems,
#  Faculty of Engineering and Architecture, Ghent University, Belgium
#
#  paper: Jianfu Bai, Yifei Li, Mingpo Zheng, Samir Khatir, Brahim Benaisa, Laith Abualigah, Magd Abdel Wahab, A Sinh Cosh Optimizer, Knowledge-Based Systems(2023).

# This function creates the first random population

import numpy as np
    
def initialization(SearchAgents_no = None,dim = None,ub = None,lb = None): 
    Boundary_no = ub.shape[2-1]
    # If the boundaries of all variables are equal
    
    if Boundary_no == 1:
        X = np.multiply(np.random.rand(SearchAgents_no,dim),(ub - lb)) + lb
    
    # If each variable has a different lb and ub
    if Boundary_no > 1:
        for i in np.arange(1,dim+1).reshape(-1):
            ub_i = ub(i)
            lb_i = lb(i)
            X[:,i] = np.multiply(np.random.rand(SearchAgents_no,1),(ub_i - lb_i)) + lb_i
    
    return X