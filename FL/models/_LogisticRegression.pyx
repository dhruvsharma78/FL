from ..optimizers.Optimizer cimport Optimizer

import numpy as np
cimport numpy as np

cdef extern from "./LogisticRegression.c":
    double obj(int N, int dim, double *X, double *y, double *w, double reg)
    void grad(int N, int dim, double *X, double *y, double *w, double reg, double* res)

cdef class LogisticRegression:

    cdef Optimizer optimizer
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.objective_func = &obj
        self.optimizer.gradient_func = &grad

    def fit(self, double[:, ::] X, double[::] y, double[::] w_star, bint parallel=False, bint local=False, communications=10, workers=1, extra_verbose=False):
        self.optimizer.w_star = &w_star[0]
        self.optimizer.parallel = parallel
        self.optimizer.communications = communications
        self.optimizer.workers = workers
        print_updates = 1 if extra_verbose == True else 0
        cdef  res;
        if local == False:
            return self.optimizer.run(&X[0,0], &y[0], X.shape[0], X.shape[1], print_updates)
        else:
            return self.optimizer.run_local(&X[0,0], &y[0], X.shape[0], X.shape[1], print_updates)
