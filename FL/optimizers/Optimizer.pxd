cdef class Optimizer:

    cdef double (*objective_func) (int N, int dim, double*, double*, double*, double)
    cdef void (*gradient_func)(int N, int dim, double*, double*, double*, double, double*)
    cdef double *w_star
    cdef int communications, workers
    cdef bint parallel
    
    cdef run(self, double* X, double* y, int N, int dim)
    cdef run_local(self, double* X, double* y, int N, int dim)
