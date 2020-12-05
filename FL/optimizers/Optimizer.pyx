cdef class Optimizer:
    
    cdef run(self, double* X, double* y, int N, int dim):
        raise NotImplementedError()

    cdef run_local(self, double* X, double* y, int N, int dim):
        raise NotImplementedError()
