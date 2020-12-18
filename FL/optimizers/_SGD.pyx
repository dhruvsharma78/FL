from .Optimizer cimport Optimizer
from ..util cimport array as flarray

cimport cython
cimport numpy as np

from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport printf

from cython.parallel import prange
cimport openmp

# Force compilation
cdef extern from "../util/random.c":
    pass

cdef extern from *:
    """
    #define START_OMP_SINGLE_PRAGMA() _Pragma("omp single") {
    #define START_OMP_PARALLEL_PRAGMA() _Pragma("omp parallel") {
    #define START_OMP_BARRIER_PRAGMA() _Pragma("omp barrier")
    #define END_OMP_PRAGMA() }
    """
    void START_OMP_SINGLE_PRAGMA() nogil
    void START_OMP_PARALLEL_PRAGMA() nogil
    void START_OMP_BARRIER_PRAGMA() nogil
    void END_OMP_PRAGMA() nogil

cdef extern from "./SGD.c":
    struct sgd_context_t:
        double* X
        double* y
        double* w_next
        double* w_star
        double reg
        double step_size
        int N
        int dim
        int decay
        int epochs
        int workers
        int parallel
        int batch_size
        int experiments
        int print_updates
        int communications
        double (*objective_func) (int N, int dim, double*, double*, double*, double)
        void   (*gradient_func)  (int N, int dim, double*, double*, double*, double, double*)
    struct result_t:
        double *obj_SGD
        double *obj_SGD_iters
        double *MSE
        double *w
    result_t run(sgd_context_t*)
    result_t run_local(sgd_context_t*)

cdef class SGD(Optimizer):

    cdef double* w_next
    cdef bint decay
    cdef int epochs, batch_size, experiments
    cdef double step_size, reg

    def __init__(self, double step_size, double reg, int epochs, int batch_size, int experiments, bint decay, double[::] w_next = None):
        self.w_next = &w_next[0] if w_next != None else NULL
        self.decay = decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.experiments = experiments
        self.step_size = step_size
        self.reg = reg

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void fill_X_batch(self, double *X, int dim, double *X_b, int *rand_idx) nogil:
        cdef int i, j
        for i in range(self.batch_size):
            for j in range(dim):
                X_b[i * dim + j] = X[rand_idx[i] * dim + j]
                
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void fill_y_batch(self, double *y, double *y_b, int *rand_idx) nogil:
        cdef int i
        for i in range(self.batch_size):
            y_b[i] = y[rand_idx[i]]

    @cython.cdivision(True)
    cdef run(self, double *X, double *y, int N, int dim, int pu):
        cdef sgd_context_t ctx
        ctx.X = X
        ctx.y = y
        ctx.w_next = NULL
        ctx.w_star = self.w_star
        ctx.reg = self.reg
        ctx.step_size = self.step_size
        ctx.N = N
        ctx.dim = dim
        ctx.decay = 1 if self.decay == True else 0
        ctx.epochs = self.epochs
        ctx.parallel = 1 if self.parallel == True else 0
        ctx.print_updates = pu
        ctx.batch_size = self.batch_size
        ctx.experiments = self.experiments
        ctx.objective_func = self.objective_func
        ctx.gradient_func = self.gradient_func

        cdef result_t res
        cdef int max_iters = N / self.batch_size

        res = run(&ctx)

        obj_SGD = flarray.ptr_to_double_np_array_1d(res.obj_SGD, self.epochs)
        obj_SGD_iters = flarray.ptr_to_double_np_array_1d(res.obj_SGD_iters, self.epochs * max_iters)
        MSE = flarray.ptr_to_double_np_array_1d(res.MSE, self.epochs * max_iters)
        w = flarray.ptr_to_double_np_array_1d(res.w, dim)

        return (obj_SGD, obj_SGD_iters, MSE, w)

    @cython.cdivision(True)
    cdef run_local(self, double *X, double *y, int N, int dim, int pu):
        cdef sgd_context_t ctx
        ctx.X = X
        ctx.y = y
        ctx.w_next = NULL
        ctx.w_star = self.w_star
        ctx.reg = self.reg
        ctx.step_size = self.step_size
        ctx.N = N
        ctx.dim = dim
        ctx.decay = 1 if self.decay == True else 0
        ctx.epochs = self.epochs
        ctx.workers = self.workers
        ctx.parallel = 1 if self.parallel == True else 0
        ctx.batch_size = self.batch_size
        ctx.experiments = self.experiments
        ctx.print_updates = pu
        ctx.communications = self.communications
        ctx.objective_func = self.objective_func
        ctx.gradient_func = self.gradient_func

        cdef result_t res
        cdef int max_iters = N / self.batch_size

        res = run_local(&ctx)

        obj_SGD = flarray.ptr_to_double_np_array_1d(res.obj_SGD, self.communications)
        MSE = flarray.ptr_to_double_np_array_1d(res.MSE, self.communications)
        w = flarray.ptr_to_double_np_array_1d(res.w, dim)

        return (obj_SGD, MSE, w)
