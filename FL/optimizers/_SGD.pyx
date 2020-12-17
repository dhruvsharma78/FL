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
    cdef run(self, double *X, double *y, int N, int dim):
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
    cdef run_local(self, double *X, double *y, int N, int dim):
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

    # @cython.wraparound(False)
    # @cython.boundscheck(False)
    # @cython.nonecheck(False)
    # @cython.initializedcheck(False)
    # @cython.cdivision(True)
    # cdef double* run(self, double* X, double*y, int N, int dim) nogil:
        
        # cdef int max_iters, k, i, j, exp, xx
        # cdef double t
        # cdef double *_obj_SGD
        # cdef double *_obj_SGD_iters
        # cdef double *_MSE
        # cdef double *_w

        # exp = self.experiments
        # max_iters = N/self.batch_size

        # _obj_SGD       = <double*> calloc(exp * self.epochs            , sizeof(double))
        # _obj_SGD_iters = <double*> calloc(exp * self.epochs * max_iters, sizeof(double))
        # _MSE           = <double*> calloc(exp * self.epochs * max_iters, sizeof(double)) 
        # _w             = <double*> calloc(exp * dim                    , sizeof(double))

        # # Thread private indexes
        # cdef double *obj_SGD
        # cdef double *obj_SGD_iters
        # cdef double *MSE
        # cdef double *w
        # cdef int    *rand_idx
        # cdef double *X_batch
        # cdef double *y_batch
        # cdef double *temp
        # cdef double step

        # if (self.run_parallel == False):
            # openmp.omp_set_num_threads(1)
        
        # for k in range(exp):
            # obj_SGD       = &_obj_SGD[k * self.epochs]
            # obj_SGD_iters = &_obj_SGD_iters[k * self.epochs * max_iters]
            # MSE           = &_MSE[k * self.epochs * max_iters]
            # w             = &_w[k * dim]
            # X_batch = <double*> malloc(sizeof(double) * self.batch_size * dim)
            # y_batch = <double*> malloc(sizeof(double) * self.batch_size)
            # temp    = <double*> malloc(sizeof(double) * dim)
            # if self.w_next == NULL:
                # flm.fill_random_doubles(dim, w)
                # flm.dscal(dim, 0.001, w, 1)
            # else:
                # flm.dcopy(dim, self.w_next, 1, w, 1)
            # for i in range(self.epochs):
                # obj_SGD[i] += self.obj(N, dim, X, y, w, self.reg)
                # if i % 10 == 0:
                    # printf("Experiment: %d/%d, Epoch %d/%d, Loss: %f\n", k+1, self.experiments, i+1, self.epochs, obj_SGD[i])
                # for j in range(max_iters):
                    # rand_idx = <int*> malloc(sizeof(int) * self.batch_size)
                    # flm.fill_random_ints(self.batch_size, rand_idx, 0, N - 1)
                    # self.fill_X_batch(X, dim, X_batch, rand_idx)
                    # self.fill_y_batch(y, y_batch, rand_idx)
                    # obj_SGD_iters[i * max_iters + j] += self.obj(N, dim, X, y, w, self.reg)
                    # # temp = w - w_star
                    # flm.vdSub(dim, w, self.w_star, temp)
                    # MSE[i * max_iters + j] += flm.dnrm2(dim, temp, 1) # Euclidian Norm
                    # step = self.step_size if self.decay == False else 0.1/(self.reg*(j+1000))
                    # # Place gradient vector in temp
                    # self.grad(self.batch_size, dim, X_batch, y_batch, w, self.reg, temp)
                    # # Multiply gradient vector by scalar `step`
                    # flm.dscal(dim, step, temp, 1)
                    # # Subtract scaled gradient from w
                    # flm.vdSub(dim, w, temp, w)
            # free(X_batch)
            # free(y_batch)
            # free(temp)
            # X_batch = NULL
            # y_batch = NULL
            # temp    = NULL

        # obj_SGD       = <double*> calloc(self.epochs, sizeof(double))
        # obj_SGD_iters = <double*> calloc(self.epochs * max_iters, sizeof(double))
        # MSE           = <double*> calloc(self.epochs * max_iters, sizeof(double))
        # w             = <double*> calloc(dim, sizeof(double))

        # for i in range(self.experiments):
            # for j in range(dim):
                # w[j] += _w[i * dim + j] / self.experiments
            # for j in range(self.epochs):
                # obj_SGD[j] += _obj_SGD[i * self.epochs + j] / self.experiments
                # for k in range(max_iters):
                    # obj_SGD_iters[k] += _obj_SGD_iters[i * (max_iters * self.epochs) + j * (max_iters) + k] / self.experiments
                    # MSE[k]           += _obj_SGD_iters[i * (max_iters * self.epochs) + j * (max_iters) + k] / self.experiments

        # free(_MSE)
        # free(_obj_SGD)
        # free(_obj_SGD_iters)
        # free(_w)

        # # For now, only returning final weights.
        # # WIll modify this to return MSE and obj_SGD as well
    
        # return w

    # @cython.wraparound(False)
    # @cython.boundscheck(False)
    # @cython.nonecheck(False)
    # @cython.initializedcheck(False)
    # @cython.cdivision(True)
    # cdef run_local(self, double* X, double*y, int N, int dim):
        # cdef int workers = self.workers
        # cdef int local_sgd_exp = self.experiments
        # cdef int num_split
        # cdef double *X_split
        # cdef double *y_split
        # cdef double *aggregate_w = <double*> malloc(sizeof(double) * dim)
        # cdef double **worker_results = <double**> malloc(sizeof(double*) * (workers))

        # # Each SGD call will only do a single experiment
        # self.experiments = 1
        # num_split = <int> (N / (workers))

        # cdef double* ttt = <double*> malloc(sizeof(double) * dim)

        # cdef int l
        # for l in range(workers):
            # worker_results[l] = NULL

        # openmp.omp_set_num_threads(workers)
        # START_OMP_PARALLEL_PRAGMA()
        # cdef double *single_result_w
        # cdef int worker, k, i, j
        # worker = openmp.omp_get_thread_num()
        # for k in range(local_sgd_exp):
            # for i in range(self.communications):
                # # for worker in prange(workers, nogil=True, num_threads=workers):
                # # printf("worker %d indexes: %d to %d\n", worker, worker*num_split, worker*num_split + num_split)
                # single_result_w = self.run(&X[worker*num_split],
                                               # &y[worker*num_split],
                                               # num_split,
                                               # dim)
                # worker_results[worker] = ttt
                # printf("Worker %d added result \n", worker)
                # # printf("work DONE\n")
                # START_OMP_BARRIER_PRAGMA()
                # # END_OMP_PRAGMA()
                # START_OMP_SINGLE_PRAGMA()
                # printf("avging\n")
                # for j in range(dim):
                    # aggregate_w[j] = 0
                # for j in range(workers):
                    # if worker_results[j] == NULL:
                        # continue
                    # printf("Added result from worker %d\n", j)
                    # flm.vdAdd(dim, aggregate_w, worker_results[j], aggregate_w)
                # flm.dscal(dim, (1.0/workers), aggregate_w, 1)
                # self.w_next = aggregate_w
                # END_OMP_PRAGMA() # single
        # END_OMP_PRAGMA() # parallel


        # free(worker_results)

        # return flarray.ptr_to_double_np_array_1d(aggregate_w, dim)
