import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef ptr_to_double_np_array_1d(double *ptr, int sz):
    cdef np.npy_intp s = sz;
    arr = np.PyArray_SimpleNewFromData(1, &s, np.NPY_FLOAT64, <void*> ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr
