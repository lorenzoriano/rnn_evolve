import numpy as np
import math
import cython
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef extern from "math.h":
        double tanh(double)
cdef extern from "math.h":
        float fabsf(float x)

@cython.boundscheck(False)
def c_net_out(np.ndarray[DTYPE_t, ndim=2] xs not None,
              np.ndarray[DTYPE_t, ndim=1] chromosome not None, int input_size, int output_size, int hidden_size):

        cdef int bias_size = hidden_size + output_size
        cdef int total_size = bias_size + input_size
        cdef int bias_starting_point = len(chromosome) - bias_size
        cdef np.ndarray[DTYPE_t, ndim=1] state = np.zeros(total_size,  dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] newstate = np.zeros(total_size,  dtype = DTYPE)
        cdef Py_ssize_t i, j, t
        cdef int timesteps = xs.shape[0]
        cdef int chromosome_len = len(chromosome)
        cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(output_size,  dtype = DTYPE)

        for t in range(timesteps):
            newstate = state.copy()
            for i in range(input_size):
                state[i] = xs[t, i]
                newstate[i] = xs[t, i]

            for i in range(input_size,  total_size):
                newstate[i] = chromosome[chromosome_len - bias_size + i - input_size]
                for j in range(0,  total_size):
                    #x[i] += w[i,j]*tanh(x[j])
                    newstate[i] += chromosome[ (i-input_size) * (total_size) + j] * tanh(state[j])

            state = newstate.copy()
        j = 0
        for i in range(input_size, input_size + output_size):
            out[j] = tanh(state[i])
            j = j+1

        return out

@cython.boundscheck(False)
def c_net_error(np.ndarray[DTYPE_t, ndim=2] xs not None,  np.ndarray[DTYPE_t, ndim=2] ys not None,
                np.ndarray[DTYPE_t, ndim=1] chromosome not None, int input_size, int output_size, int hidden_size):

        cdef int bias_size = hidden_size + output_size
        cdef int total_size = bias_size + input_size
        cdef int bias_starting_point = len(chromosome) - bias_size
        cdef np.ndarray[DTYPE_t, ndim=1] state = np.zeros(total_size,  dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] newstate = np.zeros(total_size,  dtype = DTYPE)
        cdef Py_ssize_t i, j, t
        cdef int timesteps = xs.shape[0]
        cdef int chromosome_len = len(chromosome)

        cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(output_size,  dtype = DTYPE)
        cdef double sse = 0
        cdef double error_t = 0
        cdef double __diff__ = 0

        for t in range(timesteps):
            newstate = state.copy()
            for i in range(input_size):
                state[i] = xs[t, i]
                newstate[i] = xs[t, i]

            for i in range(input_size,  total_size):
                newstate[i] = chromosome[chromosome_len - bias_size + i - input_size]
                for j in range(0,  total_size):
                    #x[i] += w[i,j]*tanh(x[j])
                    newstate[i] += chromosome[ (i-input_size) * (total_size) + j] * tanh(state[j])
            state = newstate.copy()

            #calculating the output
            j = 0
            for i in range(input_size, input_size + output_size):
                out[j] = tanh(state[i])
                j = j+1

            #calculating the error
            error_t = 0
            for i in range(output_size):
                __diff__ = out[i] - ys[t, i]
                error_t += __diff__ * __diff__
            sse += error_t

        return sse

@cython.boundscheck(False)
def c_eval_func(xs, np.ndarray[DTYPE_t, ndim=2]  ys,  net):

    cdef Py_ssize_t t, i
    cdef int timesteps = xs.shape[0]
    cdef int output_size = ys.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] net_out

    cdef double sum_squared_err = 0
    cdef double err_t = 0
    cdef double __diff__ = 0

    for t in range(timesteps):
        net_out = net(xs[t, :]).ravel()
        err_t = 0
        for i in range(output_size):
            __diff__ = ys[t, i] - net_out[i]
            err_t = err_t + __diff__*__diff__
        sum_squared_err = sum_squared_err + err_t

    return sum_squared_err
