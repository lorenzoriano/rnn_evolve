cimport numpy as np
import cython
cdef extern from "math.h":
    double log(double)

#% FUNCTION: kolmogorov.m
#% DATE: 9th Feb 2005
#% AUTHOR: Stephen Faul (stephenf@rennes.ucc.ie)
#%
#% Function for estimating the Kolmogorov Complexity as per:
#% "Easily Calculable Measure for the Complexity of Spatiotemporal Patterns"
#% by F Kaspar and HG Schuster, Physical Review A, vol 36, num 2 pg 842
#%
#% Input is a digital string, so conversion from signal to a digital stream
#% must be carried out a priori

#def kolmogorov(list s):
@cython.boundscheck(False)
def kolmogorov(np.ndarray[int, ndim=1] s not None):
    cdef int n=len(s)
    cdef int c=1
    cdef int l=1

    cdef int i=0
    cdef int k=1
    cdef int k_max=1
    cdef char stop = 0

    cdef char __ele1__, __ele2__

    while stop == 0:
        if s[i+k -1 ] != s[l+k -1]:
            if k>k_max:
                k_max=k
            i=i+1

            if i==l:
                c=c+1
                l=l+k_max
                if l+1>n:
                    stop=1
                else:
                    i=0
                    k=1
                    k_max=1
            else:
                k=1
        else:
            k=k+1
            if l+k>n:
                c=c+1
                stop=1

    cdef double b=float(n)/ (log(n) / log(2.))

#    % a la Lempel and Ziv (IEEE trans inf theory it-22, 75 (1976),
#    % h(n)=c(n)/b(n) where c(n) is the kolmogorov complexity
#    % and h(n) is a normalised measure of complexity.
    cdef double complexity=float(c)/float(b)

    return complexity

