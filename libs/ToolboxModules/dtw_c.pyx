import logging
import math
import numpy as np
cimport numpy as np
cimport cython
import cython
import ctypes
from cpython cimport array, bool
from cython import parallel
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free, abs, labs
from libc.stdio cimport printf
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, pow
from libc.stdint cimport intptr_t
from cpython.exc cimport PyErr_CheckSignals


logger = logging.getLogger("be.kuleuven.dtai.distance")


DTYPE = np.double
ctypedef np.double_t DTYPE_t
cdef double inf = np.inf

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def distance(np.ndarray[DTYPE_t, ndim=1] s1, np.ndarray[DTYPE_t, ndim=1] s2,
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0, int psi=0):
    """
    Dynamic Time Warping (keep compact matrix)
    :param s1: First sequence (np.array(np.float64))
    :param s2: Second sequence
    :param window: Only allow for shifts up to this amount away from the two diagonals
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param max_length_diff: Max length difference between the two sequences
    :param penalty: Cost incurrend when performing compression or expansion
    Returns: DTW distance
    """
    assert s1.dtype == DTYPE and s2.dtype == DTYPE
    cdef int r = len(s1)
    cdef int c = len(s2)
    if max_length_diff != 0 and abs(r-c) > max_length_diff:
        return inf
    if window == 0:
        window = max(r, c)
    if max_step == 0:
        max_step = inf
    else:
        max_step = max_step
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist *= max_dist
    penalty *= penalty
    cdef int length = min(c + 1, abs(r - c) + 2 * (window - 1) + 1 + 1 + 1)
    cdef np.ndarray[DTYPE_t, ndim=2] dtw = np.full((2, length), inf)
    # dtw[0, 0] = 0
    cdef int i
    for i in range(psi + 1):
        dtw[0, i] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int skip = 0
    cdef int skipp = 0
    cdef int i0 = 1
    cdef int i1 = 0
    cdef DTYPE_t d
    for i in range(r):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        skipp = skip
        skip = max(0, i - window + 1)
        i0 = 1 - i0
        i1 = 1 - i1
        dtw[i1 ,:] = inf
        j_start = max(0, i - max(0, r - c) - window + 1)
        j_end = min(c, i + max(0, c - r) + window)
        if dtw.shape[1] == c+ 1:
            skip = 0
        if psi != 0 and j_start == 0 and i < psi:
            dtw[i1, 0] = 0
        for j in range(j_start, j_end):
            d = abs(i - j)
            if d > max_step:
                continue
            dtw[i1, j + 1 - skip] = d + min(dtw[i0, j - skipp],
                                            dtw[i0, j + 1 - skipp] + penalty,
                                            dtw[i1, j - skip] + penalty)
            if dtw[i1, j + 1 - skip] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1, j + 1 - skip] = inf
                if prev_last_under_max_dist + 1 - skipp < j + 1 - skip:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            return inf
        if psi != 0 and j_end == len(s2) and len(s1) - 1 - i <= psi:
            psi_shortest = min(psi_shortest, dtw[i1, length - 1])
    if psi == 0:
        d = math.sqrt(dtw[i1, min(c, c + window - 1) - skip])
    else:
        ic = min(c, c + window - 1) - skip
        vc = dtw[i1, ic - psi:ic + 1]
        d = min(np.min(vc), psi_shortest)
        d = math.sqrt(d)
    # print(dtw)
    return d


def distance_nogil(double[:] s1, double[:] s2,
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0, int psi=0):
    """DTW distance.
    See distance(). This calls a pure c dtw computation that avoids the GIL.
    :param s1: First sequence (buffer of doubles)
    :param s2: Second sequence (buffer of doubles)
    """
    #return distance_nogil_c(s1, s2, len(s1), len(s2),
    # If the arrays (memoryviews) are not C contiguous, the pointer will not point to the correct array
    if isinstance(s1, (np.ndarray, np.generic)):
        if not s1.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 1 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s1 = s1.copy()
    if isinstance(s2, (np.ndarray, np.generic)):
        if not s2.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 2 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s2 = s2.copy()
    return distance_nogil_c(&s1[0], &s2[0], len(s1), len(s2),
                            window, max_dist, max_step, max_length_diff, penalty, psi)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.infer_types(False)
cdef double distance_nogil_c(
             double *s1, double *s2,
             int r, # len_s1
             int c, # len_s2
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0, int psi=0) nogil:
    """DTW distance.
    See distance(). This is a pure c dtw computation that avoid the GIL.
    """
    #printf("%i, %i\n", r, c)
    if max_length_diff != 0 and abs(r-c) > max_length_diff:
        return inf
    if window == 0:
        window = max(r, c)
    if max_step == 0:
        max_step = inf
    else:
        max_step = max_step
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist = pow(max_dist, 2)
    penalty = pow(penalty, 2)
    cdef int length = min(c+1,abs(r-c) + 2*(window-1) + 1 + 1 + 1)
    #printf("length (c) = %i\n", length)
    #cdef array.array dtw_tpl = array.array('d', [])
    #cdef array.array dtw
    #dtw = array.clone(dtw_tpl, length*2, zero=False)
    cdef double * dtw
    dtw = <double *> malloc(sizeof(double) * length * 2)
    cdef int i
    cdef int j
    for j in range(length*2):
        dtw[j] = inf
    # dtw[0] = 0
    for i in range(psi + 1):
        dtw[i] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int skip = 0
    cdef int skipp = 0
    cdef int i0 = 1
    cdef int i1 = 0
    cdef int minj
    cdef int maxj
    cdef double minv
    cdef DTYPE_t d
    cdef double tempv
    cdef double psi_shortest = inf
    cdef int iii
    for i in range(r):
        if i % 1024 == 0:
                with gil:
                    PyErr_CheckSignals()
        #printf("[ ")
        #for iii in range(length):
        #    printf("%f ", dtw[iii])
        #printf("\n")
        #for iii in range(length,length*2):
        #    printf("%f ", dtw[iii])
        #printf("]\n")
        #
        if last_under_max_dist == -1:
            prev_last_under_max_dist = inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        maxj = r - c
        if maxj < 0:
            maxj = 0
        maxj = i - maxj - window + 1
        if maxj < 0:
            maxj = 0
        skipp = skip
        skip = maxj
        i0 = 1 - i0
        i1 = 1 - i1
        for j in range(length):
            dtw[length * i1 + j] = inf
        if length == c + 1:
            skip = 0
        minj = c - r
        if minj < 0:
            minj = 0
        minj = i + minj + window
        if minj > c:
            minj = c
        if psi != 0 and maxj == 0 and i < psi:
            dtw[i1*length + 0] = 0
        for j in range(maxj, minj):
            printf('s1[i] = s1[%i] = %f , s2[j] = s2[%i] = %f\n', i, s1[i], j, s2[j])
            d = abs(i - j)
            if d > max_step:
                continue
            minv = dtw[i0*length + j - skipp]
            tempv = dtw[i0*length + j + 1 - skipp] + penalty
            if tempv < minv:
                minv = tempv
            tempv = dtw[i1*length + j - skip] + penalty
            if tempv < minv:
                minv = tempv
            #printf('d = %f, minv = %f\n', d, minv)
            dtw[i1 * length + j + 1 - skip] = d + minv
            #
            #printf('%i, %i, %i\n',i0*length + j - skipp,i0*length + j + 1 - skipp,i1*length + j - skip)
            #printf('%f, %f, %f\n',dtw[i0*length + j - skipp],dtw[i0*length + j + 1 - skipp],dtw[i1*length + j - skip])
            #printf('i=%i, j=%i, d=%f, skip=%i, skipp=%i\n',i,j,d,skip,skipp)
            #printf("[ ")
            #for iii in range(length):
            #    printf("%f ", dtw[iii])
            #printf("\n")
            #for iii in range(length,length*2):
            #    printf("%f ", dtw[iii])
            #printf("]\n")
            #
            if dtw[i1*length + j + 1 - skip] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1*length + j + 1 - skip] = inf
                if prev_last_under_max_dist + 1 - skipp < j + 1 - skip:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            free(dtw)
            return inf

        if psi != 0 and minj == c and r - 1 - i <= psi:
            if dtw[i1*length + length - 1] < psi_shortest:
                psi_shortest = dtw[i1*length + length - 1]

        # printf("[ ")
        # for iii in range(i1*length,i1*length + length):
        #    printf("%f ", dtw[iii])
        # printf("]\n")

    # print(dtw)
    if window - 1 < 0:
        c = c + window - 1
    cdef double result = sqrt(dtw[length * i1 + c - skip])
    if psi != 0:
        for i in range(c - skip - psi, c - skip + 1):  # iterate over vci
            if dtw[i1*length + i] < psi_shortest:
                psi_shortest = dtw[i1*length + i]
        result = sqrt(psi_shortest)
    free(dtw)
    return result


def warping_paths_nogil(double[:, :] dtw, double[:] s1, double[:] s2,
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0, int psi=0):
    """DTW warping paths.
    See distance(). This calls a pure c dtw computation that avoids the GIL.
    :param s1: First sequence (buffer of doubles)
    :param s2: Second sequence (buffer of doubles)
    """
    r = len(s1)
    c = len(s2)
    #return distance_nogil_c(s1, s2, len(s1), len(s2),
    # If the arrays (memoryviews) are not C contiguous, the pointer will not point to the correct array
    if isinstance(s1, (np.ndarray, np.generic)):
        if not s1.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 1 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s1 = s1.copy()
    if isinstance(s2, (np.ndarray, np.generic)):
        if not s2.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 2 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s2 = s2.copy()
    # if not isinstance(dtw, (np.ndarray, np.generic)):
    #     raise Exception("Warping paths datastructure needs to be a numpy array")
    # if not dtw.base.flags.c_contiguous:
    #     raise Exception("Warping paths datastructure is not C contiguous")
    result = warping_paths_nogil_c(&dtw[0, 0], &s1[0], &s2[0], len(s1), len(s2),
                                   window, max_dist, max_step, max_length_diff, penalty, psi, True)

    if psi == 0:
        return dtw[r, min(c, c + window - 1)]

    vr = dtw[r - psi:r, c]
    vc = dtw[r, c - psi:c]
    mir = np.argmin(vr)
    mic = np.argmin(vc)
    if vr[mir] < vc[mic]:
        dtw[r - psi + mir + 1:r + 1, c] = -1
        d = vr[mir]
    else:
        dtw[r, c - psi + mic + 1:c + 1] = -1
        d = vc[mic]
    return d


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.infer_types(False)
cdef double warping_paths_nogil_c(
            double *dtw, double *s1, double *s2,
            int r, # len_s1
            int c, # len_s2
            int window=0, double max_dist=0,
            double max_step=0, int max_length_diff=0, double penalty=0, int psi=0,
            int do_sqrt=0) nogil:
    """DTW warping paths.
    See warping_paths(). This is a pure c dtw computation that avoids the GIL.
    """
    # printf("%i, %i\n", r, c)
    if max_length_diff != 0 and abs(r-c) > max_length_diff:
        return inf
    if window == 0:
        window = max(r, c)
    if max_step == 0:
        max_step = inf
    else:
        max_step = max_step
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist = pow(max_dist, 2)
    penalty = pow(penalty, 2)
    cdef int i
    cdef int j
    for j in range(r * c):
        dtw[j] = inf
    # dtw[0] = 0
    for i in range(psi + 1):
        dtw[i] = 0
        dtw[i * (c + 1)] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int i0 = 1
    cdef int i1 = 0
    cdef int minj
    cdef int maxj
    cdef double minv
    cdef DTYPE_t d
    cdef double tempv
    cdef int iii
    for i in range(r):
        # printf("iter %i/%i\n", i, r)
        #
        #printf("[ ")
        #for iii in range(length):
        #    printf("%f ", dtw[iii])
        #printf("\n")
        #for iii in range(length,length*2):
        #    printf("%f ", dtw[iii])
        #printf("]\n")
        #
        if last_under_max_dist == -1:
            prev_last_under_max_dist = inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        i0 = i
        i1 = i + 1
        maxj = r - c
        if maxj < 0:
            maxj = 0
        maxj = i - maxj - window + 1
        if maxj < 0:
            maxj = 0
        minj = c - r
        if minj < 0:
            minj = 0
        minj = i + minj + window
        if minj > c:
            minj = c
        for j in range(maxj, minj):
            # printf('s1[i] = s1[%i] = %f , s2[j] = s2[%i] = %f\n', i, s1[i], j, s2[j])
            d = abs(i - j)
            if d > max_step:
                continue
            minv = dtw[i0 * (c + 1) + j]
            tempv = dtw[i0 * (c + 1) + j + 1] + penalty
            if tempv < minv:
                minv = tempv
            tempv = dtw[i1 * (c + 1) + j] + penalty
            if tempv < minv:
                minv = tempv
            # printf('d = %f, minv = %f\n', d, minv)
            dtw[i1 * (c + 1) + j + 1] = d + minv
            #
            #printf('%i, %i, %i\n',i0*length + j - skipp,i0*length + j + 1 - skipp,i1*length + j - skip)
            #printf('%f, %f, %f\n',dtw[i0*length + j - skipp],dtw[i0*length + j + 1 - skipp],dtw[i1*length + j - skip])
            #printf('i=%i, j=%i, d=%f, skip=%i, skipp=%i\n',i,j,d,skip,skipp)
            #printf("[ ")
            #for iii in range(length):
            #    printf("%f ", dtw[iii])
            #printf("\n")
            #for iii in range(length,length*2):
            #    printf("%f ", dtw[iii])
            #printf("]\n")
            #
            if dtw[i1 * (c + 1) + j + 1] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1 * (c + 1) + j + 1] = inf
                if prev_last_under_max_dist < j + 1:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            if do_sqrt == 1:
                for i in range((r + 1) * (c + 1)):
                    dtw[i] = sqrt(dtw[i])
            return inf

        # printf("[ ")
        # for iii in range(i1*length,i1*length + length):
        #    printf("%f ", dtw[iii])
        # printf("]\n")

    if do_sqrt == 1:
        for i in range((r + 1) * (c + 1)):
            dtw[i] = sqrt(dtw[i])
    return 0



