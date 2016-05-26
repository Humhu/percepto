# Cython declarations for basic dynamic-sized Eigen matrices
# Since we are only copying to and from numpy.ndarray, we only need
# size-getters and element access

import numpy as np
cimport numpy as np

cdef extern from "Eigen/Dense" namespace "Eigen":
    
    # Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic>
    cdef cppclass MatrixXd:
        MatrixXd() except +
        MatrixXd( int rows, int cols ) except +
        MatrixXd( const MatrixXd& other ) except +
        int rows()
        int cols()
        int size()
        double& element "operator()"(int row, int col)

cdef inline np.ndarray EigenToNumpy( const MatrixXd& mat ):
    cdef np.ndarray retval = np.empty( [mat.rows(), mat.cols()] )
    cdef int i, j
    for i in range(mat.rows()):
        for j in range(mat.cols()):
            retval[i,j] = mat.element(i,j)
    return retval

cdef inline void SetElement( MatrixXd& m, int row, int col, double el ):
    cdef double* d = &(m.element(row,col))
    d[0] = el

cdef inline MatrixXd NumpyToEigen( np.ndarray array ):
    if array.ndim > 2 :
        raise TypeError( 'Cannot convert ndarray with more than 2 dimensions!' )

    cdef int i, j
    cdef MatrixXd retval
    if array.ndim == 1:
        retval = MatrixXd( array.shape[0], 1 )
        for i in range(array.shape[0]):
            SetElement( retval, i, 0, array[i])
        return retval
    elif array.ndim == 2:
        retval = MatrixXd( array.shape[0], array.shape[1] )
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                SetElement( retval, i, j, array[i,j] )
        return retval