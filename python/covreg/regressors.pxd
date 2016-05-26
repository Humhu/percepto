# Declaration of Cython extension regressor classes

from CovregRegressors cimport LinRegd as cLinRegd
from CovregRegressors cimport ExpLinRegd as cExpLinRegd
from CovregRegressors cimport ModCholRegd as cModCholRegd
from CovregRegressors cimport DampedModCholRegd as cDampedModCholRegd
from CovregRegressors cimport TransModCholRegd as cTransModCholRegd
from CovregRegressors cimport AffineModCholRegd as cAffineModCholRegd
from CovregRegressors cimport SummedModCholRegd as cSummedModCholRegd
from CovregRegressors cimport ChainedModCholRegd as cChainedModCholRegd

from libcpp.vector cimport vector

cimport numpy as np

# cdef class LinearRegressor:
#     cdef cLinRegd* _regressor

#     cpdef unsigned int InputDim( self )
#     cpdef unsigned int OutputDim( self )
#     cpdef unsigned int ParameterDim( self )
#     cpdef SetParameters( self, np.ndarray parameters )
#     cpdef np.ndarray GetParameters( self )
#     cpdef np.ndarray Evaluate( self, np.ndarray features )

# cdef class ExponentialRegressor:
#     cpdef cExpLinRegd* _regressor
#     cpdef unsigned int InputDim( self )
#     cpdef unsigned int OutputDim( self )
#     cpdef unsigned int ParameterDim( self )
#     cpdef SetParameters( self, np.ndarray parameters )
#     cpdef np.ndarray GetParameters( self )
#     cpdef np.ndarray Evaluate( self, np.ndarray features )

cdef class ModCholRegressor:
    cpdef cModCholRegd* _regressor
    cpdef unsigned int InputDim( self )
    cpdef unsigned int OutputDim( self )
    cpdef unsigned int ParameterDim( self )
    cpdef SetParameters( self, parameters )
    cpdef GetParameters( self )
    cpdef np.ndarray Evaluate( self, features )

cdef ModCholParamsToPy( cModCholRegd.ParameterType cParams )
cdef cModCholRegd.ParameterType PyToModCholParams( pParams )

cdef ChainedModCholParamsToPy( vector[cModCholRegd.ParameterType] cParams )
cdef vector[cModCholRegd.ParameterType] PyToChainedModCholParams( pParams )

cdef ModCholInputToPy( cModCholRegd.InputType cInput )
cdef cModCholRegd.InputType PyToModCholInput( pInput )

cdef DampedModCholInputToPy( cDampedModCholRegd.InputType cInput )
cdef cDampedModCholRegd.InputType PyToDampedModCholInput( pInput )

cdef TransModCholInputToPy( cTransModCholRegd.InputType cInput )
cdef cTransModCholRegd.InputType PyToTransModCholInput( pInput )

cdef AffineModCholInputToPy( cAffineModCholRegd.InputType cInput )
cdef cAffineModCholRegd.InputType PyToAffineModCholInput( pInput )

cdef ChainedModCholInputToPy( cChainedModCholRegd.InputType cInput )
cdef cChainedModCholRegd.InputType PyToChainedModCholInput( pInput )