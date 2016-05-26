# distutils: language = c++

from EigenInterface cimport MatrixXd
from EigenInterface cimport EigenToNumpy as EigToNp
from EigenInterface cimport NumpyToEigen as NpToEig

from CovregRegressors cimport LinRegd as cLinRegd
from CovregRegressors cimport LinRegdDefaults as cLinRegdDefaults
from CovregRegressors cimport ExpLinRegd as cExpLinRegd
from CovregRegressors cimport ExpLinRegdDefaults as cExpLinRegdDefaults
from CovregRegressors cimport ModCholRegd as cModCholRegd
from CovregRegressors cimport ModCholRegdDefaults as cModCholRegdDefaults

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from collections import namedtuple
import numpy as np
cimport numpy as np

# cdef class LinearRegressor:
#     """A linear regressor class wrapper around a C++ implementation."""
    
#     def __cinit__( self, np.ndarray parameters ):
#         self._regressor = new cLinRegd( NpToEig( np.asanyarray(parameters) ) )

#     def __dealloc_( self ):
#         del self._regressor

#     cpdef unsigned int InputDim( self ):
#         return self._regressor.InputDim()

#     cpdef unsigned int OutputDim( self ):
#         return self._regressor.OutputDim()

#     cpdef unsigned int ParameterDim( self ):
#         return self._regressor.ParameterDim()

#     cpdef SetParameters( self, np.ndarray parameters ):
#         self._regressor.SetParameters( NpToEig( np.asanyarray(parameters) ) )

#     cpdef np.ndarray GetParameters( self ):
#         return EigToNp( self._regressor.GetParameters() )

#     cpdef np.ndarray Evaluate( self, np.ndarray features ):
#         cdef MatrixXd f = NpToEig( np.asanyarray(features) )
#         return EigToNp( self._regressor.Evaluate( f ) 
#     )

# cpdef create_default_linear( unsigned int inputDim, unsigned int outputDim ):
#     """Creates a LinearRegressor of the specified in/out dimensions with zero parameters."""
#     cdef MatrixXd defaultParams = cLinRegdDefaults( inputDim, outputDim )
#     return LinearRegressor( EigToNp( defaultParams ) )

# cdef class ExponentialRegressor:
#     """A linear-exponential regressor class wrapper around a C++ implementation."""

#     def __cinit__( self, np.ndarray parameters ):
#         self._regressor = new cExpLinRegd( NpToEig( np.asanyarray(parameters) ) )

#     def __dealloc__( self ):
#         del self._regressor

#     cpdef unsigned int InputDim( self ):
#         return self._regressor.InputDim()

#     cpdef unsigned int OutputDim( self ):
#         return self._regressor.OutputDim()

#     cpdef unsigned int ParameterDim( self ):
#         return self._regressor.ParameterDim()

#     cpdef SetParameters( ExponentialRegressor self, 
#                          np.ndarray parameters ):
#         self._regressor.SetParameters( NpToEig( np.asanyarray(parameters) ) )

#     cpdef np.ndarray GetParameters( ExponentialRegressor self ):
#         return EigToNp( self._regressor.GetParameters() )

#     cpdef np.ndarray Evaluate( ExponentialRegressor self,
#                                np.ndarray features ):
#         cdef MatrixXd f = NpToEig( np.asanyarray(features) )
#         return EigToNp( self._regressor.Evaluate( f ) )

# cpdef create_default_exponential( unsigned int inputDim, unsigned int outputDim ):
#     """Creates an ExponentialRegressor of the specified in/out dimensions with zero parameters."""
#     cdef MatrixXd defaultParams = cExpLinRegdDefaults( inputDim, outputDim )
#     return ExponentialRegressor( EigToNp( defaultParams ) )

cdef class ModCholRegressor:
    """A positive-definite matrix regressor that wraps a C++ implementation."""

    def __cinit__( self, parameters, np.ndarray offset ):
        self._regressor = new cModCholRegd( PyToModCholParams( parameters ), 
                                             NpToEig( np.asanyarray(offset) ) )

    cpdef unsigned int InputDim( self ):
        return self._regressor.InputDim()

    cpdef unsigned int OutputDim( self ):
        return self._regressor.OutputDim()

    cpdef unsigned int ParameterDim( self ):
        return self._regressor.ParameterDim()

    cpdef SetParameters( self, parameters ):
        self._regressor.SetParameters( PyToModCholParams( parameters ) )

    cpdef GetParameters( self ):
        return ModCholParamsToPy( self._regressor.GetParameters() )

    cpdef np.ndarray Evaluate( self, input ):
        return EigToNp( self._regressor.Evaluate( PyToModCholInput( input ) ) )

cpdef create_default_modchol( unsigned int lInputDim, unsigned int dInputDim,
                              unsigned int outputDim, np.ndarray offset ):
    """Creates a ModCholRegressor of the specified in/out dimensions with zero parameters."""
    cdef cModCholRegd.ParameterType p = cModCholRegdDefaults( lInputDim, dInputDim, outputDim )
    return ModCholRegressor( parameters = ModCholParamsToPy( p ),
                             offset = offset )

# Conversions to and from C-type structs
ModCholRegressorParameters = namedtuple( 'ModCholRegressorParameters', 
                                         ['lParameters', 'dParameters'] )

cdef ModCholParamsToPy( cModCholRegd.ParameterType cParams ):
    """Converts a C-format modified Cholesky parameter struct to Python named struct."""
    return ModCholRegressorParameters( lParameters = EigToNp( cParams.lParameters ),
                                       dParameters = EigToNp( cParams.dParameters ) )

cdef cModCholRegd.ParameterType PyToModCholParams( pParams ):
    """Converts a Python named struct to C-format modified Cholesky parameter struct."""
    cdef cModCholRegd.ParameterType cParams
    cParams.lParameters = NpToEig( np.asanyarray(pParams.lParameters) )
    cParams.dParameters = NpToEig( np.asanyarray(pParams.dParameters) )
    return cParams

cdef ChainedModCholParamsToPy( vector[cModCholRegd.ParameterType] cParams ):
    """Converts a C-format chain modified Cholesksy parameter struct to
    Python list."""
    pParams = []
    for i in range( cParams.size() ):
        pParams.append( ModCholParamsToPy(cParams[i]) )
    return pParams

cdef vector[cModCholRegd.ParameterType] PyToChainedModCholParams( pParams ):
    """Converts a Python list to C-format modified Cholesky parameter
    struct."""
    cdef vector[cModCholRegd.ParameterType] cParams
    cParams.reserve( len( pParams ) )
    for p in pParams:
        cParams.push_back( PyToModCholParams( p ) )
    return cParams

# Definition for MCReg input named tuple
ModCholInput = namedtuple( 'ModCholInput', 
                           ['lInput', 'dInput'])

cdef ModCholInputToPy( cModCholRegd.InputType cInput ):
    """Converts a C-format Modififed Cholesky input struct to a Python 
    named struct."""
    return ModCholInput( lInput = EigToNp( cInput.lInput ),
                         dInput = EigToNp( cInput.dInput ) )

cdef cModCholRegd.InputType PyToModCholInput( pInput ):
    """Converts a Python named struct to a C-format modified Cholesky 
    input struct."""
    cdef cModCholRegd.InputType cInput
    cInput.lInput = NpToEig( np.asanyarray(pInput.lInput) )
    cInput.dInput = NpToEig( np.asanyarray(pInput.dInput) )
    return cInput

DampedModCholInput = namedtuple( 'DampedModCholInput',
                                 ['lInput', 'dInput', 'offset'] )

cdef DampedModCholInputToPy( cDampedModCholRegd.InputType cInput ):
    """Converts a C-format damped modified Cholesky input struct to a Python 
    named struct."""
    return DampedModCholInput( lInput = EigToNp( cInput.input.lInput ),
                               dInput = EigToNp( cInput.input.dInput ),
                               offset = EigToNp( cInput.offset ) )

cdef cDampedModCholRegd.InputType PyToDampedModCholInput( pInput ):
    """Converts a Python named struct to a C-format damped modified Cholesky 
    input struct."""
    cdef cDampedModCholRegd.InputType cInput
    cInput.input = PyToModCholInput( pInput )
    cInput.offset = NpToEig( np.asanyarray(pInput.offset) )
    return cInput

TransModCholInput = namedtuple( 'TransModCholInput',
                                ['lInput', 'dInput', 'transform'] )

cdef TransModCholInputToPy( cTransModCholRegd.InputType cInput ):
    """Converts a C-format transformed modified Cholesky input struct
    to a Python named struct."""
    return TransModCholInput( lInput = EigToNp( cInput.input.lInput ),
                              dInput = EigToNp( cInput.input.dInput ),
                              transform = EigToNp( cInput.transform ) )

cdef cTransModCholRegd.InputType PyToTransModCholInput( pInput ):
    """Converts a Python named struct to a C-format transformed modified
    Cholesky input struct."""
    cdef cTransModCholRegd.InputType cInput
    cInput.input = PyToModCholInput( pInput )
    cInput.transform = NpToEig( np.asanyarray(pInput.transform) )
    return cInput


AffineModCholInput = namedtuple( 'AffineModCholInput',
                                 ['lInput', 'dInput', 'offset', 'transform'] )

cdef AffineModCholInputToPy( cAffineModCholRegd.InputType cInput ):
    """Converts a C-format affine modified Cholesky input struct
    to a Python named struct."""
    return AffineModCholInput( lInput = EigToNp( cInput.input.input.lInput ),
                               dInput = EigToNp( cInput.input.input.dInput ),
                               transform = EigToNp( cInput.input.transform ),
                               offset = EigToNp( cInput.offset ) )

cdef cAffineModCholRegd.InputType PyToAffineModCholInput( pInput ):
    """Converts a Python named struct to a C-format affine modified
    Cholesky input struct."""
    cdef cAffineModCholRegd.InputType cInput
    cInput.input = PyToTransModCholInput( pInput )
    cInput.offset = NpToEig( np.asanyarray(pInput.offset) )
    return cInput

ChainedModCholInput = namedtuple( 'ChainedModCholInput',
                                  ['inputs', 'offset'] )

cdef ChainedModCholInputToPy( cChainedModCholRegd.InputType cInput ):
    """Converts a C-format chain modified Cholesky input struct
    to a Python named struct."""
    subInputs = []
    for i in range(cInput.input.size()):
        if cInput.input[i].valid:
            subInputs.append( TransModCholInputToPy( cInput.input[i].input ) )
        else:
            subInputs.append( None )
    return ChainedModCholInput( inputs = subInputs, 
                                offset = EigToNp( cInput.offset ) )

cdef cChainedModCholRegd.InputType PyToChainedModCholInput( pInput ):
    """Converts a Python named struct to a C-format chain modified
    Cholesky input struct."""

    cdef cChainedModCholRegd.InputType cInput
    cInput.input.reserve( len( pInput ) )
    cdef cSummedModCholRegd.InputTuple cIn
    for p in pInput.inputs:
        if p is None:
            cIn.valid = False
        else:
            cIn.valid = True
            cIn.input = PyToTransModCholInput(p)
        cInput.input.push_back( cIn )
        cInput.offset = NpToEig( np.asanyarray(pInput.offset) )
    return cInput
