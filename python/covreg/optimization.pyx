# distutils: language = c++
# distutils: libraries = covreg

from EigenInterface cimport MatrixXd
from EigenInterface cimport EigenToNumpy as EigToNp
from EigenInterface cimport NumpyToEigen as NpToEig

from CovregRegressors cimport ModCholRegd as cModCholRegd
from regressors cimport ModCholRegressor
from regressors cimport PyToModCholParams
from regressors cimport ModCholParamsToPy
from regressors cimport ChainedModCholParamsToPy

from regressors cimport PyToModCholInput
from regressors cimport ModCholInputToPy
from regressors cimport PyToDampedModCholInput
from regressors cimport DampedModCholInputToPy
from regressors cimport PyToTransModCholInput
from regressors cimport TransModCholInputToPy
from regressors cimport PyToAffineModCholInput
from regressors cimport AffineModCholInputToPy
from regressors cimport PyToChainedModCholInput
from regressors cimport ChainedModCholInputToPy

from CovregOptimization cimport NLOptParameters as cOptParameters
from CovregOptimization cimport FittingResults as cFittingResults
from CovregOptimization cimport DirectLikelihoodData as cDirectLikelihoodData
from CovregOptimization cimport DampedLikelihoodData as cDampedLikelihoodData
from CovregOptimization cimport TransLikelihoodData as cTransLikelihoodData
from CovregOptimization cimport AffineLikelihoodData as cAffineLikelihoodData
from CovregOptimization cimport ChainedLikelihoodData as cChainedLikelihoodData

from CovregOptimization cimport batch_directll_fit as c_batch_direct_fit
from CovregOptimization cimport batch_dampedll_fit as c_batch_damped_fit
from CovregOptimization cimport batch_transll_fit as c_batch_trans_fit
from CovregOptimization cimport batch_affinell_fit as c_batch_affine_fit
from CovregOptimization cimport batch_chainll_fit as c_batch_chain_fit

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from collections import namedtuple
import numpy as np
cimport numpy as np
import math

# We use a single likelihood data tuple for all underlying types of data
LikelihoodData = namedtuple( "LikelihoodData",
                             ["sample", "input"] )

cdef cDirectLikelihoodData PyToDirectLikelihoodData( pData ) except +:
    """Converts a Python named tuple to a C-struct of direct likelihood
    data."""
    cdef cDirectLikelihoodData cData
    cData.sample = NpToEig( pData.sample )
    cData.input = PyToModCholInput( pData.input )
    return cData

cdef DirectLikelihoodDataToPy( cDirectLikelihoodData cData ):
    """Converts a C-struct of direct likelihood data to a Python 
    named tuple."""
    return LikelihoodData( sample = EigToNp( cData.sample ),
                           input = ModCholInputToPy( cData.input ) )

cdef cDampedLikelihoodData PyToDampedLikelihoodData( pData ) except +:
    """Converts a Python named tuple to a C-struct of damped likelihood
    data."""
    cdef cDampedLikelihoodData cData
    cData.sample = NpToEig( pData.sample )
    cData.input = PyToDampedModCholInput( pData.input )
    return cData

cdef DampedLikelihoodDataToPy( cDampedLikelihoodData cData ):
    """Converts a C-struct of damped likelihood data to a Python 
    named tuple."""
    return LikelihoodData( sample = EigToNp( cData.sample ),
                           input = DampedModCholInputToPy( cData.input ) )

cdef cTransLikelihoodData PyToTransLikelihoodData( pData ) except +:
    """Converts a Python named tuple to a C-struct of transformed 
    likelihood data."""
    cdef cTransLikelihoodData cData
    cData.sample = NpToEig( pData.sample )
    cData.input = PyToTransModCholInput( pData.input )
    return cData

cdef TransLikelihoodDataToPy( cTransLikelihoodData cData ):
    """Converts a C-struct of transformed likelihood data to a Python 
    named tuple."""
    return LikelihoodData( sample = EigToNp( cData.sample ),
                           input = TransModCholInputToPy( cData.input ) )

cdef cAffineLikelihoodData PyToAffineLikelihoodData( pData ) except +:
    """Converts a Python named tuple to a C-struct of affine
    likelihood data."""
    cdef cAffineLikelihoodData cData
    cData.sample = NpToEig( pData.sample )
    cData.input = PyToAffineModCholInput( pData.input )
    return cData

cdef AffineLikelihoodDataToPy( cAffineLikelihoodData cData ):
    """Converts a C-struct of affine likelihood data to a Python 
    named tuple."""
    return LikelihoodData( sample = EigToNp( cData.sample ),
                           input = AffineModCholInputToPy( cData.input ) )

cdef cChainedLikelihoodData PyToChainedLikelihoodData( pData ) except +:
    """Converts a Python named tuple to a C-struct of chain 
    likelihood data."""
    cdef cChainedLikelihoodData cData
    cData.sample = NpToEig( pData.sample )
    cData.input = PyToChainedModCholInput( pData.input )
    return cData

cdef ChainedLikelihoodDataToPy( cChainedLikelihoodData cData ):
    """Converts a C-struct of chain likelihood data to a Python
    named tuple."""
    return LikelihoodData( sample = EigToNp( cData.sample ),
                           input = ChainedModCholInputToPy( cData.input ) )

OptimizationResults = namedtuple( "OptimizationResults",
                                  ["finalObjective", "numObjectiveEvals",
                                   "numGradientEvals", "totalElapsedSecs",
                                   "totalObjectiveSecs", "totalGradientSecs"] )

cdef cOptParameters PyToOptParameters( dict pParams ):
    """Converts a Python dictionary of optimization parameters to a C
    struct."""
    # Values should be initialized to default
    cdef cOptParameters cParams
    for (key, value) in pParams.iteritems():
        if key is "objective_stop_val":
            cParams.objStopValue = value
        elif key is "abs_function_tol":
            cParams.absFuncTolerance = value
        elif key is "rel_function_tol":
            cParams.relFuncTolerance = value
        elif key is "abs_parameter_tol":
            cParams.absParamTolerance = value
        elif key is "rel_parameter_tol":
            cParams.relParamTolerance = value
        elif key is "max_function_evals":
            cParams.maxFunctionEvals = value
        elif key is "max_runtime_secs":
            cParams.maxSeconds = value
        else:
            raise ValueError("Key " + key + " does not correspond to any optimization parmaeter.")
    return cParams

cdef OptResultsToPy( cFittingResults cResults ):
    return OptimizationResults( finalObjective = cResults.finalObjective,
                                numObjectiveEvals = cResults.numObjectiveEvaluations,
                                numGradientEvals = cResults.numGradientEvaluations,
                                totalElapsedSecs = cResults.totalElapsedSecs,
                                totalObjectiveSecs = cResults.totalObjectiveSecs,
                                totalGradientSecs = cResults.totalGradientSecs )

cpdef fit_direct_likelihood( ModCholRegressor model, list dataset,
                             double lWeight, double dWeight,
                             dict optimizationParams ):
    """Fit a modified Cholesky model to a direct likelihood dataset."""

    cdef vector[cDirectLikelihoodData] cDataset
    for data in dataset:
        cDataset.push_back( PyToDirectLikelihoodData( data ) )

    cdef cFittingResults results
    cdef cOptParameters params = PyToOptParameters( optimizationParams )
    results = c_batch_direct_fit( deref( model._regressor ), cDataset, 
                                  lWeight, dWeight, params )
    return OptResultsToPy( results )

cpdef fit_damped_likelihood( ModCholRegressor model, list dataset,
                             double lWeight, double dWeight,
                             dict optimizationParams ):
    """Fit a modified Cholesky model to a damped likelihood dataset."""
    
    cdef vector[cDampedLikelihoodData] cDataset
    for data in dataset:
        cDataset.push_back( PyToDampedLikelihoodData( data ) )

    cdef cFittingResults results
    cdef cOptParameters params = PyToOptParameters( optimizationParams )
    results = c_batch_damped_fit( deref( model._regressor ), cDataset, 
                                  lWeight, dWeight, params )
    return OptResultsToPy( results )

cpdef fit_transformed_likelihood( ModCholRegressor model, list dataset,
                                  double lWeight, double dWeight,
                                  dict optimizationParams ):
    
    cdef vector[cTransLikelihoodData] cDataset
    for data in dataset:
        cDataset.push_back( PyToTransLikelihoodData( data ) )

    cdef cFittingResults results
    cdef cOptParameters params = PyToOptParameters( optimizationParams )
    results = c_batch_trans_fit( deref( model._regressor ), cDataset, 
                                  lWeight, dWeight, params )
    return OptResultsToPy( results )

cpdef fit_affine_likelihood( ModCholRegressor model, list dataset,
                             double lWeight, double dWeight,
                             dict optimizationParams ):
    
    cdef vector[cAffineLikelihoodData] cDataset
    for data in dataset:
        cDataset.push_back( PyToAffineLikelihoodData( data ) )

    cdef cFittingResults results
    cdef cOptParameters params = PyToOptParameters( optimizationParams )
    results = c_batch_affine_fit( deref( model._regressor ), cDataset, 
                                  lWeight, dWeight, params )
    return OptResultsToPy( results )

cpdef fit_chain_likelihood( list models,
                            list dataset, double lWeight, double dWeight,
                            dict optimizationParams ):
    cdef vector[cModCholRegd*] cModels
    for m in models:
        cModels.push_back( (<ModCholRegressor>m)._regressor )

    cdef vector[cChainedLikelihoodData] cDataset
    
    for data in dataset:
        cDataset.push_back( PyToChainedLikelihoodData( data ) )

    cdef cFittingResults results
    cdef cOptParameters params = PyToOptParameters( optimizationParams )
    results = c_batch_chain_fit( cModels, cDataset, lWeight, dWeight, params )
    return OptResultsToPy( results )

def gaussian_log_likelihood( x, cov ):
    x = np.asmatrix( x.flat ).T
    cov = np.asmatrix( cov )
    invProd = x.T * np.linalg.solve( a = cov, b = x )
    n = cov.shape[0]
    d = np.linalg.det(cov)
    return -0.5 * ( math.log( d ) + invProd[0,0] + n*math.log( 2*math.pi ) )