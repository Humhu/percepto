# Cython declarations for optimizers

from EigenInterface cimport MatrixXd

from CovregRegressors cimport ModCholRegd
from CovregRegressors cimport ChainedModCholRegd
from CovregRegressors cimport DampedModCholRegd
from CovregRegressors cimport TransModCholRegd
from CovregRegressors cimport AffineModCholRegd

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "covreg/NlOptInterface.hpp" namespace "covreg":

    cdef cppclass NLOptParameters:
        double objStopValue
        double absFuncTolerance
        double relFuncTolerance
        double absParamTolerance
        double relParamTolerance
        int maxFunctionEvals
        double maxSeconds
        NLOptParameters() except +

cdef extern from "covreg/LEMCRTypes.h" namespace "covreg":

    cdef cppclass FittingResults:
        double finalObjective
        unsigned int numObjectiveEvaluations
        unsigned int numGradientEvaluations
        double totalElapsedSecs
        double totalObjectiveSecs
        double totalGradientSecs

        FittingResults() except +

    cdef cppclass DirectLikelihoodData:
        MatrixXd sample
        ModCholRegd.InputType input
        DirectLikelihoodData() except +
        DirectLikelihoodData( const DirectLikelihoodData& other ) except +

    cdef cppclass DampedLikelihoodData:
        MatrixXd sample
        DampedModCholRegd.InputType input
        DampedLikelihoodData() except +
        DampedLikelihoodData( const DampedLikelihoodData& other ) except +

    cdef cppclass TransLikelihoodData:
        MatrixXd sample
        TransModCholRegd.InputType input
        TransLikelihoodData() except +
        TransLikelihoodData( const TransLikelihoodData& other ) except +

    cdef cppclass AffineLikelihoodData:
        MatrixXd sample
        AffineModCholRegd.InputType input
        AffineLikelihoodData() except +
        AffineLikelihoodData( const AffineLikelihoodData& other ) except +

    cdef cppclass ChainedLikelihoodData:
        MatrixXd sample
        ChainedModCholRegd.InputType input
        ChainedLikelihoodData() except +
        ChainedLikelihoodData( const ChainedLikelihoodData& other ) except +

    cdef FittingResults batch_directll_fit( ModCholRegd& initModel,
                                            const vector[DirectLikelihoodData]& dataset, 
                                            double lWeight, double dWeight,
                                            const NLOptParameters& params ) except +

    cdef FittingResults batch_dampedll_fit( ModCholRegd& initModel,
                                            const vector[DampedLikelihoodData]& dataset, 
                                            double lWeight, double dWeight,
                                            const NLOptParameters& params ) except +

    cdef FittingResults batch_transll_fit( ModCholRegd& initModel,
                                           const vector[TransLikelihoodData]& dataset, 
                                           double lWeight, double dWeight,
                                           const NLOptParameters& params ) except +

    cdef FittingResults batch_affinell_fit( ModCholRegd& initModel,
                                            const vector[AffineLikelihoodData]& dataset, 
                                            double lWeight, double dWeight,
                                            const NLOptParameters& params ) except +

    cdef FittingResults batch_chainll_fit( const vector[ModCholRegd*]& models,
                                           const vector[ChainedLikelihoodData]& dataset,
                                           double lWeight, double dWeight,
                                           const NLOptParameters& params ) except +