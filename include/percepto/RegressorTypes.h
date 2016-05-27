#pragma once

#include "percepto/compo/LinearRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/compo/ModifiedCholeskyWrapper.hpp"
#include "percepto/compo/AffineWrapper.hpp"
#include "percepto/compo/AdditiveWrapper.hpp"

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"
#include "percepto/optim/MeanPopulationCost.hpp"
#include "percepto/optim/StochasticPopulationCost.hpp"

namespace percepto
{

typedef ExponentialWrapper<ReLUNet> ExpReLUReg;
typedef ModifiedCholeskyWrapper<LinearRegressor, ExpReLUReg> MCReg;


typedef AffineWrapper<MCReg> AffineMCReg;
typedef AdditiveWrapper<AffineMCReg, AffineMCReg> SummedAffineMCReg;
typedef DampedRegressor<SummedModCholReg> ChainedAffineMCReg;

template <typename RegressorType>
struct FittingData
{
	VectorType sample;
	typename RegressorType::InputType input;
};

inline OptimizationResults
batch_directll_fit( ModCholReg& regressor, 
                    const std::vector<DirectLikelihoodData>& dataset,
                    double lWeight, double dWeight,
                    const NLOptParameters& params );

inline OptimizationResults
batch_affinell_fit( ModCholReg& regressor, 
                    const std::vector<AffineLikelihoodData>& dataset,
                    double lWeight, double dWeight,
                    const NLOptParameters& params );

inline OptimizationResults
batch_chainll_fit( ModCholReg& regA, ModCholReg& regB,
                   const std::vector<ChainedLikelihoodData>& dataset,
                   double lWeight, double dWeight,
                   const NLOptParameters& params );

}