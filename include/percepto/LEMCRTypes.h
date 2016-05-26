#pragma once

#include "percepto/LinearRegressor.hpp"
#include "percepto/ExponentialWrapper.hpp"
#include "percepto/ModifiedCholeskyWrapper.hpp"
#include "percepto/AffineWrapper.hpp"
#include "percepto/SummedRegressor.hpp"

#include "percepto/GaussianLogLikelihoodCost.hpp"
#include "percepto/ParameterL2Cost.hpp"
#include "percepto/MeanPopulationCost.hpp"

#include "percepto/ModelFitting.hpp"

namespace percepto
{

typedef ExponentialWrapper<LinearRegressor> ExpLinReg;
typedef ModifiedCholeskyWrapper<LinearRegressor, ExpLinReg> ModCholReg;
typedef AffineWrapper<ModCholReg> AffineModCholReg;

typedef SummedRegressor<TransModCholReg> SummedModCholReg;
typedef DampedRegressor<SummedModCholReg> ChainedModCholReg;

typedef OptimizationResults FittingResults;

typedef ModelFitting<ModCholReg, GaussianLogLikelihoodCost> DirectLikelihoodFitting;

typedef ModelFitting<AffineModCholReg, GaussianLogLikelihoodCost> AffineLikelihoodFitting;

typedef ModelFitting<ChainedModCholReg, GaussianLogLikelihoodCost> ChainedLikelihoodFitting;

FittingResults
batch_directll_fit( ModCholReg& regressor, 
                    const std::vector<DirectLikelihoodData>& dataset,
                    double lWeight, double dWeight,
                    const NLOptParameters& params );

FittingResults
batch_affinell_fit( ModCholReg& regressor, 
                    const std::vector<AffineLikelihoodData>& dataset,
                    double lWeight, double dWeight,
                    const NLOptParameters& params );

FittingResults
batch_chainll_fit( const std::vector<ModCholReg*>& regs,
                   const std::vector<ChainedLikelihoodData>& dataset,
                   double lWeight, double dWeight,
                   const NLOptParameters& params );

}