#pragma once

#include "percepto/compo/LinearRegressor.hpp"
#include "percepto/compo/ExponentialWrapper.hpp"
#include "percepto/pdreg/ModifiedCholeskyWrapper.hpp"
#include "percepto/compo/AffineWrapper.hpp"
#include "percepto/compo/AdditiveWrapper.hpp"

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"
#include "percepto/optim/MeanPopulationCost.hpp"

#include "percepto/optim/ModelFitting.hpp"

namespace percepto
{

typedef ExponentialWrapper<LinearRegressor> ExpLinReg;
typedef ModifiedCholeskyWrapper<LinearRegressor, ExpLinReg> ModCholReg;
typedef AffineWrapper<ModCholReg> AffineModCholReg;

typedef AdditiveWrapper<TransModCholReg> SummedModCholReg;
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