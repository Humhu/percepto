#include "covreg/LEMCRTypes.h"
#include <boost/foreach.hpp>

namespace covreg
{

FittingResults
batch_directll_fit( ModCholReg& regressor, 
                    const std::vector<ModCholReg::InputType>& inputs,
                    const std::vector<VectorType>& samples,
                    double lWeight, double dWeight,
                    const NLOptParameters& params )
{
	typedef DirectLikelihoodFitting DLF;

	// TODO Combine inputs, samples?
	assert( inputs.size() == samples.size() );

	std::vector<GaussianLogLikelihoodCost> costs;
	costs.reserve( inputs.size() );
	for( unsigned int i = 0; i < inputs.size(); i++ )
	{
		costs.emplace_back( inputs[i], samples[i], regressor );
	}

	VectorType weights = regressor.CreateWeightVector( lWeight, dWeight );
	return DLF::RefineModel( regressor, dataset, weights, params );
}

FittingResults
batch_dampedll_fit( ModCholReg& baseRegressor, 
                    const std::vector<DampedLikelihoodData>& dataset,
                    double lWeight, double dWeight,
                    const NLOptParameters& params )
{
	typedef DampedLikelihoodFitting DaLF;
	VectorType weights = baseRegressor.CreateWeightVector( lWeight, dWeight );
	DampedModCholReg regressor( baseRegressor );
	return DaLF::RefineModel( regressor, dataset, weights, params );
}

FittingResults
batch_transll_fit( ModCholReg& baseRegressor, 
                   const std::vector<TransLikelihoodData>& dataset,
                   double lWeight, double dWeight,
                   const NLOptParameters& params )
{
	typedef TransLikelihoodFitting TLF;
	VectorType weights = baseRegressor.CreateWeightVector( lWeight, dWeight );
	TransModCholReg regressor( baseRegressor );
	return TLF::RefineModel( regressor, dataset, weights, params );
}

FittingResults
batch_affinell_fit( ModCholReg& baseRegressor, 
                    const std::vector<AffineLikelihoodData>& dataset,
                    double lWeight, double dWeight,
                    const NLOptParameters& params )
{
	typedef AffineLikelihoodFitting ALF;
	VectorType weights = baseRegressor.CreateWeightVector( lWeight, dWeight );
	TransModCholReg tRegressor( baseRegressor );
	AffineModCholReg regressor( tRegressor );
	return ALF::RefineModel( regressor, dataset, weights, params );
}

FittingResults
batch_chainll_fit( const std::vector<ModCholReg*>& regs,
                   const std::vector<ChainedLikelihoodData>& dataset,
                   double lWeight, double dWeight, 
                   const NLOptParameters& params )
{
	typedef ChainedLikelihoodFitting CLF;
	std::vector< std::shared_ptr<TransModCholReg> > transRegPtrs;
	std::vector<TransModCholReg*> transRegRaws;
	std::vector<AffineModCholReg> affRegs;
	
	for( unsigned int i = 0; i < regs.size(); i++ )
	{
		transRegPtrs.push_back( std::make_shared<TransModCholReg>( *regs[i] ) );
		transRegRaws.push_back( transRegPtrs[i].get() );
	}
	SummedModCholReg sumRegs( transRegRaws );
	ChainedModCholReg regressor( sumRegs );

	VectorType weights = regressor.CreateWeightVector( lWeight, dWeight );
	return CLF::RefineModel( regressor, dataset, weights, params );
}

}