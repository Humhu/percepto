#pragma once

#include "percepto/optim/GaussianLogLikelihoodCost.hpp"
#include "percepto/optim/ParameterL2Cost.hpp"
#include "percepto/optim/MeanPopulationCost.hpp"
#include "percepto/optim/NlOptInterface.hpp"

namespace percepto
{

template <typename Regressor, template <typename> class Base>
class ModelFitting
{
public:

	typedef Base<Regressor> BaseCost;
	typedef MeanPopulationCost<BaseCost> MeanCost;
	typedef ParameterL2Cost<MeanCost> PenalizedMeanCost;
	
	typedef NLOptInterface<PenalizedMeanCost> Optimizer;
	typedef typename Optimizer::ResultsType OptResults;


	typedef typename BaseCost::InputType DataType;
	typedef std::vector<DataType> DatasetType;
	typedef typename Regressor::ParameterType ParameterType;

	static OptResults RefineModel( Regressor& regressor,
	                               const DatasetType& dataset,
	                               const VectorType& weights,
	                               const NLOptParameters& params )
	{
		BaseCost baseCost( regressor );
		MeanCost meanCost( baseCost, dataset );

		PenalizedMeanCost penMeanCost( meanCost, weights );

		Optimizer opt( penMeanCost, params );
		return opt.Optimize( regressor.GetParameters() );
	}

};

}