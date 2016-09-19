#pragma once

#include "valu/ValueResidualModules.h"

#include "optim/Optimizers.h"

#include "modprop/optim/ParameterL2Cost.hpp"
#include "modprop/optim/StochasticMeanCost.hpp"
#include "modprop/compo/AdditiveWrapper.hpp"

namespace percepto
{

class ApproximateValueProblem
: public NaturalOptimizationProblem
{
public:

	std::deque<BellmanResidualModule> modules;
	std::deque<SquaredLoss<double>> penalties;
	std::deque<AdditiveWrapper<double>> modSums;

	StochasticMeanCost<double> loss;
	ParameterL2Cost regularizer;
	AdditiveWrapper<double> objective;

	Parameters::Ptr parameters;
	double penaltyScale;

	ApproximateValueProblem();

	void Initialize( Parameters::Ptr params,
	                 double l2Weight,
	                 unsigned int sampleSize,
	                 double penaltyWeight );

	template <class ... Args>
	void EmplaceModule( Args&&... args )
	{
		modules.emplace_back( args... );

		penalties.emplace_back();
		penalties.back().SetSource( &modules.back().estValue->GetOutputSource() );
		penalties.back().SetTarget( 0.0 );
		penalties.back().SetScale( penaltyScale );

		modSums.emplace_back();
		modSums.back().SetSourceA( &modules.back().GetOutputSource() );
		modSums.back().SetSourceB( &penalties.back() );

		loss.AddSource( &modSums.back() );
	}

	virtual bool IsMinimization() const;
	virtual void Resample();
	virtual double ComputeObjective();
	virtual VectorType ComputeGradient();
	virtual VectorType GetParameters() const;
	virtual void SetParameters( const VectorType& params );

	virtual VectorType ComputeNaturalGradient();

	void RemoveOldest();
	size_t NumModules() const;

private:

	void Invalidate();
	void Foreprop();
	void Backprop();
	void BackpropNatural();
};

}