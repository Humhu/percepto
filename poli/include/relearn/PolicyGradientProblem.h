#pragma once

#include "modprop/optim/ParameterL2Cost.hpp"
#include "modprop/optim/StochasticMeanCost.hpp"
#include "modprop/compo/AdditiveWrapper.hpp"

#include "relearn/PolicyLogGradientModules.h"
#include "optim/Optimizers.h"

namespace percepto
{

class PolicyGradientOptimization
: public ConstrainedOptimizationProblem, 
  public NaturalOptimizationProblem
{
public:
	std::deque<ContinuousLogGradientModule> modules;
	StochasticMeanCost<double> rewards;
	ParameterL2Cost regularizer;
	AdditiveWrapper<double> objective;

	Parameters::Ptr parameters;

	PolicyGradientOptimization();

	void Initialize( Parameters::Ptr p,
	                 double l2Weight,
	                 unsigned int batchSize,
	                 double maxDivergence );

	template <class ... Args>
	void EmplaceModule( Args&&... args )
	{
		modules.emplace_back( args... );
		rewards.AddSource( modules.back().GetOutputSource() );
	}

	ContinuousLogGradientModule& GetLatestModule();
	void RemoveOldest();
	size_t NumModules() const;

	virtual bool IsMinimization() const;
	virtual void Resample();
	virtual double ComputeObjective();
	virtual VectorType ComputeGradient();
	virtual VectorType GetParameters() const;
	virtual void SetParameters( const VectorType& p );

	virtual VectorType ComputeNaturalGradient();

	virtual bool IsSatisfied();
	void ResetConstraints();

private:

	void Invalidate();
	void Foreprop();
	void ForepropAll();
	void Backprop();
	void BackpropNatural();

	// Computes the mean log-likelihood of all the modules
	double ComputeLogProb();
	double ComputeKLD();

	std::vector<VectorType> _policyMeans;
	std::vector<MatrixType> _policyInfos;

	double _initLikelihood;
	double _maxDivergence;
};

}