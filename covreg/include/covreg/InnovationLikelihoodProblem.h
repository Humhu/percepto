#pragma once

#include "covreg/KalmanFilterEpisode.h"

#include "optim/OptimInterfaces.h"
#include <modprop/optim/StochasticMeanCost.hpp>
#include <modprop/optim/ParameterL2Cost.hpp>

#include <list>

namespace percepto
{

std::ostream& operator<<( std::ostream& os, const KalmanFilterEpisode& episode );

// TODO Implement natural optimization interface
struct InnovationLikelihoodProblem
: public OptimizationProblem
{

	std::deque<KalmanFilterEpisode> episodes;
	percepto::StochasticMeanCost<double> loss;
	percepto::ParameterL2Cost regularizer;
	percepto::AdditiveWrapper<double> objective;
	percepto::Parameters::Ptr parameters;

	InnovationLikelihoodProblem( percepto::Parameters::Ptr params,
	                             double l2Weight,
	                             unsigned int batchSize );

	template <class ...Args>
	void EmplaceEpisode( Args&& ...args )
	{
		episodes.emplace_back( args... );
		loss.AddSource( episodes.back().GetLL() );
	}

	virtual void Resample();
	virtual bool IsMinimization() const;
	virtual double ComputeObjective();
	virtual VectorType ComputeGradient();
	virtual VectorType GetParameters() const;
	virtual void SetParameters( const VectorType& params );

	void RemoveOldestEpisode();

	size_t NumEpisodes() const;

	KalmanFilterEpisode* GetOldestEpisode();

	KalmanFilterEpisode* GetCurrentEpisode();

	const KalmanFilterEpisode* GetCurrentEpisode() const;

private:

	// Forbid copying
	InnovationLikelihoodProblem( const InnovationLikelihoodProblem& other );
	
	// Forbid assigning
	InnovationLikelihoodProblem& 
	operator=( const InnovationLikelihoodProblem& other );

	void Invalidate();
	void Foreprop();
	void Backprop();

	percepto::Parameters::Ptr _parameters;

};

std::ostream& operator<<( std::ostream& os, 
                          const InnovationLikelihoodProblem& problem );

}