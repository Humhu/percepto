#pragma once

#include <modprop/optim/ParameterL2Cost.hpp>

#include "optim/ModularOptimizer.h"

#include <argus_utils/utils/LinalgTypes.h>
#include <argus_utils/filters/FilterInfo.h>

#include <memory>
#include <unordered_map>

#include <boost/foreach.hpp>

#include "covreg/InnovationLikelihoodProblem.h"
#include "covreg/CovarianceEstimator.h"
#include "argus_utils/synchronization/SynchronizationTypes.h"

namespace percepto
{

// TODO Get rid of this? There aren't that many parameters
struct InnovationClipParameters
{
	unsigned int maxEpisodeLength;
	double l2Weight;
	unsigned int batchSize;

	InnovationClipParameters()
	: maxEpisodeLength( 10 ),
	l2Weight( 1E-9 ), batchSize( 30 ) {}
};

class InnovationClipOptimizer
{
public:

	InnovationClipOptimizer( CovarianceEstimator& qReg,
	                         const InnovationClipParameters& params =
	                         InnovationClipParameters() );

	void AddObservationReg( CovarianceEstimator& reg, const std::string& name );

	void AddPredict( const argus::PredictInfo& info, const VectorType& input );

	bool AddUpdate( const argus::UpdateInfo& info, const VectorType& input,
	                const std::string& name, double weight, const ros::Time& stamp );

	// Terminates the current episode and starts a new one
	void BreakCurrentEpisode();
	void RemoveEarliestEpisode();
	ros::Time GetEarliestTime();

	// Returns whether it converged or bailed early
	bool Optimize();

	size_t NumEpisodes() const;
	size_t CurrentEpisodeLength() const;

	void InitializeOptimization( const ros::NodeHandle& ph );

	double CalculateCost();
	void Print( std::ostream& os );

private:

	// Forbid copying and moving
	InnovationClipOptimizer( const InnovationClipOptimizer& other );
	InnovationClipOptimizer& operator=( const InnovationClipOptimizer& other );

	mutable argus::Mutex _mutex;
	
	// All parameters from optimized estimators
	ParameterWrapper::Ptr _paramWrapper;
	
	KalmanFilterEpisode* _currentEpisode;
	std::vector< std::pair<argus::PredictInfo,VectorType> > _predBuffer;

	// The regressors optimized
	CovarianceEstimator& _transReg;
	std::unordered_map <std::string, CovarianceEstimator&> _obsRegs;

	// The optimization problem
	InnovationLikelihoodProblem _problem;
	
	unsigned int _maxEpisodeLength;

	ModularOptimizer::Ptr _optimizer;
};

std::ostream& operator<<( std::ostream& os, InnovationClipOptimizer& opt );

}