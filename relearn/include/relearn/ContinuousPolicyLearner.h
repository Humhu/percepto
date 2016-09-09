#pragma once

#include "poli/PolicyInfoManager.h"
#include "poli/ContinuousPolicy.h"
#include "relearn/PolicyLogGradientModules.h"
#include "relearn/RelearnInterfaces.h"

#include "percepto_msgs/ContinuousAction.h"
#include "percepto_msgs/RewardStamped.h"

#include "argus_utils/synchronization/SynchronizationTypes.h"

#include <modprop/optim/OptimizerTypes.h>
#include <modprop/optim/ParameterL2Cost.hpp>
#include <modprop/optim/StochasticMeanCost.hpp>
#include <modprop/compo/AdditiveWrapper.hpp>

#include <deque>

namespace percepto
{

// TODO Templatize on the cost term module
struct PolicyGradientOptimization
{
	std::deque<ContinuousLogGradientModule> modules;
	percepto::StochasticMeanCost<double> rewards;
	percepto::ParameterL2Cost regularizer;
	percepto::AdditiveWrapper<double> objective;

	PolicyGradientOptimization();

	void Initialize( percepto::Parameters::Ptr params,
	                 double l2Weight,
	                 unsigned int batchSize );

	template <class ... Args>
	void EmplaceModule( Args&&... args )
	{
		modules.emplace_back( args... );
		rewards.AddSource( modules.back().GetOutputSource() );
	}

	ContinuousLogGradientModule& GetLatestModule();
	void RemoveOldest();
	size_t NumModules() const;

	void Invalidate();
	void Foreprop();
	void ForepropAll();
	void Backprop();
	void BackpropNatural();

	double GetOutput() const;

	// Computes the mean log-likelihood of all the modules
	double ComputeLogProb();
};

struct PolicyDivergenceChecker
{
	PolicyGradientOptimization& optimization;
	double maxDivergence;
	double startingLogLikelihood;

	PolicyDivergenceChecker( PolicyGradientOptimization& opt );

	void SetDivergenceLimit( double m );
	void ResetDivergence();
	bool ExceededLimits();
};

class ContinuousPolicyLearner
{
public:

	ContinuousPolicyLearner();

	void Initialize( ros::NodeHandle& nh, 
	                 ros::NodeHandle& ph );

private:

	mutable argus::Mutex _mutex;

	argus::LookupInterface _lookup;
	PolicyInfoManager _infoManager;
	ContinuousPolicy _policy;

	ros::ServiceClient _getCritiqueClient;
	ros::ServiceClient _setParamsClient;
	ros::Subscriber _actionSub;

	ros::Timer _updateTimer;
	ros::Time _lastOptimizationTime;

	double _logdetWeight;

	PolicyGradientOptimization _optimization;
	PolicyDivergenceChecker _optimizationChecker;

	typedef std::map<ros::Time, ContinuousAction> ActionBuffer;
	ActionBuffer _actionBuffer;

	bool _clearAfterOptimize;
	unsigned int _minModulesToOptimize;
	unsigned int _maxModulesToKeep;

	std::shared_ptr<percepto::AdamStepper> _stepper;
	std::shared_ptr<percepto::SimpleConvergence> _convergence;
	// std::shared_ptr<percepto::AdamOptimizer> _optimizer;

	std::shared_ptr<percepto::SimpleNaturalOptimizer> _optimizer;

	void ActionCallback( const percepto_msgs::ContinuousAction::ConstPtr& msg );
	void TimerCallback( const ros::TimerEvent& event );
};

}