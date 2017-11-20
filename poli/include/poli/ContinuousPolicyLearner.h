#pragma once

#include "poli/PolicyInfoManager.h"
#include "poli/ContinuousPolicy.h"
#include "relearn/RelearnInterfaces.h"
#include "relearn/PolicyGradientProblem.h"

#include "percepto_msgs/ContinuousAction.h"
#include "percepto_msgs/RewardStamped.h"

#include "argus_utils/synchronization/SynchronizationTypes.h"

#include "optim/Optimizers.h"

#include <deque>

namespace percepto
{

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

	typedef std::map<ros::Time, ContinuousAction> ActionBuffer;
	ActionBuffer _actionBuffer;

	bool _clearAfterOptimize;
	unsigned int _minModulesToOptimize;
	unsigned int _maxModulesToKeep;

	ModularOptimizer::Ptr _optimizer;

	void ActionCallback( const percepto_msgs::ContinuousAction::ConstPtr& msg );
	void TimerCallback( const ros::TimerEvent& event );
};

}