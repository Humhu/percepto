#pragma once

#include "relearn/DiscretePolicyManager.h"
#include "relearn/RewardStamped.h"
#include "relearn/DiscreteParamAction.h"
#include "relearn/PolicyLogGradientModules.h"

namespace percepto
{

class DiscretePolicyLearner
{
public:

	typedef DiscretePolicyManager::NetworkType NetworkType;

	DiscretePolicyLearner( ros::NodeHandle& nh, ros::NodeHandle& ph );

private:

	percepto::Parameters::Ptr _networkParams;

	ros::Subscriber _actionSub;
	ros::Subscriber _rewardSub;
	ros::Publisher _paramPub;

	ros::Timer _updateTimer;

	void ActionCallback( const relearn::DiscreteParamAction::ConstPtr& msg );
	void RewardCallback( const relearn::RewardStamped::ConstPtr& msg );
	void TimerCallback( const ros::TimerEvent& event );
};

}