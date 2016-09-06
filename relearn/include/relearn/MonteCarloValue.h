#pragma once

#include "relearn/RelearnInterfaces.h"
#include "relearn/RewardInterpolater.h"

namespace percepto
{

// Estimates the policy value function by summing rewards over a period of time
class MonteCarloValue
: public PolicyCritic
{
public:

	typedef std::shared_ptr<MonteCarloValue> Ptr;

	MonteCarloValue();

	void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph );

	virtual double Evaluate( const ParamAction& x ) const;

private:

	RewardInterpolater _rewardFunction;
	unsigned int _horizonSteps;
	ros::Duration _timestep;
	double _discountFactor;

};

}