#pragma once

#include "valu/ValuInterfaces.h"
#include "valu/RewardInterpolater.h"

namespace percepto
{

// Estimates the policy value function by summing rewards over a period of time
class MonteCarloValue
: public Critic
{
public:

	typedef std::shared_ptr<MonteCarloValue> Ptr;

	MonteCarloValue();

	void Initialize( RewardInterpolater::Ptr rewards, ros::NodeHandle& ph );

	virtual double GetCritique( const ros::Time& time ) const;

private:

	RewardInterpolater::Ptr _rewardFunction;
	unsigned int _horizonSteps;
	ros::Duration _timestep;
	double _discountFactor;

};

}