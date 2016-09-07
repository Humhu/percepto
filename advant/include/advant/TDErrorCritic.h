#pragma once

#include "valu/ValuInterfaces.h"
#include "valu/RewardInterpolater.h"

namespace percepto
{

class TDErrorCritic
: public Critic
{
public:

	typedef std::shared_ptr<TDErrorCritic> Ptr;

	TDErrorCritic();

	void Initialize( RewardInterpolater::Ptr rewards,
	                 Critic::Ptr values, 
	                 ros::NodeHandle& ph );

	virtual double GetCritique( const ros::Time& time ) const;

private:

	RewardInterpolater::Ptr _rewardFunction;
	Critic::Ptr _valueFunction;
	
	double _discountFactor;
	ros::Duration _timestep;

	double GetReward( const ros::Time& time ) const;
};

}