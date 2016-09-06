#pragma once

#include "relearn/RelearnInterfaces.h"
#include "relearn/RewardInterpolater.h"

namespace percepto
{

class TDErrorCritic
: public PolicyCritic
{
public:

	typedef std::shared_ptr<TDErrorCritic> Ptr;

	TDErrorCritic();

	void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph );

	double GetReward( const ros::Time& time ) const;
	virtual double Evaluate( const ParamAction& act ) const;

	ros::Duration GetTimestep() const;

private:

	ros::Publisher _estPub;

	RewardInterpolater _rewardFunction;
	PolicyCritic::Ptr _valueFunction;
	
	double _discountFactor;
	ros::Duration _timestep;
};

}