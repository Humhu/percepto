#include "advant/TDErrorCritic.h"
#include "argus_utils/utils/ParamUtils.h"

#include "percepto_msgs/RewardStamped.h"

using namespace argus;

namespace percepto
{

TDErrorCritic::TDErrorCritic() {}

void TDErrorCritic::Initialize( RewardInterpolater::Ptr rewards,
                                Critic::Ptr values,
                                ros::NodeHandle& ph )
{
	_rewardFunction = rewards;
	_valueFunction = values;

	double dt, discountRate;
	GetParamRequired( ph, "timestep", dt );
	_timestep = ros::Duration( dt );

	if( GetParam( ph, "discount_rate", discountRate ) )
	{
		_discountFactor = std::exp( dt * std::log( discountRate ) );
		ROS_INFO_STREAM( "Computed discount factor of: " << _discountFactor << 
		                 " from desired rate: " << discountRate );
	}
	else
	{
		GetParamRequired( ph, "discount_factor", _discountFactor );
	}
}

double TDErrorCritic::GetReward( const ros::Time& time ) const
{
	return _rewardFunction->IntegratedReward( time, time + _timestep );
}

double TDErrorCritic::GetCritique( const ros::Time& time ) const
{
	double reward = GetReward( time );
	double currValue = _valueFunction->GetCritique( time );
	double nextValue = _valueFunction->GetCritique( time + _timestep );

	double tdError = reward + _discountFactor * nextValue - currValue;
	return tdError;
}

}