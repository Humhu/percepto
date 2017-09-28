#include "valu/MonteCarloValue.h"

#include "argus_utils/utils/ParamUtils.h"
#include "percepto_msgs/RewardStamped.h"

using namespace argus;

namespace percepto
{

MonteCarloValue::MonteCarloValue() {}

void MonteCarloValue::Initialize( RewardInterpolater::Ptr rewards, 
                                  ros::NodeHandle& ph )
{
	_rewardFunction = rewards;

	double horizonTime, stepSize;
	GetParamRequired( ph, "horizon_time", horizonTime );
	GetParamRequired( ph, "time_step", stepSize );
	
	if( !GetParam( ph, "discount_factor", _discountFactor ) )
	{
		double discountRate;
		if( !GetParam( ph, "discount_rate", discountRate ) )
		{
			throw std::runtime_error( "MonteCarloValue: Must specify discount_factor or discount_rate." );
		}
		_discountFactor = std::exp( stepSize * std::log( discountRate ) );
		ROS_INFO_STREAM( "Computed discount factor of: " << _discountFactor << 
		                 " from desired rate: " << discountRate );
	}
	if( _discountFactor < 0.0 || _discountFactor > 1.0 )
	{
		throw std::runtime_error( "Discount factor must be between 0 and 1." );
	}

	_horizonSteps = std::floor( horizonTime / stepSize );
	_timestep = ros::Duration( stepSize );
}

double MonteCarloValue::GetCritique( const ros::Time& time ) const
{
	double gamma = 1.0;
	double acc = 0.0;
	ros::Time tAcc = time;
	for( unsigned int i = 0; i < _horizonSteps; ++i )
	{
		acc += gamma * _rewardFunction->IntegratedReward( tAcc, tAcc + _timestep );
		gamma *= _discountFactor;
		tAcc += _timestep;
	}
	return acc / _horizonSteps;
}

}