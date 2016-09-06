#pragma once

#include "relearn/RelearnInterfaces.h"
#include "percepto_msgs/RewardStamped.h"
#include <map>

namespace percepto
{

enum InterpolationMode
{
	INTERP_ZERO_ORDER_HOLD,
	INTERP_PIECEWISE_LINEAR
};

InterpolationMode StringToInterpMode( const std::string& str );

class RewardInterpolater
{
public:

	typedef std::shared_ptr<RewardInterpolater> Ptr;

	RewardInterpolater();

	void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph );

	double InstantaneousReward( const ros::Time& t ) const;
	double IntegratedReward( const ros::Time& start, const ros::Time& end ) const;

private:

	typedef std::map<ros::Time, double> RewardSeries;
	RewardSeries _rewards;

	ros::Duration _cacheTime;
	InterpolationMode _interpMode;
	ros::Subscriber _rewardSub;

	void RewardCallback( const percepto_msgs::RewardStamped::ConstPtr& msg );
	ros::Duration ComputeSpan() const;

	double InterpolateZeroOrderHold( const ros::Time& t ) const;
	double InterpolatePiecewiseLinear( const ros::Time& t ) const;
};

}