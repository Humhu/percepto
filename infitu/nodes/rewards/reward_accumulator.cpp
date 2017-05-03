#include <ros/ros.h>

#include "percepto_msgs/RewardStamped.h"
#include "infitu/SetRecording.h"

#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

class RewardAccumulator
{
public:

	RewardAccumulator( ros::NodeHandle& nh, ros::NodeHandle& ph )
		: _defaultValue( 0 )
	{
		GetParam( ph, "default_on_empty", _defaultOnEmpty, false );
		if( _defaultOnEmpty )
		{
			GetParamRequired( ph, "default_value", _defaultValue );
		}

		GetParam( ph, "time_integrate", _useIntegration, false );
		GetParam( ph, "normalize_by_time", _normalizeTime, false );
		GetParam( ph, "normalize_by_count", _normalizeCount, false );

		unsigned int buffSize;
		GetParam<unsigned int>( ph, "buffer_size", buffSize, 100 );
		_rewardSub = nh.subscribe( "reward", buffSize,
		                           &RewardAccumulator::RewardCallback, this );

		_recordServer = ph.advertiseService( "set_recording",
		                                     &RewardAccumulator::RecordingCallback,
		                                     this );
	}

private:

	void Reset()
	{
		_rewardAcc = 0;
		_rewardCount = 0;
		_rewardDur = 0;
		_lastRewardTime = ros::Time( 0 );
	}

	void RewardCallback( const percepto_msgs::RewardStamped::ConstPtr& msg )
	{
		// Indicate need for initialization with zero last time
		if( _lastRewardTime.isZero() )
		{
			_lastRewardTime = msg->header.stamp;
			_lastRewardValue = msg->reward;
		}

		double rewardInc = msg->reward;
		double dt = (msg->header.stamp - _lastRewardTime).toSec();
		if( _useIntegration )
		{
			// Simple trapezoid rule
			rewardInc = 0.5 * (_lastRewardValue + msg->reward) * dt;
		}

		_rewardAcc += rewardInc;
		_rewardCount += 1;
		_rewardDur += dt;
		_lastRewardTime = msg->header.stamp;
		_lastRewardValue = msg->reward;
	}

	bool RecordingCallback( infitu::SetRecording::Request& req,
	                        infitu::SetRecording::Response& res )
	{
		if( req.enable_recording )
		{
			Reset();
			return true;
		}

		// Else return recording
		// Check for empty
		if( _rewardCount == 0 && _defaultOnEmpty )
		{
			res.evaluation = _defaultValue;
			return true;
		}

		res.evaluation = _rewardAcc;
		if( _normalizeTime )
		{
			res.evaluation = res.evaluation / _rewardDur;
		}
		if( _normalizeCount )
		{
			res.evaluation = res.evaluation / _rewardCount;
		}
		return true;
	}

	double _rewardAcc;
	unsigned int _rewardCount;
	double _rewardDur;
	ros::Time _lastRewardTime;

	double _lastRewardValue;

	ros::Subscriber _rewardSub;
	ros::ServiceServer _recordServer;

	bool _defaultOnEmpty;
	double _defaultValue;

	bool _useIntegration;
	bool _normalizeTime;
	bool _normalizeCount;
};

int main( int argc, char**argv )
{
	ros::init( argc, argv, "reward_accumulator" );
	ros::NodeHandle nh, ph( "~" );
	RewardAccumulator ra( nh, ph );
	ros::spin();
	return 0;
}