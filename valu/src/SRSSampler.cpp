#include "valu/SRSSampler.h"
#include "percepto_msgs/SRSTuple.h"
#include "valu/ValuCommon.h"

using namespace argus;

namespace percepto
{

SRSSampler::SRSSampler() {}

void SRSSampler::Initialize( ros::NodeHandle& nh,
                             ros::NodeHandle& ph )
{
	RewardInterpolater::Ptr reward = std::make_shared<RewardInterpolater>();
	ros::NodeHandle rh( ph.resolveName( "reward_function" ) );
	reward->Initialize( nh, rh );

	BroadcastMultiReceiver::Ptr rx = std::make_shared<BroadcastMultiReceiver>();
	ros::NodeHandle ih( ph.resolveName( "input_streams" ) );
	rx->Initialize( ih );

	Initialize( reward, rx, ph );
}

void SRSSampler::Initialize( RewardInterpolater::Ptr rewards,
                             BroadcastMultiReceiver::Ptr inputs,
                             ros::NodeHandle& ph )
{
	_rewards = rewards;
	_inputs = inputs;

	double sampleRate, timestep, sampleOffset;
	GetParamRequired( ph, "sample_rate", sampleRate );
	GetParamRequired( ph, "timestep", timestep );
	GetParamRequired( ph, "sample_offset", sampleOffset );

	_timestep = ros::Duration( timestep );
	_sampleOffset = ros::Duration( sampleOffset );
	_srsPub = ph.advertise<percepto_msgs::SRSTuple>( "srs_tuple", 0 );

	_sampleTimer = ph.createTimer( ros::Duration( 1.0/sampleRate ),
	                               &SRSSampler::TimerCallback,
	                               this );
}

void SRSSampler::TimerCallback( const ros::TimerEvent& event )
{
	SRSTuple srs;
	srs.time = event.current_expected - _sampleOffset;
	srs.nextTime = srs.time + _timestep;

	StampedFeatures feat, nextFeat;
	try
	{
		_inputs->ReadStream( srs.time, feat );
		_inputs->ReadStream( srs.nextTime, nextFeat );
	}
	catch( std::out_of_range )
	{
		ROS_WARN_STREAM( "Could not read inputs." );
	}

	srs.state = feat.features;
	srs.nextState = nextFeat.features;

	_srsPub.publish( srs.ToMsg() );
}

}