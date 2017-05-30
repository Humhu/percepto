#include <ros/ros.h>

#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <percepto_msgs/RewardStamped.h>

#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

class SumSquaredErrorReward
{
public:

	// TODO Use TF or something for time interpolation
	SumSquaredErrorReward( ros::NodeHandle& nh, ros::NodeHandle& ph )
	{
		_rewardPub = ph.advertise<percepto_msgs::RewardStamped>( "reward", 10 );

		std::string estMode;
		GetParamRequired( ph, "estimate_mode", estMode );
		if( estMode == "twist" )
		{
			_estSub = nh.subscribe( "estimate", 10, &SumSquaredErrorReward::TwistEstCallback, this );
		}
		else if( estMode == "twist_stamped" )
		{
			_estSub = nh.subscribe( "estimate", 10, &SumSquaredErrorReward::TwistStampedEstCallback, this );
		}
		else if( estMode == "odom" )
		{
			_estSub = nh.subscribe( "estimate", 10, &SumSquaredErrorReward::OdomEstCallback, this );
		}

		std::string truthMode;
		GetParamRequired( ph, "truth_mode", truthMode );
		if( truthMode == "twist" )
		{
			_truthSub = nh.subscribe( "truth", 10, &SumSquaredErrorReward::TwistRefCallback, this );
		}
		else if( truthMode == "twist_stamped" )
		{
			_truthSub = nh.subscribe( "truth", 10, &SumSquaredErrorReward::TwistStampedRefCallback, this );
		}
		else if( truthMode == "odom" )
		{
			_truthSub = nh.subscribe( "truth", 10, &SumSquaredErrorReward::OdomRefCallback, this );
		}
		else
		{
			throw std::invalid_argument( "Unknown truth mode: " + truthMode );
		}

		GetParam( ph, "log_rewards", _logRewards, false );
		GetParam( ph, "min_reward", _minReward, -std::numeric_limits<double>::infinity() );
		GetParam( ph, "max_reward", _maxReward, std::numeric_limits<double>::infinity() );

		GetParam( ph, "pose_err_weights", _posErrWeights, FixedVectorType<6>::Zero() );
		GetParam( ph, "vel_err_weights", _velErrWeights, FixedVectorType<6>::Zero() );

		_lastInit = false;
	}

	void TwistRefCallback( const geometry_msgs::Twist::ConstPtr& msg )
	{
		_lastPose = PoseSE3();
		_lastVel = MsgToTangent( *msg );
		_lastInit = true;
	}

	void TwistStampedRefCallback( const geometry_msgs::TwistStamped::ConstPtr& msg )
	{
		_lastPose = PoseSE3();
		_lastVel = MsgToTangent( msg->twist );
		_lastInit = true;
	}

	void OdomRefCallback( const nav_msgs::Odometry::ConstPtr& msg )
	{
		_lastPose = MsgToPose( msg->pose.pose );
		_lastVel = MsgToTangent( msg->twist.twist );
		_lastInit = true;
	}

	void TwistEstCallback( const geometry_msgs::Twist::ConstPtr& msg )
	{
		if( !_lastInit ) { return; }

		PoseSE3::TangentVector vel = MsgToTangent( *msg );
		ProcessEstimate( PoseSE3(), vel, ros::Time::now() );
	}

	void TwistStampedEstCallback( const geometry_msgs::TwistStamped::ConstPtr& msg )
	{
		if( !_lastInit ) { return; }

		PoseSE3::TangentVector vel = MsgToTangent( msg->twist );
		ProcessEstimate( PoseSE3(), vel, msg->header.stamp );
	}

	void OdomEstCallback( const nav_msgs::Odometry::ConstPtr& msg )
	{
		if( !_lastInit ) { return; }

		PoseSE3 pose = MsgToPose( msg->pose.pose );
		PoseSE3::TangentVector vel = MsgToTangent( msg->twist.twist );
		ProcessEstimate( pose, vel, msg->header.stamp );
	}

	void ProcessEstimate( const PoseSE3& pose, const PoseSE3::TangentVector& vel,
	                      const ros::Time& stamp )
	{
		PoseSE3::TangentVector poseErr = PoseSE3::Log( pose * _lastPose.Inverse() );
		PoseSE3::TangentVector velErr = vel - _lastVel;
		PoseSE3::TangentVector poseSquaredErr = poseErr.array() * poseErr.array();
		PoseSE3::TangentVector velSquaredErr = velErr.array() * velErr.array();

		double reward = _posErrWeights.dot( poseSquaredErr ) + _velErrWeights.dot( velSquaredErr );

		if( _logRewards )
		{
			reward = std::log( reward );
		}
		// Negate the predicted error to make it a reward
		reward = -reward;
		
		reward = std::max( std::min( reward, _maxReward ), _minReward );

		percepto_msgs::RewardStamped out;
		out.header.stamp = stamp;
		out.header.frame_id = "sse";
		out.reward = reward;
		_rewardPub.publish( out );
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:

	ros::Subscriber _estSub;
	ros::Subscriber _truthSub;
	ros::Publisher _rewardPub;

	bool _logRewards;
	double _minReward;
	double _maxReward;

	bool _lastInit;
	PoseSE3 _lastPose;
	PoseSE3::TangentVector _lastVel;

	FixedVectorType<6> _posErrWeights;
	FixedVectorType<6> _velErrWeights;
};

int main( int argc, char ** argv )
{
	ros::init( argc, argv, "sse_reward" );
	ros::NodeHandle nh, ph( "~" );
	SumSquaredErrorReward sse( nh, ph );
	ros::spin();
	return 0;
}