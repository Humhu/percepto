#include <ros/ros.h>

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
		_odomSub = nh.subscribe( "odom", 10, &SumSquaredErrorReward::EstCallback, this );
		_truthSub = nh.subscribe( "odom_truth", 10, &SumSquaredErrorReward::RefCallback, this );

		GetParam( ph, "log_rewards", _logRewards, false );
		GetParam( ph, "min_reward", _minReward, -std::numeric_limits<double>::infinity() );
		GetParam( ph, "max_reward", _maxReward, std::numeric_limits<double>::infinity() );

		GetParam( ph, "pose_err_weights", _posErrWeights, FixedVectorType<6>::Zero() );
		GetParam( ph, "vel_err_weights", _velErrWeights, FixedVectorType<6>::Zero() );

		_lastInit = false;
	}

	void RefCallback( const nav_msgs::Odometry::ConstPtr& msg )
	{
		_lastPose = MsgToPose( msg->pose.pose );
		_lastVel = MsgToTangent( msg->twist.twist );
		_lastInit = true;
	}

	void EstCallback( const nav_msgs::Odometry::ConstPtr& msg )
	{
		if( !_lastInit ) { return; }

		PoseSE3 pose = MsgToPose( msg->pose.pose );
		PoseSE3::TangentVector vel = MsgToTangent( msg->twist.twist );

		PoseSE3::TangentVector poseErr = PoseSE3::Log( pose * _lastPose.Inverse() );
		PoseSE3::TangentVector velErr = vel - _lastVel;
		PoseSE3::TangentVector poseSquaredErr = poseErr.array() * poseErr.array();
		PoseSE3::TangentVector velSquaredErr = velErr.array() * velErr.array();

		double reward = _posErrWeights.dot( poseSquaredErr ) + _velErrWeights.dot( velSquaredErr );

		ROS_INFO_STREAM( "poseErr: " << poseErr.transpose() );
		ROS_INFO_STREAM( "velErr: " << velErr.transpose() );
		ROS_INFO_STREAM( "raw reward: " << reward );

		reward = std::max( std::min( reward, _maxReward ), _minReward );
		if( _logRewards )
		{
			reward = std::log( reward );
		}
		// Negate the predicted error to make it a reward
		reward = -reward;

		percepto_msgs::RewardStamped out;
		out.header = msg->header;
		out.reward = reward;
		_rewardPub.publish( out );
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:

	ros::Subscriber _odomSub;
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

int main( int argc, char **argv )
{
	ros::init( argc, argv, "sse_reward" );
	ros::NodeHandle nh, ph("~");
	SumSquaredErrorReward sse( nh, ph );
	ros::spin();
	return 0;
}