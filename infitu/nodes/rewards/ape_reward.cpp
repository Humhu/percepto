#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <percepto_msgs/RewardStamped.h>

#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

class ApproximatePosteriorReward
{
public:

	ApproximatePosteriorReward( ros::NodeHandle& nh, ros::NodeHandle& ph )
	{
		_rewardPub = ph.advertise<percepto_msgs::RewardStamped>( "reward", 10 );
		_odomSub = nh.subscribe( "odom", 10, &ApproximatePosteriorReward::OdomCallback, this );

		GetParam( ph, "log_rewards", _logRewards, false );
		GetParam( ph, "min_reward", _minReward, -std::numeric_limits<double>::infinity() );
		GetParam( ph, "max_reward", _maxReward, std::numeric_limits<double>::infinity() );

		GetParam( ph, "pose_cov_weights", _posCovWeights, FixedVectorType<6>::Zero() );
		GetParam( ph, "vel_cov_weights", _velCovWeights, FixedVectorType<6>::Zero() );
	}

	void OdomCallback( const nav_msgs::Odometry::ConstPtr& msg )
	{
		FixedMatrixType<6, 6> poseCov, velCov;
		ParseMatrix( msg->pose.covariance, poseCov );
		ParseMatrix( msg->twist.covariance, velCov );

		double reward = _posCovWeights.dot( poseCov.diagonal() ) +
		                _velCovWeights.dot( velCov.diagonal() );
		
		if( _logRewards )
		{
			reward = std::log( reward );
		}
		// Negate the predicted error to make it a reward
		reward = -reward;

		reward = std::max( std::min( reward, _maxReward ), _minReward );

		percepto_msgs::RewardStamped out;
		out.header.stamp = msg->header.stamp;
		out.header.frame_id= "ape";
		out.reward = reward;
		_rewardPub.publish( out );
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:

	ros::Subscriber _odomSub;
	ros::Publisher _rewardPub;
	bool _logRewards;
	double _minReward;
	double _maxReward;

	FixedVectorType<6> _posCovWeights;
	FixedVectorType<6> _velCovWeights;
};

int main( int argc, char **argv )
{
	ros::init( argc, argv, "ape_reward" );
	ros::NodeHandle nh, ph("~");
	ApproximatePosteriorReward ape( nh, ph );
	ros::spin();
	return 0;
}