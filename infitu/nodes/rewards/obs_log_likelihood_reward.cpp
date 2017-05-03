#include <ros/ros.h>

#include <argus_msgs/FilterStepInfo.h>
#include <percepto_msgs/RewardStamped.h>

#include "argus_utils/utils/ParamUtils.h"
#include "argus_utils/random/MultivariateGaussian.hpp"
#include "argus_utils/filter/FilterInfo.h"

using namespace argus;

class ObservationLogLikelihood
{
public:

	ObservationLogLikelihood( ros::NodeHandle& nh, ros::NodeHandle& ph )
	{
		_rewardPub = ph.advertise<percepto_msgs::RewardStamped>( "reward", 10 );
		_infoSub = nh.subscribe( "info", 10, &ObservationLogLikelihood::InfoCallback, this );

		GetParam( ph, "min_reward", _minReward, -std::numeric_limits<double>::infinity() );
		GetParam( ph, "max_reward", _maxReward, std::numeric_limits<double>::infinity() );
	}

	void InfoCallback( const argus_msgs::FilterStepInfo::ConstPtr& msg )
	{
		if( msg->info_type != argus_msgs::FilterStepInfo::UPDATE_STEP ) { return; }
		UpdateInfo info( *msg );

		double reward = GaussianLogPdf( info.obs_error_cov, info.prior_obs_error );
		reward = std::max( std::min( reward, _maxReward ), _minReward );

		percepto_msgs::RewardStamped out;
		out.header = msg->header;
		out.reward = reward;
		_rewardPub.publish( out );
	}

private:

	ros::Subscriber _infoSub;
	ros::Publisher _rewardPub;
	double _minReward;
	double _maxReward;
};

int main( int argc, char **argv )
{
	ros::init( argc, argv, "observation_log_likelihood_reward" );
	ros::NodeHandle nh, ph( "~" );
	ObservationLogLikelihood oll( nh, ph );
	ros::spin();
	return 0;
}