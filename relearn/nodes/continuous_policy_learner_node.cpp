#include "relearn/ContinuousPolicyLearner.h"

using namespace percepto;

int main( int argc, char** argv )
{
	ros::init( argc, argv, "continuous_policy_learner_node" );

	ros::NodeHandle nh, ph( "~" );
	ContinuousPolicyLearner cpl;
	cpl.Initialize( nh, ph );
	ros::spin();
	
	return 0;
}