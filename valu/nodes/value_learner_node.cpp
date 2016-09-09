#include "valu/ApproximateValueLearner.h"
#include "argus_utils/utils/ParamUtils.h"

#include <ros/ros.h>

using namespace argus;
using namespace percepto;

int main( int argc, char** argv )
{
	ros::init( argc, argv, "approximate_value_learner_node" );

	ros::NodeHandle nh, ph( "~" );
	ApproximateValueLearner avl;
	avl.Initialize( nh, ph );
	
	unsigned int numThreads;
	GetParam( ph, "num_threads", numThreads, (unsigned int) 2 );
	ros::AsyncSpinner spinner( numThreads );
	spinner.start();
	ros::waitForShutdown();

	return 0;
}