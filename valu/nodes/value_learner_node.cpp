#include "valu/ApproximateValueLearner.h"

using namespace percepto;

int main( int argc, char** argv )
{
	ros::init( argc, argv, "approximate_value_learner_node" );

	ros::NodeHandle nh, ph( "~" );
	ApproximateValueLearner avl;
	avl.Initialize( nh, ph );
	ros::spin();
	
	return 0;
}