#include <ros/ros.h>

#include "valu/SRSSampler.h"

using namespace percepto;

int main( int argc, char** argv )
{
	ros::init( argc, argv, "srs_sampler_node" );

	ros::NodeHandle nh, ph( "~" );
	SRSSampler ssn;
	ssn.Initialize( nh, ph );
	ros::spin();
	return 0;
}