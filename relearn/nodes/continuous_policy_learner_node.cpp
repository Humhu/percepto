#include "relearn/ContinuousPolicyLearner.h"

using namespace percepto;

class ContinuousPolicyLearnerNode
{
public:

	ContinuousPolicyLearnerNode( ros::NodeHandle& nh,
	                             ros::NodeHandle& ph )
	{
		_learner.Initialize( nh, ph );
	}

private:

	ContinuousPolicyLearner _learner;
};

int main( int argc, char** argv )
{
	ros::init( argc, argv, "continuous_policy_learner_node" );

	ros::NodeHandle nh, ph( "~" );
	ContinuousPolicyLearnerNode cpl( nh, ph );
	ros::spin();
	
	return 0;
}