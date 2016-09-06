#include <ros/ros.h>

#include "relearn/ApproximateValue.h"
#include "percepto_msgs/RewardStamped.h"
#include "argus_utils/utils/ParamUtils.h"

using namespace argus;
using namespace percepto;

class ApproximateValueNode
{
public:

	ApproximateValueNode( ros::NodeHandle& nh,
	                      ros::NodeHandle& ph )
	{
		_valueFunction.Initialize( nh, ph );

		_valuePub = ph.advertise<percepto_msgs::RewardStamped>( "output", 0 );

		double updateRate;
		GetParamRequired( ph, "update_rate", updateRate );
		_publishTimer = nh.createTimer( ros::Duration( 1.0/updateRate ),
		                                &ApproximateValueNode::UpdateCallback,
		                                this );
	}

private:

	ApproximateValue _valueFunction;

	ros::Publisher _valuePub;
	ros::Timer _publishTimer;

	void UpdateCallback( const ros::TimerEvent& event )
	{
		try
		{
			percepto_msgs::RewardStamped msg;
			msg.header.stamp = event.current_real;
			msg.reward = _valueFunction.Evaluate( ParamAction( event.current_real, VectorType() ) );
			_valuePub.publish( msg );
		}
		catch( std::out_of_range e )
		{
			ROS_WARN_STREAM( "Could not publish value at time: " << event.current_real );
		}
	}

};

int main( int argc, char** argv )
{
	ros::init( argc, argv, "value_approximator_node" );

	ros::NodeHandle nh, ph( "~" );
	ApproximateValueNode avn( nh, ph );
	ros::spin();

	return 0;
}