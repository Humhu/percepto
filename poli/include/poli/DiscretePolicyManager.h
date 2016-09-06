#pragma once

#include "poli/PoliInterfaces.h"
#include "poli/DiscretePolicyModules.h"
#include "argus_msgs/FloatVectorStamped.h"
#include "broadcast/BroadcastMultiReceiver.h"

#include <boost/random/mersenne_twister.hpp>

namespace percepto
{

// TODO Subscribe to parameter updates
class DiscretePolicyManager
{
public:

	typedef NormalizedPerceptron NetworkType;

	DiscretePolicyManager();

	void Initialize( DiscretePolicyInterface::Ptr& interface,
	                 ros::NodeHandle& nh, 
	                 ros::NodeHandle& ph );

	const NetworkType& GetPolicyModule() const;

private:

	DiscretePolicyInterface::Ptr _interface;
	ros::Publisher _actionPub;
	ros::Subscriber _paramSub;
	
	NetworkType::Ptr _network;
	TerminalSource<VectorType> _networkInput;
	Parameters::Ptr _networkParameters;
	
	argus::BroadcastMultiReceiver _inputStreams;
	
	boost::mt19937 _engine;
	ros::Timer _timer;

	void UpdateCallback( const ros::TimerEvent& event );
	void ParamCallback( const argus_msgs::FloatVectorStamped::ConstPtr& msg );
};

}