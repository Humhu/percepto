#include "valu/RewardInterpolater.h"
#include "broadcast/BroadcastMultiReceiver.h"

namespace percepto
{

class SRSSampler
{
public:

	SRSSampler();

	void Initialize( ros::NodeHandle& nh,
	                 ros::NodeHandle& ph );

	void Initialize( RewardInterpolater::Ptr rewards,
	                 argus::BroadcastMultiReceiver::Ptr inputs,
	                 ros::NodeHandle& ph );

private:

	RewardInterpolater::Ptr _rewards;
	argus::BroadcastMultiReceiver::Ptr _inputs;

	ros::Timer _sampleTimer;
	ros::Duration _timestep;
	ros::Duration _sampleOffset;

	ros::Publisher _srsPub;

	void TimerCallback( const ros::TimerEvent& event );
};

}