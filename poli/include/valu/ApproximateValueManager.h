#pragma once

#include "valu/ValuInterfaces.h"
#include "valu/ApproximateValue.h"
#include "valu/ValueInfoManager.h"

#include "broadcast/BroadcastMultiReceiver.h"

#include "percepto_msgs/SetParameters.h"
#include "percepto_msgs/GetParameters.h"

#include "argus_utils/synchronization/SynchronizationTypes.h"

namespace percepto
{

class ApproximateValueManager
: public Critic
{
public:

	typedef std::shared_ptr<ApproximateValueManager> Ptr;

	ApproximateValueManager();

	void Initialize( ros::NodeHandle& ph );

	void Initialize( argus::BroadcastMultiReceiver::Ptr inputs,
	                 ros::NodeHandle& ph );

	virtual double GetCritique( const ros::Time& time ) const;

private:

	mutable argus::Mutex _mutex;

	ApproximateValue _value;
	argus::BroadcastMultiReceiver::Ptr _receiver;

	argus::LookupInterface _lookup;
	ValueInfoManager _infoManager;

	ros::ServiceServer _getParamServer;
	ros::ServiceServer _setParamServer;

	VectorType GetInput( const ros::Time& time ) const;
	
	bool SetParamsCallback( percepto_msgs::SetParameters::Request& req,
	                        percepto_msgs::SetParameters::Response& res );
	bool GetParamsCallback( percepto_msgs::GetParameters::Request& req,
	                        percepto_msgs::GetParameters::Response& res );

};


}