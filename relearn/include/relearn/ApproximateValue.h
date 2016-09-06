#pragma once

#include "relearn/RelearnInterfaces.h"
#include "relearn/ValueFunctionModules.h"

#include "broadcast/BroadcastMultiReceiver.h"

#include "argus_msgs/FloatVectorStamped.h"

namespace percepto
{

class ApproximateValue
: public PolicyCritic
{
public:

	typedef std::shared_ptr<ApproximateValue> Ptr;

	ApproximateValue();

	void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph );

	virtual double Evaluate( const ParamAction& x ) const;

	ScalarFieldApproximator::Ptr CreateApproximatorModule() const;

	VectorType GetInput( const ros::Time& time ) const;
	percepto::Parameters::Ptr GetParameters() const;

private:

	ros::Subscriber _paramsSub;

	std::string _moduleType;
	mutable percepto::TerminalSource<VectorType> _approximatorInput;
	mutable ScalarFieldApproximator::Ptr _approximator;
	percepto::Parameters::Ptr _approximatorParams;

	argus::BroadcastMultiReceiver _receiver;

	void ParamsCallback( const argus_msgs::FloatVectorStamped::ConstPtr& msg );

};

}