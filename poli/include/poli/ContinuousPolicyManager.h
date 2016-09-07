#pragma once

#include "poli/ContinuousPolicy.h"
#include "poli/PoliInterfaces.h"
#include "poli/PolicyInfoManager.h"

#include "broadcast/BroadcastMultiReceiver.h"
#include "argus_utils/random/MultivariateGaussian.hpp"

#include "percepto_msgs/SetParameters.h"
#include "percepto_msgs/GetParameters.h"

namespace percepto
{

class ContinuousPolicyManager
{
public:

	typedef ContinuousPolicy::DistributionParameters DistributionParameters;

	ContinuousPolicyManager();
	
	void Initialize( ContinuousPolicyInterface* interface,
	                 ros::NodeHandle& nh, 
	                 ros::NodeHandle& ph );

	VectorType GetInput( const ros::Time& time );
	
	// NOTE Returns normalized distribution parameters
	DistributionParameters GetNormalizedDistribution( const ros::Time& time );
	DistributionParameters GetDistribution( const ros::Time& time );
	
	VectorType GetNormalizedOutput( const ros::Time& time );
	VectorType GetOutput( const ros::Time& time );

	// Samples from the policy and sets the output through the interface
	ContinuousAction Execute( const ros::Time& now );

private:

	std::string _policyName;

	argus::LookupInterface _lookup;
	PolicyInfoManager _infoManager;

	ContinuousPolicy _policy;
	ContinuousPolicyInterface* _interface;

	ros::ServiceServer _getParamServer;
	ros::ServiceServer _setParamServer;

	VectorType _policyScales;
	VectorType _policyOffsets;

	ros::Publisher _actionPub;
	ros::Publisher _normalizedActionPub;

	argus::BroadcastMultiReceiver _inputStreams;

	argus::MultivariateGaussian<> _mvg;
	double _maxSampleDevs;

	bool SetParamHandler( percepto_msgs::SetParameters::Request& req,
	                      percepto_msgs::SetParameters::Response& res );
	bool GetParamHandler( percepto_msgs::GetParameters::Request& req,
	                      percepto_msgs::GetParameters::Response& res );
};

}