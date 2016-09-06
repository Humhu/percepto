#pragma once

#include "lookup/InfoManager.h"
#include <yaml-cpp/yaml.h>

namespace percepto
{

// TODO Continuous/discrete mode enum?
struct PolicyInfo
{
	unsigned int inputDim;
	unsigned int outputDim;
	
	std::string paramQueryService;
	std::string paramSetService;

	YAML::Node networkInfo;
};

class PolicyInfoManager
: public argus::InfoManager<PolicyInfo>
{
public:

	PolicyInfoManager( argus::LookupInterface& interface );

protected:

	ros::NodeHandle _nodeHandle;

	virtual bool ParseMemberInfo( const std::string& memberNamespace,
	                              PolicyInfo& info );
	virtual void PopulateMemberInfo( const PolicyInfo& info,
	                                 const std::string& memberNamespace );
};

}