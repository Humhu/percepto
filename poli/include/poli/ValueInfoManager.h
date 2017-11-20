#pragma once

#include "lookup/InfoManager.h"
#include <yaml-cpp/yaml.h>

namespace percepto
{

struct ValueInfo
{
	unsigned int inputDim;

	std::string paramQueryService;
	std::string paramSetService;

	YAML::Node approximatorInfo;
};

class ValueInfoManager
: public argus::InfoManager<ValueInfo>
{
public:

	ValueInfoManager( argus::LookupInterface& interface );

protected:

	ros::NodeHandle _nodeHandle;

	virtual bool ParseMemberInfo( const std::string& memberNamespace,
	                              ValueInfo& info );
	virtual void PopulateMemberInfo( const ValueInfo& info,
	                                 const std::string& memberNamespace );


};

}