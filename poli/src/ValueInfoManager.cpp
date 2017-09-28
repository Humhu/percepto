#include "valu/ValueInfoManager.h"
#include "argus_utils/utils/ParamUtils.h"

#define INPUT_DIM_KEY ("input_dim")
#define PARAM_QUERY_KEY ("param_query_service")
#define PARAM_SET_KEY ("param_set_service")
#define INFO_QUERY_KEY ("approximator" )

using namespace argus;

namespace percepto
{

ValueInfoManager::ValueInfoManager( argus::LookupInterface& interface )
: InfoManager( interface )
{}

bool ValueInfoManager::ParseMemberInfo( const std::string& memberNamespace,
                                        ValueInfo& info )
{
	if( !GetParam( _nodeHandle, memberNamespace + INPUT_DIM_KEY, info.inputDim ) )
	{
		ROS_WARN_STREAM( "Could not find value input information at path " <<
		                 memberNamespace + INPUT_DIM_KEY );
		return false;
	}
	if( !GetParam( _nodeHandle, 
	               memberNamespace + PARAM_QUERY_KEY, 
	               info.paramQueryService ) )
	{
		ROS_WARN_STREAM( "Could not find value parameter query service at path " <<
		                 memberNamespace + PARAM_QUERY_KEY );
		return false;
	}
	if( !GetParam( _nodeHandle,
	               memberNamespace + PARAM_SET_KEY,
	               info.paramSetService ) )
	{
		ROS_WARN_STREAM( "Could not find value set service at path " <<
		                 memberNamespace + PARAM_SET_KEY );
		return false;
	}
	if( !GetParam( _nodeHandle, 
	               memberNamespace + INFO_QUERY_KEY,
	               info.approximatorInfo ) )
	{
		ROS_WARN_STREAM( "Could not find value network info at path " <<
		                 memberNamespace + INFO_QUERY_KEY );
		return false;
	}
	return true;
}

void ValueInfoManager::PopulateMemberInfo( const ValueInfo& info,
                                           const std::string& memberNamespace )
{
	_nodeHandle.setParam( memberNamespace + INPUT_DIM_KEY, (int) info.inputDim );
	_nodeHandle.setParam( memberNamespace + PARAM_QUERY_KEY, info.paramQueryService );
	_nodeHandle.setParam( memberNamespace + PARAM_SET_KEY, info.paramSetService );
	SetYamlParam( _nodeHandle, memberNamespace + INFO_QUERY_KEY, info.approximatorInfo );
}

}