#include "poli/PolicyInfoManager.h"
#include "argus_utils/utils/ParamUtils.h"

#define INPUT_DIM_KEY ("/input_dim")
#define OUTPUT_DIM_KEY ("/output_dim")
#define PARAM_QUERY_KEY ("/param_query_service")
#define PARAM_SET_KEY ("/param_set_service")
#define INFO_QUERY_KEY ("/network" )

using namespace argus;

namespace percepto
{

PolicyInfoManager::PolicyInfoManager( argus::LookupInterface& interface )
: InfoManager( interface )
{}

bool PolicyInfoManager::ParseMemberInfo( const std::string& memberNamespace,
                                         PolicyInfo& info )
{
	if( !GetParam( _nodeHandle, memberNamespace + INPUT_DIM_KEY, info.inputDim ) )
	{
		ROS_WARN_STREAM( "Could not find policy input information at path " <<
		                 memberNamespace + INPUT_DIM_KEY );
		return false;
	}
	if( !GetParam( _nodeHandle, memberNamespace + OUTPUT_DIM_KEY, info.outputDim ) )
	{
		ROS_WARN_STREAM( "Could not find policy output information at path" <<
		                 memberNamespace + OUTPUT_DIM_KEY );
		return false;
	}
	if( !GetParam( _nodeHandle, 
	               memberNamespace + PARAM_QUERY_KEY, 
	               info.paramQueryService ) )
	{
		ROS_WARN_STREAM( "Could not find policy parameter query service at path " <<
		                 memberNamespace + PARAM_QUERY_KEY );
		return false;
	}
	if( !GetParam( _nodeHandle,
	               memberNamespace + PARAM_SET_KEY,
	               info.paramSetService ) )
	{
		ROS_WARN_STREAM( "Could not find policy set service at path " <<
		                 memberNamespace + PARAM_SET_KEY );
		return false;
	}
	if( !GetParam( _nodeHandle, 
	               memberNamespace + INFO_QUERY_KEY,
	               info.networkInfo ) )
	{
		ROS_WARN_STREAM( "Could not find policy network info at path " <<
		                 memberNamespace + INFO_QUERY_KEY );
		return false;
	}
	return true;
}

void PolicyInfoManager::PopulateMemberInfo( const PolicyInfo& info,
                                            const std::string& memberNamespace )
{
	_nodeHandle.setParam( memberNamespace + INPUT_DIM_KEY, (int) info.inputDim );
	_nodeHandle.setParam( memberNamespace + OUTPUT_DIM_KEY, (int) info.outputDim );
	_nodeHandle.setParam( memberNamespace + PARAM_QUERY_KEY, info.paramQueryService );
	SetYamlParam( _nodeHandle, memberNamespace + INFO_QUERY_KEY, info.networkInfo );
}

}