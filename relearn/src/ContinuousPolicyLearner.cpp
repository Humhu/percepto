#include "relearn/ContinuousPolicyLearner.h"

#include "optim/OptimizerParser.h"

#include "argus_utils/utils/ParamUtils.h"
#include "argus_utils/utils/MapUtils.hpp"

#include "percepto_msgs/GetCritique.h"
#include "percepto_msgs/GetParameters.h"
#include "percepto_msgs/SetParameters.h"

#include <ros/service.h>

using namespace argus_msgs;
using namespace argus;

namespace percepto
{

ContinuousPolicyLearner::ContinuousPolicyLearner() 
: _infoManager( _lookup ) {}

void ContinuousPolicyLearner::Initialize( ros::NodeHandle& nh, 
                                          ros::NodeHandle& ph )
{
	WriteLock lock( _mutex );
	
	std::string policyName;
	GetParamRequired( ph, "policy_name", policyName );

	// Read policy information
	if( !_infoManager.CheckMemberInfo( policyName, true, ros::Duration( 10.0 ) ) )
	{
		throw std::runtime_error( "Could not find policy: " + policyName );
	}
	const PolicyInfo& info = _infoManager.GetInfo( policyName );
	_policy.Initialize( info.inputDim, info.outputDim, info.networkInfo );
	
	// Get initial policy parameters
	ros::service::waitForService( info.paramQueryService );
	ros::service::waitForService( info.paramSetService );

	percepto_msgs::GetParameters::Request req;
	percepto_msgs::GetParameters::Response res;
	if( !ros::service::call( info.paramQueryService, req, res ) )
	{
		throw std::runtime_error( "Could not query parameters at: " + info.paramQueryService );
	}
	_policy.GetParameters()->SetParamsVec( GetVectorView( res.parameters ) );
	ROS_INFO_STREAM( "Initialized policy: " << std::endl << *_policy.GetPolicyModule() );

	_setParamsClient = nh.serviceClient<percepto_msgs::SetParameters>( info.paramSetService, true );

	// Read optimization parameters
	ros::NodeHandle lh( ph.resolveName( "optimization" ) );	
	GetParamRequired( lh, "min_num_modules", _minModulesToOptimize );
	GetParam( lh, "clear_optimized_modules", _clearAfterOptimize, false );
	if( !_clearAfterOptimize )
	{
		GetParamRequired( lh, "max_num_modules", _maxModulesToKeep );
	}
	_optimizer = parse_modular_optimizer( lh );

	double l2Weight, maxDivergence;
	unsigned int batchSize;
	GetParamRequired( lh, "l2_weight", l2Weight );
	GetParamRequired( lh, "batch_size", batchSize );
	GetParamRequired( lh, "max_divergence", maxDivergence );
	_optimization.Initialize( _policy.GetParameters(), l2Weight, batchSize, maxDivergence );

	std::string critiqueTopic;
	GetParamRequired( ph, "critique_service_topic", critiqueTopic );
	ros::service::waitForService( critiqueTopic );
	_getCritiqueClient = nh.serviceClient<percepto_msgs::GetCritique>( critiqueTopic, true );

	// Subscribe to policy action feed
	_actionSub = nh.subscribe( "actions", 
	                           0, 
	                           &ContinuousPolicyLearner::ActionCallback, 
	                           this );

	_lastOptimizationTime = ros::Time::now();
	double updateRate;
	GetParamRequired( lh, "update_rate", updateRate );
	_updateTimer = nh.createTimer( ros::Duration( 1.0/updateRate ),
	                               &ContinuousPolicyLearner::TimerCallback,
	                               this );
}

void ContinuousPolicyLearner::ActionCallback( const percepto_msgs::ContinuousAction::ConstPtr& msg )
{
	if( msg->header.stamp < _lastOptimizationTime ) { return; }

	// TODO Check for identical timestamps
	ContinuousAction action( *msg );
	_actionBuffer[ msg->header.stamp ] = *msg;
}

void ContinuousPolicyLearner::TimerCallback( const ros::TimerEvent& event )
{
	// Wait until we have enough actions buffered before we begin optimization
	if( _actionBuffer.size() < _minModulesToOptimize ) 
	{
		ROS_INFO_STREAM( "Action buffer size: " << _actionBuffer.size() << 
		                 " less than min: " << _minModulesToOptimize );
		return;
	}

	// First process buffer
	ros::Time now = event.current_real;
	while( _actionBuffer.size() > 0 )
	{
		ContinuousAction action( _actionBuffer.begin()->second );

		percepto_msgs::GetCritique getCritique;
		getCritique.request.time = action.time;

		if( !_getCritiqueClient.call( getCritique ) )
		{
			ROS_WARN_STREAM( "Could not get critique for time: " << action.time );
			break;
		}
		double advantage = getCritique.response.critique;
		ROS_INFO_STREAM( "Advantage: " << advantage );

		remove_lowest( _actionBuffer );
		if( !action.input.allFinite() )
		{
			ROS_WARN_STREAM( "Received non-finite input: " << action.input.transpose() );
			continue;
		}
		if( !action.output.allFinite() )
		{
			ROS_WARN_STREAM( "Received non-finite action: " << action.output.transpose() );
			continue;
		}

		if( !std::isfinite( advantage ) )
		{
			ROS_WARN_STREAM( "Received non-finite advantage: " << advantage );
			continue;
		}
		_optimization.EmplaceModule( _policy.GetPolicyModule(),
		                             action.input, 
		                             action.output,
		                             advantage );

		ROS_INFO_STREAM( "Action: " << action.output.transpose() << 
		                 " Input: " << action.input.transpose() << 
		                 " Advantage: " << advantage );
	}

	if( _optimization.NumModules() == 0 ) 
	{
		_actionBuffer.clear();
		return;
	}

	// Perform optimization
	ROS_INFO_STREAM( "Optimizing with " << _optimization.NumModules() << " modules." );
	_optimization.ResetConstraints();
	_optimizer->ResetTerminationCheckers();
	percepto::OptimizationResults results = _optimizer->Optimize( _optimization );

	ROS_INFO_STREAM( "Result: " << results.status );
	ROS_INFO_STREAM( "Objective: " << results.finalObjective );
	ROS_INFO_STREAM( "Policy: " << *_policy.GetPolicyModule() );

	// Now that we've changed the policy, these old actions are no longer valid
	_lastOptimizationTime = ros::Time::now();
	_actionBuffer.clear();

	// Trim old data
	unsigned int targetModules = _maxModulesToKeep;
	if( _clearAfterOptimize )
	{
		targetModules = 0;
	}
	while( _optimization.NumModules() > targetModules )
	{
		_optimization.RemoveOldest();
	}

	// Set new parameters
	percepto_msgs::SetParameters srv;
	SerializeMatrix( _policy.GetParameters()->GetParamsVec(), srv.request.parameters );
	if( !_setParamsClient.call( srv ) )
	{
		throw std::runtime_error( "Could not set parameters." );
	}
}

}