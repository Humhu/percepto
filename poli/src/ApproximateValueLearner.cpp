#include "valu/ApproximateValueLearner.h"

#include "argus_utils/utils/ParamUtils.h"
#include "argus_utils/utils/MatrixUtils.h"

#include "percepto_msgs/GetParameters.h"
#include "percepto_msgs/SetParameters.h"

#include "optim/OptimizerParser.h"

#include <ros/service.h>
#include <boost/foreach.hpp>

using namespace argus;

namespace percepto
{

ApproximateValueLearner::ApproximateValueLearner() 
: _infoManager( _lookup ), _networkInitialized( false ) {}

void ApproximateValueLearner::Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph )
{
	// Initialize the value function
	std::string valueName;
	GetParamRequired( ph, "value_name", valueName );
	if( !_infoManager.CheckMemberInfo( valueName, true, ros::Duration( 10.0 ) ) )
	{
		throw std::runtime_error( "Could not find value function: " + valueName );
	}

	const ValueInfo& info = _infoManager.GetInfo( valueName );
	_value.Initialize( info.inputDim, info.approximatorInfo );

	ros::service::waitForService( info.paramQueryService );
	ros::service::waitForService( info.paramSetService );
	
	// Get the current value function parameters
	percepto_msgs::GetParameters::Request req;
	percepto_msgs::GetParameters::Response res;
	if( !ros::service::call( info.paramQueryService, req, res ) )
	{
		throw std::runtime_error( "Could not query parameters at: " + info.paramQueryService );
	}
	_value.GetParameters()->SetParamsVec( GetVectorView( res.parameters ) );
	ROS_INFO_STREAM( "Initialized value: " << std::endl << *_value.GetApproximatorModule() );

	_setParamsClient = nh.serviceClient<percepto_msgs::SetParameters>( info.paramSetService, true );

	ros::NodeHandle lh( ph.resolveName( "optimization" ) );
	GetParamRequired( lh, "min_num_modules", _minModulesToOptimize );
	GetParam( lh, "clear_optimized_modules", _clearAfterOptimize, false );
	if( !_clearAfterOptimize )
	{
		GetParamRequired( lh, "max_num_modules", _maxModulesToKeep );
	}

	_optimizer = parse_modular_optimizer( lh );
	_optimizer->ResetAll();
	_optimCounter = 0;

	double l2Weight, valuePenaltyWeight;
	unsigned int batchSize;
	GetParamRequired( lh, "l2_weight", l2Weight );
	GetParamRequired( lh, "batch_size", batchSize );
	GetParamRequired( lh, "value_penalty_weight", valuePenaltyWeight );
	_problem.Initialize( _value.GetParameters(), 
	                     l2Weight, 
	                     batchSize, 
	                     valuePenaltyWeight );

	// TODO GetParam specialization for ros::Duration and ros::Rate
	double updateRate;
	GetParamRequired( ph, "update_rate", updateRate );
	GetParamRequired( ph, "discount_rate", _discountRate );

	_updateTimer = nh.createTimer( ros::Duration( 1.0/updateRate ),
	                               &ApproximateValueLearner::UpdateCallback,
	                               this );

	_srsSub = nh.subscribe( "srs_tuple", 0, &ApproximateValueLearner::SRSCallback, this );
}

void ApproximateValueLearner::SRSCallback( const percepto_msgs::SRSTuple::ConstPtr& msg )
{
	WriteLock lock( _mutex );
	_srsBuffer.emplace_back( *msg );
}

void ApproximateValueLearner::UpdateCallback( const ros::TimerEvent& event )
{
	WriteLock lock( _mutex );
	if( _problem.NumModules() < _minModulesToOptimize &&
	    _srsBuffer.size() < _minModulesToOptimize )
	{
		ROS_INFO_STREAM( "Num modules: " << _problem.NumModules() <<
		                 " and buffer size: " << _srsBuffer.size() << 
		                 " less than min: " << _minModulesToOptimize );
		return;
	}

	if( !_networkInitialized )
	{
		InitializeNetwork();
	}

	while( !_srsBuffer.empty() )
	{
		const SRSTuple& srs = _srsBuffer.front();

		double dt = (srs.nextTime - srs.time).toSec();
		double discountFactor = std::exp( dt * std::log( _discountRate ) );

		// ROS_INFO_STREAM( srs );
		_problem.EmplaceModule( _value.GetApproximatorModule(),
		                        srs.state,
		                        _value.GetApproximatorModule(),
		                        srs.nextState,
		                        srs.reward,
		                        discountFactor );
		_srsBuffer.pop_front();
	}

	RunOptimization();
}

void ApproximateValueLearner::RunOptimization()
{
	ROS_INFO_STREAM( "Beginning optimization..." );
	_optimizer->ResetTerminationCheckers();
	OptimizationResults results = _optimizer->Optimize( _problem );
	ROS_INFO_STREAM( "Result: " << results.status );
	ROS_INFO_STREAM( "Objective: " << results.finalObjective );
	_optimCounter++;
	if( _optimCounter % 10 == 0 )
	{
		ROS_INFO_STREAM( "Approximator: " << _value.GetApproximatorModule()->Print() );
	}

	unsigned int targetNum = _maxModulesToKeep;
	if( _clearAfterOptimize )
	{
		targetNum = 0;
	}
	while( _problem.NumModules() > targetNum )
	{
		_problem.RemoveOldest();
	}

	percepto_msgs::SetParameters setParams;
	SerializeMatrix( _value.GetParameters()->GetParamsVec(), setParams.request.parameters );
	if( !_setParamsClient.call( setParams ) )
	{
		throw std::runtime_error( "Could not set parameters." );
	}
}

void ApproximateValueLearner::InitializeNetwork()
{
	if( _networkInitialized )
	{
		throw std::runtime_error( "ApproximateValueLearner: Network already initialized!" );
	}

	// Calculate constant initial value for value network from current data
	double acc = 0;
	for( unsigned int i = 0; i < _srsBuffer.size(); ++i )
	{
		const SRSTuple& srs = _srsBuffer[i];

		double dt = (srs.nextTime - srs.time).toSec();
		double discountFactor = std::exp( dt * std::log( _discountRate ) );
		acc += srs.reward / ( 1 - discountFactor );
	}

	double initValue = acc / _srsBuffer.size();
	_value.InitializeOutput( initValue );
	_networkInitialized = true;
	ROS_INFO_STREAM( "Initialized value network to offset: " << initValue );
}

}