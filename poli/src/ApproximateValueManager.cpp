#include "valu/ApproximateValueManager.h"
#include "lookup/LookupUtils.hpp"

using namespace argus;

namespace percepto
{

ApproximateValueManager::ApproximateValueManager() 
: _infoManager( _lookup ) {}

void ApproximateValueManager::Initialize( ros::NodeHandle& ph )
{
	ros::NodeHandle ih( ph.resolveName( "input_streams" ) );
	BroadcastMultiReceiver::Ptr rx = std::make_shared<BroadcastMultiReceiver>();
	rx->Initialize( ih );
	Initialize( rx, ph );
}

void ApproximateValueManager::Initialize( BroadcastMultiReceiver::Ptr inputs,
                                          ros::NodeHandle& ph )
{
	WriteLock lock( _mutex );

	_receiver = inputs;
	unsigned int inputDim = _receiver->GetDim();
	ros::NodeHandle vh( ph.resolveName( "approximator" ) );
	_value.Initialize( inputDim, vh );

	_getParamServer = ph.advertiseService( "get_params", &ApproximateValueManager::GetParamsCallback, this );
	_setParamServer = ph.advertiseService( "set_params", &ApproximateValueManager::SetParamsCallback, this );

	std::string valueName;
	GetParamRequired( ph, "value_name", valueName );
	register_lookup_target( ph, valueName );

	ValueInfo info;
	info.inputDim = _receiver->GetDim();
	info.paramQueryService = ph.resolveName( "get_params" );
	info.paramSetService = ph.resolveName( "set_params" );
	GetParamRequired( vh, "", info.approximatorInfo );
	if( !_infoManager.WriteMemberInfo( valueName, info, true, ros::Duration( 10.0 ) ) )
	{
		throw std::runtime_error( "Could not write value info!" );
	}
}

double ApproximateValueManager::GetCritique( const ros::Time& time ) const
{
	// ROS_INFO_STREAM( "Enter GetCritique" );
	WriteLock lock( _mutex );
	// ROS_INFO_STREAM( "Acquired GetCritique" );
	return _value.GetValue( GetInput( time ) );
}

VectorType ApproximateValueManager::GetInput( const ros::Time& time ) const
{
	if( !_receiver->IsReady() )
	{
		throw std::out_of_range( "Receiver is not ready." );
	}

	StampedFeatures features;
	if( !_receiver->ReadStream( time, features ) )
	{
		std::stringstream ss;
		ss << "ApproximateValueManager: Could not get input stream. Earliest time: "
		   << _receiver->EarliestTime() << " latest: " << _receiver->LatestTime();
		throw std::out_of_range( ss.str() );
	}
	return features.features;
}

bool ApproximateValueManager::SetParamsCallback( percepto_msgs::SetParameters::Request& req,
                                                 percepto_msgs::SetParameters::Response& res )
{
	WriteLock lock( _mutex );
	_value.GetParameters()->SetParamsVec( GetVectorView( req.parameters ) );
	return true;
}

bool ApproximateValueManager::GetParamsCallback( percepto_msgs::GetParameters::Request& req,
                                                 percepto_msgs::GetParameters::Response& res )
{
	ReadLock lock( _mutex );
	SerializeMatrix( _value.GetParameters()->GetParamsVec(), res.parameters );
	return true;
}

}