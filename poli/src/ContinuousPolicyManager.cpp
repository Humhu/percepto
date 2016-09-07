#include "poli/ContinuousPolicyManager.h"
#include "percepto_msgs/ContinuousAction.h"

#include "argus_utils/utils/MatrixUtils.h"
#include "modprop/utils/Randomization.hpp"

#include "lookup/LookupUtils.hpp"

#include <boost/foreach.hpp>
#include <sstream>

using namespace argus;

namespace percepto
{

ContinuousPolicyManager::ContinuousPolicyManager() 
: _infoManager( _lookup ) {}

void ContinuousPolicyManager::Initialize( ContinuousPolicyInterface* interface,
                                          ros::NodeHandle& nh,
                                          ros::NodeHandle& ph )
{
	// Read information from interface
	_interface = interface;
	const VectorType& lowerLimit = _interface->GetLowerLimits();
	const VectorType& upperLimit = _interface->GetUpperLimits();
	unsigned int outputDim = _interface->GetNumOutputs();
	std::vector<std::string> paramNames = _interface->GetParameterNames();

	// Read information from input listener
	ros::NodeHandle subh( ph.resolveName("input_streams") );
	_inputStreams.Initialize( subh );
	unsigned int inputDim = _inputStreams.GetDim();

	// Parse policy info
	ros::NodeHandle neth( ph.resolveName("policy_class") );
	_policy.Initialize( inputDim, outputDim, neth );

	// Compute initializations
	_policyScales = VectorType( outputDim );
	_policyOffsets = VectorType( outputDim );
	
	DistributionParameters initParams;
	initParams.mean = VectorType( outputDim );
	initParams.info = MatrixType::Identity( outputDim, outputDim );
	for( unsigned int i = 0; i < paramNames.size(); ++i )
	{
		if( !std::isfinite( upperLimit(i) ) || !std::isfinite( lowerLimit(i) ) )
		{
			_policyScales(i) = 1.0;
			_policyOffsets(i) = 0.0;
		}
		else
		{
			_policyScales(i) = ( upperLimit(i) - lowerLimit(i) ) / 2;
			_policyOffsets(i) = ( upperLimit(i) - lowerLimit(i) ) / 2;
		}
		initParams.mean(i) = 0;
		initParams.info(i,i) = 10;
	}
	_policy.InitializeOutput( initParams );

	ROS_INFO_STREAM( "Network initialized: " << std::endl << *_policy.GetPolicyModule() );
	ROS_INFO_STREAM( "Policy scales: " << _policyScales.transpose() << std::endl <<
	                 "Policy offsets: " << _policyOffsets.transpose() << std::endl );

	// See if any RNG seed was given
	unsigned int seed;
	if( HasParam( ph, "rng_seed" ) )
	{
		GetParamRequired( ph, "rng_seed", seed );
		_mvg = MultivariateGaussian<>( outputDim, seed );
	}
	else
	{
		_mvg = MultivariateGaussian<>( outputDim );
	}
	GetParam( ph, "max_sample_devs", _maxSampleDevs, 3.0 );

	// Create publisher topics
	_actionPub = ph.advertise<percepto_msgs::ContinuousAction>( "action_raw", 0 );
	_normalizedActionPub = ph.advertise<percepto_msgs::ContinuousAction>( "action_normalized", 0 );

	// Advertise parameter services
	std::string getParamTopic = ph.resolveName( "get_params" );
	std::string setParamTopic = ph.resolveName( "set_params" );
	_getParamServer = nh.advertiseService( getParamTopic, &ContinuousPolicyManager::GetParamHandler, this );
	_setParamServer = nh.advertiseService( setParamTopic, &ContinuousPolicyManager::SetParamHandler, this );

	GetParamRequired( ph, "policy_name", _policyName );
	register_lookup_target( ph, _policyName );

	// Register information
	PolicyInfo info;
	info.inputDim = inputDim;
	info.outputDim = outputDim;
	info.paramQueryService = getParamTopic;
	info.paramSetService = setParamTopic;
	GetParamRequired( neth, "", info.networkInfo );
	if( !_infoManager.WriteMemberInfo( _policyName, info, true, ros::Duration( 10.0 ) ) )
	{
		throw std::runtime_error( "Could not write policy info!" );
	}
}

VectorType ContinuousPolicyManager::GetInput( const ros::Time& time )
{
	StampedFeatures input;
	if( !_inputStreams.ReadStream( time, input ) )
	{
		std::stringstream ss;
		ss << "Could not read input streams at time: " << time;
		throw std::out_of_range( ss.str() );
	}
	return input.features;
}

ContinuousPolicyManager::DistributionParameters
ContinuousPolicyManager::GetNormalizedDistribution( const ros::Time& time )
{
	return _policy.GenerateOutput( GetInput( time ) );
}

ContinuousPolicyManager::DistributionParameters 
ContinuousPolicyManager::GetDistribution( const ros::Time& time )
{
	DistributionParameters initParams = GetNormalizedDistribution( time );
	initParams.mean = ( _policyScales.array() * initParams.mean.array() ).matrix();
	Eigen::LDLT<MatrixType> infoLDLT( initParams.info );
	MatrixType cov = infoLDLT.solve( MatrixType::Identity( initParams.mean.size(),
	                                                       initParams.mean.size() ) );
	cov = _policyScales.transpose() * cov * _policyScales;
	Eigen::LDLT<MatrixType> covLDLT( cov );
	initParams.info = covLDLT.solve( MatrixType::Identity( initParams.mean.size(),
	                                                   initParams.mean.size() ) );
	return initParams;
}

VectorType ContinuousPolicyManager::GetNormalizedOutput( const ros::Time& time )
{
	ContinuousPolicyManager::DistributionParameters initParams = GetNormalizedDistribution( time );
	_mvg.SetMean( initParams.mean );
	_mvg.SetInformation( initParams.info );
	return _mvg.Sample( _maxSampleDevs );
}

VectorType ContinuousPolicyManager::GetOutput( const ros::Time& time )
{
	VectorType out = GetNormalizedOutput( time );
	return ( out.array() * _policyScales.array() ).matrix() + _policyOffsets;
}

ContinuousAction ContinuousPolicyManager::Execute( const ros::Time& now )
{
	VectorType in = GetInput( now );
	VectorType out = GetOutput( now );
	_interface->SetOutput( out );

	ContinuousAction act( now, in, out );
	_actionPub.publish( act.ToMsg() );
	act.Normalize( _policyScales, _policyOffsets );
	_normalizedActionPub.publish( act.ToMsg() );
	act.Unnormalize( _policyScales, _policyOffsets );
	return act;
}

bool ContinuousPolicyManager::SetParamHandler( percepto_msgs::SetParameters::Request& req,
                                               percepto_msgs::SetParameters::Response& res )
{
	_policy.GetParameters()->SetParamsVec( GetVectorView( req.parameters ) );
	return true;
}

bool ContinuousPolicyManager::GetParamHandler( percepto_msgs::GetParameters::Request& req,
                                               percepto_msgs::GetParameters::Response& res )
{
	VectorType initParams = _policy.GetParameters()->GetParamsVec();
	res.parameters = std::vector<double>( initParams.data(), initParams.data() + initParams.size() );
	return true;
}

}