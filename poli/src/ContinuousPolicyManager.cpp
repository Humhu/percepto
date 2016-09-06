#include "poli/ContinuousPolicyManager.h"
#include "percepto_msgs/ContinuousAction.h"

#include "argus_utils/utils/MatrixUtils.h"
#include "modprop/utils/Randomization.hpp"

#include <boost/foreach.hpp>
#include <sstream>

using namespace argus;

namespace percepto
{

ContinuousPolicyManager::ContinuousPolicyManager() {}

void ContinuousPolicyManager::Initialize( ContinuousPolicyInterface* interface,
                                          ros::NodeHandle& nh,
                                          ros::NodeHandle& ph )
{

	// Initialize input listeners
	ros::NodeHandle subh( ph.resolveName("input_streams") );
	_inputStreams.Initialize( subh );
	
	_interface = interface;

	unsigned int inputDim = _inputStreams.GetDim();
	unsigned int outputDim = _interface->GetNumOutputs();

	_policy.Initialize( inputDim, outputDim, ph );

	// Read initialization if we have it
	const VectorType& lowerLimit = _interface->GetLowerLimits();
	const VectorType& upperLimit = _interface->GetUpperLimits();
	
	std::vector<std::string> paramNames = _interface->GetParameterNames();
	_policyScales = VectorType( outputDim );
	_policyOffsets = VectorType( outputDim );
	
	DistributionParameters initParams;
	initParams.mean = VectorType( outputDim );
	initParams.info = MatrixType::Identity( outputDim, outputDim );
	for( unsigned int i = 0; i < paramNames.size(); ++i )
	{
		_policyScales(i) = std::abs( upperLimit(i) - lowerLimit(i) );
		_policyOffsets(i) = ( upperLimit(i) - lowerLimit(i) ) / 2;
		initParams.mean(i) = 0;
		initParams.info(i,i) = 10;
	}
	_policy.InitializeOutput( initParams );

	ROS_INFO_STREAM( "Network initialized: " << std::endl << *_policy.GetPolicyModule() );

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

	_getParamServer = ph.advertiseService( "get_initParams", &ContinuousPolicyManager::GetParamHandler, this );
	_setParamServer = ph.advertiseService( "set_initParams", &ContinuousPolicyManager::SetParamHandler, this );
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
ContinuousPolicyManager::GetNormalizedDistributionParams( const ros::Time& time )
{
	return _policy.GenerateOutput( GetInput( time ) );
}

ContinuousPolicyManager::DistributionParameters 
ContinuousPolicyManager::GetDistributionParams( const ros::Time& time )
{
	DistributionParameters initParams = GetNormalizedDistributionParams( time );
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
	ContinuousPolicyManager::DistributionParameters initParams = GetDistributionParams( time );
	_mvg.SetMean( initParams.mean );
	_mvg.SetInformation( initParams.info );
	return _mvg.Sample( _maxSampleDevs );
}

VectorType ContinuousPolicyManager::GetOutput( const ros::Time& time )
{
	VectorType out = GetNormalizedOutput( time );
	return ( out.array() * _policyScales.array() ).matrix();
}

void ContinuousPolicyManager::Execute( const ros::Time& now )
{
	VectorType in = GetInput( now );
	VectorType out = GetOutput( now );
	_interface->SetOutput( out );

	ContinuousAction act( now, in, out );
	_actionPub.publish( act.ToMsg() );
	act.Normalize( _policyScales, _policyOffsets );
	_actionPub.publish( act.ToMsg() );
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