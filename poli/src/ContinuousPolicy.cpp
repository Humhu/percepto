#include "poli/ContinuousPolicy.h"
#include "modprop/utils/Randomization.hpp"
#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

namespace percepto
{

ContinuousPolicy::ContinuousPolicy() {}

void ContinuousPolicy::Initialize( unsigned int inputDim,
                                   unsigned int outputDim,
                                   ros::NodeHandle& info )
{
	ReadInitialization( inputDim, outputDim, info );
}

void ContinuousPolicy::Initialize( unsigned int inputDim,
                                   unsigned int outputDim,
                                   const YAML::Node& info )
{
	ReadInitialization( inputDim, outputDim, info );
} 

template <typename InfoType>
void ContinuousPolicy::ReadInitialization( unsigned int inputDim,
                                           unsigned int outputDim,
                                           const InfoType& info )
{
	bool useCorr;
	GetParamRequired( info, "enable_correlations", useCorr );

	GetParamRequired( info, "type", _moduleType );
	if( _moduleType == "constant_gaussian" )
	{
		_network = std::make_shared<ConstantGaussian>( outputDim, useCorr );
	}
	else if( _moduleType == "linear_gaussian" )
	{
		unsigned int numHiddenLayers, layerWidth;
		GetParamRequired( info, "num_hidden_layers", numHiddenLayers );
		GetParamRequired( info, "layer_width", layerWidth );
		_network = std::make_shared<LinearGaussian>( inputDim, outputDim, useCorr );
	}
	else if( _moduleType == "fixed_variance_gaussian" )
	{
		unsigned int numHiddenLayers, layerWidth;
		GetParamRequired( info, "num_hidden_layers", numHiddenLayers );
		GetParamRequired( info, "layer_width", layerWidth );
		_network = std::make_shared<FixedVarianceGaussian>( inputDim, 
		                                                    outputDim, 
		                                                    numHiddenLayers, 
		                                                    layerWidth,
		                                                    useCorr );
	}
	else if( _moduleType == "variable_variance_gaussian" )
	{
		unsigned int numHiddenLayers, layerWidth;
		GetParamRequired( info, "num_hidden_layers", numHiddenLayers );
		GetParamRequired( info, "layer_width", layerWidth );
		_network = std::make_shared<VariableVarianceGaussian>( inputDim, 
		                                                       outputDim, 
		                                                       numHiddenLayers, 
		                                                       layerWidth,
		                                                       useCorr );
	}
	else
	{
		throw std::invalid_argument( "Unknown policy type: " + _moduleType );
	}

	_network->SetInputSource( &_networkInput );
	_networkParameters = _network->CreateParameters();

	double initScale;
	GetParam( info, "initial_magnitude", initScale, 1.0 );
	VectorType paramVec = _networkParameters->GetParamsVec();
	randomize_vector( paramVec, -initScale, initScale );
	_networkParameters->SetParamsVec( paramVec );
}

ContinuousPolicyModule::Ptr
ContinuousPolicy::GetPolicyModule() const
{
	if( _moduleType == "constant_gaussian" )
	{
		ConstantGaussian::Ptr net = std::dynamic_pointer_cast<ConstantGaussian>( _network );
		return std::make_shared<ConstantGaussian>( *net );
	}
	if( _moduleType == "linear_gaussian" )
	{
		LinearGaussian::Ptr net = std::dynamic_pointer_cast<LinearGaussian>( _network );
		return std::make_shared<LinearGaussian>( *net );
	}
	else if( _moduleType == "fixed_variance_gaussian" )
	{
		FixedVarianceGaussian::Ptr net = std::dynamic_pointer_cast<FixedVarianceGaussian>( _network );
		return std::make_shared<FixedVarianceGaussian>( *net );
	}
	else if( _moduleType == "variable_variance_gaussian" )
	{
		VariableVarianceGaussian::Ptr net = std::dynamic_pointer_cast<VariableVarianceGaussian>( _network );
		return std::make_shared<VariableVarianceGaussian>( *net );
	}
	else
	{
		throw std::invalid_argument( "Unknown policy type: " + _moduleType );
	}
}

ContinuousPolicy::DistributionParameters
ContinuousPolicy::GenerateOutput( const VectorType& input )
{
	_networkInput.SetOutput( input );
	_networkInput.Invalidate();
	_network->Invalidate();
	_networkInput.Foreprop();
	_network->Foreprop(); // TODO Clunky!

	DistributionParameters out;
	out.mean = _network->GetMeanSource().GetOutput();
	out.info = _network->GetInfoSource().GetOutput();
	ROS_INFO_STREAM( "Policy: " << out );
	return out;
}

percepto::Parameters::Ptr ContinuousPolicy::GetParameters()
{
	return _networkParameters;
}

void ContinuousPolicy::InitializeOutput( const DistributionParameters& params )
{
	_network->InitializeMean( params.mean );
	_network->InitializeInformation( params.info );
}

std::ostream& operator<<( std::ostream& os, 
                          const ContinuousPolicy::DistributionParameters& params )
{
	os << "Mean: " << params.mean.transpose() << std::endl;
	os << "Information: " << std::endl << params.info;
	return os;
}

}
