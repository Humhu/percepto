#include "valu/ApproximateValue.h"
#include "percepto_msgs/RewardStamped.h"
#include "argus_utils/utils/ParamUtils.h"
#include "modprop/utils/Randomization.hpp"

using namespace argus;

namespace percepto
{

ApproximateValue::ApproximateValue() {}

void ApproximateValue::Initialize( unsigned int inputDim, 
                                   ros::NodeHandle& info )
{
	ReadInitialization( inputDim, info );
}

void ApproximateValue::Initialize( unsigned int inputDim, 
                                   const YAML::Node& info )
{
	ReadInitialization( inputDim, info );
}

template <typename InfoType>
void ApproximateValue::ReadInitialization( unsigned int inputDim,
                                           const InfoType& info )
{
	// Create module
	GetParamRequired( info, "type", _moduleType );
	if( _moduleType == "perceptron" )
	{
		unsigned int numHiddenLayers, layerWidth;
		GetParamRequired( info, "num_hidden_layers", numHiddenLayers );
		GetParamRequired( info, "layer_width", layerWidth );
		_approximator = std::make_shared<PerceptronFunctionApproximator>( inputDim,
		                                                                  1,
		                                                                  numHiddenLayers,
		                                                                  layerWidth );
	}
	else
	{
		throw std::invalid_argument( "ApproximateValue: Unknown network type: " + _moduleType );
	}

	// Configure pipeline
	_approximator->SetInputSource( &_approximatorInput );
	_approximatorParams = _approximator->CreateParameters();

	// Initialize parameters
	double initialValue, initialMagnitude;
	GetParamRequired( info, "initial_magnitude", initialMagnitude );
	GetParamRequired( info, "initial_value", initialValue );

	VectorType w = _approximatorParams->GetParamsVec();
	percepto::randomize_vector( w, -initialMagnitude, initialMagnitude );
	_approximatorParams->SetParamsVec( w );
	_approximator->InitializeOutput( initialValue );

	ROS_INFO_STREAM( "ApproximateValue: Network initialized: " << std::endl << _approximator->Print() );
}

double ApproximateValue::GetValue( const VectorType& input ) const
{
	_approximatorInput.Invalidate();
	_approximator->Invalidate();
	_approximatorInput.SetOutput( input );
	_approximatorInput.Foreprop();
	_approximator->Foreprop();
	return _approximator->GetOutputSource().GetOutput();
}



percepto::Parameters::Ptr ApproximateValue::GetParameters() const
{
	return _approximatorParams;
}

ScalarFieldApproximator::Ptr 
ApproximateValue::GetApproximatorModule() const
{
	if( _moduleType == "perceptron" )
	{
		PerceptronFunctionApproximator::Ptr net = 
			std::dynamic_pointer_cast<PerceptronFunctionApproximator>( _approximator );
		return std::make_shared<PerceptronFunctionApproximator>( *net );
	}
	else
	{
		throw std::invalid_argument( "ApproximateValue: Unknown network type: " + _moduleType );
	}
}

void ApproximateValue::InitializeOutput( double out )
{
	_approximator->InitializeOutput( out );
}

}