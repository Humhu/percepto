#pragma once

#include "percepto/neural/NullActivation.hpp"
#include "percepto/compo/Interfaces.h"
#include "percepto/compo/Parametric.hpp"

#include <sstream>

namespace percepto
{

// TODO Have a nicer interface for pulling the entire network configuration,
// not just the parameter vector
template <template<typename> class Layer, typename Activation>
class FullyConnectedNet
{
public:

	enum OutputLayerMode
	{
		OUTPUT_UNRECTIFIED = 0,
		OUTPUT_RECTIFIED
	};

	typedef Activation ActivationType;
	typedef Layer<ActivationType> RectifiedLayerType;
	typedef typename RectifiedLayerType::InputType InputType;
	typedef typename RectifiedLayerType::OutputType OutputType;
	typedef Layer<NullActivation> UnrectifiedLayerType;
	typedef Source<InputType> InputSourceType;
	typedef Source<VectorType> OutputSourceType;

	FullyConnectedNet( unsigned int inputDim, 
	                   unsigned int outputDim,
	                   unsigned int numHiddenLayers, 
	                   unsigned int layerWidth,
	                   const ActivationType& activation, 
	                   OutputLayerMode outputMode = OUTPUT_UNRECTIFIED )
	: _outputMode( outputMode ),
	  _unrectifiedLayer( layerWidth, outputDim, NullActivation() )
	{
		if( numHiddenLayers == 0 )
		{
			throw std::runtime_error( "Need at least one hidden layer." );
		}

		_layers.reserve( numHiddenLayers + 1 );

		// Create first layer
		_layers.emplace_back( inputDim, layerWidth, activation );

		// Create middle layers
		for( unsigned int i = 1; i < numHiddenLayers; i++ )
		{
			_layers.emplace_back( layerWidth, layerWidth, activation );
		}

		// Unrectified layer is created automatically
		if( _outputMode == OUTPUT_UNRECTIFIED )
		{
			_unrectifiedLayer.SetSource( &_layers.back() );
		}
		else if( _outputMode == OUTPUT_RECTIFIED )
		{
			_layers.emplace_back( layerWidth, outputDim, activation );
		}
		else
		{
			throw std::runtime_error( "Unknown output mode received." );
		}

		// Connect all layers
		for( unsigned int i = 1; i < _layers.size(); i++ )
		{
			_layers[i].SetSource( &_layers[i-1] );
		}
	}

	FullyConnectedNet( const FullyConnectedNet& other )
	: _outputMode( other._outputMode ),
	  // _layers( other._layers ),
	  _unrectifiedLayer( other._unrectifiedLayer ),
	  _paramSets( other._paramSets ),
	  _params( other._params )
	{
		for( unsigned int i = 0; i < other._layers.size(); ++i )
		{
			_layers.emplace_back( other._layers[i] );
		}
		for( unsigned int i = 1; i < _layers.size(); ++i )
		{
			_layers[i].SetSource( &_layers[i-1] );
		}
		if( _outputMode == OUTPUT_UNRECTIFIED )
		{
			_unrectifiedLayer.SetSource( &_layers.back() );
		}
	}

	Parameters::Ptr CreateParameters()
	{
		_paramSets.clear();
		_params = ParameterWrapper();
		_paramSets.reserve( NumLayers() );
		for( unsigned int i = 0; i < _layers.size(); i++ )
		{
			_paramSets.push_back( _layers[i].CreateParameters() );
			_params.AddParameters( _paramSets.back() );
		}
		if( _outputMode == OUTPUT_UNRECTIFIED )
		{
			_paramSets.push_back( _unrectifiedLayer.CreateParameters() );
			_params.AddParameters( _paramSets.back() );
		}
		return std::make_shared<ParameterWrapper>( _params );
	}

	const std::vector<Parameters::Ptr>& GetParameterSets() const
	{
		return _paramSets;
	}

	void SetParameterSets( const std::vector<Parameters::Ptr>& params )
	{
		if( params.size() != NumLayers() )
		{
			std::stringstream ss;
			ss << "FullyConnectedNet: Received " << params.size() << 
			      " param sets but expected " << NumLayers();
			throw std::runtime_error( ss.str() );
		}

		for( unsigned int i = 0; i < _layers.size(); i++ )
		{
			_layers[i].SetParameters( params[i] );
		}
		if( _outputMode == OUTPUT_UNRECTIFIED )
		{
			_unrectifiedLayer.SetParameters( params.back() );
		}
	}

	void SetSource( InputSourceType* _base ) 
	{ 
		_layers.front().SetSource( _base ); 
	}

	unsigned int NumLayers() const
	{
		if( _outputMode == OUTPUT_UNRECTIFIED )
		{
			return _layers.size() + 1;
		}
		else //( _outputMode == OUTPUT_RECTIFIED )
		{
			return _layers.size();
		}
	}

	Source<VectorType>& GetOutputSource()
	{
		if( _outputMode == OUTPUT_UNRECTIFIED )
		{
			return _unrectifiedLayer;
		}
		// Constructor guarantees that this is true
		else //( _outputMode == OUTPUT_RECTIFIED )
		{
			return _layers.back();
		}
	}

	const Source<VectorType>& GetOutputSource() const
	{
		if( _outputMode == OUTPUT_UNRECTIFIED )
		{
			return _unrectifiedLayer;
		}
		// Constructor guarantees that this is true
		else //( _outputMode == OUTPUT_RECTIFIED )
		{
			return _layers.back();
		}
	}


	OutputType GetOutput() const 
	{
		if( _outputMode == OUTPUT_UNRECTIFIED )
		{
			return _unrectifiedLayer.GetOutput();
		}
		else
		{
			return _layers.back().GetOutput(); 
		}
	}

	unsigned int NumHiddenLayers() const { return _layers.size(); }
	unsigned int OutputDim() const { return _layers.back().OutputDim(); }
	unsigned int InputDim() const { return _layers.front().InputDim(); }
	const ActivationType& GetActivation() const { return _layers.front().GetActivation(); }

private:

	OutputLayerMode _outputMode;
	std::vector<RectifiedLayerType> _layers;
	UnrectifiedLayerType _unrectifiedLayer; // May be needed for unrectified out

	std::vector<Parameters::Ptr> _paramSets;
	ParameterWrapper _params;

};

}
