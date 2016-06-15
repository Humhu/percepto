#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/compo/Parametric.hpp"

#include <sstream>
#include <iostream>
namespace percepto
{

// TODO Have a nicer interface for pulling the entire network configuration,
// not just the parameter vector
template <template<typename> class Layer, typename Activation>
class FullyConnectedNet
{
public:

	typedef Layer<Activation> LayerType;
	typedef Activation ActivationType;
	typedef typename LayerType::InputType InputType;
	typedef typename LayerType::OutputType OutputType;
	typedef Source<InputType> SourceType;

	FullyConnectedNet( unsigned int inputDim, unsigned int outputDim,
	                   unsigned int numHiddenLayers, unsigned int layerWidth,
	                   const ActivationType& activation )
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

		// Create last layer
		_layers.emplace_back( layerWidth, outputDim, activation );

		// Connect all layers
		for( unsigned int i = 1; i < _layers.size(); i++ )
		{
			_layers[i].SetSource( &_layers[i-1] );
		}
	}

	FullyConnectedNet( const FullyConnectedNet& other )
	{
		_layers.reserve( other._layers.size() );
		for( unsigned int i = 0; i < other._layers.size(); i++ )
		{
			_layers.emplace_back( other._layers[i] );
		}
		for( unsigned int i = 1; i < _layers.size(); i++ )
		{
			_layers[i].SetSource( &_layers[i-1] );
		}
	}

	std::vector<Parameters::Ptr> CreateParameters()
	{
		std::vector<Parameters::Ptr> params;
		params.reserve( _layers.size() );
		for( unsigned int i = 0; i < _layers.size(); i++ )
		{
			params.push_back( _layers[i].CreateParameters() );
		}
		return params;
	}

	void SetParameters( const std::vector<Parameters::Ptr>& params )
	{
		if( params.size() != _layers.size() )
		{
			throw std::runtime_error( "FullyConnectedNet: Invalid number of param objects." );
		}
		for( unsigned int i = 0; i < _layers.size(); i++ )
		{
			_layers[i].SetParameters( params[i] );
		}
	}

	void SetSource( SourceType* _base ) 
	{ 
		_layers.front().SetSource( _base ); 
	}

	Source<VectorType>& GetOutputSource()
	{
		return _layers.back();
	}

	OutputType GetOutput() const { return _layers.back().GetOutput(); }

	unsigned int NumHiddenLayers() const { return _layers.size(); }
	unsigned int OutputDim() const { return _layers.back().OutputDim(); }
	unsigned int InputDim() const { return _layers.front().InputDim(); }
	const ActivationType& GetActivation() const { return _layers.front().GetActivation(); }

private:

	std::vector<LayerType> _layers;

};

}
