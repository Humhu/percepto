#pragma once

#include "percepto/compo/SeriesWrapper.hpp"
#include "percepto/compo/Parametric.hpp"
#include "percepto/compo/InputWrapper.hpp"

#include <sstream>

namespace percepto
{

template <template<typename> class Layer, typename Activation>
class FullyConnectedNet
: public ParametricWrapper
{
public:

	typedef Layer<Activation> LayerType;
	typedef Activation ActivationType;
	
	typedef SequenceWrapper<InputWrapper<LayerType>, LayerType> HiddenType;
	typedef typename HiddenType::ContainerType HiddenContainerType;

	typedef typename LayerType::InputType InputType;
	typedef typename LayerType::OutputType OutputType;

	FullyConnectedNet( unsigned int inputDim, unsigned int outputDim,
	                   unsigned int numHiddenLayers, unsigned int layerWidth,
	                   const ActivationType& activation )
	: _inputUnit( inputDim, layerWidth, activation ), _inputWrapper( _inputUnit ),
	_net( _inputWrapper, _units )
	{
		if( numHiddenLayers < 1 ) 
		{
			throw std::runtime_error( "FullyConnectedNet: Need at least 1 hidden layer." );
		}

		ParametricWrapper::AddParametric( &_inputUnit );

		for( int i = 0; i < numHiddenLayers - 1; i++ )
		{
			_units.emplace_back( layerWidth, layerWidth, activation );
			ParametricWrapper::AddParametric( &_units.back() );
		}
		_units.emplace_back( layerWidth, outputDim, activation );
		ParametricWrapper::AddParametric( &_units.back() );
	}

	MatrixSize OutputSize() const { return MatrixSize( OutputDim(), 1 ); }
	unsigned int OutputDim() const { return _units.back().OutputDim(); }

	OutputType Evaluate( const InputType& input )
	{
		_inputWrapper.SetInput( input );
		return _net.Evaluate();
	}

	MatrixType Backprop( const InputType& input, const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "FullyConnectedNet: Backprop dim error." );
		}

		_inputWrapper.SetInput( input );
		return _net.Backprop( nextDodx );
	}

private:

	LayerType _inputUnit;
	InputWrapper<LayerType> _inputWrapper;
	HiddenContainerType _units;
	HiddenType _net;

};

}
