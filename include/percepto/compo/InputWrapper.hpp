#pragma once

#include "percepto/PerceptoTypes.h"
#include <iostream>

namespace percepto
{

/**
 * \brief A wrapper that applies an input to the base.
 */
template <typename Base>
class InputWrapper
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	typedef Base BaseType;
	typedef typename BaseType::InputType InputType;
	typedef typename BaseType::OutputType OutputType;

	InputWrapper( BaseType& b )
	: _base( b ) {}

	InputWrapper( BaseType& b, const InputType& in )
	: _base( b ), _input( in ) {}

	MatrixSize OutputSize() const { return _base.OutputSize(); }
	unsigned int OutputDim() const { return _base.OutputDim(); }

	void SetInput( const InputType& in ) { _input = in; }
	const InputType& GetInput() const { return _input; }

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "InputWrapper: Backprop dim error." );
		}
		return _base.Backprop( _input, nextDodx );
	}

	MatrixType Backprop( const InputType& input, 
	               const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "InputWrapper: Backprop dim error." );
		}
		return _base.Backprop( input, nextDodx );
	}

	OutputType Evaluate() const
	{
		return _base.Evaluate( _input );
	}

	OutputType Evaluate( const InputType& input )
	{
		_input = input;
		return _base.Evaluate( input );
	}

private:

	BaseType& _base;
	InputType _input;

};

template <typename InputBase, typename OutputBase>
class InputChainWrapper
{
public:

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	typedef InputBase InputBaseType;
	typedef typename InputBaseType::InputType InputType;
	typedef OutputBase OutputBaseType;
	typedef typename OutputBaseType::OutputType OutputType;

	InputChainWrapper( InputBaseType& ibase, OutputBaseType& obase )
	: _ibase( ibase ), _obase( obase ) {}

	InputChainWrapper( InputBaseType& ibase, OutputBaseType& obase,
	                   const InputType& input )
	: _ibase( ibase ), _obase( obase ) 
	{
		SetInput( input );
	}

	MatrixSize OutputSize() const { return _obase.OutputSize(); }
	unsigned int OutputDim() const { return _obase.OutputDim(); }

	void SetInput( const InputType& in ) { _ibase.SetInput( in );
	}

	const InputType& GetInput() const { return _ibase.GetInput(); }

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "InputChainWrapper: Backprop dim error." );
		}
		return _obase.Backprop( nextDodx );
	}

	MatrixType Backprop( const InputType& input, 
	               const MatrixType& nextDodx )
	{
		SetInput( input );
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "InputChainWrapper: Backprop dim error." );
		}
		return _obase.Backprop( nextDodx );
	}

	OutputType Evaluate() const
	{
		return _obase.Evaluate();
	}

	OutputType Evaluate( const InputType& input )
	{
		SetInput( input );
		return _obase.Evaluate();
	}

private:

	InputBaseType& _ibase;
	OutputBaseType& _obase;
};

}