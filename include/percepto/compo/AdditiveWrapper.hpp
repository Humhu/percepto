#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/utils/MatrixUtils.hpp"

namespace percepto
{

/*! \brief Adds two non-regressor bases together. */
template <typename BaseA, typename BaseB>
class AdditiveWrapper
{
public:

	typedef BaseA BaseAType;
	typedef BaseB BaseBType;
	typedef typename BaseAType::OutputType OutputType;

	AdditiveWrapper( BaseAType& baseA, BaseBType& baseB )
	: _baseA( baseA ), _baseB( baseB )
	{}

	MatrixSize OutputSize() const { return _baseA.OutputSize(); }
	unsigned int OutputDim() const { return _baseA.OutputDim(); }

	OutputType Evaluate() const
	{
		return _baseA.Evaluate() + _baseB.Evaluate();
	}

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "AdditiveWrapper: Backprop dim error." );
		}

		// Pass the info to the constituents
		// This module's inputs are A and B, respectively
		return ConcatenateHor( _baseA.Backprop( nextDodx ),
		                       _baseB.Backprop( nextDodx ) );
	}

private:

	BaseAType& _baseA;
	BaseBType& _baseB;
};

/*! \brief Adds a number of non-regressor bases together. */
template <typename Base, template<typename,typename> class Container = std::vector>
class AdditiveSumWrapper
{
public:

	typedef Base BaseType;
	typedef Container<Base, std::allocator<Base>> ContainerType;
	typedef typename BaseType::OutputType OutputType;

	AdditiveSumWrapper( ContainerType& bases )
	: _bases( bases ) {}

	MatrixSize OutputSize() const { return _bases[0].OutputSize(); }
	unsigned int OutputDim() const { return _bases[0].OutputDim(); }

	OutputType Evaluate() const
	{
		OutputType out = _bases[0].Evaluate();
		for( unsigned int i = 1; i < _bases.size(); ++i )
		{
			out += _bases[i].Evaluate();
		}
		return out;
	}

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "AdditiveSumWrapper: Backprop dim error." );
		}

		MatrixType thisDodx = MatrixType( nextDodx.rows(), 
		                            _bases.size() * nextDodx.cols() );
		for( unsigned int i = 0; i < _bases.size(); ++i )
		{
			_bases[i].Backprop( nextDodx );
			thisDodx.block( 0, 
			                nextDodx.cols() * i, 
			                nextDodx.rows(), 
			                nextDodx.cols() ) = nextDodx;
		}
		return thisDodx;
	}

private:

	ContainerType& _bases;
};

}