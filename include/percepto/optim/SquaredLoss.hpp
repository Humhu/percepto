#pragma once

#include "percepto/compo/BackpropInfo.hpp"
#include <iostream>

namespace percepto
{

/*! \brief Provides a squared loss on a vector regression target. */
template <typename Base>
class SquaredLoss
{
public:

	typedef Base BaseType;
	typedef VectorType TargetType;
	typedef double OutputType;

	SquaredLoss( BaseType& r, const TargetType& target, double scale = 1.0 )
	: _base( r ), _target( target ), _scale( scale ) {}

	MatrixSize OutputSize() const { return MatrixSize(1,1); }
	unsigned int OutputDim() const { return 1; }

	OutputType Evaluate() const
	{
		VectorType err = ComputeError();
		return 0.5 * err.dot( err ) * _scale;
	}

	MatrixType Backprop( const MatrixType& nextDodx ) const
	{
		assert( nextDodx.cols() == OutputDim() );

		BackpropInfo midInfo;
		
		VectorType err = ComputeError();
		MatrixType thisDodx = _scale * nextDodx * err.transpose();
		// MatrixType thisDodx = MatrixType( nextDodx.rows(), err.size() );
		// for( unsigned int i = 0; i < nextDodx.rows(); i++ )
		// {
		// 	thisDodx.row(i) = nextInfo.dodx(i,0) * _scale * err.transpose();
		// }

		return _base.Backprop( thisDodx );
	}

private:

	BaseType& _base;
	const TargetType _target;
	const double _scale;

	inline VectorType ComputeError() const
	{
		return _base.Evaluate() - _target;
	}

};

}