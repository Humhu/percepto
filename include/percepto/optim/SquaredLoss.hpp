#pragma once

#include "percepto/compo/BackpropInfo.hpp"

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

	unsigned int OutputDim() const { return 1; }
	unsigned int ParamDim() const { return _base.ParamDim(); }

	void SetParamsVec( const VectorType& v )
	{
		_base.SetParamsVec( v );
	}

	VectorType GetParamsVec() const
	{
		return _base.GetParamsVec();
	}

	OutputType Evaluate() const
	{
		VectorType err = ComputeError();
		return err.dot( err ) * _scale;
	}

	BackpropInfo Backprop( const BackpropInfo& nextInfo ) const
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo midInfo;
		
		VectorType err = ComputeError();
		midInfo.dodx = MatrixType( nextInfo.SystemOutputDim(), err.size() );
		for( unsigned int i = 0; i < nextInfo.SystemOutputDim(); i++ )
		{
			midInfo.dodx.row(i) = nextInfo.dodx(i) * _scale * err.transpose();
		}

		return _base.Backprop( midInfo );
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