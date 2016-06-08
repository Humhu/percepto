#pragma once

#include "percepto/compo/Parametric.hpp"

namespace percepto
{

/** 
 * \brief A regressor that returns a constant output.
 */
class ConstantRegressor
: public Parametric
{
public:

	typedef MatrixType ParamType;
	typedef MatrixType OutputType;

	/*! \brief Creates a regressor with params set to zero. */
	ConstantRegressor( unsigned int outputRows, unsigned int outputCols )
	: _W( ParamType::Zero( outputRows, outputCols ) ) {}

	ConstantRegressor( const ParamType& params )
	: _W( params ) {}

	unsigned int OutputDim() const { return _W.size(); }
	MatrixSize OutputSize() const { return MatrixSize( _W.rows(), _W.cols() ); }

	unsigned int ParamDim() const { return _W.size(); }

	void SetParams( const ParamType& p )
	{
		assert( p.rows() == _W.rows() &&
		        p.cols() == _W.cols() );
		_W = p;
	}

	virtual void SetParamsVec( const VectorType& vec )
	{
		assert( vec.size() == _W.size() );
		_W = Eigen::Map<const ParamType>( vec.data(), _W.rows(), _W.cols() );
	}

	ParamType GetParams() const
	{
		return _W;
	}

	virtual VectorType GetParamsVec() const
	{
		return Eigen::Map<const VectorType>( _W.data(), _W.size(), 1 );
	}

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "ConstantRegressor: Backprop dim error." );
		}
		// This module's dodw is just the next one's dodx since 
		// its params are its outputs
		// This module has no inputs, so no input derivative
		Parametric::AccumulateWeightDerivs( nextDodx );

		// Pretend it has 1 input
		return MatrixType::Zero( nextDodx.rows(), 1 );
	}

	OutputType Evaluate() const
	{
		return _W;
	}

private:

	ParamType _W;

};

}