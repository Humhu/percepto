#pragma once

#include "percepto/compo/BackpropInfo.hpp"

namespace percepto
{

/** 
 * \brief A regressor that returns a constant output.
 */
class ConstantRegressor
{
public:

	typedef MatrixType InputType; // Won't actually be used, but we need an input
	typedef MatrixType ParamType;
	typedef MatrixType OutputType;

	/*! \brief Creates a regressor with params set to zero. */
	ConstantRegressor( unsigned int outputRows, unsigned int outputCols )
	: _W( ParamType::Zero( outputRows, outputCols ) ) {}

	ConstantRegressor( const ParamType& params )
	: _W( params ) {}

	unsigned int InputDim() const { return 0; }
	MatrixSize InputSize() const { return MatrixSize( 0, 0 ); }

	unsigned int OutputDim() const { return _W.size(); }
	MatrixSize OutputSize() const { return MatrixSize( _W.rows(), _W.cols() ); }

	unsigned int ParamDim() const { return _W.size(); }

	void SetParams( const ParamType& p )
	{
		assert( p.rows() == _W.rows() &&
		        p.cols() == _W.cols() );
		_W = p;
	}

	void SetParamsVec( const VectorType& vec )
	{
		assert( vec.size() == _W.size() );
		_W = Eigen::Map<const ParamType>( vec.data(), _W.rows(), _W.cols() );
	}

	ParamType GetParams() const
	{
		return _W;
	}

	VectorType GetParamsVec() const
	{
		return Eigen::Map<const VectorType>( _W.data(), _W.size(), 1 );
	}

	BackpropInfo Backprop( const InputType& input, const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo thisInfo;
		// thisInfo.dodx is empty since this regressor takes no inputs
		thisInfo.dodx = MatrixType( nextInfo.SystemOutputDim(), 0 );
		thisInfo.dodw = nextInfo.dodx;
		return thisInfo;
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _W;
	}

private:

	ParamType _W;

};

}