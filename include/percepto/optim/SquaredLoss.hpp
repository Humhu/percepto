#pragma once

#include "percepto/compo/BackpropInfo.hpp"

namespace percepto
{

/*! \brief Provides a squared loss on a vector regression target. */
template <typename VectorRegressor>
class SquaredLoss
{
public:

	typedef VectorRegressor RegressorType;
	typedef typename RegressorType::InputType InputType;
	typedef VectorType TargetType;
	typedef double OutputType;

	SquaredLoss( const InputType& input, const TargetType& target, 
	             RegressorType& regressor, double scale = 1.0 )
	: _input( input ), _target( target ), _regressor( regressor ),
	_scale ( scale ) {}

	unsigned int OutputDim() { return 1; }
	unsigned int ParamDim() { return _regressor.ParamDim(); }

	void SetParamsVec( const VectorType& v )
	{
		_regressor.SetParamsVec( v );
	}

	VectorType GetParamsVec() const
	{
		return _regressor.GetParamsVec();
	}


	OutputType Evaluate() const
	{
		VectorType err = ComputeError();
		return err.dot( err ) * _scale;
	}

	BackpropInfo Backprop( const BackpropInfo& nextInfo ) const
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo thisInfo;
		
		VectorType err = ComputeError();
		thisInfo.dodx = MatrixType( nextInfo.SystemOutputDim(), err.size() );
		for( unsigned int i = 0; i < nextInfo.SystemOutputDim(); i++ )
		{
			thisInfo.dodx.row(i) = nextInfo.dodx(i) * _scale * err.transpose();
		}
		// thisLayers.dodw is empty
		return thisInfo;
	}

	RegressorType& GetRegressor() const { return _regressor; }

private:

	InputType _input;
	TargetType _target;
	RegressorType& _regressor;
	double _scale;

	VectorType ComputeError() const
	{
		return _regressor.Evaluate( _input ) - _target;
	}

};

}