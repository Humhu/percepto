#pragma once

#include "percepto/PerceptoTypes.hpp"
#include "percepto/PerceptoUtils.h"
#include "percepto/Backprop.h"

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

	void EvaluateAndGradient( OutputType& output, VectorType& grad ) const
	{
		VectorType err = ComputeError();
		output = Evaluate( err );
		grad = Gradient( grad );
	}

	OutputType Evaluate() const
	{
		return Evaluate( ComputeError() );
	}

	VectorType Gradient() const
	{
		return Gradient( ComputeError() );
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

	OutputType Evaluate( const VectorType& err ) const
	{
		return err.dot( err ) * _scale;
	}

	VectorType Gradient( const VectorType& err ) const
	{
		BackpropInfo thisLayers;
		thisLayers.sysOutDim = 1;
		thisLayers.dodx = _scale * err.transpose();

		BackpropInfo regressorsInfo = _regressor.Backprop( _input, thisLayers );
		return regressorsInfo.dodw.transpose();
	}

};

}