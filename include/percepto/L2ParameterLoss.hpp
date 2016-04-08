#pragma once

#include "percepto/PerceptoTypes.hpp"

namespace percepto
{

/*! \brief Adds an L2 cost to a base cost. */
template <class Cost>
class L2ParameterLoss
{
public:

	typedef Cost CostType;
	typedef double OutputType;
	typedef typename CostType::RegressorType RegressorType;

	// Stores the base cost by reference
	L2ParameterLoss( const CostType& base, double scale )
	: _base( base ), _scale( scale ) {}

	void EvaluateAndGradient( OutputType& output, VectorType& grad ) const
	{
		VectorType params = _base.GetRegressor().GetParamsVec();
		output = Evaluate( params );
		grad = Gradient( params );
	}

	OutputType Evaluate() const
	{
		VectorType params = _base.GetRegressor().GetParamsVec();
		return Evaluate( params );
	}

	VectorType Gradient() const
	{
		VectorType params = _base.GetRegressor().GetParamsVec();
		return Gradient( params );
	}

	RegressorType& GetRegressor() const { return _base.GetRegressor(); }

private:

	const CostType& _base;
	double _scale;

	OutputType Evaluate( const VectorType& params ) const
	{
		return _scale * params.dot( params ) + _base.Evaluate();
	}

	VectorType Gradient( const VectorType& params ) const
	{
		VectorType grad = _base.Gradient();
		return grad + _scale * _base.GetRegressor().GetParamsVec();
	}

};

}