#pragma once

#include "percepto/BackpropInfo.hpp"

namespace percepto
{

/*! \brief A cost function that adds a weighted L2-norm to an
 * existing cost. For maximization tasks, the weight should be negative. */
template <typename CostType>
class ParameterL2Cost
{
public:

	typedef typename CostType::RegressorType RegressorType;
	typedef typename RegressorType::ParamType ParamType;
	typedef ScalarType OutputType;

	ParameterL2Cost( CostType& c, ScalarType w )
	: _base( c )
	{
		_weights = w * VectorType::Ones( _base.ParamDim() );
	}

	ParameterL2Cost( CostType& c, const VectorType& weights )
	: _base( c ), _weights( weights ) {}

	unsigned int OutputDim() const { return 1; }
	unsigned int ParamDim() const
	{
		return _base.ParamDim();
	}

	void SetParamsVec( const VectorType& v )
	{
		_base.SetParamsVec( v );
	}

	VectorType GetParamsVec() const
	{
		return _base.GetParamsVec();
	}

	void Backprop( const BackpropInfo& nextInfo, BackpropInfo& thisInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		// Add the L2 cost into the dodw
		VectorType current = _base.GetParamsVec();
		_base.Backprop( nextInfo, thisInfo );
		for( unsigned int i = 0; i < nextInfo.SystemOutputDim(); i++ )
		{
			thisInfo.dodw.row(i) += ( _weights.array()*current.array() ).matrix().transpose();
		}
	}

	VectorType Gradient() const
	{
		VectorType current = _base.GetParamsVec();
		VectorType costGrad = _base.Gradient();

		return costGrad + ( _weights.array()*current.array() ).matrix();
	}

	VectorType Derivative( unsigned int ind ) const
	{
		VectorType current = _base.GetParamsVec();
		VectorType costGrad = _base.Derivative( ind );
		costGrad(ind) += _weights(ind)*current(ind);
		return costGrad;
	}

	OutputType Evaluate() const
	{
		VectorType current = _base.GetParamsVec();
		OutputType base = _base.Evaluate();
		return base + 0.5 * (_weights.array() * current.array() * current.array() ).sum();
	}

private:

	CostType& _base;
	VectorType _weights;

};


}