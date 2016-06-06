#pragma once

#include "percepto/compo/BackpropInfo.hpp"

namespace percepto
{

/*! \brief A cost function that adds a weighted L2-norm to an
 * existing cost. For maximization tasks, the weight should be negative. */
template <typename BaseType>
class ParameterL2Cost
{
public:

	typedef ScalarType OutputType;

	ParameterL2Cost( BaseType& b, ScalarType w )
	: _base( b ), _w( w )
	{}

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

	BackpropInfo Backprop( const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		// Add the L2 cost into the dodw
		VectorType current = _base.GetParamsVec();
		BackpropInfo thisInfo = _base.Backprop( nextInfo );
		for( unsigned int i = 0; i < nextInfo.SystemOutputDim(); i++ )
		{
			thisInfo.dodw.row(i) += ( _w * current.array() ).matrix().transpose();
		}
		return thisInfo;
	}

	OutputType Evaluate() const
	{
		VectorType current = _base.GetParamsVec();
		OutputType base = _base.Evaluate();
		return base + 0.5 * ( _w * current.array() * current.array() ).sum();
	}

private:

	BaseType& _base;
	ScalarType _w;

};


}