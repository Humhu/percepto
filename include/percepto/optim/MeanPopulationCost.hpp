#pragma once

#include <boost/foreach.hpp>
#include "percepto/PerceptoTypes.h"

namespace percepto
{

template <typename BaseCost>
class MeanPopulationCost
{
public:

	typedef BaseCost BaseCostType;
	typedef ScalarType OutputType;

	/*! \brief Creates a cost by averaging costs on a poulation of
	 * inputs. Assumes all costs use the same regressor. */
	MeanPopulationCost( std::vector<BaseCostType>& costs )
	: _costs( costs )
	{}

	unsigned int OutputDim() const { return 1; }
	unsigned int ParamDim() const
	{
		return _costs[0].ParamDim();
	}

	void SetParamsVec( const VectorType& v )
	{
		_costs[0].SetParamsVec( v );
	}

	VectorType GetParamsVec() const
	{
		return _costs[0].GetParamsVec();
	}

	BackpropInfo Backprop( const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo midInfo, costInfo;
		midInfo.dodx = nextInfo.dodx / _costs.size();
		BackpropInfo thisInfo = _costs[0].Backprop( midInfo );
		for( unsigned int i = 1; i < _costs.size(); i++ )
		{
			costInfo = _costs[i].Backprop( midInfo );
			thisInfo.dodx += costInfo.dodx;
			thisInfo.dodw += costInfo.dodw;
		}
		return thisInfo;
	}

	/*! \brief Calculate the objective function by averaging the 
	 * underlying cost function over the population. */
	OutputType Evaluate() const
	{
		OutputType acc = 0;
		BOOST_FOREACH( const BaseCostType& cost, _costs )
		{
			acc += cost.Evaluate();
		}
		return acc / _costs.size();
	}

	OutputType EvaluateMax() const
	{
		OutputType largest = -std::numeric_limits<OutputType>::infinity();
		BOOST_FOREACH( const BaseCostType& cost, _costs )
		{
			OutputType out = cost.Evaluate();
			if( out > largest ) { largest = out; }
		}
		return largest;
	}

protected:

	std::vector<BaseCostType>& _costs;

};

}