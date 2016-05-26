#pragma once

#include <boost/foreach.hpp>
#include "percepto/PerceptoTypes.h"

namespace percepto
{

template <typename CostType>
class MeanPopulationCost
{
public:

	typedef typename CostType::RegressorType RegressorType;
	typedef ScalarType OutputType;

	/*! \brief Creates a cost by averaging costs on a poulation of
	 * inputs. Assumes all costs use the same regressor. */
	MeanPopulationCost( std::vector<CostType>& costs )
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

	void Backprop( const BackpropInfo& nextInfo, BackpropInfo& thisInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo midInfo, costInfo;
		midInfo.dodx = nextInfo.dodx / _costs.size();
		_costs[0].Backprop( midInfo, thisInfo );
		for( unsigned int i = 1; i < _costs.size(); i++ )
		{
			_costs[i].Backprop( midInfo, costInfo );
			thisInfo.dodx += costInfo.dodx;
			thisInfo.dodw += costInfo.dodw;
		}
	}

	/*! \brief Calculate the objective function by averaging the 
	 * underlying cost function over the population. */
	OutputType Evaluate() const
	{
		OutputType acc = 0;
		BOOST_FOREACH( const CostType& cost, _costs )
		{
			acc += cost.Evaluate();
		}
		return acc / _costs.size();
	}

private:

	std::vector<CostType>& _costs;

};

}