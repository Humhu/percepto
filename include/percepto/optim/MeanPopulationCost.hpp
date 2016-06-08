#pragma once

#include <boost/foreach.hpp>
#include <deque>
#include "percepto/PerceptoTypes.h"

namespace percepto
{

template <typename BaseCost, template<typename,typename> class Container = std::vector>
class MeanPopulationCost
{
public:

	typedef BaseCost BaseCostType;
	typedef ScalarType OutputType;
	typedef Container<BaseCost, std::allocator<BaseCost>> ContainerType;

	/*! \brief Creates a cost by averaging costs on a poulation of
	 * inputs. Assumes all costs use the same regressor. Keeps a reference
	 * to the container so it can be changed easily. */
	MeanPopulationCost( ContainerType& costs )
	: _costs( costs )
	{}

	unsigned int OutputDim() const { return 1; }

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		assert( nextDodx.cols() == OutputDim() );

		MatrixType thisDodx = nextDodx / _costs.size();
		MatrixType indDodx = _costs[0].Backprop( thisDodx );
		for( unsigned int i = 0; i < _costs.size(); i++ )
		{
			indDodx += _costs[i].Backprop( thisDodx );
		}
		return indDodx;
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

	ContainerType& _costs;

};

}