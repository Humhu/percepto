#pragma once

#include "percepto/PerceptoTypes.hpp"

namespace percepto
{

template <class Cost>
class PopulationLoss
{
public:

	typedef Cost CostType;
	typedef double OutputType;
	typedef typename CostType::RegressorType RegressorType;

	PopulationLoss( const std::vector<CostType>& costs )
	: _costs( costs ) {}

	void EvaluateAndGradient( OutputType& output, VectorType& grad ) const
	{
		_costs[0].EvaluateAndGradient( output, grad );
		for( unsigned int i = 1; i < _costs.size(); ++i )
		{
			OutputType o;
			VectorType g;
			_costs[i].EvaluateAndGradient( o, g );
			output += o;
			grad += g;
		}
		output = output / _costs.size();
		grad = grad / _costs.size();
	}

	VectorType Gradient() const
	{
		VectorType acc = _costs[0].Gradient();
		for( unsigned int i = 1; i < _costs.size(); ++i )
		{
			acc += _costs[i].Gradient();
		}
		return acc / _costs.size();
	}

	OutputType Evaluate() const
	{
		OutputType acc = 0;
		for( unsigned int i = 0; i < _costs.size(); ++i )
		{
			acc += _costs[i].Evaluate();
		}
		return acc / _costs.size();
	}

	OutputType EvaluateMax() const
	{
		OutputType largest = -std::numeric_limits<OutputType>::infinity();
		for( unsigned int i = 0; i < _costs.size(); ++i )
		{
			OutputType out = _costs[i].Evaluate();
			if( out > largest ) { largest = out; }
		}
		return largest;
	}

	RegressorType& GetRegressor() const { return _costs[0].GetRegressor(); }

protected:

	const std::vector<CostType>& _costs;

};

}