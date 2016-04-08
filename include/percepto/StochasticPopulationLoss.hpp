#pragma once

#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "percepto/SubsetSamplers.hpp"
#include "percepto/PerceptoTypes.hpp"
#include "percepto/PopulationLoss.hpp"

namespace percepto
{

/*! \brief Randomly samples subsets of the population to return
 * estimates of the objective and gradient. 
 *
 * Resampling is executed on calls to Evaluate() only, since adding an
 * explicit Resample() method would break the cost function interface. */
template <class Cost>
class StochasticPopulationLoss
: public PopulationLoss<Cost>
{
public:

	typedef Cost CostType;
	typedef PopulationLoss<CostType> ParentCost;
	typedef double OutputType;
	typedef typename CostType::RegressorType RegressorType;

	/*! \brief Creates a cost by averaging costs on a poulation of
	 * inputs. Does not copy the underlying data.*/
	StochasticPopulationLoss( const std::vector<CostType>& costs,
	                          unsigned int subsize )
	: ParentCost( costs ), _subsetSize( subsize )
	{
		boost::random::random_device rng;
		_generator.seed( rng );
	}

	void EvaluateAndGradient( OutputType& output, VectorType& grad ) const
	{
		output = Evaluate();
		grad = Gradient();
	}

	// TODO Combine gradient and evaluate into a single call
	/*! \brief Calculate the gradient of the cost w.r.t. parameters. 
	 * Uses underlying cost gradient averaged over the population. */
	VectorType Gradient() const
	{
		VectorType acc = ParentCost::_costs[ _activeInds[0] ].Gradient();
		for( unsigned int i = 1; i < _subsetSize; ++i )
		{
			acc += ParentCost::_costs[ _activeInds[i] ].Gradient();
		}
		return acc / _subsetSize;
	}

	/*! \brief Calculate the objective function by averaging the 
	 * underlying cost function over the population. Resamples the
	 * active population at the beginning of the call. When making sequential
	 * calls to Evaluate() and Gradient(), make sure to call Evaluate()
	 * first to get an equivalent Gradient(). */
	OutputType Evaluate() const
	{
		RandomSample();

		OutputType acc = 0;
		for( unsigned int i = 0; i < _subsetSize; ++i )
		{
			acc += ParentCost::_costs[ _activeInds[i] ].Evaluate();
		}

		return acc / _subsetSize;
	}

	RegressorType& GetRegressor() const { return ParentCost::GetRegressor(); }

private:

	unsigned int _subsetSize;
	mutable std::vector<unsigned int> _activeInds;

	// Used for random selection
	mutable boost::random::mt19937 _generator;

	void RandomSample() const
	{
		BitmapSampling( ParentCost::_costs.size(), _subsetSize, 
		                _activeInds, _generator );
	}

};

}