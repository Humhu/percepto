#pragma once

#include <boost/foreach.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "percepto/MeanPopulationCost.hpp"
#include "percepto/utils/SubsetSamplers.hpp"

namespace percepto
{

/*! \brief Randomly samples subsets of the population to return
 * estimates of the objective and gradient. 
 *
 * Resampling is executed on calls to Evaluate() only, since adding an
 * explicit Resample() method would break the cost function interface. */
template <typename CostType>
class StochasticPopulationCost
: public MeanPopulationCost<CostType>
{
public:

	typedef Scalar OutputType;
	typedef MeanPopulationCost<CostType> ParentCost;

	/*! \brief Creates a cost by averaging costs on a poulation of
	 * inputs. */
	StochasticPopulationCost( std::vector<CostType>& costs, unsigned int subsize )
	: ParentCost( costs ), _subsetSize( subsize )
	{
		boost::random::random_device rng;
		_generator.seed( rng );
	}

	void Backprop( const BackpropInfo& nextInfo, BackpropInfo& thisInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );
		
		BackpropInfo midInfo, costInfo;
		midInfo.dodx = nextInfo.dodx / _subsetSize;
		ParentCost::_costs[ _activeInds[0] ].Backprop( midInfo, thisInfo );
		for( unsigned int i = 1; i < _subsetSize; i++ )
		{
			ParentCost::_costs[ _activeInds[i] ].Backprop( midInfo, costInfo );
			thisInfo.dodx += costInfo.dodx;
			thisInfo.dodw += costInfo.dodw;
		}
	}

	/*! \brief Calculate the objective function by averaging the 
	 * underlying cost function over the population. Resamples the
	 * active population at the beginning of the call. When making sequential
	 * calls to Evaluate() and Derivative(), make sure to call Evaluate()
	 * first to get an equivalent Derivative(). */
	OutputType Evaluate()
	{
		RandomSample();

		OutputType acc = 0;
		for( unsigned int i = 0; i < _subsetSize; i++ )
		{
			acc += ParentCost::_costs[ _activeInds[i] ].Evaluate();
		}

		return acc / _activeInds.size();
	}

private:

	// Used for random selection
	boost::random::mt19937 _generator;

	unsigned int _subsetSize;
	std::vector<unsigned int> _activeInds;

	void RandomSample()
	{
		BitmapSampling( _population.size(), _subsetSize, 
		                   _activeInds, _generator );
	}

};

}