#pragma once

#include <boost/foreach.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "percepto/optim/MeanPopulationCost.hpp"
#include "percepto/utils/SubsetSamplers.hpp"

namespace percepto
{

/*! \brief Randomly samples subsets of the population to return
 * estimates of the objective and gradient. 
 *
 * Resampling is executed on calls to Evaluate() only, since adding an
 * explicit Resample() method would break the cost function interface. 
 *
 * The container holding the costs can be altered to change the
 * population. Make sure to call Evaluate() after alteration to set
 * the active indices.
 */
template <typename CostType, template<typename,typename> class Container = std::vector>
class StochasticPopulationCost
: public MeanPopulationCost<CostType, Container>
{
public:

	typedef ScalarType OutputType;
	typedef MeanPopulationCost<CostType, Container> ParentCost;
	typedef typename ParentCost::ContainerType ContainerType;

	/*! \brief Creates a cost by averaging costs on a poulation of
	 * inputs. */
	StochasticPopulationCost( ContainerType& costs, unsigned int subsize )
	: ParentCost( costs ), _subsetSize( subsize )
	{
		boost::random::random_device rng;
		_generator.seed( rng );
	}

	unsigned int OutputDim() const { return 1; }

	BackpropInfo Backprop( const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );
		
		BackpropInfo midInfo, costInfo;
		midInfo.dodx = nextInfo.dodx / _subsetSize;
		BackpropInfo thisInfo = ParentCost::_costs[ _activeInds[0] ].Backprop( midInfo );
		for( unsigned int i = 1; i < _subsetSize; i++ )
		{
			BackpropInfo costInfo = ParentCost::_costs[ _activeInds[i] ].Backprop( midInfo );
			thisInfo.dodx += costInfo.dodx;
			thisInfo.dodw += costInfo.dodw;
		}
		return thisInfo;
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
		BitmapSampling( ParentCost::_costs.size(), _subsetSize, 
		                   _activeInds, _generator );
	}

};

}