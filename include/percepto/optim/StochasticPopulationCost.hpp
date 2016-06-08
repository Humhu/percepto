#pragma once

#include <boost/foreach.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <iostream>

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

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		assert( nextDodx.cols() == OutputDim() );
		
		MatrixType thisDodx = nextDodx / _activeInds.size();
		MatrixType indDodx = ParentCost::_costs[ _activeInds[0] ].Backprop( thisDodx );
		for( unsigned int i = 1; i < _activeInds.size(); i++ )
		{
			indDodx += ParentCost::_costs[ _activeInds[i] ].Backprop( thisDodx );
		}
		return indDodx;
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
		for( unsigned int i = 0; i < _activeInds.size(); i++ )
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
		if( ParentCost::_costs.size() > _subsetSize )
		{
			BitmapSampling( ParentCost::_costs.size(), _subsetSize, 
			                   _activeInds, _generator );
			return;
		}

		// If there isn't enough data, just use all of it
		_activeInds.clear();
		_activeInds.reserve( ParentCost::_costs.size() );
		for( unsigned int i = 0; i < ParentCost::_costs.size(); ++i )
		{
			_activeInds.push_back( i );
		}
	}

};

}