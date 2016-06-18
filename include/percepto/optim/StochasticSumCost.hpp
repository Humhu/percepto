#pragma once

#include <boost/foreach.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <iostream>

#include "percepto/optim/SumCost.hpp"
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
template <typename DataType>
class StochasticSumCost
: public SumCost<DataType>
{
public:

	typedef DataType OutputType;
	typedef SumCost<DataType> ParentCost;

	StochasticSumCost()
	: _hasResampled( false ), _subsetSize( 0 ), _ssRatio( 0 )
	{
		boost::random::random_device rng;
		_generator.seed( rng );
	}

	void SetBatchSize( unsigned int ss ) 
	{ 
		_ssRatio = 0;
		_subsetSize = ss; 
	}
	void SetSubsample( double ss ) 
	{ 
		_ssRatio = std::min( ss, 1.0 );
		_subsetSize = 0;
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( _activeInds.size() == 0 ) { return; }
		for( unsigned int i = 0; i < _activeInds.size(); i++ )
		{
			ParentCost::_sinks[ _activeInds[i] ].Backprop( nextDodx );
		}
	}

	void Resample()
	{
		RandomSample();
		_hasResampled = true;
	}

	const std::vector<unsigned int>& GetActiveInds() const
	{
		return _activeInds;
	}

	/*! \brief Calculate the objective function by averaging the 
	 * underlying cost function over the population. Resamples the
	 * active population at the beginning of the call. When making sequential
	 * calls to Evaluate() and Derivative(), make sure to call Evaluate()
	 * first to get an equivalent Derivative(). */
	virtual void Foreprop()
	{
		if( !_hasResampled ) 
		{
			RandomSample(); 
			_hasResampled = true;
		}

		if( _activeInds.size() == 0 ) 
		{
			 // TODO Generalize to other datatypes
			ParentCost::SourceType::SetOutput( 0 );
			ParentCost::SourceType::Foreprop();
			return; 
		}
		for( unsigned int i = 0; i < _activeInds.size(); i++ )
		{
			if( !ParentCost::_sinks[ _activeInds[i] ].IsValid() ) { return; }
		}

		OutputType acc = ParentCost::_sinks[ _activeInds[0] ].GetInput();
		for( unsigned int i = 1; i < _activeInds.size(); i++ )
		{
			acc += ParentCost::_sinks[ _activeInds[i] ].GetInput();
		}

		ParentCost::SourceType::SetOutput( acc );
		ParentCost::SourceType::Foreprop();
	}

	virtual void Invalidate()
	{
		_hasResampled = false;
		ParentCost::SourceType::Invalidate();
	}

private:

	// Used for random selection
	mutable boost::random::mt19937 _generator;

	mutable bool _hasResampled;
	unsigned int _subsetSize;
	double _ssRatio;
	mutable std::vector<unsigned int> _activeInds;

	void RandomSample() const
	{
		unsigned int ss;
		if( _subsetSize > 0 )
		{
			ss = _subsetSize;
		}
		else
		{
			ss = (unsigned int) std::ceil( _ssRatio * ParentCost::_sinks.size() );
		}
		if( ss < 1 ) { ss = 1; }

		if( ParentCost::_sinks.size() > ss )
		{
			BitmapSampling( ParentCost::_sinks.size(), ss, 
			                   _activeInds, _generator );
			return;
		}

		// If there isn't enough data, just use all of it
		_activeInds.clear();
		_activeInds.reserve( ParentCost::_sinks.size() );
		for( unsigned int i = 0; i < ParentCost::_sinks.size(); ++i )
		{
			_activeInds.push_back( i );
		}
	}

};

}