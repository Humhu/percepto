#pragma once

#include <Eigen/Dense>

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace percepto
{
	
// For initializing vectors to random in a range
template <typename Derived>
void randomize_vector( Eigen::DenseBase<Derived>& mat, 
                       double minRange = -1.0, double maxRange = 1.0 )
{
	boost::random::mt19937 generator;
	boost::random::random_device rng;
	generator.seed( rng );
	boost::random::uniform_real_distribution<> xDist( minRange, maxRange );

	for( unsigned int i = 0; i < mat.rows(); ++i )
	{
		for( unsigned int j = 0; j < mat.cols(); ++j )
		{
			mat(i,j) = xDist( generator );
		}
	}
}

}