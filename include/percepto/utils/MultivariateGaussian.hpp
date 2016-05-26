#pragma once

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <boost/random/random_device.hpp>
#include <boost/random/random_number_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "percepto/PerceptoTypes.h"

namespace percepto 

{

/*! \brief Simple multivariate normal sampling and PDF class. */
template <typename Engine = boost::mt19937, typename Scalar = double>
class MultivariateGaussian 
{
public:

	typedef Scalar ScalarType;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

	typedef boost::normal_distribution<Scalar> UnivariateNormal;
	typedef boost::variate_generator<Engine&, UnivariateNormal> RandAdapter;
	
	/*! \brief Seeds the engine using a true random number. */
	MultivariateGaussian( const VectorType& u, const MatrixType& S )
	: _distribution( 0.0, 1.0 ), 
	  _adapter( _generator, _distribution ),
	  _mean( u ), _covariance( S )
	{
		boost::random::random_device rng;
		_generator.seed( rng );
		Initialize();
	}
    
	/*! \brief Seeds the engine using a specified seed. */
	MultivariateGaussian( const VectorType& u, const MatrixType& S, unsigned long seed )
	: _distribution( 0.0, 1.0 ), 
	  _adapter( _generator, _distribution ),
	  _mean( u ), _covariance( S )
	{
		_generator.seed( seed );
		Initialize();
	}
    
    void SetMean( const VectorType& u ) { _mean = u; }
    void SetCovariance( const MatrixType& S )
	{
		_covariance = S;
		Initialize();
	}
    
	const VectorType& GetMean() const { return _mean; }
	const MatrixType& GetCovariance() const { return _mean; }
	const MatrixType& GetCholesky() const { return _L; }
	
	/*! \brief Generate a sample truncated at a specified number of standard deviations. */
	VectorType Sample( double v = 3.0 )
	{
		VectorType samples( _mean.size() );
		for( unsigned int i = 0; i < _mean.size(); i++ )
		{
			double s;
			do
			{
				s = _adapter();
			}
			while( std::abs( s ) > v );
			samples(i) = s;
		}
		
		return _mean + _L*samples;
	}

	/*! \brief Evaluate the multivariate normal PDF for the specified sample. */
    double EvaluateProbability( const VectorType& x ) const
	{
		VectorType diff = x - _mean;
		Eigen::Matrix<Scalar, 1, 1> exponent = -0.5 * diff.dot( _LLT.solve( diff ) );
		return _z * std::exp( exponent(0) );
	}
    
protected:
	
	Engine _generator;
	UnivariateNormal _distribution;
	RandAdapter _adapter;

	VectorType _mean;
	MatrixType _covariance;

	MatrixType _L;
	Eigen::LLT<MatrixType> _LLT;

	double _z; // Normali_zation constant;
	
	void Initialize()
	{
		assert( _mean.rows() == _covariance.rows() );
		assert( _mean.rows() == _covariance.cols() );
		_LLT = Eigen::LLT<MatrixType>( _covariance );
		_L = _LLT.matrixL();
		_z = std::pow( 2*M_PI, -_mean.size()/2.0 ) * std::pow( _covariance.determinant(), -0.5 );
	}

};

}
