#pragma once

#include <stdexcept>
#include <string>
#include <cassert>
#include "percepto/PerceptoTypes.h"

namespace percepto 
{

template <typename DerivedVector, typename DerivedMatrix>
void VecToDiagonal( const Eigen::DenseBase<DerivedVector>& vec,
                    Eigen::DenseBase<DerivedMatrix>& mat )
{
	assert( (vec.size() == mat.rows()) && (mat.rows() == mat.cols()) );
	for( unsigned int i = 0; i < vec.size(); i++ )
	{
		mat(i,i) = vec(i);
	}
}

template <typename DerivedMatrix, typename DerivedVector>
void DiagonalToVec( const Eigen::DenseBase<DerivedMatrix>& mat,
                    Eigen::DenseBase<DerivedVector>& vec )
{
	assert( vec.size() == mat.rows() == mat.cols() );
	for( unsigned int i = 0; i < vec.size(); i++ )
	{
		vec(i) = mat(i,i);
	}
}

/*! \brief Class that provides fast mapping between linear vectors and 
 * lower-triangular matrices using a mapping table. */
class TriangularMapping
{
public:

	typedef std::pair<unsigned int, unsigned int> Index;

	TriangularMapping( unsigned int dim = 0 )
	: _dim( dim )
	{
		SetDim( dim );
	}

	void SetDim( unsigned int dim ) 
	{
		_dim = dim;
		unsigned int numPositions = (_dim * (_dim + 1)) / 2;
		_inds.clear();
		_inds.resize( numPositions );

		unsigned int i = 0, j = 0;
		for( unsigned int pos = 0; pos < numPositions; pos++ )
		{
			_inds[pos] = Index(i,j);
			
			i++;
			if( i >= _dim )
			{
				j++;
				i = j;
			}
		}
	}

	unsigned int NumPositions() const { return _inds.size(); }

	static unsigned int num_positions( unsigned int matDim ) 
	{ 
		return ( matDim * (matDim + 1) ) / 2;
	}

	template <typename DerivedVector, typename DerivedMatrix>
	void VecToLowerTriangular( const Eigen::DenseBase<DerivedVector>& vec,
	                           Eigen::DenseBase<DerivedMatrix>& mat ) const
	{
		CheckLowerTriangularInputs( mat, vec );

		// as a r-value
		Eigen::Block<DerivedMatrix> bl = mat.bottomLeftCorner( _dim, _dim );
		VecToTriangular( vec, bl );
	}

	template <typename DerivedVector, typename DerivedMatrix>
	void VecToTriangular( const Eigen::DenseBase<DerivedVector>& vec,
	                      Eigen::DenseBase<DerivedMatrix>& mat ) const
	{
		CheckTriangularInputs( mat, vec );

		for( unsigned int pos = 0; pos < _inds.size(); pos++ )
		{
			mat( _inds[pos].first, _inds[pos].second ) = vec(pos);
		}
	}

	template <typename DerivedMatrix, typename DerivedVector>
	void LowerTriangularToVec( const Eigen::DenseBase<DerivedMatrix>& mat,
	                           Eigen::DenseBase<DerivedVector>& vec ) const
	{
		CheckLowerTriangularInputs( mat, vec );

		TriangularToVec( mat.bottomLeftCorner( _dim, _dim ),
		                 vec );
	}

	template <typename DerivedMatrix, typename DerivedVector>
	void TriangularToVec( const Eigen::DenseBase<DerivedMatrix>& mat,
	                      Eigen::DenseBase<DerivedVector>& vec ) const
	{
		CheckTriangularInputs( mat, vec );

		for( unsigned int pos = 0; pos < _inds.size(); pos++ )
		{
			vec[pos] = mat( _inds[pos].first, _inds[pos].second );
		}
	}

	const Index& PosToIndex( unsigned int pos ) const 
	{ 
		return _inds[pos]; 
	}

	unsigned int IndexToPos( Index ind ) const
	{
		if( ind.second > ind.first )
		{
			throw std::out_of_range( "Index (" + std::to_string(ind.first) +
			                         ", " + std::to_string(ind.second) +
			                         ") not lower triangular." );
		}
		return ind.second * ( _dim - 1 - (ind.second - 1) / 2 ) + ind.first;
	}

private:

	unsigned int _dim;
	std::vector<Index> _inds;

	template <typename DerivedMatrix, typename DerivedVector>
	void CheckLowerTriangularInputs( const Eigen::DenseBase<DerivedMatrix>& mat,
	                                 const Eigen::DenseBase<DerivedVector>& vec ) const
	{
		assert( vec.size() == _inds.size() );
		assert( mat.rows() == mat.cols() );
		assert( (mat.rows() * (mat.rows() + 1)) / 2 >= _inds.size() );
	}

	template <typename DerivedMatrix, typename DerivedVector>
	void CheckTriangularInputs( const Eigen::DenseBase<DerivedMatrix>& mat,
	                            const Eigen::DenseBase<DerivedVector>& vec ) const
	{
		assert( vec.size() == _inds.size() );
		assert( mat.rows() == mat.cols() );
		assert( (mat.rows() * (mat.rows() + 1)) / 2 == _inds.size() );
	}

};

} // end namespace percepto