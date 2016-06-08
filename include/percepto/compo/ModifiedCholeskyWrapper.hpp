#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/utils/MatrixUtils.hpp"

namespace percepto
{

/*! \brief Positive-definite matrix regressor that regresses the L and D matrices
 * of a modified Cholesky decomposition and reforms them into a matrix. Uses
 * a different regressor for the L and D terms, but the same features. Orders
 * concatenated parameters with L parameters first, then D. */
template <typename LBase, typename DBase>
class ModifiedCholeskyWrapper
{
public:

	typedef LBase LBaseType;
	typedef DBase DBaseType;
	typedef MatrixType OutputType;

	/*! \brief Instantiate an estimator by giving it regressors for 
	 * the modified Cholesky predictors. Makes copies of the regressors. */
	ModifiedCholeskyWrapper( LBaseType& l, DBaseType& d )
	: _lBase( l ), _dBase( d ), _tmap( _dBase.OutputDim() - 1 )
	{
		InitCheckDimensions();
	}

	MatrixSize OutputSize() const
	{
		return MatrixSize( _dBase.OutputDim(), _dBase.OutputDim() );
	}
	unsigned int OutputDim() const 
	{ 
		return _dBase.OutputDim() * _dBase.OutputDim(); 
	}

	// Assuming that dodx is given w.r.t. matrix col-major ordering
	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "ModifiedCholeskyWrapper: Backprop dim error." );
		}

		MatrixType D = EvaluateD();
		MatrixType L = EvaluateL();

		// Calculate output matrix deriv wrt L inputs
		MatrixType d = MatrixType::Zero( OutputSize().rows, OutputSize().cols );
		MatrixType dSdl = MatrixType( OutputDim(), _tmap.NumPositions() );
		for( unsigned int i = 0; i < _tmap.NumPositions(); i++ )
		{
			const TriangularMapping::Index& ind = _tmap.PosToIndex( i );
			// Have to add one to get the offset from the diagonal
			d( ind.first + 1, ind.second ) = 1;
			MatrixType dSdi = d * D * L.transpose() + L * D * d.transpose();
			Eigen::Map<VectorType> dSdiVec( dSdi.data(), dSdi.size(), 1 );
			dSdl.col(i) = dSdiVec;
			d( ind.first + 1, ind.second ) = 0;
		}
		MatrixType midLInfoDodx = nextDodx * dSdl;
		_lBase.Backprop( midLInfoDodx );

		// Perform D backprop
		MatrixType dSdd = MatrixType( OutputDim(), OutputSize().rows );
		for( unsigned int i = 0; i < OutputSize().rows; i++ )
		{
			// TODO Make more efficient with row product
			d( i, i ) = 1;
			MatrixType dSdi = L * d * L.transpose();
			Eigen::Map<VectorType> dSdiVec( dSdi.data(), dSdi.size(), 1 );
			dSdd.col(i) = dSdiVec;
			d( i, i ) = 0;
		}
		MatrixType midDInfoDodx = nextDodx * dSdd;
		_dBase.Backprop( midDInfoDodx );

		return ConcatenateHor( midLInfoDodx, midDInfoDodx );
	}

	OutputType Evaluate() const
	{
		MatrixType& L = EvaluateL();
		DiagonalType& D = EvaluateD();
		return L * D * L.transpose();
	}
	
private:
	
	LBaseType& _lBase;
	DBaseType& _dBase;
	TriangularMapping _tmap;

	typedef Eigen::DiagonalMatrix <ScalarType, 
	                               Eigen::Dynamic, 
	                               Eigen::Dynamic> DiagonalType;

	MatrixType& EvaluateL() const
	{
		_tmap.VecToLowerTriangular( _lBase.Evaluate(), _L );
		return _L;
	}

	DiagonalType& EvaluateD() const
	{
		_D.diagonal() = _dBase.Evaluate();
		return _D;
	}

	void InitCheckDimensions()
	{
		assert( _lBase.OutputDim() == _tmap.NumPositions() );

		_L =  MatrixType::Identity( OutputSize().rows, OutputSize().cols );
		_D = DiagonalType( _dBase.OutputDim() );
		_D.setZero();
	}

	// Workspace
	mutable MatrixType _L;
	mutable DiagonalType _D;

};

}

