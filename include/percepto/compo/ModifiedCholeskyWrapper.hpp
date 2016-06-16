#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/PerceptoTypes.h"
#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/utils/MatrixUtils.hpp"

namespace percepto
{

// TODO Deprecate TriangularMapping
/*! \brief Positive-definite matrix regressor that regresses the L and D matrices
 * of a modified Cholesky decomposition and reforms them into a matrix. Uses
 * a different regressor for the L and D terms, but the same features. Orders
 * concatenated parameters with L parameters first, then D. */
class ModifiedCholeskyWrapper
: public Source<MatrixType>
{
public:

	typedef Source<VectorType> InputSourceType;
	typedef Source<MatrixType> OutputSourceType;
	typedef Sink<VectorType> SinkType;
	typedef MatrixType OutputType;

	ModifiedCholeskyWrapper() 
	: _lInput( this ), _dInput( this ) {}

	ModifiedCholeskyWrapper( const ModifiedCholeskyWrapper& other ) 
	: _lInput( this ), _dInput( this ) {}

	void SetLSource( InputSourceType* l ) { l->RegisterConsumer( &_lInput ); }
	void SetDSource( InputSourceType* d ) { d->RegisterConsumer( &_dInput ); }

	// Assuming that dodx is given w.r.t. matrix col-major ordering
	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// std::cout << "ModifiedCholeskyWrapper backprop" << std::endl;
		MatrixType dody = nextDodx;
		if( nextDodx.size() == 0 )
		{
			dody = MatrixType::Identity( _D.size(), _D.size() );
		}

		// Calculate output matrix deriv wrt L inputs
		MatrixType d = MatrixType::Zero( _D.rows(), _D.cols() );
		MatrixType dSdl = MatrixType( _D.size(), _tmap.NumPositions() );
		for( unsigned int i = 0; i < _tmap.NumPositions(); i++ )
		{
			const TriangularMapping::Index& ind = _tmap.PosToIndex( i );
			// Have to add one to get the offset from the diagonal
			d( ind.first + 1, ind.second ) = 1;
			MatrixType dSdi = d * _D * _L.transpose() + _L * _D * d.transpose();
			Eigen::Map<VectorType> dSdiVec( dSdi.data(), dSdi.size(), 1 );
			dSdl.col(i) = dSdiVec;
			d( ind.first + 1, ind.second ) = 0;
		}
		MatrixType midLInfoDodx = dody * dSdl;
		_lInput.Backprop( midLInfoDodx );

		// Perform D backprop
		MatrixType dSdd = MatrixType( _D.size(), _D.rows() );
		for( unsigned int i = 0; i < _D.rows(); i++ )
		{
			// TODO Make more efficient with row product
			d( i, i ) = 1;
			MatrixType dSdi = _L * d * _L.transpose();
			Eigen::Map<VectorType> dSdiVec( dSdi.data(), dSdi.size(), 1 );
			dSdd.col(i) = dSdiVec;
			d( i, i ) = 0;
		}
		MatrixType midDInfoDodx = dody * dSdd;
		_dInput.Backprop( midDInfoDodx );
	}

	virtual void Foreprop()
	{
		if( _lInput.IsValid() && _dInput.IsValid() )
		{
			const VectorType& lVec = _lInput.GetInput();
			const VectorType& dVec = _dInput.GetInput();
			unsigned int N = dVec.size();

			// Initialize workspace
			if( lVec.size() != _tmap.NumPositions() )
			{
				_tmap.SetDim( N - 1 );
				_L = MatrixType::Identity( N, N );
				_D = DiagonalType( N );
				_D.setZero();
			}

			_tmap.VecToLowerTriangular( lVec, _L );
			_D.diagonal() = dVec;
			OutputSourceType::SetOutput( _L * _D * _L.transpose() );
			OutputSourceType::Foreprop();
		}
	}
	
private:
	
	SinkType _lInput;
	SinkType _dInput;
	TriangularMapping _tmap;

	typedef Eigen::DiagonalMatrix <ScalarType,
	                               Eigen::Dynamic,
	                               Eigen::Dynamic> DiagonalType;

	// Workspace
	MatrixType _L;
	DiagonalType _D;
};

}

