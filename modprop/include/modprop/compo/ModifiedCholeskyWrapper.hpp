#pragma once

#include "modprop/compo/Interfaces.h"
#include "modprop/ModpropTypes.h"
#include "modprop/utils/LowerTriangular.hpp"
#include "modprop/utils/MatrixUtils.hpp"

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
	
	typedef Eigen::DiagonalMatrix <ScalarType,
	                               Eigen::Dynamic,
	                               Eigen::Dynamic> DiagonalType;

	ModifiedCholeskyWrapper() 
	: _lInput( this ), _dInput( this ), _initialized( false ) {}

	ModifiedCholeskyWrapper( const ModifiedCholeskyWrapper& other ) 
	: _lInput( this ), _dInput( this ), _initialized( false ) {}

	void SetLSource( InputSourceType* l ) { l->RegisterConsumer( &_lInput ); }
	void SetDSource( InputSourceType* d ) { d->RegisterConsumer( &_dInput ); }

	// Assuming that dodx is given w.r.t. matrix col-major ordering
	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// clock_t start = clock();

		MatrixType dody = nextDodx;
		if( nextDodx.size() == 0 )
		{
			dody = MatrixType::Identity( _D.size(), _D.size() );
		}

		if( dody.cols() != _D.rows() * _D.cols() )
		{
			std::cout << "nextDodx cols: " << dody.cols() << std::endl;
			std::cout << "D size: " << _D.size() << std::endl;
			throw std::runtime_error( "ModifiedCholeskyWrapper: Backprop dim error." );
		}

		if( !_initialized )
		{
			// Calculate output matrix deriv wrt L inputs
			_dSdl = MatrixType( _D.size(), _tmap.NumPositions() );
			MatrixType DLT = _D * _L.transpose();
			for( unsigned int i = 0; i < _tmap.NumPositions(); i++ )
			{
				const TriangularMapping::Index& ind = _tmap.PosToIndex( i );
				// Have to add one to get the offset from the diagonal
				MatrixType dSdi = MatrixType::Zero( _D.rows(), _D.cols() );
				dSdi.row( ind.first + 1 ) = DLT.row( ind.second );
				dSdi.col( ind.first + 1 ) += DLT.row( ind.second );
				
				Eigen::Map<const VectorType> dSdiVec( dSdi.data(), dSdi.size(), 1 );
				_dSdl.col(i) = dSdiVec;
			}

			// Perform D backprop
			_dSdd = MatrixType( _D.size(), _D.rows() );
			for( unsigned int i = 0; i < _D.rows(); i++ )
			{
				MatrixType dSdi = MatrixType::Zero( _D.rows(), _D.cols() );
				dSdi.row(i) = _L.col(i);
				dSdi = _L * dSdi;

				Eigen::Map<const VectorType> dSdiVec( dSdi.data(), dSdi.size(), 1 );
				_dSdd.col(i) = dSdiVec;
			}
			_initialized = true;
		}

		// std::cout << "MC backprop: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;

		MatrixType midLInfoDodx = dody * _dSdl;
		_lInput.Backprop( midLInfoDodx );
		
		MatrixType midDInfoDodx = dody * _dSdd;
		_dInput.Backprop( midDInfoDodx );
		// std::cout << "MC return: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;
	}

	const MatrixType& GetLOutput() const
	{
		return _L;
	}

	const DiagonalType& GetDOutput() const
	{
		return _D;
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

	bool _initialized;
	MatrixType _dSdl;
	MatrixType _dSdd;

	// Workspace
	MatrixType _L;
	DiagonalType _D;
};

}

