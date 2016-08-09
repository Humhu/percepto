#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/PerceptoTypes.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <iostream>
namespace percepto
{

// NOTE This only works for PSD matrices!
template <typename Mat>
using EigLDL = Eigen::LDLT<Mat, Eigen::Lower>;

template <template <typename> class Solver = Eigen::ColPivHouseholderQR>
class InverseWrapper
: public Source<MatrixType>
{
public:

	typedef Source<MatrixType> SourceType;
	typedef Sink<MatrixType> SinkType;
	typedef Solver<MatrixType> SolverType;
	typedef MatrixType OutputType;

	InverseWrapper() 
	: _input( this ), _initialized( false ) {}

	InverseWrapper( const InverseWrapper& other ) 
	: _input( this ), _initialized( false ) {}

	void SetSource( SourceType* src ) { src->RegisterConsumer( &_input ); }

	virtual void Foreprop()
	{
		MatrixType in = _input.GetInput();
		SolverType solver( _input.GetInput() );
		_initialized = false;
		SourceType::SetOutput( solver.solve( MatrixType::Identity( in.rows(), 
		                                                           in.cols() ) ) );
		SourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		// clock_t start = clock();

		MatrixType Sinv = SourceType::GetOutput(); // Current output
		if( nextDodx.size() != 0 && nextDodx.cols() != Sinv.size() )
		{
			throw std::runtime_error( "InverseWrapper: Backprop dim error." );
		}

		if( !_initialized )
		{
			_dSdx = MatrixType( Sinv.size(), Sinv.size() );
			MatrixType d = MatrixType::Zero( Sinv.rows(),
			                                 Sinv.cols() );
			for( unsigned int i = 0; i < Sinv.size(); i++ )
			{
				d(i) = 1;
				MatrixType temp = Sinv * d * Sinv.transpose();
				_dSdx.col(i) = -Eigen::Map<VectorType>( temp.data(), temp.size(), 1 );
				d(i) = 0;
			}
			_initialized = true;
		}
		
		// std::cout << "Inverse backprop: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;

		if( nextDodx.size() == 0 )
		{
			// std::cout << "Inverse dSdx: " << dSdx << std::endl;
			_input.Backprop( _dSdx );
		}
		else
		{
			// std::cout << "Inverse nextDodx * dSdx: " << nextDodx * dSdx << std::endl;
			_input.Backprop( nextDodx * _dSdx );
		}
		// std::cout << "Inverse return: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;

	}

private:

	SinkType _input;
	bool _initialized;
	MatrixType _dSdx;
};

}