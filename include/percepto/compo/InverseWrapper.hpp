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
	: _input( this ) {}

	InverseWrapper( const InverseWrapper& other ) 
	: _input( this ) {}

	void SetSource( SourceType* src ) { src->RegisterConsumer( &_input ); }

	virtual void Foreprop()
	{
		MatrixType in = _input.GetInput();
		SolverType solver( _input.GetInput() );
		SourceType::SetOutput( solver.solve( MatrixType::Identity( in.rows(), 
		                                                           in.cols() ) ) );
		SourceType::Foreprop();
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		MatrixType Sinv = SourceType::GetOutput(); // Current output
		if( nextDodx.cols() != Sinv.size() )
		{
			throw std::runtime_error( "InverseWrapper: Backprop dim error." );
		}
		MatrixType dSdx( Sinv.size(), Sinv.size() );
		MatrixType d = MatrixType::Zero( Sinv.rows(),
		                                 Sinv.cols() );
		for( unsigned int i = 0; i < Sinv.size(); i++ )
		{
			d(i) = 1;
			MatrixType temp = Sinv * d * Sinv.transpose();
			dSdx.col(i) = -Eigen::Map<VectorType>( temp.data(), temp.size(), 1 );
			d(i) = 0;
		}

		if( nextDodx.size() == 0 )
		{
			// std::cout << "Inverse dSdx: " << dSdx << std::endl;
			_input.Backprop( dSdx );
		}
		else
		{
			// std::cout << "Inverse nextDodx * dSdx: " << nextDodx * dSdx << std::endl;
			_input.Backprop( nextDodx * dSdx );
		}
	}

private:

	SinkType _input;
};

}