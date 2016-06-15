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

	void SetSource( SourceType* src ) { src->RegisterConsumer( &_input ); }

	virtual void Foreprop()
	{
		SolverType solver( _input.GetInput() );
		SourceType::SetOutput( solver.inverse() );
		SourceType::Foreprop();
	}

	virtual void Backprop( const MatrixType& nextDodx )
	{
		MatrixType Sinv = SourceType::GetOutput(); // Current output
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
		MatrixType thisDodx = nextDodx * dSdx;
		_input.Backprop( thisDodx );
	}

private:

	SinkType _input;
};

}