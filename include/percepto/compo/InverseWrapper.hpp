#pragma once

#include "percepto/PerceptoTypes.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>

namespace percepto
{

template <typename Base, 
          template <typename> class Solver = Eigen::ColPivHouseholderQR>
class InverseWrapper
{
public:

	typedef Base BaseType;
	typedef Solver<MatrixType> SolverType;
	typedef typename BaseType::OutputType OutputType;

	InverseWrapper( BaseType& base )
	: _base( base ) {}

	unsigned int InputDim() const { return _base.OutputDim(); }
	unsigned int OutputDim() const { return _base.OutputDim(); }
	MatrixSize OutputSize() const { return _base.OutputSize(); }

	MatrixType Evaluate()
	{
		SolverType solver( _base.Evaluate() );
		return solver.inverse();
	}

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "InverseWrapper: Backprop dim error." );
		}
		SolverType solver( _base.Evaluate() );
		MatrixType Sinv = solver.inverse();
		MatrixType dSdx( OutputDim(), OutputDim() ); // OutDim == InDim
		MatrixType d = MatrixType::Zero( _base.OutputSize().rows,
		                                 _base.OutputSize().cols );
		for( unsigned int i = 0; i < InputDim(); i++ )
		{
			d(i) = 1;
			MatrixType temp = Sinv * d * Sinv.transpose();
			dSdx.col(i) = -Eigen::Map<VectorType>( temp.data(), temp.size(), 1 );
			d(i) = 0;
		}
		MatrixType thisDodx = nextDodx * dSdx;
		_base.Backprop( thisDodx );
		return thisDodx;
	}

private:

	BaseType& _base;
};

template <typename Base, 
          template <typename,int> class Solver = Eigen::LDLT>
class PSDInverseWrapper
{
public:

	typedef Base BaseType;
	typedef Solver<MatrixType, Eigen::Lower> SolverType;
	typedef typename BaseType::OutputType OutputType;

	PSDInverseWrapper( BaseType& base )
	: _base( base ) {}

	unsigned int InputDim() const { return _base.OutputDim(); }
	unsigned int OutputDim() const { return _base.OutputDim(); }
	MatrixSize OutputSize() const { return _base.OutputSize(); }

	MatrixType Evaluate()
	{
		SolverType solver( _base.Evaluate() );
		return solver.solve( MatrixType::Identity( _base.OutputSize().rows,
		                                           _base.OutputSize().cols ) );
	}

	MatrixType Backprop( const MatrixType& nextDodx )
	{
		if( nextDodx.cols() != OutputDim() )
		{
			throw std::runtime_error( "PSDInverseWrapper: Backprop dim error." );
		}
		SolverType solver( _base.Evaluate() );
		MatrixType Sinv = solver.solve( MatrixType::Identity( _base.OutputSize().rows,
		                                _base.OutputSize().cols ) );
		MatrixType dSdx( OutputDim(), OutputDim() ); // OutDim == InDim
		MatrixType d = MatrixType::Zero( _base.OutputSize().rows,
		                                 _base.OutputSize().cols );
		for( unsigned int i = 0; i < InputDim(); i++ )
		{
			d(i) = 1;
			MatrixType temp = Sinv * d * Sinv.transpose();
			dSdx.col(i) = -Eigen::Map<VectorType>( temp.data(), temp.size(), 1 );
			d(i) = 0;
		}
		MatrixType thisDodx = nextDodx * dSdx;
		_base.Backprop( thisDodx );
		return thisDodx;
	}

private:

	BaseType& _base;
};

}