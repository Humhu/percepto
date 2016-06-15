#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/compo/Parametric.hpp"
#include <Eigen/Dense>

namespace percepto 
{

// TODO Support matrix inputs?
/*! \brief A simple linear regression class. Outputs Weights matrix * input. 
 * The matrix and input are dynamically-sized. Assumes row-major ordering
 * for vectorizing the weights matrix. */
class LinearRegressor
: public Parametric, 
public Producer<VectorType>
{
public:
	
	typedef Eigen::Matrix<double,-1,-1,Eigen::RowMajor> ParamType;
	typedef VectorType InputType;
	typedef VectorType OutputType;
	typedef Producer<VectorType> InputModule;

	LinearRegressor() {}

	LinearRegressor( unsigned int inputDim, unsigned int outputDim )
	: _W( ParamType::Zero( outputDim, inputDim ) ) {}

	/*! \brief Create a linear regressor with specified weights. */
	LinearRegressor( const ParamType& weightMat )
	: _W( weightMat ) {}
	
	LinearRegressor( InputModule* b, const ParamType& weightMat ) {}

	void SetBase( InputModule* b ) { _base = b; }

	unsigned int OutputDim() const { return _W.rows(); }

	/*! \brief Retrieve the parameter vector by returning a vector view of the
	 * matrix. */
	void SetParams( const ParamType& p ) 
	{ 
		assert( p.rows() == _W.rows() &&
		        p.cols() == _W.cols() );
		_W = p;
	}

	virtual void SetParamsVec( const VectorType& vec ) 
	{
		assert( vec.size() == _W.size() );
		_W = Eigen::Map<const ParamType>( vec.data(), OutputDim(), InputDim() );
	}

	ParamType GetParams() const 
	{ 
		return _W;
	}

	virtual VectorType GetParamsVec() const
	{
		return Eigen::Map<const VectorType>( _W.data(), _W.size(), 1 );
	}
	
	MatrixType Backprop( const MatrixType& nextDodx )
	{
		assert( nextDodx.cols() == OutputDim() );

		InputType input = _base.Evaluate();
		MatrixType thisDodw = MatrixType( nextDodx.rows(), ParamDim() );
		for( unsigned int i = 0; i < nextDodx.rows(); i++ )
		{
			for( unsigned int j = 0; j < OutputDim(); j++ )
			{
				thisDodw.block(i, j*InputDim(), 1, InputDim()) =
					nextDodx(i,j) * input.transpose();
			}
		}
		MatrixType thisDodx = nextInfo.dodx * _W;
		Parametric::AccumulateWeightDerivs( thisDodw );
		// Parametric::AccumulateInputDerivs( thisDodx );

		return thisDodx;
	}
	
	/*! \brief Produce the output. */
	OutputType Evaluate() const 
	{ 
		return _W * _base.Evaluate();
	}

private:
	
	// NOTE Since the matrix is dynamic-sized we don't have to use the 
	// eigen aligned new operator
	MatrixType _W; // The weight vector
	BaseModule* _base;
	
};

}

