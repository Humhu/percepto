#pragma once

#include "percepto/compo/BackpropInfo.hpp"
#include <Eigen/Dense>

namespace percepto 
{

/*! \brief A simple linear regression class. Outputs Weights matrix * input. 
 * The matrix and input are dynamically-sized. Assumes row-major ordering
 * for vectorizing the weights matrix. */
class LinearRegressor
{
public:
	
	typedef Eigen::Matrix<double,-1,-1,Eigen::RowMajor> ParamType;
	typedef VectorType InputType;
	typedef VectorType OutputType;

	static ParamType create_zeros( unsigned int inputDim,
	                               unsigned int outputDim )
	{
		return ParamType::Zero( outputDim, inputDim );
	}

	/*! \brief Create a linear regressor with specified weights. */
	LinearRegressor( const ParamType& weightMat )
	: _W( weightMat ) {}
	
	unsigned int InputDim() const { return _W.cols(); }
	unsigned int OutputDim() const { return _W.rows(); }
	unsigned int ParamDim() const { return _W.size(); }

	/*! \brief Retrieve the parameter vector by returning a vector view of the
	 * matrix. */
	void SetParams( const ParamType& p ) 
	{ 
		assert( p.size() == _W.size() );
		_W = p;
	}

	void SetParamsVec( const VectorType& vec ) 
	{
		assert( vec.size() == _W.size() );
		_W = Eigen::Map<const ParamType>( vec.data(), OutputDim(), InputDim() );
	}

	ParamType GetParams() const 
	{ 
		return _W;
	}

	VectorType GetParamsVec() const
	{
		return Eigen::Map<const VectorType>( _W.data(), _W.size(), 1 );
	}
	
	BackpropInfo Backprop( const InputType& input, const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo thisInfo;
		thisInfo.dodx = nextInfo.dodx * _W;
		thisInfo.dodw = MatrixType( nextInfo.SystemOutputDim(), ParamDim() );
		for( unsigned int i = 0; i < nextInfo.SystemOutputDim(); i++ )
		{
			for( unsigned int j = 0; j < OutputDim(); j++ )
			{
				thisInfo.dodw.block(i, j*InputDim(), 1, InputDim()) =
					nextInfo.dodx(i,j) * input.transpose();
			}
		}
		return thisInfo;
	}
	
	/*! \brief Produce the output. */
	OutputType Evaluate( const InputType& input ) const 
	{ 
		assert( input.size() == InputDim() );
		return _W * input;
	}

private:
	
	// NOTE Since the matrix is dynamic-sized we don't have to use the 
	// eigen aligned new operator
	MatrixType _W; // The weight vector
	
};

}

