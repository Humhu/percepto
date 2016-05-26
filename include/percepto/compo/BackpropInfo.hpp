#pragma once

#include "percepto/PerceptoTypes.h"

namespace percepto
{

/*! Backpropagation in/out info. */
struct BackpropInfo
{
	// System output index corresponds to row index
	// Derivative index corresponds to col index

	// Derivative of system outputs wrt this layer's inputs
	// If empty, layer assumes it is the output layer
	MatrixType dodx;
	// Derivative of system outputs wrt this layer's parameters
	MatrixType dodw;

	BackpropInfo() {}

	unsigned int ModuleInputDim() const { return dodx.cols(); }
	unsigned int ModuleParamDim() const { return dodw.cols(); }
	unsigned int SystemOutputDim() const { return dodx.rows(); }
};

template <class R>
MatrixType BackpropGradient( R& r, const typename R::InputType& rIn )
{
	BackpropInfo info;
	info.dodx = MatrixType::Identity( r.OutputDim(), r.OutputDim() );
	BackpropInfo output = r.Backprop( rIn, info );
	return output.dodw.transpose();
}

template <class R>
MatrixType BackpropGradient( R& r )
{
	BackpropInfo info;
	info.dodx = MatrixType::Identity( r.OutputDim(), r.OutputDim() );
	BackpropInfo output = r.Backprop( info );
	return output.dodw.transpose();
}

}