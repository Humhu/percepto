#pragma once

#include "percepto/PerceptoTypes.h"

namespace percepto
{

// TODO Initialize sysOutDim to zero
/*! Backpropagation in/out info. The system output dimension is unknown,
 * so dynamic matrices must be used. */
struct BackpropInfo
{
	// System output index corresponds to row index
	// Derivative index corresponds to col index

	// System output dimensionality
	unsigned int sysOutDim;
	// Derivative of system outputs wrt this layer's inputs
	// If empty, layer assumes it is the output layer
	MatrixType dodx;
	// Derivative of system outputs wrt this layer's parameters
	MatrixType dodw;

	BackpropInfo()
	: sysOutDim( 0 ) {}
};

}