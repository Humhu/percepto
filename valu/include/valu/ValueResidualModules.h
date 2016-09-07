#pragma once

#include "valu/ValueFunctionModules.h"
#include "modprop/compo/ScaleWrapper.hpp"
#include "modprop/compo/OffsetWrapper.hpp"
#include "modprop/optim/SquaredLoss.hpp"
#include "modprop/compo/DifferenceWrapper.hpp"

namespace percepto
{

struct BellmanResidualModule
{

	percepto::TerminalSource<VectorType> input;
	ScalarFieldApproximator::Ptr estValue;

	percepto::TerminalSource<VectorType> nextInput;
	ScalarFieldApproximator::Ptr nextValue;
	percepto::ScaleWrapper<double> discountedNextValue;
	percepto::OffsetWrapper<double> targetValue;
	
	percepto::DifferenceWrapper<double> residual;
	percepto::SquaredLoss<double> loss;

	BellmanResidualModule( ScalarFieldApproximator::Ptr estValueModule,
	                       const VectorType& currIn,
	                       ScalarFieldApproximator::Ptr nextValueModule,
	                       const VectorType& nextIn,
	                       double reward,
	                       double gamma );

	void Foreprop();
	void Invalidate();

	percepto::Source<double>& GetOutputSource();
};

}