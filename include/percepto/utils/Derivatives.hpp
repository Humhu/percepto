#pragma once

#include "percepto/PerceptoTypes.h"

namespace percepto
{

template <typename Regressor>
std::vector<double> EvalMatDeriv( Regressor& r, Parametric& p )
{
	static double stepSize = 1E-3;

	std::vector<double> errors( p.ParamDim() );

	typedef typename Regressor::OutputType OutType;

	OutType output, trueOutput;
	VectorType paramsVec = p.GetParamsVec();
	MatrixType sysDodx = MatrixType::Identity( r.OutputDim(), r.OutputDim() );
	p.ResetAccumulators();
	r.Backprop( sysDodx );
	MatrixType dodw = p.GetAccWeightDerivs();

	output = r.Evaluate();

	for( unsigned int ind = 0; ind < p.ParamDim(); ind++ )
	{
		// Take a step and check the output
		VectorType testParamsVec = paramsVec;
		testParamsVec[ind] += stepSize;
		p.SetParamsVec( testParamsVec );
		trueOutput = r.Evaluate();
		p.SetParamsVec( paramsVec );
		
		VectorType gi = dodw.col(ind);
		Eigen::Map<MatrixType> dS( gi.data(), r.OutputSize().rows, r.OutputSize().cols );
		MatrixType predOut = output + dS*stepSize;
		double predErr = ( predOut - trueOutput ).norm();

		errors[ind] = predErr;
	}
	return errors;
}

template <typename Cost>
std::vector<double> EvalCostDeriv( Cost& cost, Parametric& p )
{
	static double stepSize = 1E-3;

	std::vector<double> errors( p.ParamDim() );

	typedef typename Cost::OutputType OutType;

	OutType output, trueOutput;
	VectorType paramsVec = p.GetParamsVec();
	MatrixType sysDodx = MatrixType::Identity(1,1);
	p.ResetAccumulators();
	MatrixType dodx = cost.Backprop( sysDodx );
	MatrixType dodw = p.GetAccWeightDerivs();

	output = cost.Evaluate();

	for( unsigned int ind = 0; ind < dodw.size(); ind++ )
	{
		// Take a step and check the output
		VectorType testParamsVec = paramsVec;
		testParamsVec[ind] += stepSize;
		p.SetParamsVec( testParamsVec );
		trueOutput = cost.Evaluate();
		p.SetParamsVec( paramsVec );

		// double dOut = output - trueOutput;
		// if( dOut == 0 ) { dOut = 1; }
		double predOut = output + dodw(ind)*stepSize;
		double predErr = std::abs( predOut - trueOutput );

		// std::cout << "err: " << predErr << " true - nom: " << trueOutput - output
		//           << " pred - nom: " << predOut - output << std::endl;

		// if( predErr > 0.5 ) 
		// { 
		// 	std::cerr << "Too much error!" << std::endl;
		// }

		errors[ind] = predErr;
	}
	return errors;
}

}