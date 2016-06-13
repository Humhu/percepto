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

	OutType startOutput, trueOutput;
	VectorType paramsVec = p.GetParamsVec();
	MatrixType sysDodx = MatrixType::Identity( r.OutputDim(), r.OutputDim() );
	p.ResetAccumulators();
	r.Backprop( sysDodx );
	MatrixType dodw = p.GetAccWeightDerivs();

	startOutput = r.Evaluate();

	for( unsigned int ind = 0; ind < p.ParamDim(); ind++ )
	{
		// Take a step and check the startOutput
		VectorType testParamsVec = paramsVec;
		testParamsVec[ind] += stepSize;
		p.SetParamsVec( testParamsVec );
		trueOutput = r.Evaluate();
		p.SetParamsVec( paramsVec );
		
		VectorType gi = dodw.col(ind);
		Eigen::Map<MatrixType> dS( gi.data(), r.OutputSize().rows, r.OutputSize().cols );
		MatrixType predOut = startOutput + dS*stepSize;
		double predErr = ( predOut - trueOutput ).norm();

		// std::cout << "nominal: " << std::endl << startOutput << std::endl;
		// std::cout << "true - nom: " << std::endl << trueOutput - startOutput << std::endl;
		// std::cout << "pred - nom: " << std::endl << predOut - startOutput << std::endl;

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

	OutType startOutput, trueOutput;
	VectorType paramsVec = p.GetParamsVec();
	MatrixType sysDodx = MatrixType::Identity(1,1);
	p.ResetAccumulators();
	MatrixType dodx = cost.Backprop( sysDodx );
	MatrixType dodw = p.GetAccWeightDerivs();

	startOutput = cost.Evaluate();

	for( unsigned int ind = 0; ind < dodw.size(); ind++ )
	{
		// Take a step and check the startOutput
		VectorType testParamsVec = paramsVec;
		testParamsVec[ind] += stepSize;
		p.SetParamsVec( testParamsVec );
		trueOutput = cost.Evaluate();
		p.SetParamsVec( paramsVec );

		// double dOut = startOutput - trueOutput;
		// if( dOut == 0 ) { dOut = 1; }
		double predOut = startOutput + dodw(ind)*stepSize;
		double predErr = std::abs( predOut - trueOutput );

		// std::cout << "err: " << predErr << " true - nom: " << trueOutput - startOutput
		//           << " pred - nom: " << predOut - startOutput << std::endl;

		// if( predErr > 0.5 ) 
		// { 
		// 	std::cerr << "Too much error!" << std::endl;
		// }

		errors[ind] = predErr;
	}
	return errors;
}

}