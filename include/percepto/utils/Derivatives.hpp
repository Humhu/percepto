#pragma once

#include "percepto/PerceptoTypes.h"

namespace percepto
{

template <typename Regressor, typename Module>
std::vector<double> EvalMatDeriv( Regressor& r, Module& m, Parameters& p )
{
	static double stepSize = 1E-3;

	std::vector<double> errors( p.ParamDim() );

	typedef typename Module::OutputType OutType;

	OutType startOutput, trueOutput;
	VectorType initParams = p.GetParamsVec();
	r.Invalidate();
	r.Foreprop();
	startOutput = m.GetOutput();
	p.ResetAccumulators();
	m.Backprop( MatrixType() );
	MatrixType dodw = p.GetDerivs();

	for( unsigned int ind = 0; ind < p.ParamDim(); ind++ )
	{
		// Take a step and check the startOutput
		VectorType testParamsVec = initParams;
		testParamsVec[ind] += stepSize;
		p.SetParamsVec( testParamsVec );

		r.Invalidate();
		r.Foreprop();
		trueOutput = m.GetOutput();
		
		p.SetParamsVec( initParams );
		
		VectorType gi = dodw.col(ind);
		Eigen::Map<MatrixType> dS( gi.data(), trueOutput.rows(), trueOutput.cols() );
		MatrixType predOut = startOutput + dS*stepSize;
		double predErr = ( predOut - trueOutput ).norm();

		// std::cout << "nominal: " << std::endl << startOutput << std::endl;
		// std::cout << "true - nom: " << std::endl << trueOutput - startOutput << std::endl;
		// std::cout << "pred - nom: " << std::endl << predOut - startOutput << std::endl;

		errors[ind] = predErr;
	}
	return errors;
}

template <typename Regressor, typename Module>
std::vector<double> EvalCostDeriv( Regressor& r, Module& m, Parameters& p )
{
	static double stepSize = 1E-3;

	std::vector<double> errors( p.ParamDim() );


	double startOutput, trueOutput;
	VectorType initParams = p.GetParamsVec();
	r.Invalidate();
	r.Foreprop();
	startOutput = m.GetOutput();	
	p.ResetAccumulators();
	m.Backprop( MatrixType() );
	MatrixType dodw = p.GetDerivs();

	for( unsigned int ind = 0; ind < dodw.size(); ind++ )
	{
		// Take a step and check the startOutput
		VectorType testParamsVec = initParams;
		testParamsVec[ind] += stepSize;
		p.SetParamsVec( testParamsVec );

		r.Invalidate();
		r.Foreprop();
		trueOutput = m.GetOutput();
		
		p.SetParamsVec( initParams );

		double predOut = startOutput + dodw(ind)*stepSize;
		double predErr = std::abs( predOut - trueOutput );

		// std::cout << "err: " << predErr << " true - nom: " << trueOutput - startOutput
		//           << " pred - nom: " << predOut - startOutput << std::endl;

		errors[ind] = predErr;
	}
	return errors;
}

}