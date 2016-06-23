#pragma once

#include <ctime>
#include "percepto/PerceptoTypes.h"

namespace percepto
{

//template <typename ParameterType>
struct OptimizationResults
{
	bool converged; // Whether it converged or bailed

	// Fields relating to optimization results
	//ParameterType initialParameters;
	//ParameterType finalParameters;
	double finalObjective;

	// Fields relating to optimization performance
	unsigned int numObjectiveEvaluations;
	unsigned int numGradientEvaluations;
	double totalElapsedSecs;
	double totalObjectiveSecs;
	double totalGradientSecs;
};

class OptimizationProfiler
{
public:

	OptimizationProfiler() {}

	OptimizationResults GetResults() 
	{ 
		_results.totalObjectiveSecs = TicksToSecs( _objAcc );
		_results.totalGradientSecs = TicksToSecs( _gradAcc );
		return _results; 
	}

	void Reset() 
	{ 
		_results.numObjectiveEvaluations = 0;
		_results.numGradientEvaluations = 0;
		_results.totalElapsedSecs = 0;
		_results.totalObjectiveSecs = 0;
		_results.totalGradientSecs = 0;
		_objAcc = 0;
		_gradAcc = 0;
	}

	void StartOverall()
	{
		Reset();
		_allStart = clock();
	}

	void StopOverall()
	{
		_results.totalElapsedSecs = TicksToSecs( clock() - _allStart );
	}

	/*! \brief Records an untimed objective evaluation. */
	void RecordObjectiveEvaluation()
	{
		_results.numObjectiveEvaluations++;
	}

	/*! \brief Records an untimed gradient evaluation. */
	void RecordGradientEvaluation()
	{
		_results.numGradientEvaluations++;
	}

	/*! \brief Begin timing objective evaluation. */
	void StartObjectiveCall() 
	{ 
		_objStart = clock(); 
	}

	/*! \brief End timing objective evaluation. Also records 
	 * an objective evaluation. */
	void FinishObjectiveCall()
	{
		_objAcc += clock() - _objStart;
		RecordObjectiveEvaluation();
	}

	/*! \brief Begin timing gradient evaluation. */
	void StartGradientCall()
	{
		_gradStart = clock();
	}

	/*! \brief End timing gradient evaluation. Also records
	 * a gradient evaluation. */
	void FinishGradientCall()
	{
		_gradAcc += clock() - _gradStart;
		RecordGradientEvaluation();
	}


private:

	double TicksToSecs( clock_t t )
	{
		return  1.0 * t / CLOCKS_PER_SEC;
	}

	OptimizationResults _results;
	clock_t _allStart; // Clock counter on optimization start
	clock_t _objStart; // Clock counter on objective evaluation start
	clock_t _objAcc; // Accumulated objective ticks
	clock_t _gradStart; // Clock counter on gradient evaluation start
	clock_t _gradAcc; // Accumulated gradient ticks

};

}