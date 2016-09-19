#pragma once

#include <ctime>
#include "optim/OptimInterfaces.h"

namespace percepto
{

struct ProfileResults
{
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

	OptimizationProfiler();

	ProfileResults GetResults();

	void Reset();

	void StartOverall();

	void StopOverall();

	/*! \brief Records an untimed objective evaluation. */
	void RecordObjectiveEvaluation();

	/*! \brief Records an untimed gradient evaluation. */
	void RecordGradientEvaluation();

	/*! \brief Begin timing objective evaluation. */
	void StartObjectiveCall();

	/*! \brief End timing objective evaluation. Also records 
	 * an objective evaluation. */
	void FinishObjectiveCall();

	/*! \brief Begin timing gradient evaluation. */
	void StartGradientCall();

	/*! \brief End timing gradient evaluation. Also records
	 * a gradient evaluation. */
	void FinishGradientCall();


private:

	static double TicksToSecs( clock_t t );

	ProfileResults _results;
	clock_t _allStart; // Clock counter on optimization start
	clock_t _objStart; // Clock counter on objective evaluation start
	clock_t _objAcc; // Accumulated objective ticks
	clock_t _gradStart; // Clock counter on gradient evaluation start
	clock_t _gradAcc; // Accumulated gradient ticks

};

}