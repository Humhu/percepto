#include "optim/OptimizationProfiler.h"

namespace percepto
{

OptimizationProfiler::OptimizationProfiler() {}

ProfileResults OptimizationProfiler::GetResults() 
{ 
	_results.totalObjectiveSecs = TicksToSecs( _objAcc );
	_results.totalGradientSecs = TicksToSecs( _gradAcc );
	return _results; 
}

void OptimizationProfiler::Reset() 
{ 
	_results.numObjectiveEvaluations = 0;
	_results.numGradientEvaluations = 0;
	_results.totalElapsedSecs = 0;
	_results.totalObjectiveSecs = 0;
	_results.totalGradientSecs = 0;
	_objAcc = 0;
	_gradAcc = 0;
}

void OptimizationProfiler::StartOverall()
{
	Reset();
	_allStart = clock();
}

void OptimizationProfiler::StopOverall()
{
	_results.totalElapsedSecs = TicksToSecs( clock() - _allStart );
}

/*! \brief Records an untimed objective evaluation. */
void OptimizationProfiler::RecordObjectiveEvaluation()
{
	_results.numObjectiveEvaluations++;
}

/*! \brief Records an untimed gradient evaluation. */
void OptimizationProfiler::RecordGradientEvaluation()
{
	_results.numGradientEvaluations++;
}

/*! \brief Begin timing objective evaluation. */
void OptimizationProfiler::StartObjectiveCall() 
{ 
	_objStart = clock(); 
}

/*! \brief End timing objective evaluation. Also records 
 * an objective evaluation. */
void OptimizationProfiler::FinishObjectiveCall()
{
	_objAcc += clock() - _objStart;
	RecordObjectiveEvaluation();
}

/*! \brief Begin timing gradient evaluation. */
void OptimizationProfiler::StartGradientCall()
{
	_gradStart = clock();
}

/*! \brief End timing gradient evaluation. Also records
 * a gradient evaluation. */
void OptimizationProfiler::FinishGradientCall()
{
	_gradAcc += clock() - _gradStart;
	RecordGradientEvaluation();
}


double OptimizationProfiler::TicksToSecs( clock_t t )
{
	return  1.0 * t / CLOCKS_PER_SEC;
}

}