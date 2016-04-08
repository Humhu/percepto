#include "percepto/SimpleConvergence.h"

namespace percepto
{

SimpleConvergenceCriteria::SimpleConvergenceCriteria()
: maxRuntime( std::numeric_limits<double>::infinity() ),
maxIterations( 0 ),
minElementDelta( -std::numeric_limits<double>::infinity() ),
minAverageDelta( -std::numeric_limits<double>::infinity() ),
minElementGradient( -std::numeric_limits<double>::infinity() ),
minAverageGradient( -std::numeric_limits<double>::infinity() ),
minObjectiveDelta( -std::numeric_limits<double>::infinity() )
{}

SimpleConvergence::SimpleConvergence( const SimpleConvergenceCriteria& criteria )
: _criteria( criteria ), _initialized( false ), _iteration( 0 ),
_startTicks( clock() ) {}

bool SimpleConvergence::Converged( double objective, const VectorType& params,
                                   const VectorType& gradient )
{
	_iteration++;
	if( _criteria.maxIterations > 0 &&
	   _iteration >= _criteria.maxIterations ) { return true; }

	clock_t now = clock();
	double timeSinceStart = ((double) now - _startTicks ) / CLOCKS_PER_SEC;
	if( timeSinceStart > _criteria.maxRuntime ) { return true; }

	if( !_initialized )
	{
		_lastObjective = objective;
		_lastGradient = gradient;
		_lastParams = params;
		return false;
	}

	VectorType deltaParams = (params - _lastParams).array().abs().matrix();
	double maxDeltaParams = deltaParams.maxCoeff();
	if( maxDeltaParams < _criteria.minElementDelta ) { return true; }
	double avgDeltaParams = deltaParams.sum() / deltaParams.size();
	if( avgDeltaParams < _criteria.minAverageDelta ) { return true; }

	VectorType gradAbs = gradient.array().abs().matrix();
	double maxGradient = gradAbs.maxCoeff();
	if( maxGradient < _criteria.minElementGradient ) { return true; }
	double avgGradient = gradAbs.sum() / gradAbs.size();
	if( avgGradient < _criteria.minAverageGradient ) { return true; }

	double deltaObj = std::abs( objective - _lastObjective );
	if( deltaObj < _criteria.minObjectiveDelta ) { return true; }

	_lastParams = params;
	_lastGradient = gradient;
	_lastObjective = objective;

	return false;
}

}