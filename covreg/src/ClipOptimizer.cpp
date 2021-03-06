#include "covreg/ClipOptimizer.h"
#include "optim/OptimizerParser.h"

#include <iostream>
#include <sstream>

using namespace argus;

namespace percepto
{

InnovationClipOptimizer::InnovationClipOptimizer( CovarianceEstimator& qReg,
                                                  const InnovationClipParameters& params )
: _paramWrapper( std::make_shared<percepto::ParameterWrapper>() ),
  _currentEpisode( nullptr ),
  _transReg( qReg ),
  _problem( _paramWrapper, params.l2Weight, params.batchSize ),
  _maxEpisodeLength( params.maxEpisodeLength )
{
	_paramWrapper->AddParameters( qReg.GetParamSet() );
}

void InnovationClipOptimizer::AddObservationReg( CovarianceEstimator& reg,
                                                 const std::string& name )
{
	WriteLock lock( _mutex );
	_obsRegs.emplace( name, reg );
	_paramWrapper->AddParameters( reg.GetParamSet() );
}

void InnovationClipOptimizer::AddPredict( const PredictInfo& info,
                                          const VectorType& input )
{
	WriteLock lock( _mutex );

	_predBuffer.emplace_back( info, input );
}

ros::Time InnovationClipOptimizer::GetEarliestTime()
{
	KalmanFilterEpisode* earliest = _problem.GetOldestEpisode();
	if( !earliest )
	{
		return ros::Time::now();
	}
	return earliest->startTime;
}

void InnovationClipOptimizer::RemoveEarliestEpisode()
{
	_problem.RemoveOldestEpisode();
}

bool InnovationClipOptimizer::AddUpdate( const UpdateInfo& info,
                                         const VectorType& input,
                                         const std::string& name,
                                         double scale,
                                         const ros::Time& stamp )
{
	WriteLock lock( _mutex );

	if( _obsRegs.count( name ) == 0 ) 
	{
		throw std::runtime_error( "Received unregistered update." );
	}
	CovarianceEstimator& est = _obsRegs.at( name );

	if( _predBuffer.size() == 0 )
	{
		std::cout << "Received update with no predicts!" << std::endl;
		return false;
	}

	_currentEpisode = _problem.GetCurrentEpisode();
	if( _currentEpisode == nullptr ||
	    _currentEpisode->NumUpdates() >= _maxEpisodeLength )
	{
		std::pair<PredictInfo,VectorType>& item = _predBuffer.front();
		PredictInfo& info = item.first;
		_problem.EmplaceEpisode( info.prior_state_cov, stamp );
		_currentEpisode = _problem.GetCurrentEpisode();
	}

	for( unsigned int i = 0; i < _predBuffer.size(); i++ )
	{
		std::pair<PredictInfo,VectorType>& item = _predBuffer[i];
		PredictInfo& info = item.first;
		VectorType& input = item.second;
		_currentEpisode->EmplacePredict( _currentEpisode->GetTailCov(), 
		                                _transReg.GetModule(),
		                                // info.Q / info.dt,
		                                info.step_dt,
		                                input,
		                                info.trans_jacobian );
	}
	_predBuffer.clear();

	_currentEpisode->EmplaceUpdate( name,
	                               scale,
	                               _currentEpisode->GetTailCov(), 
	                               est.GetModule(),
	                               input,
	                               info.obs_jacobian,
	                               info.prior_obs_error );
	return true;
}

void InnovationClipOptimizer::BreakCurrentEpisode()
{
	WriteLock lock( _mutex );
	_currentEpisode = nullptr;
}

size_t InnovationClipOptimizer::NumEpisodes() const
{
	return _problem.NumEpisodes();
}

size_t InnovationClipOptimizer::CurrentEpisodeLength() const
{
	if( NumEpisodes() == 0 || _currentEpisode == nullptr ) { return 0; }
	return _problem.GetCurrentEpisode()->NumUpdates();
}

void InnovationClipOptimizer::InitializeOptimization( const ros::NodeHandle& ph )
{
	WriteLock lock( _mutex );
	_optimizer = parse_modular_optimizer( ph );
	_optimizer->ResetAll();
}

bool InnovationClipOptimizer::Optimize()
{	
	WriteLock lock( _mutex );
	_optimizer->Optimize( _problem );
	return false;
}

double InnovationClipOptimizer::CalculateCost()
{
	WriteLock lock( _mutex );
	return _problem.ComputeObjective();
}

void InnovationClipOptimizer::Print( std::ostream& os )
{
	// TODO
	WriteLock lock( _mutex );
	os << "Optimization problem: " << std::endl << _problem;
}

std::ostream& operator<<( std::ostream& os, InnovationClipOptimizer& opt )
{
	opt.Print( os );
	return os;
}

}