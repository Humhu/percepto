#include "valu/ApproximateValueLearner.h"

#include "argus_utils/utils/ParamUtils.h"
#include "argus_utils/utils/MatrixUtils.h"

#include "percepto_msgs/GetParameters.h"
#include "percepto_msgs/SetParameters.h"

#include <ros/service.h>
#include <boost/foreach.hpp>

using namespace argus;

namespace percepto
{

ApproximateValueProblem::ApproximateValueProblem() {}

void ApproximateValueProblem::Initialize( percepto::Parameters::Ptr params,
                                          double l2Weight,
                                          unsigned int sampleSize,
                                          double penaltyWeight )
{
	penaltyScale = penaltyWeight;
	regularizer.SetParameters( params );
	regularizer.SetWeight( l2Weight );
	objective.SetSourceA( &loss );
	objective.SetSourceB( &regularizer );
	loss.SetBatchSize( sampleSize );
}

void ApproximateValueProblem::RemoveOldest()
{
	loss.RemoveOldestSource();
	modules.pop_front();
	penalties.pop_front();
	modSums.pop_front();
}

size_t ApproximateValueProblem::NumModules() const
{
	return modules.size();
}

void ApproximateValueProblem::Invalidate()
{
	regularizer.Invalidate();
	for( unsigned int i = 0; i < NumModules(); ++i )
	{
		modules[i].Invalidate();
		penalties[i].Invalidate();
	}
}

void ApproximateValueProblem::Foreprop()
{
	regularizer.Foreprop();
	
	loss.Resample();
	const std::vector<unsigned int>& inds = loss.GetActiveInds();
	BOOST_FOREACH( unsigned int ind, inds )
	// for( unsigned int ind = 0; ind < modules.size(); ind++ )
	{
		modules[ind].Foreprop();
		penalties[ind].Foreprop();
	}
}

void ApproximateValueProblem::Backprop()
{
	objective.Backprop( MatrixType::Identity(1,1) );
}

void ApproximateValueProblem::BackpropNatural()
{
	MatrixType back = MatrixType::Identity(1,1) / modules.size();
	BOOST_FOREACH( BellmanResidualModule& module, modules )
	{
		module.Foreprop();
		// NOTE estValue is used in the regularizer term, so we can't backprop it directly
		module.nextValue->GetOutputSource().Backprop( back );
	}
}

double ApproximateValueProblem::GetOutput() const
{
	double out = objective.GetOutput();
	// ROS_INFO_STREAM( "Objective: " << out );
	return out;
}

ApproximateValueLearner::ApproximateValueLearner() 
: _infoManager( _lookup ) {}

void ApproximateValueLearner::Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph )
{
	// Initialize the value function
	std::string valueName;
	GetParamRequired( ph, "value_name", valueName );
	if( !_infoManager.CheckMemberInfo( valueName, true, ros::Duration( 10.0 ) ) )
	{
		throw std::runtime_error( "Could not find value function: " + valueName );
	}

	const ValueInfo& info = _infoManager.GetInfo( valueName );
	_value.Initialize( info.inputDim, info.approximatorInfo );

	ros::service::waitForService( info.paramQueryService );
	ros::service::waitForService( info.paramSetService );
	
	// Get the current value function parameters
	percepto_msgs::GetParameters::Request req;
	percepto_msgs::GetParameters::Response res;
	if( !ros::service::call( info.paramQueryService, req, res ) )
	{
		throw std::runtime_error( "Could not query parameters at: " + info.paramQueryService );
	}
	_value.GetParameters()->SetParamsVec( GetVectorView( res.parameters ) );
	ROS_INFO_STREAM( "Initialized value: " << std::endl << *_value.GetApproximatorModule() );

	_setParamsClient = nh.serviceClient<percepto_msgs::SetParameters>( info.paramSetService, true );

	ros::NodeHandle lh( ph.resolveName( "optimization" ) );
	GetParamRequired( lh, "min_num_modules", _minModulesToOptimize );
	GetParam( lh, "clear_optimized_modules", _clearAfterOptimize, false );
	if( !_clearAfterOptimize )
	{
		GetParamRequired( lh, "max_num_modules", _maxModulesToKeep );
	}

	percepto::SimpleConvergenceCriteria criteria;
	GetParam( lh, "convergence/max_time", criteria.maxRuntime, std::numeric_limits<double>::infinity() );
	GetParam( lh, "convergence/max_iters", criteria.maxIterations, std::numeric_limits<unsigned int>::max() );
	GetParam( lh, "convergence/min_avg_delta", criteria.minAverageDelta, -std::numeric_limits<double>::infinity() );
	GetParam( lh, "convergence/min_avg_grad", criteria.minAverageGradient, -std::numeric_limits<double>::infinity() );
	_convergence = std::make_shared<percepto::SimpleConvergence>( criteria );

	// percepto::NaturalStepperParameters stepperParams;
	percepto::AdamParameters stepperParams;
	GetParam( lh, "stepper/step_size", stepperParams.alpha, 1E-3 );
	GetParam( lh, "stepper/max_step", stepperParams.maxStepElement, 1.0 );
	GetParam( lh, "stepper/beta1", stepperParams.beta1, 0.9 );
	GetParam( lh, "stepper/beta2", stepperParams.beta2, 0.99 );
	GetParam( lh, "stepper/epsilon", stepperParams.epsilon, 1E-7 );
	
	// double windowRatio;
	// GetParam( lh, "stepper/window_ratio", windowRatio, 1.0 );
	// stepperParams.windowLen = std::ceil( windowRatio * _value.GetParameters()->ParamDim() );

	GetParam( lh, "stepper/enable_decay", stepperParams.enableDecay, false );
	GetParam( lh, "stepper/reset_after_optimization", _resetStepperAfter, false );
	_stepper = std::make_shared<percepto::AdamStepper>( stepperParams );
	_optimizer = std::make_shared<percepto::AdamOptimizer>( *_stepper, 
	                                                          *_convergence,
	                                                          *_value.GetParameters(),
	                                                          percepto::OPT_MINIMIZATION );
	// _optimizer = std::make_shared<percepto::SimpleNaturalOptimizer>( *_convergence,
	//                                                                  *_value.GetParameters(),
	//                                                                  stepperParams,
	//                                                                  percepto::OPT_MINIMIZATION );
	_optimCounter = 0;

	double l2Weight, valuePenaltyWeight;
	unsigned int batchSize;
	GetParamRequired( lh, "l2_weight", l2Weight );
	GetParamRequired( lh, "batch_size", batchSize );
	GetParamRequired( lh, "value_penalty_weight", valuePenaltyWeight );
	_problem.Initialize( _value.GetParameters(), 
	                     l2Weight, 
	                     batchSize, 
	                     valuePenaltyWeight );

	// TODO GetParam specialization for ros::Duration and ros::Rate
	double updateRate;
	GetParamRequired( ph, "update_rate", updateRate );
	GetParamRequired( ph, "discount_rate", _discountRate );

	_updateTimer = nh.createTimer( ros::Duration( 1.0/updateRate ),
	                               &ApproximateValueLearner::UpdateCallback,
	                               this );

	_srsSub = nh.subscribe( "srs_tuple", 0, &ApproximateValueLearner::SRSCallback, this );
}

void ApproximateValueLearner::SRSCallback( const percepto_msgs::SRSTuple::ConstPtr& msg )
{
	WriteLock lock( _mutex );
	_srsBuffer.emplace_back( *msg );
}

void ApproximateValueLearner::UpdateCallback( const ros::TimerEvent& event )
{
	WriteLock lock( _mutex );
	while( !_srsBuffer.empty() )
	{
		const SRSTuple& srs = _srsBuffer.front();

		double dt = (srs.nextTime - srs.time).toSec();
		double discountFactor = std::exp( dt * std::log( _discountRate ) );

		ROS_INFO_STREAM( srs );
		_problem.EmplaceModule( _value.GetApproximatorModule(),
		                        srs.state,
		                        _value.GetApproximatorModule(),
		                        srs.nextState,
		                        srs.reward,
		                        discountFactor );
		_srsBuffer.pop_front();
	}
	lock.unlock();

	RunOptimization();
}

void ApproximateValueLearner::RunOptimization()
{
	if( _problem.NumModules() < _minModulesToOptimize )
	{
		ROS_INFO_STREAM( "Num modules: " << _problem.NumModules() <<
		                 " less than min: " << _minModulesToOptimize );
		return;
	}

	if( _resetStepperAfter )
	{
		// _stepper->Reset();
	}

	ROS_INFO_STREAM( "Beginning optimization..." );
	percepto::OptimizationResults results = _optimizer->Optimize( _problem );
	ROS_INFO_STREAM( "Objective: " << results.finalObjective );
	_optimCounter++;
	if( _optimCounter % 10 == 0 )
	{
		ROS_INFO_STREAM( "Approximator: " << _value.GetApproximatorModule()->Print() );
	}

	unsigned int targetNum = _maxModulesToKeep;
	if( _clearAfterOptimize )
	{
		targetNum = 0;
	}
	while( _problem.NumModules() > targetNum )
	{
		_problem.RemoveOldest();
	}

	ROS_INFO_STREAM( "Setting parameters..." );
	percepto_msgs::SetParameters setParams;
	SerializeMatrix( _value.GetParameters()->GetParamsVec(), setParams.request.parameters );
	if( !_setParamsClient.call( setParams ) )
	{
		throw std::runtime_error( "Could not set parameters." );
	}
	ROS_INFO_STREAM( "Setting parameters done." );
}

}