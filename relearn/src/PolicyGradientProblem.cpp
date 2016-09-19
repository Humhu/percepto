#include "relearn/PolicyGradientProblem.h"
#include "poli/PoliCommon.h"

namespace percepto
{

PolicyGradientOptimization::PolicyGradientOptimization() 
{}

void PolicyGradientOptimization::Initialize( percepto::Parameters::Ptr params,
                                             double l2Weight,
                                             unsigned int batchSize,
                                             double maxDivergence )
{
	_maxDivergence = maxDivergence;
	parameters = params;
	regularizer.SetParameters( params );
	regularizer.SetWeight( l2Weight );
	objective.SetSourceA( &rewards );
	objective.SetSourceB( &regularizer );
	rewards.SetBatchSize( batchSize );
}

size_t PolicyGradientOptimization::NumModules() const
{
	return modules.size();
}

bool PolicyGradientOptimization::IsMinimization() const
{
	return false;
}

void PolicyGradientOptimization::Resample() 
{
	rewards.Resample();
}

double PolicyGradientOptimization::ComputeObjective()
{
	Invalidate();
	Foreprop();
	// std::cout << "Objective: " << objective.GetOutput() << std::endl;
	return objective.GetOutput();
}

VectorType PolicyGradientOptimization::ComputeGradient()
{
	Invalidate();
	Foreprop();
	Backprop();
	return parameters->GetDerivs();
}

VectorType PolicyGradientOptimization::GetParameters() const
{
	return parameters->GetParamsVec();
}

void PolicyGradientOptimization::SetParameters( const VectorType& p )
{
	parameters->SetParamsVec( p );
}

VectorType PolicyGradientOptimization::ComputeNaturalGradient()
{
	Invalidate();
	Foreprop();
	BackpropNatural();
	return parameters->GetDerivs();
}

bool PolicyGradientOptimization::IsSatisfied()
{
	// double divergence = std::abs( ComputeLogProb() - _initLikelihood );
	double divergence = ComputeKLD();
	std::cout << "Divergence: " << divergence << std::endl;
	return divergence < _maxDivergence;
}

void PolicyGradientOptimization::ResetConstraints()
{
	_policyMeans.clear();
	_policyInfos.clear();
	ForepropAll();
	// _initLikelihood = ComputeLogProb();
	BOOST_FOREACH( ContinuousLogGradientModule& mod, modules )
	{
		_policyMeans.push_back( mod.network->GetMeanSource().GetOutput() );
		_policyInfos.push_back( mod.network->GetInfoSource().GetOutput() );
	}
}

void PolicyGradientOptimization::Invalidate()
{
	BOOST_FOREACH( ContinuousLogGradientModule& mod, modules )
	{
		mod.Invalidate();
	}
	regularizer.Invalidate();
	parameters->ResetAccumulators();
}

void PolicyGradientOptimization::Foreprop()
{
	const std::vector<unsigned int>& inds = rewards.GetActiveInds();
	BOOST_FOREACH( unsigned int ind, inds )
	{
		modules[ind].Foreprop();
	}
	regularizer.Foreprop();
}

void PolicyGradientOptimization::ForepropAll()
{
	BOOST_FOREACH( ContinuousLogGradientModule& mod, modules )
	{
		mod.Foreprop();
	}
	regularizer.Foreprop();
}

void PolicyGradientOptimization::Backprop()
{
	objective.Backprop( MatrixType::Identity(1,1) );
}

void PolicyGradientOptimization::BackpropNatural()
{
	const std::vector<unsigned int>& inds = rewards.GetActiveInds();
	MatrixType back = MatrixType::Identity(1,1) / inds.size();
	BOOST_FOREACH( unsigned int ind, inds )
	{
		modules[ind].GetLogProbSource()->Backprop( back, true );
	}
}

double PolicyGradientOptimization::ComputeKLD()
{
	Invalidate();
	ForepropAll();
	double acc = 0;
	for( unsigned int i = 0; i < modules.size(); ++i )
	{
		const VectorType& startingMean = _policyMeans[i];
		const MatrixType& startingInfo = _policyInfos[i];
		VectorType currMean = modules[i].network->GetMeanSource().GetOutput();
		MatrixType currInfo = modules[i].network->GetInfoSource().GetOutput();
		acc += gaussian_kl_divergence( startingMean, startingInfo,
		                               currMean, currInfo );
	}
	return acc;
}

double PolicyGradientOptimization::ComputeLogProb()
{
	Invalidate();
	ForepropAll();
	double acc = 0;
	BOOST_FOREACH( ContinuousLogGradientModule& mod, modules )
	{
		acc += mod.GetLogProbSource()->GetOutput();
	}
	return acc; // / modules.size();
}

void PolicyGradientOptimization::RemoveOldest()
{
	rewards.RemoveOldestSource();
	modules.pop_front();
}

ContinuousLogGradientModule& PolicyGradientOptimization::GetLatestModule()
{
	return modules.back();
}

}