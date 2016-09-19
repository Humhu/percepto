#include "covreg/InnovationLikelihoodProblem.h"

namespace percepto
{

InnovationLikelihoodProblem::InnovationLikelihoodProblem( percepto::Parameters::Ptr params,
                                                          double l2Weight,
                                                          unsigned int batchSize )
: parameters( params )
{
	loss.SetBatchSize( batchSize );
	regularizer.SetParameters( parameters );
	regularizer.SetWeight( l2Weight );
	objective.SetSourceA( &loss );
	objective.SetSourceB( &regularizer );
}

bool InnovationLikelihoodProblem::IsMinimization() const
{
	return false;
}

void InnovationLikelihoodProblem::Resample()
{
	loss.Resample();
}

double InnovationLikelihoodProblem::ComputeObjective()
{
	Invalidate();
	Foreprop();
	return loss.GetOutput();
}

VectorType InnovationLikelihoodProblem::ComputeGradient()
{
	Invalidate();
	Foreprop();
	return _parameters->GetDerivs();
}

VectorType InnovationLikelihoodProblem::GetParameters() const
{
	return _parameters->GetParamsVec();
}

void InnovationLikelihoodProblem::SetParameters( const VectorType& params )
{
	_parameters->SetParamsVec( params );
}

void InnovationLikelihoodProblem::RemoveOldestEpisode()
{
	if( episodes.empty() ) { return; }
	loss.RemoveOldestSource();
	episodes.pop_front();
}

size_t InnovationLikelihoodProblem::NumEpisodes() const
{
	return episodes.size(); 
}

KalmanFilterEpisode* 
InnovationLikelihoodProblem::GetOldestEpisode()
{
	if( episodes.empty() ) { return nullptr; }
	return &episodes.front();
}


KalmanFilterEpisode* 
InnovationLikelihoodProblem::GetCurrentEpisode()
{
	if( episodes.empty() ) { return nullptr; }
	return &episodes.back();
}

const KalmanFilterEpisode* 
InnovationLikelihoodProblem::GetCurrentEpisode() const
{
	if( episodes.empty() ) { return nullptr; }
	return &episodes.back();
}

void InnovationLikelihoodProblem::Invalidate()
{
	regularizer.Invalidate();
	BOOST_FOREACH( KalmanFilterEpisode& ep, episodes )
	{
		ep.Invalidate();
	}
}

void InnovationLikelihoodProblem::Foreprop()
{
	const std::vector<unsigned int>& activeEpsInds = loss.GetActiveInds();
	BOOST_FOREACH( const unsigned int& ind, activeEpsInds )
	{
		episodes[ind].Foreprop();
	}
	regularizer.Foreprop();
}

void InnovationLikelihoodProblem::Backprop()
{
	objective.Backprop( MatrixType::Identity(1,1) );
}

std::ostream& operator<<( std::ostream& os, const InnovationLikelihoodProblem& problem )
{
	os << "Episodes: " << std::endl;
	for( unsigned int i = 0; i < problem.episodes.size(); i++ )
	{
		os << problem.episodes[i] << std::endl;
	}

	os << "regularizer penalty: " << problem.regularizer.GetOutput() << std::endl;
	os << "mean loss: " << problem.loss.ParentCost::GetOutput() << std::endl;
	os << "stochastic objective: " << problem.objective.GetOutput() << std::endl;
	return os;
}

}