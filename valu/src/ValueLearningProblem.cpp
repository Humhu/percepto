#include "valu/ValueLearningProblem.h"

namespace percepto
{

ApproximateValueProblem::ApproximateValueProblem() {}

void ApproximateValueProblem::Initialize( percepto::Parameters::Ptr params,
                                          double l2Weight,
                                          unsigned int sampleSize,
                                          double penaltyWeight )
{
	parameters = params;
	penaltyScale = penaltyWeight;
	regularizer.SetParameters( params );
	regularizer.SetWeight( l2Weight );
	objective.SetSourceA( &loss );
	objective.SetSourceB( &regularizer );
	loss.SetBatchSize( sampleSize );
}

bool ApproximateValueProblem::IsMinimization() const
{
	return true;
}

void ApproximateValueProblem::Resample()
{
	loss.Resample();
}

double ApproximateValueProblem::ComputeObjective()
{
	Invalidate();
	Foreprop();
	return objective.GetOutput();
}

VectorType ApproximateValueProblem::ComputeGradient()
{
	Invalidate();
	Foreprop();
	Backprop();
	return parameters->GetDerivs();
}

VectorType ApproximateValueProblem::ComputeNaturalGradient()
{
	Invalidate();
	Foreprop();
	BackpropNatural();
	return parameters->GetDerivs();
}

VectorType ApproximateValueProblem::GetParameters() const
{
	return parameters->GetParamsVec();
}

void ApproximateValueProblem::SetParameters( const VectorType& params )
{
	parameters->SetParamsVec( params );
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
	parameters->ResetAccumulators();
}

void ApproximateValueProblem::Foreprop()
{
	const std::vector<unsigned int>& inds = loss.GetActiveInds();
	BOOST_FOREACH( unsigned int ind, inds )
	{
		modules[ind].Foreprop();
		penalties[ind].Foreprop();
	}
	regularizer.Foreprop();
}

void ApproximateValueProblem::Backprop()
{
	objective.Backprop( MatrixType::Identity(1,1) );
}

void ApproximateValueProblem::BackpropNatural()
{
	const std::vector<unsigned int>& inds = loss.GetActiveInds();
	MatrixType back = MatrixType::Identity( 1, 1 ) / inds.size();
	BOOST_FOREACH( unsigned int ind, inds )
	{
		modules[ind].estValue->GetOutputSource().Backprop( back, true );
	}
}

}