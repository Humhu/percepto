#include "relearn/ValueResidualModules.h"

namespace percepto
{

BellmanResidualModule::BellmanResidualModule( ScalarFieldApproximator::Ptr estValueModule,
                                              const VectorType& prevIn,
                                              ScalarFieldApproximator::Ptr nextValueModule,
                                              const VectorType& nextIn,
                                              double reward,
                                              double gamma )
: estValue( estValueModule ),
  nextValue( nextValueModule )
{
	input.SetOutput( prevIn );
	estValue->SetInputSource( &input );

	nextInput.SetOutput( nextIn );
	nextValue->SetInputSource( &nextInput );

	discountedNextValue.SetSource( &nextValue->GetOutputSource() );
	discountedNextValue.SetScale( gamma );
	targetValue.SetSource( &discountedNextValue );
	targetValue.SetOffset( reward );

	residual.SetPlusSource( &estValue->GetOutputSource() );
	residual.SetMinusSource( &targetValue );
	loss.SetSource( &residual );
	loss.SetTarget( 0.0 );
}

void BellmanResidualModule::Foreprop()
{
	input.Foreprop();
	nextInput.Foreprop();
}

void BellmanResidualModule::Invalidate()
{
	input.Invalidate();
	nextInput.Invalidate();
}

percepto::Source<double>& BellmanResidualModule::GetOutputSource()
{
	return loss;
}

}