#include "valu/ValueResidualModules.h"
#include <iostream>

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

	estValue->GetOutputSource().modName = "est_value";
	input.modName = "input";
	nextValue->GetOutputSource().modName = "next_value";
	nextInput.modName = "next_input";
	discountedNextValue.modName = "discounted_next_value";
	targetValue.modName = "target_value";
	residual.modName = "residual";
	loss.modName = "loss";
}

void BellmanResidualModule::Foreprop()
{
	input.Foreprop();
	nextInput.Foreprop();
	if( !std::isfinite( residual.GetOutput() ) )
	{
		throw std::runtime_error( "Non-finite residual." );
	}

	// std::cout << "Residual: " << residual.GetOutput() << " reward: " << targetValue.GetOffset() 
	          // << " estValue: " << estValue->GetOutputSource().GetOutput() << std::endl;
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