#include "percepto/SquaredLoss.h"

namespace percepto
{

SquaredLoss::SquaredLoss( const TargetType& target, double scale )
: _target( target ), _scale( scale ) {}

SquaredLoss::OutputType SquaredLoss::Evaluate( const InputType& input ) const
{
	InputType err = input - _target;
	return _scale * err.dot( err );
}

void SquaredLoss::BackpropDerivs( const InputType& input,
                                  const BackpropInfo& nextLayers,
                                  BackpropInfo& thisLayers ) const
{
	thisLayers.sysOutDim = nextLayers.sysOutDim;
	MatrixType dody = nextLayers.dodx;
	if( dody.size() == 0 )
	{
		thisLayers.sysOutDim = OutputDim();
		dody = MatrixType::Identity( OutputDim(), OutputDim() );
	}
	else
	{
		runtime_assert( dody.cols() == OutputDim(),
		                "Backprop dimension mismatch." );
	}

	InputType err = input - _target;
	// thisLayers.dodw is empty since loss has no parameters
	thisLayers.dodx = MatrixType( thisLayers.sysOutDim, InputDim() );
	for( unsigned int i = 0; i < thisLayers.sysOutDim; i++ )
	{
		thisLayers.dodx.row(i) = dody(i) * err.transpose() * scale;
	}
}

unsigned int SquaredLoss::InputDim() const { return _target.size(); }
unsigned int SquaredLoss::OutputDim() const { return 1; }

}