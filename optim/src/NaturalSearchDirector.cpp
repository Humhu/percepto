#include "optim/NaturalSearchDirector.h"
#include "argus_utils/utils/ParamUtils.h"
#include <iostream>
#include <sstream>

namespace argus
{

NaturalOptimizationProblem::NaturalOptimizationProblem() {}

NaturalOptimizationProblem::~NaturalOptimizationProblem() {}

NaturalSearchDirector::NaturalSearchDirector() 
: _initialized( false ), _bufferRatio( 1.0 ), _epsilon( 1.0 ) {}

template <typename InfoType>
void NaturalSearchDirector::InitializeFromInfo( const InfoType& info )
{
	double eps, rat;
	if( GetParam( info, "epsilon", eps ) )
	{
		SetEpsilon( eps );
	}
	if( GetParam( info, "buffer_size_ratio", rat ) )
	{
		SetGradientBufferRatio( rat );
	}
}

void NaturalSearchDirector::Initialize( const ros::NodeHandle& ph )
{
	InitializeFromInfo( ph );
}

void NaturalSearchDirector::Initialize( const YAML::Node& node )
{
	InitializeFromInfo( node );
}

void NaturalSearchDirector::SetEpsilon( double eps )
{
	_epsilon = eps;
}

void NaturalSearchDirector::SetGradientBufferRatio( double r )
{
	_bufferRatio = r;
}

void NaturalSearchDirector::Reset()
{
	_initialized = false;
}

VectorType NaturalSearchDirector::ComputeSearchDirection( OptimizationProblem& problem )
{
	NaturalOptimizationProblem& naturalProblem = 
	    dynamic_cast<NaturalOptimizationProblem&>( problem );

	// Get both gradients for the problem
	VectorType naturalGradient = naturalProblem.ComputeNaturalGradient();
	VectorType gradient = naturalProblem.ComputeGradient();
	if( naturalGradient.size() != gradient.size() )
	{
		std::stringstream ss;
		ss << "Natural gradient dim " << naturalGradient.size() << 
		      " does not match gradient dim " << gradient.size();
		throw std::runtime_error( ss.str() );
	}

	if( !naturalGradient.allFinite() )
	{
		std::stringstream ss;
		ss << "Non-finite natural gradient: " << naturalGradient.transpose();
		throw std::runtime_error( ss.str() );
	}

	// First update the Fisher information matrix
	if( !_initialized )
	{
		_maxBufferSize = std::round( _bufferRatio * gradient.size() );
		_ldlt = Eigen::LDLT<MatrixType>( _epsilon * MatrixType::Identity( gradient.size(),
		                                                                  gradient.size() ) );
		_initialized = true;
	}

	_ldlt.rankUpdate( naturalGradient );
	_gradientBuffer.push_back( naturalGradient );
	while( _gradientBuffer.size() > _maxBufferSize )
	{
		_ldlt.rankUpdate( _gradientBuffer.front(), -1 );
		_gradientBuffer.pop_front();
	}

	VectorType out = _ldlt.solve( _gradientBuffer.size() * gradient );
	
	if( problem.IsMinimization() )
	{
		return -out;
	}
	else
	{
		return out;
	}
}

}