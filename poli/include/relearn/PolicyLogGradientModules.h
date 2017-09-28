#pragma once

#include "poli/DiscretePolicyModules.h"
#include "poli/ContinuousPolicyModules.h"

#include "modprop/compo/ProductWrapper.hpp"
#include "modprop/compo/ScaleWrapper.hpp"
#include "modprop/relearn/DiscreteLogProbability.hpp"
#include "modprop/relearn/GaussianLogProbability.hpp"

namespace percepto
{

struct DiscreteLogGradientModule
{
	typedef percepto::Source<double> SourceType;
	typedef percepto::Sink<double> SinkType;

	DiscreteLogGradientModule( DiscretePolicyModule::Ptr net, 
	                           const VectorType& input,
	                           unsigned int actionIndex, 
	                           double actionAdvantage );

	percepto::TerminalSource<VectorType> networkInput;
	DiscretePolicyModule::Ptr network;
	percepto::DiscreteLogProbability logProb;
	percepto::ScaleWrapper<double> logExpectedAdvantage;

	void Foreprop();
	void Invalidate();
	SourceType* GetOutputSource();
};

struct ContinuousLogGradientModule
{
	typedef percepto::Source<double> SourceType;
	typedef percepto::Sink<double> SinkType;

	ContinuousLogGradientModule( ContinuousPolicyModule::Ptr net, 
	                             const VectorType& input,
	                             const VectorType& action, 
	                             double actionAdvantage );

	percepto::TerminalSource<VectorType> networkInput;
	ContinuousPolicyModule::Ptr network;
	percepto::GaussianLogProbability logProb;
	percepto::ScaleWrapper<double> logExpectedAdvantage;

	void Foreprop();
	void Invalidate();

	SourceType* GetOutputSource();
	SourceType* GetLogProbSource();

};

}