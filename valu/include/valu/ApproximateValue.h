#pragma once

#include "valu/ValuInterfaces.h"
#include "valu/ValueFunctionModules.h"

#include <yaml-cpp/yaml.h>

#include <lwpr.h>

namespace percepto
{

class ApproximateValue
: public ValueFunction
{
public:

	typedef std::shared_ptr<ApproximateValue> Ptr;

	ApproximateValue();

	void Initialize( unsigned int inputDim, const YAML::Node& info );
	void Initialize( unsigned int inputDim, ros::NodeHandle& info );

	// Create a copy of the value module/network
	ScalarFieldApproximator::Ptr GetApproximatorModule() const;
	
	// Retrieve the approximator parameters
	percepto::Parameters::Ptr GetParameters() const;

	virtual double GetValue( const VectorType& state ) const;

	// Set offsets for the value network
	void InitializeOutput( double out );

private:

	std::string _moduleType;
	
	mutable percepto::TerminalSource<VectorType> _approximatorInput;
	mutable ScalarFieldApproximator::Ptr _approximator;

	percepto::Parameters::Ptr _approximatorParams;

	template <typename InfoType>
	void ReadInitialization( unsigned int inputDim, const InfoType& info );

};

}