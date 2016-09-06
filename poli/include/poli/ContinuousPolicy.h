#pragma once

#include "poli/PoliCommon.h"
#include "poli/ContinuousPolicyModules.h"
#include <yaml-cpp/yaml.h>

namespace percepto
{

// Wraps a continuous policy represented by a multivariate Gaussian distribution
class ContinuousPolicy
{
public:

	struct DistributionParameters
	{
		VectorType mean;
		MatrixType info;
	};

	ContinuousPolicy();

	void Initialize( unsigned int inputDim, 
	                 unsigned int outputDim,
	                 ros::NodeHandle& info );
	void Initialize( unsigned int inputDim,
	                 unsigned int outputDim,
	                 const YAML::Node& info );

	// Create a copy of the policy module/network
	ContinuousPolicyModule::Ptr GetPolicyModule() const;

	// Retrieve the policy parameters
	percepto::Parameters::Ptr GetParameters();

	unsigned int OutputDim() const;
	unsigned int InputDim() const;

	DistributionParameters GenerateOutput( const VectorType& input );

	// Set offsets for the policy network
	void InitializeOutput( const DistributionParameters& params );

private:

	std::string _moduleType;
	ContinuousPolicyModule::Ptr _network;

	percepto::TerminalSource<VectorType> _networkInput;
	percepto::Parameters::Ptr _networkParameters;

	template <typename InfoType>
	void ReadInitialization( unsigned int inputDim,
	                         unsigned int outputDim,
	                         const InfoType& info );
};

}