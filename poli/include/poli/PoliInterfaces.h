#pragma once

#include <ros/ros.h>
#include <memory>

#include "poli/PoliCommon.h"

namespace percepto
{

// Interface class that should be inherited by modules that will wrap continuous policies
class ContinuousPolicyInterface
{
public:

	typedef std::shared_ptr<ContinuousPolicyInterface> Ptr;

	ContinuousPolicyInterface() {};
	virtual ~ContinuousPolicyInterface() {};

	virtual unsigned int GetNumOutputs() const = 0;

	// Get the ordered parameter names
	virtual std::vector<std::string> GetParameterNames() const = 0;

	// Set each parameter to the specified parameter
	virtual void SetOutput( const VectorType& outputs ) = 0;

	virtual const VectorType& GetLowerLimits() const = 0;
	virtual const VectorType& GetUpperLimits() const = 0;
	
};

class DiscretePolicyInterface
{
public:

	typedef std::shared_ptr<DiscretePolicyInterface> Ptr;
	
	DiscretePolicyInterface();
	~DiscretePolicyInterface();

	// The number of outputs (dimensionality)
	virtual unsigned int GetNumOutputs() const = 0;

	// The number of indices for each output dimension
	virtual std::vector<unsigned int>& GetOutputSizes() const = 0;

	// Set outputs to each specified categorical index
	virtual void SetOutput( const Eigen::VectorXi& out ) = 0;

};

}