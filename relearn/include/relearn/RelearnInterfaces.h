#pragma once

#include "poli/PoliCommon.h"
#include "modprop/ModpropTypes.h"
#include <memory>
#include <ros/ros.h>

namespace percepto
{

class PolicyCritic
{
public:

	typedef std::shared_ptr<PolicyCritic> Ptr;

	PolicyCritic() {}
	virtual ~PolicyCritic() {}

	virtual double Evaluate( const ParamAction& act ) const = 0;
	virtual void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph ) = 0;

};

}