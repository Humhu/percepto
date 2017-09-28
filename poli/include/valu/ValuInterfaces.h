#pragma once

#include "modprop/ModpropTypes.h"
#include <memory>
#include <ros/ros.h>

namespace percepto
{

class Critic
{
public:

	typedef std::shared_ptr<Critic> Ptr;

	Critic() {}
	virtual ~Critic() {}

	virtual double GetCritique( const ros::Time& time ) const = 0;
};

class ValueFunction
{
public:

	typedef std::shared_ptr<ValueFunction> Ptr;

	ValueFunction() {}
	virtual ~ValueFunction() {}

	virtual double GetValue( const VectorType& state ) const = 0;
};

}