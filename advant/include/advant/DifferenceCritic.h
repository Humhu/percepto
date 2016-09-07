#pragma once

#include "valu/RelearnInterfaces.h"

#include <unordered_map>

namespace percepto
{

class DifferenceCritic
: public Critic
{
public:

	typedef std::shared_ptr<DifferenceCritic> Ptr;

	DifferenceCritic();

	void Initialize( Critic::Ptr value, Critic::Ptr baseline );

	virtual double GetCritique( const ros::Time& time ) const;

private:

	Critic::Ptr _valueFunction;
	Critic::Ptr _baselineFunction;
};

}