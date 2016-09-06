#pragma once

#include "relearn/RelearnInterfaces.h"

#include <unordered_map>

namespace percepto
{

class DifferenceCritic
: public PolicyCritic
{
public:

	typedef std::shared_ptr<DifferenceCritic> Ptr;

	DifferenceCritic();

	void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph );

	virtual double Evaluate( const ParamAction& act ) const;

private:

	PolicyCritic::Ptr _valueFunction;
	PolicyCritic::Ptr _baselineFunction;
};

}