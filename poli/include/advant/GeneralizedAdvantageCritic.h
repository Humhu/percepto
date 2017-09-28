#pragma once

#include "advant/TDErrorCritic.h"
#include "valu/ValuInterfaces.h"

namespace percepto
{

// Based on work by Schulman et. al. (2015)
class GeneralizedAdvantageCritic
: public Critic
{
public:

	typedef std::shared_ptr<GeneralizedAdvantageCritic> Ptr;

	GeneralizedAdvantageCritic();

	void Initialize( TDErrorCritic::Ptr td,
	                 ros::NodeHandle& ph );

	virtual double GetCritique( const ros::Time& time ) const;

private:

	TDErrorCritic::Ptr _tdErr;
	double _lambda;
	double _discountFactor;
	ros::Duration _timestep;
	ros::Duration _horizon;
};

}