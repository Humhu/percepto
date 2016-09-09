#include "advant/GeneralizedAdvantageCritic.h"
#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

namespace percepto
{

GeneralizedAdvantageCritic::GeneralizedAdvantageCritic() {}

void GeneralizedAdvantageCritic::Initialize( TDErrorCritic::Ptr td,
                                             ros::NodeHandle& ph )
{
	_tdErr = td;

	GetParamRequired( ph, "lambda", _lambda );
	
	double timestep, horizon, discountRate;
	GetParamRequired( ph, "timestep", timestep );
	_timestep = ros::Duration( timestep );
	GetParamRequired( ph, "horizon", horizon );
	_horizon = ros::Duration( horizon );
	GetParamRequired( ph, "discount_rate", discountRate );
	_discountFactor = std::exp( timestep * std::log( discountRate ) );
}

double GeneralizedAdvantageCritic::GetCritique( const ros::Time& time ) const
{
	ros::Time current = time;
	double gacc = 1.0;
	double racc = 0.0;
	while( current < time + _horizon )
	{
		racc += gacc * _tdErr->GetCritique( current );
		gacc *= _lambda * _discountFactor;
		current += _timestep;
	}
	return racc;
}

}