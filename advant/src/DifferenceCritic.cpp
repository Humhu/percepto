#include "advant/DifferenceCritic.h"
#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

namespace percepto
{

DifferenceCritic::DifferenceCritic()
{}

void DifferenceCritic::Initialize( Critic::Ptr value, Critic::Ptr baseline )
{
	_valueFunction = value;
	_baselineFunction = baseline;
}

double DifferenceCritic::GetCritique( const ros::Time& time ) const
{
	return _valueFunction->GetCritique( time ) - _baselineFunction->GetCritique( time );
}

}