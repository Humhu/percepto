#pragma once

#include "optim/OptimInterfaces.h"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

namespace percepto
{

class AdamSearchDirector
: public SearchDirector
{
public:

	typedef std::shared_ptr<AdamSearchDirector> Ptr;

	AdamSearchDirector();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetBeta1( double b1 );
	void SetBeta2( double b2 );
	void SetEpsilon( double eps );

	virtual void Reset();
	virtual VectorType ComputeSearchDirection( OptimizationProblem& problem );

private:

	double _beta1;
	double _beta2;
	double _epsilon;

	double _beta1Acc;
	double _beta2Acc;
	VectorType _m;
	VectorType _v;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );
};

}