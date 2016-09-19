#pragma once

#include "optim/OptimInterfaces.h"

#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Cholesky>
#include <deque>

namespace percepto
{

class NaturalOptimizationProblem
: virtual public OptimizationProblem
{
public:

	NaturalOptimizationProblem();
	virtual ~NaturalOptimizationProblem();

	// Compute the gradient with respect to the natural metric (wording?)
	virtual VectorType ComputeNaturalGradient() = 0;

};

class NaturalSearchDirector
: public SearchDirector
{
public:

	typedef std::shared_ptr<NaturalSearchDirector> Ptr;

	NaturalSearchDirector();

	void Initialize( const ros::NodeHandle& ph );
	void Initialize( const YAML::Node& node );

	void SetEpsilon( double eps );

	// Set gradient buffer size as r * # parameters
	void SetGradientBufferRatio( double r );

	virtual void Reset();

	virtual VectorType ComputeSearchDirection( OptimizationProblem& problem );

private:

	double _bufferRatio;
	double _epsilon;
	unsigned int _maxBufferSize;

	bool _initialized;
	Eigen::LDLT<MatrixType> _ldlt;
	std::deque<VectorType> _gradientBuffer;

	template <typename InfoType>
	void InitializeFromInfo( const InfoType& info );
};

}