#pragma once

#include <ros/ros.h>
#include "argus_utils/filters/FilterInfo.h"
#include "argus_utils/filters/FilterUtils.h"
#include "modprop/ModpropTypes.h"

namespace percepto
{

// Based on Mohamed and Schwarz 1999
class AdaptiveTransitionCovarianceEstimator
{
public:

	AdaptiveTransitionCovarianceEstimator();
	void Initialize( ros::NodeHandle& ph );

	MatrixType GetQ() const;

	void Update( const argus::PredictInfo& predict, 
	             const argus::UpdateInfo& update );

	void Reset();

private:

	unsigned int _windowLength;
	bool _useDiag;

	VectorType _prodWeights;

	std::deque<MatrixType> _delXOuterProds;
	MatrixType _lastFSpostFT;
	MatrixType _currSpost;
	MatrixType _lastF;
	MatrixType _offset;
	double _lastDt;
	MatrixType _initCov;
};

class AdaptiveObservationCovarianceEstimator
{
public:

	AdaptiveObservationCovarianceEstimator();

	void Initialize( ros::NodeHandle& ph );

	MatrixType GetR() const;

	void Update( const argus::UpdateInfo& update );

	void Reset();

private:

	unsigned int _windowLength;
	bool _useDiag;

	VectorType _prodWeights;

	std::deque<MatrixType> _innoOuterProds;
	MatrixType _lastHPHT;
	MatrixType _initCov;
};

}