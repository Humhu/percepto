#include "covreg/AdaptiveCovarianceEstimator.h"
#include <boost/foreach.hpp>
#include "argus_utils/utils/ParamUtils.h"
#include "argus_utils/utils/MatrixUtils.h"

using namespace argus_msgs;
using namespace argus;

namespace percepto
{

AdaptiveTransitionCovarianceEstimator::AdaptiveTransitionCovarianceEstimator() 
: _lastDt( 1.0 ) {}

void AdaptiveTransitionCovarianceEstimator::Initialize( ros::NodeHandle& ph )
{
	unsigned int dim;
	GetParamRequired<unsigned int>( ph, "dim", dim );
	GetParamRequired( ph, "window_length", _windowLength );
	GetParam( ph, "use_diag", _useDiag, true );

	_initCov = MatrixType::Identity( dim, dim );
	if( !GetMatrixParam<double>( ph, "initial_cov", _initCov ) )
	{
		if( !GetDiagonalParam<double>( ph, "initial_cov", _initCov ) )
		{
			ROS_WARN_STREAM( "No initial covariance specified. Using identity.");
		}
	}

	_offset = 1E-9 * MatrixType::Identity( dim, dim );
	if( !GetMatrixParam<double>( ph, "offset", _offset ) )
	{
		if( !GetDiagonalParam<double>( ph, "offset", _offset ) )
		{
			ROS_WARN_STREAM( "No offset specified. Using 1E-9.");
		}
	}

	_currSpost = MatrixType::Zero( dim, dim );
	_lastFSpostFT = MatrixType::Zero( dim, dim );

	double decayRate;
	GetParam( ph, "decay_rate", decayRate, 1.0 );
	
	_prodWeights = VectorType( _windowLength );
	_prodWeights(0) = 1.0;
	for( unsigned int i = 1; i < _windowLength; ++i )
	{
		_prodWeights(i) = _prodWeights(i-1) * decayRate;
	}
	_prodWeights = _prodWeights / _prodWeights.sum();
}

MatrixType AdaptiveTransitionCovarianceEstimator::GetQ() const
{
	MatrixType acc = MatrixType::Zero( _currSpost.rows(), _currSpost.cols() );
	for( unsigned int i = 0; i < _delXOuterProds.size(); i++ )
	{
		acc += _delXOuterProds[i] * _prodWeights(i);
	}
	MatrixType adaptQ = acc + _currSpost + _lastFSpostFT + _offset;

	// Check for diagonal
	if( _useDiag )
	{
		adaptQ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>( adaptQ.diagonal() );
	}

	// Smoothly slew between adaptQ and initCov
	double adaptWeight = _delXOuterProds.size() / (double) _windowLength;
	double initWeight = 1.0 - adaptWeight;

	MatrixType ret = adaptWeight * adaptQ + initWeight * _initCov;
	return ret;
}

void AdaptiveTransitionCovarianceEstimator::ProcessInfo( const FilterStepInfo& msg )
{
	// NOTE We should always have a predict first
	if( msg.isPredict )
	{
		PredictInfo info = MsgToPredict( msg );
		_lastF = info.F;
		_lastDt = info.dt;
	}
	else
	{
		UpdateInfo info = MsgToUpdate( msg );
		// TODO This should really be time since last update
		MatrixType op = info.delta_x * info.delta_x.transpose() / _lastDt;
		_delXOuterProds.push_front( op );
		while( _delXOuterProds.size() > _windowLength )
		{
			_delXOuterProds.pop_back();
		}

		_lastFSpostFT = _lastF * _currSpost * _lastF.transpose();
		_currSpost = info.Spost;
	}

}

void AdaptiveTransitionCovarianceEstimator::Reset()
{
	_delXOuterProds.clear();
}


AdaptiveObservationCovarianceEstimator::AdaptiveObservationCovarianceEstimator() {}

void AdaptiveObservationCovarianceEstimator::Initialize( ros::NodeHandle& ph )
{
	unsigned int dim;
	GetParamRequired<unsigned int>( ph, "dim", dim );
	GetParamRequired( ph, "window_length", _windowLength );
	GetParam( ph, "use_diag", _useDiag, true );

	_initCov = MatrixType::Identity( dim, dim );
	if( !GetMatrixParam<double>( ph, "initial_cov", _initCov ) )
	{
		if( !GetDiagonalParam<double>( ph, "initial_cov", _initCov ) )
		{
			ROS_WARN_STREAM( "No initial covariance specified. Using identity.");
		}
	}

	double decayRate;
	GetParam( ph, "decay_rate", decayRate, 1.0 );
	
	_prodWeights = VectorType( _windowLength );
	_prodWeights(0) = 1.0;
	for( unsigned int i = 1; i < _windowLength; ++i )
	{
		_prodWeights(i) = _prodWeights(i-1) * decayRate;
	}
	_prodWeights = _prodWeights / _prodWeights.sum();
}

MatrixType AdaptiveObservationCovarianceEstimator::GetR() const
{
	MatrixType acc = MatrixType::Zero( _lastHPHT.rows(), _lastHPHT.cols() );
	for( unsigned int i = 0; i < _innoOuterProds.size(); ++i )
	{
		acc += _innoOuterProds[i] * _prodWeights(i);
	}
	MatrixType adaptR = acc + _lastHPHT;
	
	// Check for diagonal
	if( _useDiag )
	{
		adaptR = Eigen::DiagonalMatrix<double, Eigen::Dynamic>( adaptR.diagonal() );
	}
	
	// Smoothly slew between initCov and adaptR
	double adaptWeight = _innoOuterProds.size() / (double) _windowLength;
	double initWeight = 1.0 - adaptWeight;
	return adaptWeight * adaptR + initWeight * _initCov;
}

void AdaptiveObservationCovarianceEstimator::ProcessInfo( const argus_msgs::FilterStepInfo& msg )
{
	// Update is R = Cv+ + H * P+ * H^
	UpdateInfo info = MsgToUpdate( msg );
	_lastHPHT = info.H * info.Spost * info.H.transpose();

	MatrixType op = info.post_innovation * info.post_innovation.transpose();
	_innoOuterProds.push_front( op );
	while( _innoOuterProds.size() > _windowLength )
	{
		_innoOuterProds.pop_back();
	}
}

void AdaptiveObservationCovarianceEstimator::Reset()
{
	_innoOuterProds.clear();
}

}